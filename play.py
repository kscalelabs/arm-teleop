import argparse
import asyncio
from collections import OrderedDict as ODict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import os
import time
from typing import Dict, List, OrderedDict, Tuple
import uuid

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray
import pybullet as p
import pybullet_data
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, ImageBackground, PointLight, Urdf

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="5dof")
parser.add_argument("--camera", type=str, default="webcam")
parser.add_argument("--name", type=str, default="test", help="logging name")
args = parser.parse_args()

# ------ robots

@dataclass
class Robot:
    urdf_path: str
    bimanual: bool
    start_q: OrderedDict[str, float]
    eer_link: str = None
    eel_link: str = None
    eer_chain: List[str] = None
    eel_chain: List[str] = None
    pb_ik_q_list: List[str] = []
    # start positions for robot
    pb_start_pos: NDArray = np.array([0, 0, 0])
    pb_start_eul: NDArray = np.array([0, 0, 0])
    vuer_start_pos: NDArray = np.array([0, 0, 0])
    vuer_start_eul: NDArray = np.array([0, 0, 0])
    # start positions for end effectors
    pb_start_pos_eer: NDArray = np.array([0, 0, 0])
    pb_start_eul_eer: NDArray = np.array([0, 0, 0])
    pb_start_pos_eel: NDArray = np.array([0, 0, 0])
    pb_start_eul_eel: NDArray = np.array([0, 0, 0])

ROBOTS: OrderedDict[str, Robot] = ODict()
ROBOTS['5dof'] = Robot(
    urdf_path="5dof/5dof.urdf"
)
ROBOTS['torso'] = Robot(
    urdf_path="7dof/7dof.urdf"
)

assert args.robot in ROBOTS, f"robot {args.robot} not found"
robot: Robot = ROBOTS[args.robot]
robot_lock: asyncio.Lock = asyncio.Lock()
robot_q: Dict[str, float] = deepcopy(robot.start_q)
q_len: int = len(robot_q)
robot_pos: NDArray = robot.vuer_start_pos
robot_orn: NDArray = robot.vuer_start_eul

print("🤖 loading robot")
print(f"\t robot: {args.robot}")

# ------ camera

@dataclass
class Camera:
    name: str  # name
    w: int  # image width
    h: int  # image height
    c: int  # image channels
    fps: int # frames per second
    fl: int  # focal length
    pp: Tuple[int]  # principal point
    low: int = 0
    high: int = 255
    dtype = np.uint8
    device_id: int = 0

CAMS: OrderedDict[str, Camera] = ODict()
CAMS['webcam'] = Camera(
    name="webcam",
    w=1280,
    h=720,
    c=3,
    fps=60,
    fl=1280,
    pp=(640, 360),
)

assert args.camera in CAMS, f"camera {args.camera} not found"
camera: Camera = CAMS[args.camera]
aspect_ratio: float = camera.w / camera.h
BGR_TO_RGB: NDArray = np.array([2, 1, 0], dtype=np.uint8)

print("📸 starting camera")
print(f"\t camera: {camera.name}")
print(f"\t device: {camera.device_id}")
print(f"\t resolution: {camera.w}x{camera.h}")
print(f"\t fps: {camera.fps}")

img_lock: asyncio.Lock = asyncio.Lock()
img: NDArray = np.zeros((camera.w, camera.h, camera.c), dtype=camera.dtype)

cam_cv2: cv2.VideoCapture = cv2.VideoCapture(camera.device_id)
cam_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cam_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)
cam_cv2.set(cv2.CAP_PROP_FPS, camera.fps)

async def update_image() -> None:
    global cam_cv2
    if not cam_cv2.isOpened():
        raise ValueError("Camera is not available")
    start = time.time()
    ret, frame = cam_cv2.read()
    if ret:
        async with img_lock:
            global img
            img = frame[:, :, BGR_TO_RGB]
    else:
        print("failed to read frame")
    print(f"🕒 update_image() took {time.time() - start}s")

# ------ transformations

# PyBullet, Vuer, and Scipy/Rerun use different quaternion conventions
# https://github.com/clemense/quaternion-conventions
XYZW_2_WXYZ: NDArray = np.array([3, 0, 1, 2])
WXYZ_2_XYZW: NDArray = np.array([1, 2, 3, 0])
MJ_TO_VUER_ROT: R = R.from_euler("z", np.pi) * R.from_euler("x", np.pi / 2)
VUER_TO_MJ_ROT: R = MJ_TO_VUER_ROT.inv()

def mj2vuer_pos(pos: NDArray) -> NDArray:
    return MJ_TO_VUER_ROT.apply(pos)


def mj2vuer_orn(orn: NDArray, offset: NDArray = None) -> NDArray:
    rot = R.from_quat(orn[XYZW_2_WXYZ]) * MJ_TO_VUER_ROT
    if offset is not None:
        rot = R.from_quat(offset[XYZW_2_WXYZ]) * rot
    return rot.as_euler("xyz")


def vuer2mj_pos(pos: NDArray) -> NDArray:
    return VUER_TO_MJ_ROT.apply(pos)


def vuer2mj_orn(orn: R) -> NDArray:
    rot = orn * VUER_TO_MJ_ROT
    return rot.as_quat()[WXYZ_2_XYZW]

# ------ pybullet (used for ik)

HEADLESS: bool = True
# damping determines which joints are used for ik
DAMPING_CHAIN: float = 0.1
DAMPING_NON_CHAIN: float = 10.0

print("🔫 starting pybullet")
print(f"\t headless: {HEADLESS}")
if HEADLESS:
    clid = p.connect(p.DIRECT)
else:
    clid = p.connect(p.SHARED_MEMORY)
    if clid < 0:
        p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_robot_id = p.loadURDF(robot.urdf_path, robot.pb_start_pos, useFixedBase=True)
p.setGravity(0, 0, 0)
p.resetBasePositionAndOrientation(
    pb_robot_id,
    robot.pb_start_pos,
    p.getQuaternionFromEuler(robot.pb_start_eul),
)
pb_num_joints: int = p.getNumJoints(pb_robot_id)
print(f"\t number of joints: {pb_num_joints}")
pb_joint_names: List[str] = [""] * pb_num_joints
pb_child_link_names: List[str] = [""] * pb_num_joints
pb_joint_upper_limit: List[float] = [0.0] * pb_num_joints
pb_joint_lower_limit: List[float] = [0.0] * pb_num_joints
pb_joint_ranges: List[float] = [0.0] * pb_num_joints
pb_start_q: List[float] = [0.0] * pb_num_joints
pb_damping: List[float] = [0.0] * pb_num_joints
pb_q_map: Dict[str, int] = {}
for i in range(pb_num_joints):
    info = p.getJointInfo(pb_robot_id, i)
    name = info[1].decode("utf-8")
    pb_joint_names[i] = name
    pb_child_link_names[i] = info[12].decode("utf-8")
    pb_joint_lower_limit[i] = info[9]
    pb_joint_upper_limit[i] = info[10]
    pb_joint_ranges[i] = abs(info[10] - info[9])
    if name in robot.start_q:
        pb_start_q[i] = robot.start_q[name]
    if name in robot.pb_ik_q_list:
        pb_q_map[name] = i
    if name in robot.eer_chain:
        pb_damping[i] = DAMPING_CHAIN
    elif robot.bimanual and name in robot.eel_chain:
        pb_damping[i] = DAMPING_CHAIN
    else:
        pb_damping[i] = DAMPING_NON_CHAIN
    p.resetJointState(pb_robot_id, i, pb_start_q[i])

action_lock: asyncio.Lock = asyncio.Lock()
pb_eer_id: int = pb_child_link_names.index(robot.eer_link)
goal_pos_eer: NDArray = robot.pb_start_pos_eer
goal_orn_eer: NDArray = p.getQuaternionFromEuler(robot.pb_start_eul_eer)
if robot.bimanual:
    pb_eel_id: int = pb_child_link_names.index(robot.eel_link)
    goal_pos_eel: NDArray = robot.pb_start_pos_eel
    goal_orn_eel: NDArray = p.getQuaternionFromEuler(robot.pb_start_eul_eel)


async def ik(arm: str) -> None:
    _start: float = time.time()
    if arm == "right":
        global goal_pos_eer, goal_orn_eer
        ee_id = pb_eer_id
        ee_chain = robot.eer_chain
        pos = goal_pos_eer
        orn = goal_orn_eer
    elif robot.bimanual and arm == "left":
        global goal_pos_eel, goal_orn_eel
        ee_id = pb_eel_id
        ee_chain = robot.eel_chain
        pos = goal_pos_eel
        orn = goal_orn_eel
    else:
        raise ValueError(f"arm {arm} not found")
    # print(f"ik {arm} {pos} {orn}")
    pb_q = p.calculateInverseKinematics(
        pb_robot_id,
        ee_id,
        pos,
        orn,
        pb_joint_lower_limit,
        pb_joint_upper_limit,
        pb_joint_ranges,
        pb_start_q,
    )
    async with robot_lock:
        global robot_q
        for i, val in enumerate(pb_q):
            joint_name = robot.pb_ik_q_list[i]
            if joint_name in ee_chain:
                robot_q[joint_name] = val
                p.resetJointState(pb_robot_id, pb_q_map[joint_name], val)
    print(f"🕒 ik({arm}) took {(time.time() - _start) * 1000:.2f}ms")


# ------ vuer

MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0
VUER_IMG_QUALITY: int = 20
VUER_CAM_DISTANCE: int = 5
VUER_IMAGE_PLANE_POS: NDArray = np.array([0, 0, -10])
VUER_IMAGE_PLANE_EUL: NDArray = np.array([0, 0, 0])
# Vuer hand tracking and pinch detection params
HAND_FPS: int = 30
FINGER_INDEX: int = 9
FINGER_THUMB: int = 4
FINGER_MIDLE: int = 14
FINGER_PINKY: int = 24
PINCH_OPEN: float = 0.10  # 10cm
PINCH_CLOSE: float = 0.01  # 1cm

print("🎨 starting vuer")
vuer_app = Vuer()

@vuer_app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    _start: float = time.time()
    global hr_pos, hr_orn, eer_pos, eer_orn, grip_r, reset
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_INDEX])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_THUMB])
    # orientation is calculated from wrist rotation matrix
    rwrist_orn: NDArray = np.array(event.value["rightHand"])
    rwrist_orn = rwrist_orn.reshape(4, 4)[:3, :3]
    rwrist_orn = R.from_matrix(rwrist_orn).as_euler("xyz")
    # index finger to thumb pinch turns on tracking
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    if rpinch_dist < PINCH_CLOSE:
        print("👌 pinch detected in right hand")
        # pinching with middle finger controls gripper
        rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_MIDLE])
        rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_OPEN
        # async with async_lock:
        #     global hr_pos, hr_orn, eer_pos, eer_orn, grip_r
        eer_pos = np.clip(hr_pos - rthumb_pos, -1, 1)
        print(f"eer_pos action {eer_pos}")
        eer_orn = np.clip(hr_orn - rwrist_orn, -1, 1)
        print(f"eer_orn action {eer_orn}")
        grip_r = rgrip_dist
        print(f"grip_r action {grip_r}")
    # pinky to thumb resets the environment (starts recording new episode)
    rpinky_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_PINKY])
    rpinky_dist: NDArray = np.linalg.norm(rthumb_pos - rpinky_pos)
    if rpinky_dist < PINCH_CLOSE:
        print("Reset detected in right hand")
        # async with async_lock:
        #     global reset, hr_pos, hr_orn
        reset = True
        # reset the hand indicator to the pinky
        hr_pos = rthumb_pos
        hr_orn = rwrist_orn
    if robot.bimanual:
        global hl_pos, hl_orn, eel_pos, eel_orn, grip_l
        # left hand
        lindex_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_INDEX])
        lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_THUMB])
        lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
        # orientation is calculated from wrist rotation matrix
        lwrist_orn: NDArray = np.array(event.value["leftHand"])
        lwrist_orn = lwrist_orn.reshape(4, 4)[:3, :3]
        lwrist_orn = R.from_matrix(lwrist_orn).as_euler("xyz")
        # index finger to thumb pinch turns on tracking
        if lpinch_dist < PINCH_CLOSE:
            print("👌 pinch detected in left hand")
            # pinching with middle finger controls gripper
            lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_MIDLE])
            lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_OPEN
            # async with async_lock:
            #     global hl_pos, hl_orn, eel_pos, eel_orn, grip_l
            eel_pos = np.clip(hl_pos - lthumb_pos, -1, 1)
            print(f"eel_pos action {eel_pos}")
            eel_orn = np.clip(hl_orn - lwrist_orn, -1, 1)
            print(f"eel_orn action {eel_orn}")
            grip_l = lgrip_dist
            print(f"grip_l action {grip_l}")
        # pinky to thumb resets the environment (starts recording new episode)
        lpinky_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_PINKY])
        lpinky_dist: NDArray = np.linalg.norm(lthumb_pos - lpinky_pos)
        if lpinky_dist < PINCH_CLOSE:
            print("Reset detected in left hand")
            # async with async_lock:
            #     global hl_pos
            # reset the hand indicator
            hl_pos = lthumb_pos
            hl_orn = lwrist_orn
    print(f"🕒 hand_handler() took {(time.time() - _start) * 1000:.2f}ms")

# ------ h5py

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
DATE_FORMAT: str = "%mm%dd%Yy_%Hh%Mm"
H5PY_CHUNK_SIZE_BYTES: int = 1024**2 * 2
MAX_EPISODE_STEPS: int = 64

logdir_name: str = "{}.{}.{}".format(
    args.name,
    str(uuid.uuid4())[:6],
    datetime.now().strftime(DATE_FORMAT),
)
logdir_path = os.path.join(DATA_DIR, logdir_name)
os.makedirs(logdir_path, exist_ok=True)

data_lock: asyncio.Lock = asyncio.Lock()
reset_h5py: bool = True
episode_idx: int = 0
step: int = 0
f: h5py.File = None

print("📦 starting h5py")
print(f"\t logdir: {logdir_path}")
print(f"\t max episode steps: {MAX_EPISODE_STEPS}")

async def record_h5py() -> None:
    _start: float = time.time()
    global f, episode_idx, step, reset_h5py, img, robot_q
    if reset_h5py:
        async with data_lock:
            if f is not None:
                f.close()
                episode_idx += 1
            log_path: str = os.path.join(logdir_path, f"episode_{episode_idx}.hdf5")
            f = h5py.File(log_path, "w", rdcc_nbytes=H5PY_CHUNK_SIZE_BYTES)
            print(f"📝 new h5py file {log_path}")
            f.attrs["robot"] = robot
            f.attrs["camera"] = camera
            f.create_group("observations/images")
            f.create_dataset("observations/q_pos", (MAX_EPISODE_STEPS, q_len))
            f.create_dataset("observations/q_vel", (MAX_EPISODE_STEPS, q_len))
            f.create_dataset("action/goal_pos_eer", (MAX_EPISODE_STEPS, 3))
            f.create_dataset("action/goal_orn_eer", (MAX_EPISODE_STEPS, 4))
            f.create_dataset("action/grip_r", (MAX_EPISODE_STEPS, 1))
            if robot.bimanual:
                f.create_dataset("action/goal_pos_eel", (MAX_EPISODE_STEPS, 3))
                f.create_dataset("action/goal_orn_eel", (MAX_EPISODE_STEPS, 4))
                f.create_dataset("action/grip_l", (MAX_EPISODE_STEPS, 1))
            g = f.create_group(f"metadata/{camera.name}")
            g.attrs["resolution"] = [camera.w, camera.h]
            g.attrs["focal_length"] = camera.fl
            g.attrs["principal_point"] = camera.pp
            g.attrs["fps"] = camera.fps
            f.create_dataset(
                f"/observations/images/{camera.name}",
                (MAX_EPISODE_STEPS, camera.h, camera.w, camera.c),
                dtype=camera.dtype,
                chunks=(1, camera.h, camera.w, camera.c),
            )
            reset_h5py = False
    if f is not None:
        async with data_lock:
            id: int = info["step"] - 1
            async with action_lock:
                f["action/goal_pos_eer"][id] = goal_pos_eer
                f["action/goal_orn_eer"][id] = goal_orn_eer
                f["action/grip_r"][id] = grip_r
                if robot.bimanual:
                    f["action/goal_pos_eel"][id] = goal_pos_eel
                    f["action/goal_orn_eel"][id] = goal_orn_eel
                    f["action/grip_l"][id] = grip_l
            async with robot_lock:
                f["observations/q_pos"][id] = robot_q
                f["observations/q_vel"][id] = robot_q
            async with img_lock:
                f[f"/observations/images/{camera.name}"][id] = img
            f.flush()
            step += 1
    print(f"🕒 record_h5py() took {(time.time() - _start) * 1000:.2f}ms")
    return None

# ------ rerun


# Blueprint stores the GUI layout for ReRun
blueprint: rrb.Blueprint = None
reset_rr: bool = True

print("📊 starting rerun")

async def record_rerun() -> None:
    _start: float = time.time()
    global blueprint, reset_rr
    if blueprint is None:
        time_series_views: List[rrb.SpaceView] = []
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_pos", name="q_pos"),
        )
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_vel", name="q_vel"),
        )
        time_series_views.append(
            rrb.TimeSeriesView(origin="/action", name="action"),
        )
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        origin="/world",
                        name="scene",
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            origin=camera.name,
                            name=camera.name,
                        )
                    ),
                ),
                rrb.Vertical(*time_series_views),
            ),
        )
        rr.init(robot.name, default_blueprint=blueprint)
    if reset_rr:
        log_path: str = os.path.join(logdir_path, f"episode_{episode_idx}.rrd")
        rr.save(log_path, default_blueprint=blueprint)
        rr.send_blueprint(blueprint=blueprint)
        print(f"📝 new rerun file {log_path}")
        reset_rr = False
    rr.log(
        f"world/camera/{camera.name}",
        rr.Pinhole(
            resolution=[camera.w, camera.h],
            focal_length=camera.fl,
            principal_point=camera.pp,
        ),
    )
    rr.set_time_seconds("cpu_time", time.time())
    rr.set_time_sequence("episode", episode_idx)
    rr.set_time_sequence("step", step)
    rr.log(f"state/q_pos/{key}", rr.Scalar(observation["q_pos"][i]))
    rr.log(f"state/q_vel/{key}", rr.Scalar(observation["q_vel"][i]))
    rr.log(
        "world/eer",
        rr.Transform3D(
            translation=action["eer_pos"],
            # rotation=rr.Quaternion(xyzw=action["eer_orn"][k.WXYZ_2_XYZW]),
        ),
    )
    rr.log("action/grip_r", rr.Scalar(action["grip_r"]))
    if robot.bimanual:
        rr.log(
            "world/eel",
            rr.Transform3D(
                translation=action["eel_pos"],
                # rotation=rr.Quaternion(xyzw=action["eel_orn"][k.WXYZ_2_XYZW]),
            ),
        )
        rr.log("action/grip_l", rr.Scalar(action["grip_l"]))
    for cam in info["cameras"]:
        rr.log(cam.log_name, rr.Image(observation[cam.log_name]))
        rr.log(
            f"world/{cam.name}",
            rr.Transform3D(
                translation=self.mj_env.physics.data.camera(cam.name).xpos,
                rotation=rr.Quaternion(xyzw=_quat[k.WXYZ_2_XYZW]),
            ),
        )
    print(f"🕒 record_rerun() took {(time.time() - _start) * 1000:.2f}ms")
    return None

# ------ main loop

@vuer_app.spawn(start=True)
async def main(session: VuerSession):
    global q
    global cube_pos, cube_orn
    global hr_pos, hr_orn
    global hl_pos, hl_orn
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=robot.urdf_path,
        jointValues=robot.start_q,
        position=robot_pos,
        rotation=robot_orn,
        key="robot",
    )
    print("🚀 starting main loop")
    while True:
        await asyncio.gather(
            ik("left"),  # ~1ms
            ik("right"),  # ~1ms
            record_h5py(), # ~10ms
            record_rerun(), # ~10ms
            update_image(),  # ~10ms
            asyncio.sleep(1 / MAX_FPS),  # ~16ms @ 60fps
        )
        async with robot_lock:
            _start: float = time.time()
            session.upsert @ Urdf(
                jointValues=robot_q,
                position=robot_pos,
                rotation=robot_orn,
                key="robot",
            )
            print(f"🕒 URDF upsert took {(time.time() - _start) * 1000:.2f}ms")
        async with img_lock:
            _start: float = time.time()
            session.upsert(
                ImageBackground(
                    img,
                    format="jpg",
                    quality=VUER_IMG_QUALITY,
                    interpolate=True,
                    fixed=True,
                    aspect=aspect_ratio,
                    distanceToCamera=VUER_CAM_DISTANCE,
                    position=VUER_IMAGE_PLANE_POS,
                    rotation=VUER_IMAGE_PLANE_EUL,
                    key="video",
                ),
                to="bgChildren",
            )
            print(f"🕒 ImageBackground upsert took {(time.time() - _start) * 1000:.2f}ms")