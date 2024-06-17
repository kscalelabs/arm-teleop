import argparse
import asyncio
from collections import OrderedDict as ODict
from copy import deepcopy
from dataclasses import dataclass
import os
import time
import math
from typing import Dict, List, OrderedDict, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, ImageBackground, Plane, PointLight, Urdf


parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="5dof")
parser.add_argument("--cam", type=str, default="webcam")
args = parser.parse_args()

# ------ robots

@dataclass
class Robot:
    urdf_path: str

ROBOTS: ODict[str, Robot] = OrderedDict()
ROBOTS['5dof'] = Robot(urdf="5dof/5dof.urdf")
ROBOTS['torso'] = Robot(urdf="7dof/7dof.urdf")

assert args.robot in ROBOTS, f"robot {args.robot} not found"
robot: Robot = ROBOTS[args.robot]

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

CAMS: ODict[str, Camera] = OrderedDict()
CAMS['webcam'] = Camera(
    name="webcam",
    w=1280,
    h=720,
    c=3,
    fps=60,
    fl=1280,
    pp=(640, 360),
)

assert args.cam in CAMS, f"camera {args.cam} not found"
cam: Camera = CAMS[args.cam]
aspect_ratio: float = cam.w / cam.h
BGR_TO_RGB: NDArray = np.array([2, 1, 0], dtype=np.uint8)

print("📸 starting camera")
print(f"\t camera: {cam.name}")
print(f"\t device: {cam.device_id}")
print(f"\t resolution: {cam.w}x{cam.h}")
print(f"\t fps: {cam.fps}")

img_lock: asyncio.Lock = asyncio.Lock()
img: NDArray[cam.dtype] = np.zeros((cam.w, cam.h, cam.c), dtype=cam.dtype)

cam: cv2.VideoCapture = cv2.VideoCapture(cam.device_id)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam.w)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.h)
cam.set(cv2.CAP_PROP_FPS, cam.fps)

async def update_image() -> None:
    global cam
    if not cam.isOpened():
        raise ValueError("Camera is not available")
    start = time.time()
    ret, frame = cam.read()
    if ret:
        async with img_lock:
            global img
            img = frame[:, :, BGR_TO_RGB]
    else:
        print("failed to read frame")
    print(f"Time to update image: {time.time() - start}")

# ------ h5py

print("📦 starting h5py")

# ------ rerun

print("📊 starting rerun")

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
pb_robot_id = p.loadURDF(robot.urdf_path, [0, 0, 0], useFixedBase=True)
p.setGravity(0, 0, 0)
p.resetBasePositionAndOrientation(
    pb_robot_id,
    START_POS_TRUNK_PYBULLET,
    p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
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
    if name in START_Q:
        pb_start_q[i] = START_Q[name]
    if name in EER_CHAIN_ARM or name in EEL_CHAIN_ARM:
        pb_damping[i] = DAMPING_CHAIN
    else:
        pb_damping[i] = DAMPING_NON_CHAIN
    if name in IK_Q_LIST:
        pb_q_map[name] = i
pb_eer_id = pb_child_link_names.index(EER_LINK)
pb_eel_id = pb_child_link_names.index(EEL_LINK)
for i in range(pb_num_joints):
    p.resetJointState(pb_robot_id, i, pb_start_q[i])

# global variables get updated by various async functions
q_lock = asyncio.Lock()
q: Dict[str, float] = deepcopy(START_Q)
goal_pos_eer: NDArray = START_POS_EER_VUER
goal_orn_eer: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)
goal_pos_eel: NDArray = START_POS_EEL_VUER
goal_orn_eel: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)


async def ik(arm: str) -> None:
    start_time = time.time()
    if arm == "right":
        global goal_pos_eer, goal_orn_eer
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        pos = goal_pos_eer
        orn = goal_orn_eer
    else:
        global goal_pos_eel, goal_orn_eel
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        pos = goal_pos_eel
        orn = goal_orn_eel
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
    async with q_lock:
        global q
        for i, val in enumerate(pb_q):
            joint_name = IK_Q_LIST[i]
            if joint_name in ee_chain:
                q[joint_name] = val
                p.resetJointState(pb_robot_id, pb_q_map[joint_name], val)
    print(f"ik {arm} took {time.time() - start_time} seconds")


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

# MuJoCo and Scipy/Rerun use different quaternion conventions
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

print("🎨 starting vuer")
app = Vuer()

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
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
        print("Pinch detected in right hand")
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
    if BIMANUAL:
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
            print("Pinch detected in left hand")
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

@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    global cube_pos, cube_orn
    global hr_pos, hr_orn
    global hl_pos, hl_orn
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB_PATH,
        jointValues=env.unwrapped.q_dict,
        position=k.mj2vuer_pos(robot_pos),
        rotation=k.mj2vuer_orn(robot_orn),
        key="robot",
    )
    session.upsert @ Box(
        args=cube_size,
        position=k.mj2vuer_pos(cube_pos),
        rotation=k.mj2vuer_orn(cube_orn),
        materialType="standard",
        material=dict(color="#ff0000"),
        key="cube",
    )
    session.upsert @ Plane(
        args=TABLE_SIZE,
        position=k.mj2vuer_pos(table_pos),
        rotation=TABLE_ROT,
        materialType="standard",
        material=dict(color="#cbc1ae"),
        key="table",
    )
    session.upsert @ Sphere(
        args=SPHERE_ARGS,
        position=hr_pos,
        rotation=hr_orn,
        materialType="standard",
        material=dict(color="#0000ff"),
        key="hand_r",
    )
    if BIMANUAL:
        session.upsert @ Sphere(
            args=SPHERE_ARGS,
            position=hl_pos,
            rotation=hl_orn,
            materialType="standard",
            material=dict(color="#ff0000"),
            key="hand_l",
        )
    while True:
        await asyncio.gather(
            ik("left"),  # ~1ms
            ik("right"),  # ~1ms
            # record_h5py(),
            # record_rerun(),
            update_image(),  # ~10ms
            asyncio.sleep(1 / MAX_FPS),  # ~16ms @ 60fps
        )
        async with async_lock:
            session.upsert @ Urdf(
                jointValues=q,
                position=k.mj2vuer_pos(robot_pos),
                rotation=k.mj2vuer_orn(robot_orn),
                key="robot",
            )
        async with img_lock:
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
            session.upsert @ Box(
                position=k.mj2vuer_pos(cube_pos),
                rotation=k.mj2vuer_orn(cube_orn),
                key="cube",
            )
            session.upsert @ Sphere(
                position=hr_pos,
                rotation=hr_orn,
                key="hand_r",
            )
            if BIMANUAL:
                session.upsert @ Sphere(
                    position=hl_pos,
                    rotation=hl_orn,
                    key="hand_l",
                )