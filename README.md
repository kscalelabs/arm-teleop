<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/teleop/blob/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/kscale-onshape-library)](https://pypi.org/project/kscale-onshape-library/)
[![Discord](https://dcbadge.limes.pink/api/server/k5mSvCkYQh?style=flat)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)

</div>
<h1 align="center">
    <p>arm-teleop</p>
</h1>

---

Bi-manual robotic teleoperation system using VR hand tracking and camera streaming. Saves data to `.h5` files for training. Visualizes the robot using `rerun`.

### Setup

```bash
git clone https://github.com/kscalelabs/arm-teleop.git && cd arm-teleop
conda create -y -n arm-teleop python=3.8 && conda activate arm-teleop
pip install -r requirements.txt
```

### Usage - Demo Mode

Start the server on the robot computer.

```bash
python play.py --robot="5dof" --camera="0"
```

Start ngrok on the robot computer.

```bash
ngrok http 8012
```

Open the browser app on the HMD and go to the ngrok URL.

### Usage - Recording Data

Start the server on the robot computer.

```bash

```

🤗 [K-Scale HuggingFace Datasets](https://huggingface.co/kscalelabs)

### Dependencies

- [Vuer](https://github.com/vuer-ai/vuer) is used for visualization
- [PyBullet](https://pybullet.org/wordpress/) is used for IK
- [ngrok](https://ngrok.com/download) is used for networking
- [Rerun](https://github.com/rerun-io/rerun/) is used for visualization
- [H5Py](https://docs.h5py.org/en/stable/) is used for logging datasets
- [HuggingFace](https://huggingface.co/) is used for dataset & model storage 

### Citation

```
@misc{teleop-2024,
  title={Bi-Manual Remote Robotic Teleoperation},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/kscalelabs/teleop}
}
```
