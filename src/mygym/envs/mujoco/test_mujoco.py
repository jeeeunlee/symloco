import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
FREQ = 200 
xml_path = "src/mygym/envs/mujoco/unitree_a1/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco_renderer = MujocoRenderer(
    model, data, DEFAULT_CAMERA_CONFIG)


for t in range(1000):
    
    # print(f"position: {len(data.qpos)}")
    # print(data.qpos)

    # print(f"velocity: {len(data.qvel)}")
    # print(data.qvel)
    
    qpos = data.qpos[7:]
    qvel = data.qvel[6:]
    
    acc = [0.0]*3
    gyro = data.qvel[3:6]
    bpos = data.qpos[0:3]
    bquat = data.qpos[3:7]

    imu = np.concatenate((acc, gyro, bpos, bquat)).ravel()
    concat = np.concatenate((qpos, qvel, imu)).ravel()
    
    print(f"sensordata: {len(data.sensordata)}")
    print(f"concat: {len(concat)}")

    print(data.sensordata)
    print(concat)
    mujoco.mj_step(model, data)
    mujoco_renderer.render("human")



