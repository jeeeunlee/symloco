import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
FREQ = 200 
A1_XML_PATH = "src/mygym/envs/mujoco/unitree_a1/scene.xml"

init_qpos = [0,0,0.35,1,0,0,0, 0.3,1,-1.9, 0.3,1,-1.9, 0.3,1,-1.9, 0.3,1,-1.9]

if __name__ == "__main__":
        
    model = mujoco.MjModel.from_xml_path(A1_XML_PATH)
    data = mujoco.MjData(model)
    mujoco_renderer = MujocoRenderer(
        model, data, DEFAULT_CAMERA_CONFIG)

    data.qpos = init_qpos

    for t in range(1000):

        mujoco.mj_step(model, data)
        mujoco_renderer.render("human")
        
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
        
        # print(f"sensordata: {len(data.sensordata)}")
        # print(f"concat: {len(concat)}")

        print(data.qpos)
        print(data.sensordata)
        
        




