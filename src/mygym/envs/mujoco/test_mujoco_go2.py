import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
FREQ = 200 
GO2_XML_PATH = "src/mygym/envs/mujoco/unitree_go2/scene.xml"

# init_qpos = [0,0,0.35,
#                 1, 0, 0, 0, 
#                 0.3, 0.8, -1.6,
#                 0.3, 0.8, -1.6,
#                 0.3, 0.8, -1.6, 
#                 0.3, 0.8, -1.6]

init_qpos = [0,0,0.3, 1,0,0,0, 
             0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6, 
            -0.2, 0.8, -1.6]

if __name__ == "__main__":
        
    model = mujoco.MjModel.from_xml_path(GO2_XML_PATH)
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

        data.qpos = init_qpos
        
        qpos = data.qpos[7:]
        qvel = data.qvel[6:]
        
        acc = [0.0]*3
        gyro = data.qvel[3:6]
        bpos = data.qpos[0:3]
        bvel = data.qvel[0:3]
        bquat = data.qpos[3:7]

        fake_imu = np.concatenate((acc, gyro, bpos, bquat, bvel)).ravel()

        sen_qpos = data.sensordata[0:12]
        sen_qvel = data.sensordata[12:24]
        sen_qtrq = data.sensordata[24:36]

        imu_quat = data.sensordata[36:40] #4
        imu_gyro = data.sensordata[40:43] #3
        imu_acc = data.sensordata[43:46] #3
        frame_pos = data.sensordata[46:49] #3
        frame_vel = data.sensordata[49:] #3

        print("===============================")
        print("pos: --------------------------")
        print(sen_qpos)
        print(qpos)
        print("vel: --------------------------")
        print(sen_qvel)
        print(qvel)
        print("acc: --------------------------")
        print(imu_acc)
        print(acc)
        print("gyro: --------------------------")
        print(imu_gyro)
        print(gyro)
        print("imu_quat: --------------------------")
        print(imu_quat)
        print(bquat)
        print("frame_pos: --------------------------")
        print(frame_pos)
        print(bpos)
        print("frame_vel: --------------------------")
        print(frame_vel)
        print(bvel)
        
        # print(f"sensordata: {len(data.sensordata)}")
        # print(f"concat: {len(concat)}")
        # print(data.qpos)
        # print(data.sensordata)
        
        




