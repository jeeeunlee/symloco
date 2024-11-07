import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
GO2_XML_PATH = "src/mygym/envs/mujoco/unitree_go2/scene.xml"
INIT_PRINT = True

# init_qpos = [0,0,0.35,
#                 1, 0, 0, 0, 
#                 0.3, 0.8, -1.6,
#                 0.3, 0.8, -1.6,
#                 0.3, 0.8, -1.6, 
#                 0.3, 0.8, -1.6]
init_z = 0.05
init_qpos_reverted = [0, 0, 0.275, 1,0,0,0, 
             0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6, 
            -0.2, 0.8, -1.6]
init_qpos = [0, 0, 0.275, 1,0,0,0, 
             -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6, 
            0.2, 0.8, -1.6]
qdim = 12
timestep = 2e-3
prev_joint_pos = np.array([0]*12)
prev_sen_qpos = np.array([0]*12)

if __name__ == "__main__":
        
    model = mujoco.MjModel.from_xml_path(GO2_XML_PATH)
    data = mujoco.MjData(model)
    mujoco_renderer = MujocoRenderer(
        model, data, DEFAULT_CAMERA_CONFIG)

    data.qpos = init_qpos_reverted
    data.qpos[2] += init_z

    for t in range(10000):

        mujoco.mj_step(model, data)
        mujoco_renderer.render("human")
        
        data.ctrl = np.array(init_qpos[7:]) + np.random.uniform(
            low=-0.05, 
            high=0.05, 
            size=qdim
        )
        # data.qpos = init_qpos
        
        qpos = data.qpos[7:]
        qvel = data.qvel[6:]
        qveldiff = (data.qpos[7:] - prev_joint_pos)/timestep
        prev_joint_pos = data.qpos[7:].flat.copy()
        
        acc = [0.0]*3
        gyro = data.qvel[3:6]
        bpos = data.qpos[0:3]
        bvel = data.qvel[0:3]
        bquat = data.qpos[3:7]

        # fake_imu = np.concatenate((acc, gyro, bpos, bquat, bvel)).ravel()

        sen_qpos = data.sensordata[0:12]
        sen_qvel = data.sensordata[12:24]
        sen_qtrq = data.sensordata[24:36]
        sen_qvel_diff =  (sen_qpos - prev_sen_qpos)/timestep
        prev_sen_qpos = sen_qpos.flat.copy()

        imu_quat = data.sensordata[36:40] #4
        imu_gyro = data.sensordata[40:43] #3
        imu_acc = data.sensordata[43:46] #3
        frame_pos = data.sensordata[46:49] #3
        frame_vel = data.sensordata[49:] #3

        # if(INIT_PRINT):
        if(t%100 == 0):
            INIT_PRINT = False
            num_sensors = model.nsensor
            num_sensordata = model.nsensordata

            print("===============================")
            print(f"t={t}")
            print("pos: --------------------------")
            print(sen_qpos)
            print(qpos)
            print("vel: --------------------------")
            print(sen_qvel)
            print(sen_qvel_diff)
            print(qvel)
            # print(qveldiff)
            
            # print("acc: --------------------------")
            # print(imu_acc)
            # print(acc)
            # print("gyro: --------------------------")
            # print(imu_gyro)
            # print(gyro)
            # print("imu_quat: --------------------------")
            # print(imu_quat)
            # print(bquat)
            # print("frame_pos: --------------------------")
            # print(frame_pos)
            # print(bpos)
            # print("frame_vel: --------------------------")
            # print(frame_vel)
            # print(bvel)
        
        




