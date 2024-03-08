import mujoco
import time
import numpy as np

from mujoco_py import MjSim
from mujoco_py import MjViewer

FREQ = 200 
model = mujoco.MjModel.from_xml_path("src/mygym/envs/mujoco/unitree_a1/a1.xml")
data = mujoco.MjData(model)
sim = MjSim(model)
viewer = MjViewer(sim)
viewer._render_every_frame = True

# for n in model.names:
#     print(n)


for t in range(5):
    
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
    # mujoco.mj_step(model, data)
    
    # sim.forward()
    # time.sleep(1 / FREQ)
    # viewer.render() 



