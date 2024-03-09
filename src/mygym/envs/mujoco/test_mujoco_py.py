import mujoco
import time
import numpy as np

import mujoco_py


FREQ = 200 
xml_path = "src/mygym/envs/mujoco/unitree_a1_py/xml/a1.xml"
pymodel = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(pymodel)
viewer = mujoco_py.MjViewer(sim)
viewer._render_every_frame = True

for t in range(1000):
    
    # print(f"position: {len(data.qpos)}")
    # print(data.qpos)

    # print(f"velocity: {len(data.qvel)}")
    # print(data.qvel)
    
    qpos = sim.data.qpos[7:]
    qvel = sim.data.qvel[6:]
    
    acc = [0.0]*3
    gyro = sim.data.qvel[3:6]
    bpos = sim.data.qpos[0:3]
    bquat = sim.data.qpos[3:7]

    imu = np.concatenate((acc, gyro, bpos, bquat)).ravel()
    concat = np.concatenate((qpos, qvel, imu)).ravel()
    
    print(f"sensordata: {len(sim.data.sensordata)}")
    print(f"concat: {len(concat)}")

    print(sim.data.sensordata)
    print(concat)

    
    sim.forward()
    time.sleep(1 / FREQ)
    viewer.render() 



