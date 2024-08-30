import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
FREQ = 200 
A1_XML_PATH = "src/mygym/envs/mujoco/unitree_a1/scene.xml"

# init_qpos = [0,0,0.26,
#             1,0,0,0, 
#             0.3,0.8,-1.6, 
#             0.3,0.8,-1.6,
#             0.3,0.8,-1.6, 
#             0.3,0.8,-1.6]

init_qpos = [0,0,0.28,
            1,0,0,0, 
            -0.2,0.8,-1.6, 
            0.2,0.8,-1.6,
            -0.2,0.8,-1.6, 
            0.2,0.8,-1.6]
   

if __name__ == "__main__":
        
    model = mujoco.MjModel.from_xml_path(A1_XML_PATH)
    data = mujoco.MjData(model)
    mujoco_renderer = MujocoRenderer(
        model, data, DEFAULT_CAMERA_CONFIG)

    data.qpos = init_qpos

    for t in range(1000):

        mujoco.mj_step(model, data)
        mujoco_renderer.render("human")
        # data.qpos = init_qpos  
        data.ctrl = init_qpos[7:]      
       
        qpos = data.qpos[7:]
        qvel = data.qvel[6:]

        sen_qpos = data.sensordata[0:12]
        sen_qvel = data.sensordata[12:24]
        
        acc = [0.0]*3
        gyro = data.qvel[3:6]
        bpos = data.qpos[0:3]
        bquat = data.qpos[3:7]

        print("===============================")
        print("pos: --------------------------")
        print(sen_qpos)
        print(qpos)
        print("vel: --------------------------")
        print(sen_qvel)
        print(qvel)
        
        




