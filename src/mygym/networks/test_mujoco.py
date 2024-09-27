import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import os
        
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
FREQ = 200 

XML_FILE_PATH = os.path.join(os.path.dirname(__file__),"half_cheetah_sym.xml")

   

if __name__ == "__main__":
        
    model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
    data = mujoco.MjData(model)
    mujoco_renderer = MujocoRenderer(
        model, data, DEFAULT_CAMERA_CONFIG)
    

    init_qpos = data.qpos.ravel().copy()
    init_qvel = data.qvel.ravel().copy()

    data.qpos = init_qpos

    for t in range(1000):

        mujoco.mj_step(model, data)
        mujoco_renderer.render("human")
        # data.qpos = init_qpos  
        # data.ctrl = init_qpos[7:]
        # data.qpos[3:] = init_qpos[3:]
        # data.qpos[0:3] = [0,0,0]
        data.qpos[3:6] = [0.1, 0.1, 0.1]
        data.qpos[6:9] = [-0.1, -0.1, -0.1]
       
        qpos = data.qpos
        qvel = data.qvel       

        print("===============================")
        print("pos: --------------------------")
        print(qpos)
        print("vel: --------------------------")
        print(qvel)
        
        




