import os
import sys
cwd=os.getcwd()
sys.path.append(cwd)


import mujoco
import time
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from src.mygym.networks.target_velcity_generator import (
    SinusoidalVelcocityGenerator, BiasedSinusoidalVelcocityGenerator,
)

        
class SimpleHalfCheetahEnv:
    def __init__(self,
                 velocity_profile = "oneway"):
        DEFAULT_CAMERA_CONFIG = {"distance": 4.0,}
        FREQ = 200 
        XML_FILE_PATH = os.path.join(os.path.dirname(__file__),"half_cheetah_sym.xml")

        self.dt = 1./FREQ
        self.model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
        self.data = mujoco.MjData(self.model)
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG)

        # for mirrored model rendering
        self.model_m = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
        self.data_m = mujoco.MjData(self.model_m)       
        self.mujoco_renderer_mirrored = MujocoRenderer(
            self.model_m, self.data_m, DEFAULT_CAMERA_CONFIG)
        
        self._time = 0        
        self._action_dim = 2
        
        if(velocity_profile ==  "oneway"):
            self.tv_gen = BiasedSinusoidalVelcocityGenerator(self._action_dim)
        elif(velocity_profile ==  "bothway"):
            self.tv_gen = SinusoidalVelcocityGenerator(self._action_dim)
        else:
            self.tv_gen = BiasedSinusoidalVelcocityGenerator(self._action_dim)
        self.target_velocity = self.tv_gen.get_target_velocity(self._time)

        self.init_robot(self.data, self.model)
        self.init_robot(self.data_m, self.model_m)

    def init_robot(self, 
                   data: mujoco.MjData,
                   model: mujoco.MjModel):
        data.qpos[0:3] = [0,0,0]
        data.qpos[3:6] = [0.1, 0.1, 0.1]
        data.qpos[6:9] = [-0.1, -0.1, -0.1]
        mujoco.mj_step(model, data)

    def step_target_veloccity(self,
                            obs,
                            data: mujoco.MjData,
                            model: mujoco.MjModel):
        data.qpos[0:9] = obs[0:9]
        data.qvel[0:9] = obs[9:18]
        data.qvel[0] = obs[18]
        data.qvel[2] = obs[19] 
        mujoco.mj_step(model, data)

    def step(self):
        self._time += self.dt
        self.target_velocity = self.tv_gen.get_target_velocity(self._time)
        observation = self._get_obs()
        mirrored_obs = self.get_mirrored_states(observation)

        self.step_target_veloccity(observation, self.data, self.model)
        self.step_target_veloccity(mirrored_obs, self.data_m, self.model_m)

        # mujoco.mj_step(self.model, self.data)
        # self.mujoco_renderer.render("human")
        self.mujoco_renderer_mirrored.render("human")

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        observation = np.concatenate((position, velocity, self.target_velocity)).ravel()
        return observation
    
    def get_mirrored_states(self, 
                            feature: np.ndarray): # (20,) = 9+9+2
        
        rootpx = feature[0]  # Shape [2]
        rootpz = feature[1] # Shape [2]
        rootry = feature[2] # Shape [2]        
        bfoot_pos = feature[3:6]  # Shape [3]
        ffoot_pos = feature[6:9]  # Shape [3]

        rootvx = feature[9]  # Shape [2]
        rootvz = feature[10] # Shape [2]
        rootwy = feature[11] # Shape [2]
        bfoot_vel = feature[12:15]  # Shape [3]
        ffoot_vel = feature[15:18]  # Shape [3]

        target_vel = feature[18:20]  # Shape [2]

        mirrored_obs = np.concatenate(
            [[-rootpx, rootpz, -rootry], 
                -ffoot_pos, bfoot_pos,
                [-rootvx, rootvz, -rootwy],
                -ffoot_vel, bfoot_vel,
                -target_vel])

        return mirrored_obs
    
    def restruct_features(self, feature): 
        rootx = feature[0, 9]  # Shape [2]
        rootz = feature[1, 10] # Shape [2]
        rooty = feature[2, 11] # Shape [2]
        bfoot_pos = feature[3:6]  # Shape [3]
        ffoot_pos = feature[6:9]  # Shape [3]
        bfoot_vel = feature[12:15]  # Shape [3]
        ffoot_vel = feature[15:18]  # Shape [3]
        target_vel = feature[18]  # Shape [2]

        feature_left = np.cat( [rootx, rootz, rooty, 
             bfoot_pos, bfoot_vel, target_vel], dim=1
        )
        feature_right = np.cat(
            [-rootx, rootz, -rooty, 
             -ffoot_pos, -ffoot_vel, -target_vel], dim=1
        )
        structured_features = np.stack([feature_left, feature_right], dim=1)
        return structured_features

    def destruct_actions_fn(self, structured_actions):  # shape [n,2,3]
        actions = np.cat(
            (structured_actions[:, 0, :], structured_actions[:, 1, :]), dim=1
        )
        return actions  # shape [n,6]


if __name__ == "__main__":
    
    cheetah = SimpleHalfCheetahEnv()   

   

    for t in range(1000):
        cheetah.step()
        
        
        




