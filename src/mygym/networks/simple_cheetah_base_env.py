#this is used for base representation in left& right frame
#also the code is set up the target velocity make robot follow it
import numpy as np
import os
import mujoco
import torch as th
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


XML_FILE_PATH = os.path.join(os.path.dirname(__file__), "half_cheetah_sym.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


# symmetric inverted double pendulum

class SymCheetahBaseEnv(MujocoEnv, utils.EzPickle):
    """
    ## Action Space
    The action space is a `ndarray` with shape `(12,)`
    the joint position of left and right legs totally 6 joints

    ## Observation Space
    The observation is a `ndarray` with shape `(30,)` where the elements correspond to the following:
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        # target_qvel=2.0,#set the target velovity is 2m/s along the x-axis positive direction
        target_qvel=-2.0,#set the target velovity is 2m/s along the x-axis positive direction

        velocity_reward_weight=10,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self.target_qvel=target_qvel
        self.velocity_reward_weight=velocity_reward_weight
        # self.init_qpos=[0,0,0.35,0.0,0.0,0.0000,-0.,-0.0,-0.0]
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            XML_FILE_PATH,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.init_sym_structure_param()

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        #this is the base position if we use the left and right representation it should be 
        x_position_before = self.data.qpos[0]
        print(self.data.qpos[1])
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    def is_fallen(self):
        """
        Check if the robot has fallen based on its height or orientation.
        For example, if the z position of the base is too low or the pitch angle is too large.
        """
        position = self.data.qpos.flat.copy()
        height = abs(position[1]) # the second is z-axis(should pay attention to
        print(height)

        
        if height < 0.01:  # If height is below 0.001 meters, consider it as fallen
            return True

        return False

    def velocity_reward(self):
        current_velocity=self.data.qvel.flat.copy()[0]
        reward=self.velocity_reward_weight*(abs(current_velocity)-abs(self.target_qvel))
        return reward
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # print("position",position)
        # Transform base information to each leg frame
        base_angular_velocity = velocity[0:3]  # [rx_dot, rz_dot, ry_dot]
        base_linear_acceleration = self.data.qacc[0:3]  # [x_ddot, z_ddot, y_ddot]
        gravity_vector = np.array([0, 0, -9.81])

        # Now we have 2 legs , we transform base info for each
        # transformed_base_info = []
        observation_info = []
        for leg_idx in range(2):
            Rib = self.get_leg_rotation_matrix(leg_idx)  # Rotation matrix from base to leg frame
            angular_velocity_leg = Rib @ base_angular_velocity
            linear_acceleration_leg = Rib @ base_linear_acceleration
            gravity_leg = Rib @ gravity_vector
            # pi = Rib @ (self.data.qpos[:3] - self.data.qpos[3*(leg_idx+1):3*(leg_idx+2)])
            # #this represent the position of the base in joint frame
            # transformed_base_info.extend([angular_velocity_leg, linear_acceleration_leg, gravity_leg, pi])
            leg_joint_angles = position[3*(leg_idx+1):3*(leg_idx+2)]
            leg_joint_velocities = velocity[3*(leg_idx+1):3*(leg_idx+2)] 
            observation_info.extend([leg_joint_angles,leg_joint_velocities,angular_velocity_leg, linear_acceleration_leg, gravity_leg])
            
            # transformed_base_info.extend([angular_velocity_leg, linear_acceleration_leg, gravity_leg])
        observation = np.concatenate(observation_info).ravel()
        # # transformed_base_info = np.concatenate(transformed_base_info)
        # leg_joint_angles = position[3:9]  
        # leg_joint_velocities = velocity[3:9] 

        # observation = np.concatenate((leg_joint_angles, leg_joint_velocities, transformed_base_info)).ravel()
        return observation

    def reset_model(self):
        # ground_geom_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # # Get the position of the ground geom
        # ground_height = model.geom_pos[ground_geom_index][2]  # The z-coordinate represents the height

        # print(f"Ground height: {ground_height}")
        # self.init_qpos=[0,0,0.7,0.1,0.1,0.1,-0.1,-0.1,-0.1]

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        #reset the target value sample from the 1.5m/s~2.5m/s
        self.target_qvel=self.target_qvel-self._reset_noise_scale * self.np_random.uniform(-5, 5, self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    def init_sym_structure_param(self):
        self.restructured_feature_dim = 12 # 6 for body + 6 for legs
        self.restructured_action_dim = 6 # 3x2(for four legs)

    def get_leg_rotation_matrix(self, leg_idx):
        # Placeholder function: return appropriate rotation matrix for each leg
        # now only use the unit matrix
        #to-do replace it with the real one from the enironment
        #the order of the along the x-axis the ffoot is one the right
        Rib = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]])
        if leg_idx in [2]:  # 
            Rib[1, 2] = -1  # Flip y-axis for fronyt legs to maintain symmetry
        return Rib
    

    # def restruct_features_fn(self, feature):
    #     # feature shape [n, 24]
    #     rootx = feature[:, [0, 12]]  # Shape [n, 2]
    #     rootz = feature[:, [1, 13]]  # Shape [n, 2]
    #     rooty = feature[:, [2, 14]]  # Shape [n, 2]

    #     bfoot_pos = feature[:, 3:6]  # Shape [n, 3]
    #     ffoot_pos = feature[:, 6:9]  # Shape [n, 3]
    #     bfoot_vel = feature[:, 15:18]  # Shape [n, 3]
    #     ffoot_vel = feature[:, 18:21]  # Shape [n, 3]

    #     feature_left = th.cat([rootx, rootz, rooty, 
    #                            bfoot_pos, bfoot_vel], dim=1)
    #     feature_right = th.cat([-rootx, rootz, -rooty, 
    #                             -ffoot_pos, -ffoot_vel], dim=1)
    #     structured_features = th.stack([feature_left, feature_right], dim=1)
    #     return structured_features

    # def destruct_actions_fn(self, structured_actions): # shape [n, 4, 3]
    #     actions = th.cat((structured_actions[:, 0, :], 
    #                       structured_actions[:, 1, :],
    #                       structured_actions[:, 2, :],
    #                       structured_actions[:, 3, :]), dim=1)
    #     return actions # shape [n, 12]
