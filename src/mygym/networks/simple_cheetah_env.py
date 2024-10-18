import numpy as np
import os

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

# DEFAULT_VELOCITY_PROFILE = [
#     (2, 0.0),  # forward at 0.3m/s
#     (-2, 0.0),  # backward at 0.3m/s
# ]

DEFAULT_VELOCITY_PROFILE = {
    "freq": (0.1, 0.1), # 0.2Hz
     "mag" : (2, -0.3), # 2m/s
}


# symmetric inverted double pendulm


class SymCheetahEnv(MujocoEnv, utils.EzPickle):
    """
    ## Action Space
    The action space is a `ndarray` with shape `(6,)`

    ## Observation Space
    The observation is a `ndarray` with shape `(20,)` where the elements correspond to the following:

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
        velocity_profile=None,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._velocity_profile = (
            velocity_profile
            if velocity_profile is not None
            else DEFAULT_VELOCITY_PROFILE
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        self._time = 0
        self.target_velocity = self.get_target_velocity(self._time)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            XML_FILE_PATH,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.init_sym_structure_param()

    def get_target_velocity(self, t=0.): # -> tuple[float, float]
        assert (
            self._velocity_profile is not None and len(self._velocity_profile) > 0
        ), "Invalid velocity trajectory"
        mag = self._velocity_profile["mag"]
        freq = self._velocity_profile["freq"]
        x_d = mag[0] * np.sin(2.* np.pi * freq[0]*t)
        ry_d = mag[1] * np.sin(2.* np.pi * freq[1]*t)
        return np.array([x_d, ry_d])
        # return self._velocity_profile[1]

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def movement_reward(self, xz_velocity):
        # if self._time % 5 + self.dt > 5:
        #     self._velocity_profile.append(self._velocity_profile.pop(0))
        return self._forward_reward_weight * np.linalg.norm(
            self.target_velocity - xz_velocity
        )

    def step(self, action):
        xz_position_before = self.data.qpos[[0, 2]]
        self.do_simulation(action, self.frame_skip)
        xz_position_after = self.data.qpos[[0, 2]]
        xz_velocity = (xz_position_after - xz_position_before) / self.dt
        self.target_velocity = self.get_target_velocity(self._time)

        ctrl_cost = self.control_cost(action)

        movement_reward = self.movement_reward(xz_velocity)

        observation = self._get_obs()
        reward = - movement_reward - ctrl_cost

        self._time += self.dt

        terminated = False
        info = {
            "xz_position": xz_position_after,
            "xz_velocity": xz_velocity,
            "reward_run": -movement_reward,
            "reward_ctrl": -ctrl_cost,
            "target_velocity": self.target_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        # if self._exclude_current_positions_from_observation:
        #     position = position[1:]

        observation = np.concatenate((position, velocity, self.target_velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self._time = 0

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def init_sym_structure_param(self):
        self.restructured_feature_dim = 14  # 6 for body + 6 for leg + 2 for target vel
        self.restructured_action_dim = 3  # 3x2(left, right)

    def restruct_features_fn(self, feature):
        # feature shape [n,20]
        rootx = feature[:, [0, 9]]  # Shape [n, 2]
        rootz = feature[:, [1, 10]]  # Shape [n, 2]
        rooty = feature[:, [2, 11]]  # Shape [n, 2]

        bfoot_pos = feature[:, 3:6]  # Shape [n, 3]
        ffoot_pos = feature[:, 6:9]  # Shape [n, 3]
        bfoot_vel = feature[:, 12:15]  # Shape [n, 3]
        ffoot_vel = feature[:, 15:18]  # Shape [n, 3]

        target_vel = feature[:, 18]  # Shape [n, 2]

        feature_left = th.cat(
            [rootx, rootz, rooty, bfoot_pos, bfoot_vel, target_vel], dim=1
        )
        feature_right = th.cat(
            [-rootx, rootz, -rooty, -ffoot_pos, -ffoot_vel, -target_vel], dim=1
        )
        structured_features = th.stack([feature_left, feature_right], dim=1)
        return structured_features

    def destruct_actions_fn(self, structured_actions):  # shape [n,2,3]
        actions = th.cat(
            (structured_actions[:, 0, :], structured_actions[:, 1, :]), dim=1
        )
        return actions  # shape [n,6]
