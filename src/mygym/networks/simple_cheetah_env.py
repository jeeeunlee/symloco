import numpy as np
import os

import torch as th
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from mygym.networks.target_velocity_generator import (
    SinusoidalVelocityGenerator,
    BiasedSinusoidalVelocityGenerator,
)


XML_FILE_PATH = os.path.join(os.path.dirname(__file__), "half_cheetah_sym.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

TOUCH_SENSOR_NOISE = 0.01


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
        velocity_profile="oneway",
        weight_run=-1.0,
        weight_ctrl=-0.1,
        weight_gait=-0.05,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            velocity_profile,
            weight_run,
            weight_ctrl,
            weight_gait,
            reset_noise_scale,
            **kwargs,
        )
        self._weight_run = weight_run
        self._weight_ctrl = weight_ctrl
        self._weight_gait = weight_gait
        self._reset_noise_scale = reset_noise_scale

        self._time = 0
        self._action_dim = 2
        # target velocity generator
        if velocity_profile == "oneway":
            self.tv_gen = BiasedSinusoidalVelocityGenerator(self._action_dim)
        elif velocity_profile == "bothway":
            self.tv_gen = SinusoidalVelocityGenerator(self._action_dim)
        else:
            self.tv_gen = BiasedSinusoidalVelocityGenerator(self._action_dim)

        self.target_velocity = self.tv_gen.get_target_velocity(self._time)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            XML_FILE_PATH,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.init_sym_structure_param()

    def reward_ctrl(self, action):
        return self._weight_ctrl * np.sum(np.square(action))

    def reward_run(self, xz_velocity):
        return self._weight_run * np.linalg.norm(self.target_velocity - xz_velocity)

    def reward_gait(self, ftouch, btouch):
        return self._weight_gait * int(
            (ftouch <= TOUCH_SENSOR_NOISE) != (btouch <= TOUCH_SENSOR_NOISE)
        )

    def step(self, action):
        xz_position_before = self.data.qpos[[0, 2]]
        self.do_simulation(action, self.frame_skip)
        xz_position_after = self.data.qpos[[0, 2]]
        xz_velocity = (xz_position_after - xz_position_before) / self.dt
        sensordata = self.data.sensordata.flat.copy()
        self.target_velocity = self.tv_gen.get_target_velocity(self._time)

        # Note: all reward weights are currently negative
        reward_ctrl = self.reward_ctrl(action)
        reward_run = self.reward_run(xz_velocity)
        reward_gait = self.reward_gait(sensordata[0], sensordata[1])

        observation = self._get_obs()
        reward = reward_run + reward_ctrl + reward_gait

        self._time += self.dt

        terminated = False
        info = {
            "xz_position": xz_position_after,
            "xz_velocity": xz_velocity,
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_gait": reward_gait,
            "command": self.target_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        sensordata = self.data.sensordata.flat.copy()

        observation = np.concatenate(
            (position, velocity, sensordata, self.target_velocity)
        ).ravel()
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
        self.restructured_feature_dim = (
            14  # 6 for body + 6 for leg + 1 for touch + 1 for target vel
        )
        self.restructured_action_dim = 3  # 3x2(left, right)

    def restruct_features_fn(
        self,
        feature: th.Tensor,  # Shape [n,18]
    ) -> th.Tensor:  # Shape [n,18]
        rootx = feature[:, [0, 9]]  # Shape [n, 2]
        rootz = feature[:, [1, 10]]  # Shape [n, 2]
        rooty = feature[:, [2, 11]]  # Shape [n, 2]
        bfoot_pos = feature[:, 3:6]  # Shape [n, 3]
        ffoot_pos = feature[:, 6:9]  # Shape [n, 3]
        bfoot_vel = feature[:, 12:15]  # Shape [n, 3]
        ffoot_vel = feature[:, 15:18]  # Shape [n, 3]

        ftouch = feature[:, 18:19]  # Shape [n, 1]
        btouch = feature[:, 19:20]  # Shape [n, 1]

        target_vel = feature[:, 20:21]  # Shape [n, 1]

        feature_left = th.cat(
            [rootx, rootz, rooty, bfoot_pos, bfoot_vel, btouch, target_vel], dim=1
        )
        feature_right = th.cat(
            [-rootx, rootz, -rooty, -ffoot_pos, -ffoot_vel, ftouch, -target_vel], dim=1
        )
        structured_features = th.stack([feature_left, feature_right], dim=1)
        return structured_features

    def destruct_actions_fn(self, structured_actions):  # shape [n,2,3]
        actions = th.cat(
            (structured_actions[:, 0, :], structured_actions[:, 1, :]), dim=1
        )
        return actions  # shape [n,6]
