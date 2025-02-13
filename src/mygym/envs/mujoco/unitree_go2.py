import os
from mygym.utils.target_velocity_generator import (
    get_velocity_generator,
    TargetVelocityGenerator,
    VelocityProfile,
)
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

UNITREE_GO2_PATH = os.path.join(
    os.path.dirname(__file__), "mygym/envs/mujoco/unitree_go2/scene.xml"
)
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
init_z = 0.005

# fmt: off
init_qpos = [0, 0, 0.275 + init_z, 
            1,0,0,0, 
             -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6, 
            0.2, 0.8, -1.6]
# fmt: on


class Go2Env(MujocoEnv, utils.EzPickle):
    """

    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                             | Ctrl Min | Ctrl Max | Name (in XML ) | Joint     | Unit        |
    | --- | ---------------------------------- | -------- | -------- | -------------- | --------- | ----------- |
    | 0   | Pos Cmd on Front Right hip joint   | -1.0472  | 1.0472   | FR_hip_joint   | abduction | angle (rad) |
    | 1   | Pos Cmd on Front Right thigh joint | -1.5708  | 3.4907   | FR_thigh_joint | hip       | angle (rad) |
    | 2   | Pos Cmd on Front Right calf joint  | -2.7227  |-0.83776  | FR_calf_joint  | knee      | angle (rad) |
    | 3   | Pos Cmd on Front Left hip joint    | -1.0472  | 1.0472   | FL_hip_joint   | abduction | angle (rad) |
    | 4   | Pos Cmd on Front Left thigh joint  | -1.5708  | 3.4907   | FL_thigh_joint | hip       | angle (rad) |
    | 5   | Pos Cmd on Front Left calf joint   | -2.7227  |-0.83776  | FL_calf_joint  | knee      | angle (rad) |
    | 6   | Pos Cmd on Rear Right hip joint    | -1.0472  | 1.0472   | RR_hip_joint   | abduction | angle (rad) |
    | 7   | Pos Cmd on Rear Right thigh joint  | -0.5236  | 4.5379   | RR_thigh_joint | hip       | angle (rad) |
    | 8   | Pos Cmd on Rear Right calf joint   | -2.7227  |-0.83776  | RR_calf_joint  | knee      | angle (rad) |
    | 9   | Pos Cmd on Rear Left hip joint     | -1.0472  | 1.0472   | RL_hip_joint   | abduction | angle (rad) |
    | 10  | Pos Cmd on Rear Left thigh joint   | -0.5236  | 4.5379   | RL_thigh_joint | hip       | angle (rad) |
    | 11  | Pos Cmd on Rear Left calf joint    | -2.7227  |-0.83776  | RL_calf_joint  | knee      | angle (rad) |

    ## Observation Space
    By default, the observation is a `Box(-Inf, Inf, (53,), float64)` where the elements correspond to the following:

    | Num | Action              | Ctrl Min | Ctrl Max | Name (in XML ) | Joint     | Unit        |
    | --- | --------------------| -------- | -------- | -------------- | --------- | ----------- |
    | 0~11| Pos Cmd on joint    |    -     |     -    |       -        |           | angle (rad) |
    |12~23| ang vel of joint    |   -inf   |    inf   |       -        |     -     | vel (rad/s) |
    |23~35| ang trq of joint    |   -inf   |    inf   |       -        |     -     | trq (?)     |
    |36~39| Quaternion in IMU   |   -inf   |    inf   | imu_quat       | sensor    | quat        |
    |40~42| Gyroscope in IMU    |   -inf   |    inf   | imu_gyro       | sensor    | vel (rad/s) |
    |43~45| Acclerometer in IMU |   -inf   |    inf   | imu_acc        | sensor    | acc (m/s2)  |
    |46~48| Position in IMU     |   -inf   |    inf   | frame_pos      | sensor    | pos (m)     |
    |49~52| Velocity in IMU     |   -inf   |    inf   | frame_vel      | sensor    | vel (m/s)   |

    + additional observation (previous histories)

    ## Rewards
    reward = forward_reward - ctrl_cost
    - *forward_reward*: A reward of moving forward
                        = forward_reward_weight * ( x[t+1] - x[t] ) / dt
    default dt = 5(frame_skip) * 0.01(frametime) = 0.05.
    - *ctrl_cost*: A cost for penalising large actions
        = ctrl_cost_weight * sum(action^2)
    default ctrl_cost_weight = 0.1

    ## Noise
    init_qpos = [0,0,0.3,           pos
                1,0,0,0,            quat
                0.2, 0.8, -1.6,     FR
                -0.2, 0.8, -1.6,    FL
                0.2, 0.8, -1.6,     RR
                -0.2, 0.8, -1.6]    RL

    inital observations : [0*12, 0*12,
    12 positions with a noise in the range of [-`reset_noise_scale`, `reset_noise_scale`]
    12 velocities with a standard normal noise with a mean of 0 and standard deviation of `reset_noise_scale`
    13 for IMU

    ## Episode End
    The episode truncates when the episode length is greater than 1000.

    ## Arguments
    ```python
    import gymnasium as gym
    env = gym.make('A1-v1', ctrl_cost_weight=, ...)
    ```
    | Parameter                 | Type      | Default              | Description                                                                                                                                                       |
    | ------------------------- | --------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                | **str**   | `"half_cheetah.xml"` | Path to a MuJoCo model                                                                                                                                            |
    | `forward_reward_weight`   | **float** | `1.0`                | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`        | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see section on reward)                                                                                                             |
    | `reset_noise_scale`       | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file=UNITREE_GO2_PATH,
        velocity_profile: VelocityProfile = "oneway",
        weight_balance=-10.0,
        weight_run=-2.0,
        weight_ctrl=-0.1,
        weight_safety=-0.1,
        weight_smooth=-0.1,
        reset_noise_scale=0.05,
        init_qpos=init_qpos,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            velocity_profile,
            weight_balance,
            weight_run,
            weight_ctrl,
            weight_safety,
            weight_smooth,
            reset_noise_scale,
            init_qpos,
            **kwargs,
        )
        # reward weights
        self._weight_balance = weight_balance
        self._weight_run = weight_run
        self._weight_ctrl = weight_ctrl
        self._weight_safety = weight_safety
        self._weight_smooth = weight_smooth
        # noise
        self._reset_noise_scale = reset_noise_scale

        self._time = 0
        self._action_dim = 3

        # target velocity generator
        self.tv_gen: TargetVelocityGenerator = get_velocity_generator(velocity_profile)(
            self._action_dim
        )
        self.target_velocity = self.tv_gen.get_target_velocity(self._time)

        # additional observation
        # self.dim_action = 12
        # self.prev_joint_velocity = np.zeros(12)
        # self.prev_joint_acceleration = np.zeros(12)
        self.dim_action = 6
        self.dim_obs = 52 + 24  # 76
        # init prev cmd
        # self.prev_joint_cmd = np.zeros(12)
        self.init_qpos = init_qpos
        self.init_qpos_inverted = init_qpos.copy()
        self._revert_sign_abduction(self.init_qpos_inverted)

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.dim_obs,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # self.action_space = Box(
        #     low=-3.0, high=3.0, shape=(self.dim_action,), dtype=np.float64
        # )

    def _revert_sign_abduction(self, q_value) -> None:
        # don't know why but sign reverted for abduction joint
        base_ind = 7 if len(q_value) > 12 else 0
        for abd_ind in [0, 3, 6, 9]:
            q_value[abd_ind + base_ind] *= -1

    def reward_ctrl(self, action):
        return np.exp(-self._weight_ctrl * np.linalg.norm(action))

    def reward_run(self, xyz_velocity):
        return np.exp(
            self._weight_run * np.linalg.norm(self.target_velocity - xyz_velocity)
        )

    def reward_balance(self) -> tuple[float, float, float]:
        qw = self.data.qpos[3]
        qxyz = self.data.qpos[4:7]
        r = R.from_quat([*qxyz, qw]).as_rotvec()
        dr = np.linalg.norm(r)
        dz = self.init_qpos[2] - self.data.qpos[2]
        balance_reward = np.exp(self._weight_balance * np.linalg.norm([dr, 2.5 * dz]))
        return balance_reward, dr, dz

    def reward_smooth(self):
        penalty_acc = np.linalg.norm(self.joint_acceleration)
        penalty_vel = np.linalg.norm(self.joint_velocity)
        return np.exp(self._weight_smooth * (penalty_acc + 0.1 * penalty_vel))

    def reward_safety(self):
        joint_limits = self.model.jnt_range[1:]
        qpos = self.data.sensordata[:12]
        safety_reward = 0.0
        for i, (jnt_min, jnt_max) in enumerate(joint_limits):
            dist_to_limit = np.min([jnt_max - qpos[i], qpos[i] - jnt_min]) / (
                jnt_max - jnt_min
            )
            if dist_to_limit < 0.1:
                safety_reward -= (1 - 10 * dist_to_limit) ** 2
        return np.exp(self._weight_safety * safety_reward)

    def step(self, action):
        # despos = self.prev_joint_cmd + actions * self.dt

        xyz_pos_before = self.data.qpos[:3]
        self.do_simulation(action, self.frame_skip)
        xyz_pos_after = self.data.qpos[:3]
        xyz_velocity = (xyz_pos_after - xyz_pos_before) / self.dt

        # x_position_before = self.data.qpos[0]
        # self.do_simulation(despos, self.frame_skip)
        # x_position_after = self.data.qpos[0]
        # x_velocity = (x_position_after - x_position_before) / self.dt
        self._update_prev_obs(action)
        self.target_velocity = self.tv_gen.get_target_velocity(self._time)

        reward_ctrl = self.reward_ctrl(action)
        reward_run = self.reward_run(xyz_velocity)
        reward_balance, dr, dz = self.reward_balance()
        reward_smooth = self.reward_smooth()
        reward_safety = self.reward_safety()

        observation = self._get_obs()
        reward = (
            reward_ctrl * reward_run * reward_balance * reward_smooth * reward_safety
        )

        self._time += self.dt

        info = {
            "pos": xyz_pos_after,
            "vel": xyz_velocity,
            "reward_ctrl": reward_ctrl,
            "reward_run": reward_run,
            "reward_balance": reward_balance,
            "reward_smooth": reward_smooth,
            "reward_safety": reward_safety,
            "command_x": self.target_velocity[0],
            "command_ry": self.target_velocity[1],
            "command_z": self.target_velocity[2],
        }

        if self.render_mode == "human":
            self.render()

        # termination condition
        terminated = dz > 0.25 or dr > 0.75
        if dz > 0.25:
            print(f"{dz=}")
        if dr > 0.75:
            print(f"{dr=}")

        return observation, reward, terminated, False, info

    def _update_prev_obs(self, action):
        self.joint_acceleration = (action - self.joint_velocity) / self.dt
        self.joint_velocity = action

    def _get_obs(self):
        # qpos (12,); qvel (12,); qtrq (12,); imu (16,)
        sensordata = self.data.sensordata.flat.copy()  # sensordata shape: (52,)
        observation = np.concatenate(
            (
                sensordata,
                self.joint_velocity,
                self.joint_acceleration,
                self.target_velocity,
            )
        ).ravel()
        return observation

    def reset_model(self):
        qpos = self.init_qpos_inverted + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        self.init_joint_velocity = qvel[6:].flat.copy()
        self.init_joint_cmd = qpos[7:].flat.copy()
        self._revert_sign_abduction(self.init_joint_velocity)
        self._revert_sign_abduction(self.init_joint_cmd)

        # self.prev_joint_acceleration = np.zeros(12)
        # self.prev_joint_velocity = self.init_joint_velocity.flat.copy()
        # self.prev_joint_cmd = self.init_joint_cmd.flat.copy()

        observation = self._get_obs()
        return observation

    def init_sym_structure_param(self):
        self.restructured_feature_dim = 16
        self.restructured_action_dim = 6  # 6x2(left, right)

    def restruct_features_fn(self, feature: torch.Tensor) -> torch.Tensor:
        # TODO: check which axis this should be
        # rootx = feature[:, [46, 49]]
        # rootz = feature[:, [47, ]]
        # rootx = feature[:, [0, 12]]
        # rootz = feature[:, [1, 13]]
        # rooty = feature[:, [2, 14]]
        # fr_pos = feature[:, []]
        pass
