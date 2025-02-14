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

UNITREE_GO2_PATH = os.path.join(os.path.dirname(__file__), "unitree_go2/scene.xml")
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
init_z = 0.05

# fmt: off
init_pos = [0, 0, 0.275 + init_z, 
            1,0,0,0, 
             -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6, 
            0.2, 0.8, -1.6]
# fmt: on

VELOCITY_PROFILE = {"freq": [0.1, 0.0, 0.1], "mag": [2, 0, 0]}


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
        weight_lin_vel=1.0,
        weight_rot_vel=0.2,
        weight_z=-50.0,
        weight_pose=-0.1,
        weight_action_rate=-0.005,
        weight_vel_z=-1.0,
        weight_stability=-0.1,
        reset_noise_scale=0.05,
        init_pos=init_pos,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            velocity_profile,
            weight_lin_vel,
            weight_z,
            weight_pose,
            weight_action_rate,
            weight_vel_z,
            weight_stability,
            reset_noise_scale,
            init_pos,
            **kwargs,
        )
        # reward weights
        self._weight_lin_vel = weight_lin_vel
        self._weight_rot_vel = weight_rot_vel
        self._weight_z = weight_z
        self._weight_pose = weight_pose
        self._weight_action_rate = weight_action_rate
        self._weight_vel_z = weight_vel_z
        self._weight_stability = weight_stability
        # noise
        self._reset_noise_scale = reset_noise_scale

        # termination conditions
        self._max_roll = np.deg2rad(10)
        self._max_pitch = np.deg2rad(10)
        self._min_z = 0.0

        # target velocity generator (returns command as [v_x, v_y, w_z])
        self.COMMAND_DIM = 3
        self._tv_gen: TargetVelocityGenerator = get_velocity_generator(
            velocity_profile
        )(self.COMMAND_DIM, freq=VELOCITY_PROFILE["freq"], mag=VELOCITY_PROFILE["mag"])

        # additional observation
        self.DIM_OBS = 48
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.DIM_OBS,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # init data
        self._pos_init = init_pos
        self.qvel_init = np.zeros(self.model.nv)
        self._revert_sign_abduction(self.qpos_init)
        self._revert_sign_abduction(self.qvel_init)

        self.reset_model()

    @property
    def target_velocity(self) -> np.ndarray:
        return self._tv_gen.get_target_velocity(self._time)

    @property
    def qpos_init(self) -> np.ndarray:
        return init_pos[7:]

    @property
    def z_init(self) -> float:
        return init_pos[2]

    ###############
    # Sensor data #
    ###############

    @property
    def sensordata(self) -> np.ndarray:
        # Only copy sensordata when new step has been observed
        if self._new_step:
            self._new_step = False
            self._sensordata = self.data.sensordata.flat.copy()
        assert self._sensordata is not None, "No valid sensordata to read"
        return self._sensordata

    @property
    def qpos(self) -> np.ndarray:
        return self.sensordata[:12]

    @property
    def qvel(self) -> np.ndarray:
        return self.sensordata[12:24]

    @property
    def qtorque(self) -> np.ndarray:
        return self.sensordata[24:36]

    @property
    def ori_rpy(self) -> np.ndarray:
        return R.from_quat(self.sensordata[36:40], scalar_first=True).as_euler("yxz")

    @property
    def rot_vel(self) -> np.ndarray:
        return self.sensordata[40:43]

    @property
    def lin_acc(self) -> np.ndarray:
        return self.sensordata[43:46]

    @property
    def pos(self) -> np.ndarray:
        return self.sensordata[46:49]

    @property
    def lin_vel(self) -> np.ndarray:
        return self.sensordata[49:52]

    ####################
    # Reward functions #
    ####################

    def _reward_lin_vel(self):
        return self._weight_lin_vel * np.exp(
            -(np.linalg.norm(self.target_velocity[:2] - self.lin_vel[:2]) ** 2)
        )

    def _reward_rot_vel(self):
        return self._weight_rot_vel * np.exp(
            -(np.linalg.norm(self.target_velocity[2] - self.rot_vel[2]) ** 2)
        )

    def _reward_z(self):
        return self._weight_z * (self.pos[-1] - self.z_init) ** 2

    def _reward_pose(self):
        return self._weight_pose * np.linalg.norm(self.qpos - self.qpos_init) ** 2

    def _reward_action_rate(self, action):
        return (
            self._weight_action_rate * np.linalg.norm(action - self._last_action) ** 2
        )

    def _reward_vel_z(self):
        return self._weight_vel_z * self.lin_vel[2] ** 2

    def _reward_stability(self):
        return self._weight_stability * np.linalg.norm(self.ori_rpy[:2]) ** 2

    def _revert_sign_abduction(self, q_value) -> None:
        # don't know why but sign reverted for abduction joint
        base_ind = 7 if len(q_value) > 12 else 0
        for abd_ind in [0, 3, 6, 9]:
            q_value[abd_ind + base_ind] *= -1

    def _get_obs(self):
        """
        **Observation space:** `Box(-Inf, Inf, (48,), float64)`

        | Index | Observation                    | Name (in XML)  | Unit             |
        | ----- | ------------------------------ | -------------- | ---------------- |
        | 0-2   | linear velocity in IMU         | frame_vel      | vel (m/s)        |
        | 3-5   | rotational velocity in IMU     | imu_gyro       | vel (rad/s)      |
        | 6-7   | roll and pitch of IMU          | imu_quat       | angle (rad)      |
        | 8-19  | joint positions                | *jointpos      | quat             |
        | 20-31 | joint velocities               | *jointvel      | vel (rad/s)      |
        | 32-43 | position commands to joints    |       -        | angle (rad)      |
        | 44-46 | velocity command v_x, v_y, w_z |       -        | vel (m/s, rad/s) |
        | 47    | height (z) command             |       -        | pos (m)          |
        """
        observation = np.concatenate(
            (
                self.lin_vel,
                self.rot_vel,
                self.ori_rpy[:2],
                self.qpos,
                self.qvel,
                self._last_action,
                self.target_velocity,
                [self.z_init],
            )
        ).ravel()
        return observation

    def _check_terminate(self):
        """
        Terminate if |roll| > max_roll, |pitch| > max_pitch, z < min_z
        """
        return (
            abs(self.ori_rpy[0]) > self._max_roll
            or abs(self.ori_rpy[1]) > self._max_pitch
            or self.lin_vel[2] < self._min_z
        )

    def step(self, action_res):
        action = self.qpos_init + action_res
        self.do_simulation(action, self.frame_skip)
        self._new_step = True
        self._time += self.dt

        # Calculate rewards
        reward_lin_vel = self._reward_lin_vel()
        reward_rot_vel = self._reward_rot_vel()
        reward_z = self._reward_z()
        reward_pose = self._reward_pose()
        reward_action_rate = self._reward_action_rate(action_res)
        reward_vel_z = self._reward_vel_z()
        reward_stability = self._reward_stability()

        observation = self._get_obs()
        reward = (
            reward_lin_vel
            + reward_rot_vel
            + reward_z
            + reward_pose
            + reward_action_rate
            + reward_vel_z
            + reward_stability
        )

        info = {
            "pos_x": self.pos[0],
            "pos_y": self.pos[1],
            "pos_z": self.pos[2],
            "vel_x": self.lin_vel[0],
            "vel_y": self.lin_vel[1],
            "vel_z": self.lin_vel[2],
            "w_x": self.rot_vel[0],
            "w_y": self.rot_vel[1],
            "w_z": self.rot_vel[2],
            "reward": reward,
            "reward_lin_vel": reward_lin_vel,
            "reward_rot_vel": reward_rot_vel,
            "reward_z": reward_z,
            "reward_pose": reward_pose,
            "reward_action_rate": reward_action_rate,
            "reward_vel_z": reward_vel_z,
            "reward_stability": reward_stability,
            "command_x": self.target_velocity[0],
            "command_y": self.target_velocity[1],
            "command_wz": self.target_velocity[2],
        }

        self._last_action = action_res

        if self.render_mode == "human":
            self.render()

        return observation, reward, self._check_terminate(), False, info

    def reset_model(self) -> np.ndarray:
        """
        Reset joint pos/vel and other utility variables
        """
        qpos = self._pos_init + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        qvel = (
            self.qvel_init
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        self._new_step = True
        self._sensordata = None
        self._time = 0.0
        self._last_action = np.zeros(self.action_space.shape)

        return self._get_obs()

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
