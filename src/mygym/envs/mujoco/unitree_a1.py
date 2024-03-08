import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class A1Env(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is based on A1. The goal is to apply a torque
    on the joints to make the cheetah run forward (right) as fast as possible,
    with a positive reward allocated based on the distance moved forward and a
    negative reward allocated for moving backward. The torso and head of the
    cheetah are fixed, and the torque can only be applied on the other 6 joints
    over the front and back thighs (connecting to the torso), shins
    (connecting to the thighs) and feet (connecting to the shins).

    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                             | Ctrl Min | Ctrl Max | Name (in XML ) | Joint     | Unit        |
    | --- | ---------------------------------- | -------- | -------- | -------------- | --------- | ----------- |
    | 0   | Pos Cmd on Front Right hip joint   |-0.802851 | 0.802851 | FR_hip_joint   | abduction | angle (rad) |
    | 1   | Pos Cmd on Front Right thigh joint | -1.0472  | 4.18879  | FR_thigh_joint | hip       | angle (rad) |
    | 2   | Pos Cmd on Front Right calf joint  | -2.69653 |-0.916298 | FR_calf_joint  | knee      | angle (rad) |
    | 3   | Pos Cmd on Front Left hip joint    |-0.802851 | 0.802851 | FL_hip_joint   | abduction | angle (rad) |
    | 4   | Pos Cmd on Front Left thigh joint  | -1.0472  | 4.18879  | FL_thigh_joint | hip       | angle (rad) |
    | 5   | Pos Cmd on Front Left calf joint   | -2.69653 |-0.916298 | FL_calf_joint  | knee      | angle (rad) |
    | 6   | Pos Cmd on Rear Right hip joint    |-0.802851 | 0.802851 | RR_hip_joint   | abduction | angle (rad) |
    | 7   | Pos Cmd on Rear Right thigh joint  | -1.0472  | 4.18879  | RR_thigh_joint | hip       | angle (rad) |
    | 8   | Pos Cmd on Rear Right calf joint   | -2.69653 |-0.916298 | RR_calf_joint  | knee      | angle (rad) |
    | 9   | Pos Cmd on Rear Left hip joint     |-0.802851 | 0.802851 | RL_hip_joint   | abduction | angle (rad) |
    | 10  | Pos Cmd on Rear Left thigh joint   | -1.0472  | 4.18879  | RL_thigh_joint | hip       | angle (rad) |
    | 11  | Pos Cmd on Rear Left calf joint    | -2.69653 |-0.916298 | RL_calf_joint  | knee      | angle (rad) |

    ## Observation Space
    Observations consist of positional values of different body parts of the
    cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    However, by default, the observation is a `Box(-Inf, Inf, (17,), float64)` where the elements correspond to the following:

    | Num | Observation         | Min      | Max      | Name (in XML)  | Joint     | Unit        |
    | --- | ------------------- | -------- | -------- | -------------- | --------- | ----------- |
    | 0   | angle of FR_hip     |-0.802851 | 0.802851 | FR_hip_joint   | abduction | angle (rad) |
    | 1   | angle of FR_thigh   | -1.0472  | 4.18879  | FR_thigh_joint | hip       | angle (rad) |
    | 2   | angle of FR_calf    | -2.69653 |-0.916298 | FR_calf_joint  | knee      | angle (rad) |
    | 3   | angle of FL_hip     |-0.802851 | 0.802851 | FL_hip_joint   | abduction | angle (rad) |
    | 4   | angle of FL_thigh   | -1.0472  | 4.18879  | FL_thigh_joint | hip       | angle (rad) |
    | 5   | angle of FL_calf    | -2.69653 |-0.916298 | FL_calf_joint  | knee      | angle (rad) |
    | 6   | angle of RR_hip     |-0.802851 | 0.802851 | RR_hip_joint   | abduction | angle (rad) |
    | 7   | angle of RR_thigh   | -1.0472  | 4.18879  | RR_thigh_joint | hip       | angle (rad) |
    | 8   | angle of RR_calf    | -2.69653 |-0.916298 | RR_calf_joint  | knee      | angle (rad) |
    | 9   | angle of RL_hip     |-0.802851 | 0.802851 | RL_hip_joint   | abduction | angle (rad) |
    | 10  | angle of RL_thigh   | -1.0472  | 4.18879  | RL_thigh_joint | hip       | angle (rad) |
    | 11  | angle of RL_calf    | -2.69653 |-0.916298 | RL_calf_joint  | knee      | angle (rad) |
    | 12  | ang vel of FR_hip   |   -inf   |    inf   | FR_hip_joint   | abduction | vel (rad/s) |
    | 13  | ang vel of FR_thigh |   -inf   |    inf   | FR_thigh_joint | hip       | vel (rad/s) |
    | 14  | ang vel of FR_calf  |   -inf   |    inf   | FR_calf_joint  | knee      | vel (rad/s) |
    | 15  | ang vel of FL_hip   |   -inf   |    inf   | FL_hip_joint   | abduction | vel (rad/s) |
    | 16  | ang vel of FL_thigh |   -inf   |    inf   | FL_thigh_joint | hip       | vel (rad/s) |
    | 17  | ang vel of FL_calf  |   -inf   |    inf   | FL_calf_joint  | knee      | vel (rad/s) |
    | 18  | ang vel of RR_hip   |   -inf   |    inf   | RR_hip_joint   | abduction | vel (rad/s) |
    | 19  | ang vel of RR_thigh |   -inf   |    inf   | RR_thigh_joint | hip       | vel (rad/s) |
    | 20  | ang vel of RR_calf  |   -inf   |    inf   | RR_calf_joint  | knee      | vel (rad/s) |
    | 21  | ang vel of RL_hip   |   -inf   |    inf   | RL_hip_joint   | abduction | vel (rad/s) |
    | 22  | ang vel of RL_thigh |   -inf   |    inf   | RL_thigh_joint | hip       | vel (rad/s) |
    | 23  | ang vel of RL_calf  |   -inf   |    inf   | RL_calf_joint  | knee      | vel (rad/s) |
    | 24-6| Accleration in IMU  |   -inf   |    inf   | Body_Acc       | sensor    | acc (m/s2)  |
    | 27-9| Gyroscope in IMU    |   -inf   |    inf   | Body_Gyro      | sensor    | vel (rad/s) |
    | 30-2| Position in IMU     |   -inf   |    inf   | Body_Pos       | sensor    | pos (m)     |
    | 33-6| Quaternion in IMU   |   -inf   |    inf   | Body_Quat      | sensor    | quat        |

    
    ## Rewards
    reward = forward_reward - ctrl_cost 
    - *forward_reward*: A reward of moving forward 
                        = forward_reward_weight * ( x[t+1] - x[t] ) / dt
    default dt = 5(frame_skip) * 0.01(frametime) = 0.05. 
    - *ctrl_cost*: A cost for penalising large actions 
        = ctrl_cost_weight * sum(action^2)
    default ctrl_cost_weight = 0.1

    ## Noise
    inital state: [0.   0.   0.43    pos
                  1.   0.   0.   0. quat
                  0.   0.   0.      FR
                  0.   0.   0.      FL
                  0.   0.   0.      RR
                  0.   0.   0. ]    RL
    inital observations : [0*12, 0*12, 0,0,-9.8, 0,0,0, 0,0,0.43, 1,0,0,0]
    12 positions with a noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] 
    12 velocities with a standard normal noise with a mean of 0 and standard deviation of `reset_noise_scale` 
    13 for IMU

    ## Episode End
    The episode truncates when the episode length is greater than 1000.

    ## Arguments
    ```python
    import gymnasium as gym
    env = gym.make('unitreeA1-v1', ctrl_cost_weight=, ...)
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
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="unitree_a1/a1.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
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
    
    def _get_sensor_data(self):
        return self.data.sensordata

    def _get_obs(self):
        # observation = self.data.sensordata
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        sensordata = self.data.sensordata
        # removing floating base states and add imu data
        qpos = position[7:]
        qvel = velocity[6:]
        imu = sensordata[24:]
        observation = np.concatenate((qpos, qvel, imu)).ravel()
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

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
