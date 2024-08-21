import os
dirname = os.path.dirname(__file__)
for i in range(3):
    path = os.path.abspath(dirname)
    dirname = os.path.dirname(path)
# print(dirname)
UNITREE_GO2_PATH = os.path.join(dirname, "mygym/envs/mujoco/unitree_go2/scene.xml")
# print('PATTTTT',UNITREE_GO2_PATH)
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class Go2Env(MujocoEnv, utils.EzPickle):
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
    Observations consist of positional values of different body parts of the
    cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    However, by default, the observation is a `Box(-Inf, Inf, (17,), float64)` where the elements correspond to the following:

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
    inital state: [0.   0.   0.29    pos
                  1.   0.   0.   0. quat
                  0.   0.   0.      FR
                  0.   0.   0.      FL
                  0.   0.   0.      RR
                  0.   0.   0. ]    RL
    inital observations : [0*12, 0*12, 0,0,-9.8, 0,0,0, 0,0,0.29, 1,0,0,0]
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
        balance_reward_weight=5.0,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.01,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            balance_reward_weight,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self._balance_reward_weight = balance_reward_weight
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.action_space = Box(
            low=-20.0, high=20.0, shape=(12,), dtype=np.float64
        )

    def control_cost(self, action):
        control_cost = np.exp(-self._ctrl_cost_weight * np.linalg.norm(action))
        return control_cost
    
    def forward_reward(self, x_velocity):
        forward_reward = np.exp(self._forward_reward_weight * x_velocity)
        return forward_reward
    
    def balance_reward(self):
        qw = self.data.qpos[3]
        qx = self.data.qpos[4]
        qy = self.data.qpos[5]
        qz = self.data.qpos[6]
        quat = [qx, qy, qz, qw]
        r = R.from_quat(quat).as_rotvec()
        dr = np.linalg.norm(r)
        dz = (0.29 - self.data.qpos[2])
        balance_reward = np.exp(-self._balance_reward_weight * np.linalg.norm([dr, dz]))
        return balance_reward        

    def step(self, actions):
        observation = self._get_obs()
        despos = observation[:12] + actions * self.dt
        x_position_before = self.data.qpos[0]
        self.do_simulation(despos, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(actions)
        forward_reward = self.forward_reward(x_velocity)
        balance_reward = self.balance_reward()

        observation = self._get_obs()
        reward = balance_reward * forward_reward * ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,        
        }

        if self.render_mode == "human":
            self.render()

        qw = self.data.qpos[3]
        qx = self.data.qpos[4]
        qy = self.data.qpos[5]
        qz = self.data.qpos[6]
        quat = [qx, qy, qz, qw]
        r = R.from_quat(quat).as_rotvec()
        dr = np.linalg.norm(r)
        dz = np.abs(0.29 - self.data.qpos[2])
        if dz > 0.2: 
            terminated = True
            print("dz =", dz)
        if dr > 0.8: 
            terminated = True
            print("dr =", r)

        return observation, reward, terminated, False, info
    
    def _get_sensor_data(self):
        return self.data.sensordata

    def _get_obs(self):
        position = self.data.qpos.flat.copy()#position shape: (19,)
        velocity = self.data.qvel.flat.copy()#velocity shape: (18,)
        sensordata = self.data.sensordata.flat.copy()#sensordata shape: (52,)
        # num_sensors = self.model.nsensor
        # sensor_names = self.model.sensor_names
        # for i in range(num_sensors):
        #     sensor_id = self.model.sensor_id[i]
        #     sensor_name = self.model.sensor_id2name(sensor_id)
        #     print(f"Sensor {i}: Name = {sensor_name}, ID = {sensor_id}")

        qpos = position[7:]#qpos shape: (12,)
        qvel = velocity[6:]#qvel shape: (12,)
        # imu = sensordata[36:49]#imu shape: (13,)
        imu = sensordata[36:] #imu shape: (16,)

        observation = np.concatenate((qpos, qvel, imu)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        init_qpos = [0.,0.,0.3, 1.,0.,0.,0., 
                    0.2, 0.8, -1.6,
                    -0.2, 0.8, -1.6,
                    0.2, 0.8, -1.6, 
                    -0.2, 0.8, -1.6]


        qpos = init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
