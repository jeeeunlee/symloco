import os
dirname = os.path.dirname(__file__)
for i in range(3):
    path = os.path.abspath(dirname)
    dirname = os.path.dirname(path)
# print(dirname)
UNITREE_GO2_PATH = os.path.join(dirname, "mygym/envs/mujoco/unitree_go2/scene.xml")
init_qpos = [0, 0, 0.28, 1,0,0,0, 
             -0.2, 0.8, -1.6,
            0.2, 0.8, -1.6,
            -0.2, 0.8, -1.6, 
            0.2, 0.8, -1.6]

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
        balance_reward_weight=5.0,
        forward_reward_weight=2.0,
        ctrl_cost_weight=0.01,
        safety_reward_weight=0.1,
        smooth_reward_weight=0.01,
        reset_noise_scale=0.05,
        init_qpos=init_qpos,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            balance_reward_weight,
            forward_reward_weight,
            ctrl_cost_weight,
            safety_reward_weight,
            smooth_reward_weight,
            reset_noise_scale,
            **kwargs,
        )
        # reward weights
        self._balance_reward_weight = balance_reward_weight
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._safety_reward_weight=safety_reward_weight
        self._smooth_reward_weight=smooth_reward_weight
        # noise
        self._reset_noise_scale = reset_noise_scale
        # additional observation
        self.prev_joint_velocity = np.zeros(12) 
        self.prev_joint_acceleration = np.zeros(12)
        self.dim_obs = 52 + 24 #76
        # init prev cmd
        self.prev_joint_cmd = np.zeros(12)
        self.dim_action = 12
        self.init_qpos = init_qpos
        self.init_qpos_inverted = init_qpos.copy()
        self.revert_sign_abduction(self.init_qpos_inverted)        

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

        self.action_space = Box(
            low=-20.0, high=20.0, shape=(self.dim_action,), dtype=np.float64
        )

    def revert_sign_abduction(self, q_value):
        # don't know why but sign reverted for abduction joint
        if(len(q_value)>12): base_ind = 7 
        else: base_ind=0
        for abd_ind in [0,3,6,9]:
            q_value[abd_ind+base_ind] *= -1

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
        dz = (self.init_qpos[2] - self.data.qpos[2])
        balance_reward = np.exp(-self._balance_reward_weight * np.linalg.norm([dr, dz]))
        return balance_reward, dr, dz
    
    def smooth_control_reward(self):    
        smoothness_penalty_acc = np.sum(np.square(self.joint_accleration))
        acceleration_changes = self.joint_accleration - self.prev_joint_acceleration
        smoothness_penalty_accchg = np.sum(np.square(acceleration_changes))
        smoothness_penalty = smoothness_penalty_acc + 0.1*smoothness_penalty_accchg
        # print(f"smoothness_penalty:{smoothness_penalty}")        
        smooth_reward = np.exp(-self._smooth_reward_weight*smoothness_penalty)
        # print(f"smooth_reward:{smooth_reward}")
        return smooth_reward
    
    def safety_reward(self):
            joint_limits = self.model.jnt_range[1:]

            # qpos = self.data.qpos[7:]
            qpos = self.data.sensordata[:12]
            safety_reward = 0.0  # initial safety reward

            for i, limits in enumerate(joint_limits):
                # 0 ~ 0.5
                dist_to_limit = (np.min([limits[1]-qpos[i], qpos[i]-limits[0]]))/(limits[1]-limits[0])                
                safety_reward  = 3*(dist_to_limit-0.25) # bigger distance bigger reward
            # print(f"first_Safety Reward: {safety_reward}")

            # penalize singularity when calf joint angle is zero
            calf_indices = [2,5,8,11]            
            calf_angles = qpos[calf_indices]            
            for calf_angle in calf_angles:
                if np.abs(calf_angle) < 5e-2:
                    safety_reward -= 1
            
            safety_reward = np.exp(self._safety_reward_weight * safety_reward)
            # print(f"final_Safety Reward: {safety_reward}")
            return safety_reward



    def step(self, actions):
        observation = self._get_obs()
        despos = self.prev_joint_cmd + actions*self.dt
        x_position_before = self.data.qpos[0]
        self.do_simulation(despos, self.frame_skip)
        self.prev_joint_cmd = despos.flat.copy()
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(actions)
        forward_reward = self.forward_reward(x_velocity)
        balance_reward, dr, dz = self.balance_reward()
        smooth_control_reward = self.smooth_control_reward()
        safety_reward = self.safety_reward()

        observation = self._get_obs(b_update_prev=True)
        reward = balance_reward * forward_reward * ctrl_cost * smooth_control_reward * safety_reward
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "safety_reward": safety_reward,
            "smooth_control_reward":smooth_control_reward,
        }

        if self.render_mode == "human":
            self.render()

        # termination condition
        if dz > 0.3: 
            terminated = True
            print("dz =", dz)
        if dr > 0.8: 
            terminated = True
            print("dr =", dr)

        return observation, reward, terminated, False, info
    
    def _get_sensor_data(self):
        return self.data.sensordata

    def _get_obs(self, b_update_prev=False):        
        # position = self.data.qpos.flat.copy() #position shape: (19,)
        # velocity = self.data.qvel.flat.copy() #velocity shape: (18,)
        # qpos = position[7:]#qpos shape: (12,)
        # qvel = velocity[6:]#qvel shape: (12,)

        sensordata = self.data.sensordata.flat.copy() #sensordata shape: (52,)
        qpos = sensordata[:12]#qpos shape: (12,)
        qvel = sensordata[12:24]#qvel shape: (12,)
        qtrq = sensordata[24:36]#trq shape: (12,)
        imu = sensordata[36:] #imu shape: (16,)
        self.joint_accleration = (qvel-self.prev_joint_velocity)/self.dt

        observation = np.concatenate((qpos, qvel, qtrq, imu,
                            self.prev_joint_velocity, 
                            self.prev_joint_acceleration)).ravel()
        
        # update prev_joint_velocity, prev_joint_acceleration
        if(b_update_prev):
            self.prev_joint_acceleration = self.joint_accleration
            self.prev_joint_velocity = qvel

        return observation

    def reset_model(self):        
        qpos = self.init_qpos_inverted + self.np_random.uniform(
            low=-self._reset_noise_scale, 
            high=self._reset_noise_scale, 
            size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        
        self.prev_joint_acceleration = np.zeros(12)
        self.prev_joint_velocity = qvel[6:].flat.copy()
        self.prev_joint_cmd = qpos[7:].flat.copy()
        self.revert_sign_abduction(self.prev_joint_cmd)
        observation = self._get_obs()

        return observation
