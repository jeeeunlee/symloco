import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
GO2_URDF_PATH = "/home/xingru/symloco-main/src/mygym/envs/mujoco/unitree_go2_py/urdf/go2_description.urdf"

robot_id = p.loadURDF(GO2_URDF_PATH, basePosition=[0, 0, 0.28], baseOrientation=[0, 0, 0, 1])

num_joints = p.getNumJoints(robot_id)


leg_joint_indices = {
    'FL': [2, 3, 4],  # （hip, thigh, calf）
    'FR': [8, 9, 10], # （hip, thigh, calf）
    'RL': [14, 15, 16],# （hip, thigh, calf）
    'RR': [20, 21, 22] # （hip, thigh, calf）
}


initial_joint_angles = [-0.2, 0.8, -1.6, 0.2, 0.8, -1.6, -0.2, 0.8, -1.6, 0.2, 0.8, -1.6]

# set other joints to 0
for joint_index in range(num_joints):
    p.resetJointState(robot_id, joint_index, 0.0)

# set int_angles to 12 joints
angle_index = 0
for leg, indices in leg_joint_indices.items():
    for i, joint_index in enumerate(indices):
        p.resetJointState(robot_id, joint_index, initial_joint_angles[angle_index])
        angle_index += 1

# give some time to finish the initialization
time.sleep(1)

# print all 28 joints
for joint_index in range(num_joints):
    joint_state = p.getJointState(robot_id, joint_index)
    joint_angle = joint_state[0]  #get joints_values from pybullet environment
    print(f"Joint {joint_index} angle: {joint_angle}")

# 延时1分钟
time.sleep(60)

# 断开仿真
p.disconnect()
