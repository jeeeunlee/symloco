import numpy as np
from scipy.spatial.transform import Rotation as R

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class TreeEnv(MujocoEnv, utils.EzPickle):
    pass