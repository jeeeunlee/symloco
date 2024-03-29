import os
import sys
cwd=os.getcwd()
sys.path.append(os.path.join(cwd,"src"))

from gymnasium.envs.mujoco.mujoco_env import MujocoEnv, MuJocoPyEnv  # isort:skip
from mygym.envs.mujoco.unitree_a1 import A1Env
from gymnasium.envs.registration import register




# register A1 env
register(
    # unique identifier for the env `name-version`
    id="A1-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mygym.envs.mujoco.unitree_a1:A1Env",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)