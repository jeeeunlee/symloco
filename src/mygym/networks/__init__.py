import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '/home/xingru/symloco-main/src'))
from gymnasium.envs.registration import register
from simple_idp_env import SymIDPEnv
from simple_cheetah_env import SymCheetahEnv
from mygym.networks.simple_cheetah_base_env import SymCheetahBaseEnv

register(
    # unique identifier for the env `name-version`
    id="simple_idp",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="simple_idp_env:SymIDPEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)

register(
    # unique identifier for the env `name-version`
    id="simple_cheetah",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="simple_cheetah_env:SymCheetahEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)


register(
    # unique identifier for the env `name-version`
    id="simple_cheetah_base",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="simple_cheetah_base_env:SymCheetahBaseEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)
