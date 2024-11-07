#
import os
import sys
<<<<<<< HEAD
=======
import shutil
dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

>>>>>>> b6e4cbb (THE LOSS FUNCTION)
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, TD3, A2C, PPO
<<<<<<< HEAD
# add path home/xingru/symloco-main/symloco-mian/src

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
# print(dirname)
sys.path.append(dirname)

# Register the custom environment
register(
    id='GO2-v1',
    entry_point='mygym.envs.mujoco.unitree_go2:Go2Env',
    max_episode_steps=500,
)
=======
from src.mygym.envs.mujoco import unitree_go2


>>>>>>> b6e4cbb (THE LOSS FUNCTION)

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
<<<<<<< HEAD
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
=======
>>>>>>> b6e4cbb (THE LOSS FUNCTION)

def train(env: VecEnv, sb3_algo, modelname=''):
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return
<<<<<<< HEAD

    TIMESTEPS = 5000
=======
        
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{modelname}"): shutil.rmtree(f"{model_dir}/{modelname}")
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    TIMESTEPS = 10000
>>>>>>> b6e4cbb (THE LOSS FUNCTION)
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{model_dir}/{modelname}/{sb3_algo}_{TIMESTEPS*iters}")

def test(env: VecEnv, sb3_algo: str, path_to_model: str):
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()
    done = False
<<<<<<< HEAD
    n_envs = env.num_envs
=======
    n_envs = env.num_envs    
>>>>>>> b6e4cbb (THE LOSS FUNCTION)
    extra_steps = [500]*n_envs
    while True:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        for i, done in enumerate(dones):
            if done:
                extra_steps[i] -= 1
            if extra_steps[i] < 0:
                break

if __name__ == '__main__':
<<<<<<< HEAD
    gymenv = make_vec_env('GO2-v1', n_envs=4)
    # train(gymenv, 'SAC', modelname='GO2-exprewards')
    test(gymenv, sb3_algo='SAC', path_to_model='models/GO2-exprewards/SAC_265000.zip')
=======
    gymenv = make_vec_env('GO2-v1', n_envs=1)
    train(gymenv, 'TD3', modelname='GO2-exprewards-240906-TD3')
    # test(gymenv, sb3_algo='TD3', path_to_model='models/GO2-exprewards-240906-TD3/TD3_330000.zip')
    
>>>>>>> b6e4cbb (THE LOSS FUNCTION)
