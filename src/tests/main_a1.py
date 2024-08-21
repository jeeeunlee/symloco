import os
import sys
# add path home/jelee/my_ws/RL/symloco/src
dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
# print(dirname)
sys.path.append(dirname)
import mygym.envs.mujoco
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, TD3, A2C, PPO
import os


# Create directories to hold models and logs
model_dir = "models"
# log_dir = "logs"

os.makedirs(model_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)
def train(env:VecEnv, sb3_algo, modelname=''):
    log_dir = f"logs/{modelname}"
    os.makedirs(log_dir, exist_ok=True)

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

    TIMESTEPS = 5000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{model_dir}/{modelname}/{sb3_algo}_{TIMESTEPS*iters}")

def test(env:VecEnv,
         sb3_algo:str, 
         path_to_model:str):

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
    n_envs = env.num_envs
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

    # gymenv = gym.make('Humanoid-v4', render_mode='human')
    gymenv = make_vec_env('A1-v1', n_envs=4)
    # train(gymenv, 'SAC', modelname='A1-240719')

    test(gymenv, sb3_algo='SAC', path_to_model='models/A1-240717/SAC_5000.zip')
