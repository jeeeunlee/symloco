#
import os
import sys
import shutil

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, TD3, A2C

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())


# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"


def train(env: VecEnv, sb3_algo, modelname=""):
    match sb3_algo:
        case "SAC":
            model = SAC(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "TD3":
            model = TD3(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "A2C":
            model = A2C(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case _:
            print("Algorithm not found")
            return

    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{modelname}"):
        shutil.rmtree(f"{model_dir}/{modelname}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{model_dir}/{modelname}/{sb3_algo}_{TIMESTEPS*iters}")


def test(env: VecEnv, sb3_algo: str, path_to_model: str):
    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "TD3":
            model = TD3.load(path_to_model, env=env)
        case "A2C":
            model = A2C.load(path_to_model, env=env)
        case _:
            print("Algorithm not found")
            return

    obs = env.reset()
    done = False
    n_envs = env.num_envs
    extra_steps = [500] * n_envs
    while True:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        for i, done in enumerate(dones):
            if done:
                extra_steps[i] -= 1
            if extra_steps[i] < 0:
                break


if __name__ == "__main__":
    gymenv = make_vec_env("GO2-v1", n_envs=1)
    # train(gymenv, 'TD3', modelname='GO2-exprewards-240906-TD3')
    test(
        gymenv,
        sb3_algo="TD3",
        path_to_model="models/GO2-exprewards-240906-TD3/TD3_320000.zip",
    )
