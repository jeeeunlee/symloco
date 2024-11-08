import os
import shutil
from time import sleep
import argparse
from typing import Any

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure


def get_args(prog_name: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog=prog_name)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("-n", "--model-name", nargs=1, type=str)
    parser.add_argument("-c", "--use_sym_policy", action="store_true")
    parser.add_argument("-mp", "--model-path", nargs=1, type=argparse.FileType("rb"))
    return parser.parse_args()


def make_model(env: VecEnv, sb3_algo: str) -> BaseAlgorithm:
    match sb3_algo:
        case "SAC":
            return SAC("MlpPolicy", env, verbose=1, device="cuda")
        case "TD3":
            return TD3("MlpPolicy", env, verbose=1, device="cuda")
        case "A2C":
            return A2C("MlpPolicy", env, verbose=1, device="cuda")
        case "PPO":
            return PPO("MlpPolicy", env, verbose=1, device="cuda")
        case _:
            raise ValueError(f"Algorithm '{sb3_algo}' not found")


def load_model(env: VecEnv, path_to_model: str, sb3_algo: str) -> BaseAlgorithm:
    match sb3_algo:
        case "SAC":
            return SAC.load(path_to_model, env=env)
        case "TD3":
            return TD3.load(path_to_model, env=env)
        case "A2C":
            return A2C.load(path_to_model, env=env)
        case "PPO":
            return PPO.load(path_to_model, env=env)
        case _:
            raise ValueError(f"Algorithm '{sb3_algo}' not found")


def train(
    model: BaseAlgorithm,
    sb3_algo: str,
    *,
    model_name: str,
    n_timesteps: int = 10000,
    max_iters: int | None = None,
    model_dir: str = "models",
    log_dir: str = "logs",
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{model_name}"):
        shutil.rmtree(f"{model_dir}/{model_name}")
    log_subdir = f"{log_dir}/{model_name}"
    if os.path.exists(log_subdir):
        shutil.rmtree(log_subdir)
    os.makedirs(log_subdir)

    logger = configure(log_subdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    iters = 0
    while max_iters is None or iters < max_iters:
        iters += 1
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/{model_name}/{sb3_algo}_{n_timesteps * iters}")


def test(model: BaseAlgorithm, env: VecEnv, *, fps: int = 100) -> None:
    obs = env.reset()
    done = False
    n_envs = env.num_envs
    extra_steps = [500] * n_envs
    while True:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        sleep(1 / fps)  # maybe a better way to do this?
        for i, done in enumerate(dones):
            if done:
                extra_steps[i] -= 1
            if extra_steps[i] < 0:
                break
