import os
import shutil
from time import sleep

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure


def make_model(env: VecEnv, sb3_algo: str, *, log_dir: str) -> BaseAlgorithm:
    match sb3_algo:
        case "SAC":
            return SAC(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "TD3":
            return TD3(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "A2C":
            return A2C(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
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
    n_timesteps: int = 10000,
    max_iters: int | None = None,
    model_dir: str = "models",
    log_dir: str = "logs",
    model_name: str = "",
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{model_name}"):
        shutil.rmtree(f"{model_dir}/{model_name}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    iters = 0
    while max_iters is None or iters < max_iters:
        iters += 1
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=True)
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
