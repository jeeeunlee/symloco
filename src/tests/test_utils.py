import os
import shutil
from time import sleep
import argparse
from argparse import ArgumentTypeError as err
from typing import Any

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class PathType(object):
    def __init__(self, exists=True, type="file", dash_ok=True):
        """exists:
             True: a path that does exist
             False: a path that does not exist, in a valid parent directory
             None: don't care
        type: file, dir, symlink, None, or a function returning True for valid paths
             None: don't care
        dash_ok: whether to allow "-" as stdin/stdout"""

        assert exists in (True, False, None)
        assert type in ("file", "dir", "symlink", None) or hasattr(type, "__call__")

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        if string == "-":
            # the special argument "-" means sys.std{in,out}
            if self._type == "dir":
                raise err("standard input/output (-) not allowed as directory path")
            elif self._type == "symlink":
                raise err("standard input/output (-) not allowed as symlink path")
            elif not self._dash_ok:
                raise err("standard input/output (-) not allowed")
        else:
            e = os.path.exists(string)
            if self._exists:
                if not e:
                    raise err("path does not exist: '%s'" % string)

                if self._type is None:
                    pass
                elif self._type == "file":
                    if not os.path.isfile(string):
                        raise err("path is not a file: '%s'" % string)
                elif self._type == "symlink":
                    if not os.path.symlink(string):
                        raise err("path is not a symlink: '%s'" % string)
                elif self._type == "dir":
                    if not os.path.isdir(string):
                        raise err("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise err("path not valid: '%s'" % string)
            else:
                if not self._exists and e:
                    raise err("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or "."
                if not os.path.isdir(p):
                    raise err("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise err("parent directory does not exist: '%s'" % p)

        return string


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str, scalars: list[tuple[str, str]]):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self._scalars = scalars

    def _on_step(self) -> bool:
        info = self.locals["infos"][-1]

        for name, k in self._scalars:
            self.writer.add_scalar(name, info[k], self.num_timesteps)

        # self.writer.add_scalar("reward/run", info["reward_run"], self.num_timesteps)
        # self.writer.add_scalar("reward/ctrl", info["reward_ctrl"], self.num_timesteps)
        # self.writer.add_scalar("reward/gait", info["reward_gait"], self.num_timesteps)
        # self.writer.add_scalar(
        #     "train/command_x", info["command"][0], self.num_timesteps
        # )
        # self.writer.add_scalar(
        #     "train/command_ry", info["command"][1], self.num_timesteps
        # )

        return True


def get_brax_args(prog_name: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog=prog_name)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("-n", "--run_name", type=str)
    parser.add_argument("-s", "--use_sym_policy", action="store_true")
    parser.add_argument("-c", "--checkpoint_name", type=str)
    return parser.parse_args()
    # parser = argparse.ArgumentParser(prog=prog_name)
    # parser.add_argument("-n", "--run_name", type=str, required=True)
    # subparsers = parser.add_subparsers(help="subcommand help")

    # train = subparsers.add_parser(
    #     "train", description="Train a locomotion policy", help="train help"
    # )
    # train.add_argument("-s", "--use_sym_policy", action="store_true")

    # test = subparsers.add_parser(
    #     "test", description="Test a trained policy", help="test help"
    # )
    # test.add_argument("-c", "--checkpoint_name", type=str, required=True)
    # return parser.parse_args()


def get_args(prog_name: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog=prog_name)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("-n", "--model_name", type=str)
    parser.add_argument("-s", "--use_sym_policy", action="store_true")
    parser.add_argument("-mp", "--model_path", type=argparse.FileType("rb"))
    parser.add_argument("-e", "--n_envs", default=16, type=int)
    parser.add_argument(
        "-v", "--velocity_profile", choices=["oneway", "bothway"], default="oneway"
    )
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
    logging_keys: list[tuple[str, str]] | None = None,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{model_name}"):
        shutil.rmtree(f"{model_dir}/{model_name}")
    log_subdir = f"{log_dir}/{model_name}"
    if os.path.exists(log_subdir):
        shutil.rmtree(log_subdir)
    os.makedirs(log_subdir)

    callback = (
        RewardLoggerCallback(log_dir=log_subdir, scalars=logging_keys)
        if logging_keys is not None
        else None
    )
    logger = configure(log_subdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    iters = 0
    while max_iters is None or iters < max_iters:
        iters += 1
        if callback:
            model.learn(
                total_timesteps=n_timesteps,
                reset_num_timesteps=False,
                callback=callback,
            )
        else:
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
