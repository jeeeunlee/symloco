import os
import sys
import io

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

from stable_baselines3.common.vec_env import VecEnv  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from src.tests.test_utils import (  # noqa: E402
    train as _train,
    test as _test,
    make_model,
    load_model,
    get_args,
)
from src.mygym.envs.mujoco import unitree_go2  # noqa: F401, E402


SB3_ALGO = "TD3"


def _make_env() -> VecEnv:
    return make_vec_env("GO2-v1", n_envs=4)


def train():
    env = _make_env()
    model = make_model(env, SB3_ALGO)
    _train(model, SB3_ALGO, n_timesteps=10000, max_iters=10)


def test(model_path: io.BytesIO):
    env = _make_env()
    model = load_model(env, model_path, SB3_ALGO)
    _test(model, env, fps=env.metadata["render_fps"])


if __name__ == "__main__":
    args = get_args("main_go2")
    if args.mode == "train":
        train()
    else:
        assert len(args.model_path) == 1, "Model file required for testing"
        test(args.model_path[0])
