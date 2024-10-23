import os
import sys
import io

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

from stable_baselines3.common.vec_env import VecEnv  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from src.mygym.networks.simple_sym_network import CustomActorCriticPolicy  # noqa: E402
from src.tests.test_utils import train as _train, test as _test, load_model, get_args  # noqa: E402


SB3_ALGO = "PPO"


def _make_env() -> VecEnv:
    return make_vec_env("simple_cheetah", n_envs=4)


def train():
    env = _make_env()
    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs={"env": env.envs[0]},
    )
    _train(model, SB3_ALGO, n_timesteps=10000, max_iters=10)


def test(model_path: io.BytesIO):
    env = _make_env()
    model = load_model(env, model_path, SB3_ALGO)
    _test(model, env, fps=env.metadata["render_fps"])


if __name__ == "__main__":
    args = get_args("main_cheetah")
    if args.mode == "train":
        train()
    else:
        assert len(args.model_path) == 1, "Model file required for testing"
        test(args.model_path[0])
