import os
import sys
import io

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

from stable_baselines3.common.vec_env import VecEnv  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from src.mygym.networks.simple_sym_network import (  # noqa: E402
    CustomActorCriticPolicy as SymActorCriticPolicy,
)
from src.mygym.networks.custom_network_example import (  # noqa: E402
    CustomActorCriticPolicy as NonSymActorCriticPolicy,
)
from src.tests.test_utils import (  # noqa: E402
    train as _train,
    test as _test,
    load_model,
    get_args,
)


SB3_ALGO = "PPO"


def _make_env() -> VecEnv:
    return make_vec_env("simple_cheetah", n_envs=4)


def train(model_name: str, use_sym_policy: bool):
    env = _make_env()
    model = (
        PPO(
            SymActorCriticPolicy,
            env,
            verbose=1,
            device="cuda",
            policy_kwargs={"env": env.envs[0]},
        )
        if use_sym_policy
        else PPO(
            NonSymActorCriticPolicy,
            env,
            verbose=1,
            device="cuda",
        )
    )
    _train(model, SB3_ALGO, model_name=model_name, n_timesteps=50000, max_iters=200)


def test(model_path: io.BytesIO):
    env = _make_env()
    model = load_model(env, model_path, SB3_ALGO)
    _test(model, env, fps=env.metadata["render_fps"])


if __name__ == "__main__":
    args = get_args("main_cheetah")
    if args.mode == "train":
        assert len(args.model_name) == 1, "Must provide model name"
        train(args.model_name[0], args.use_sym_policy)
    else:
        assert len(args.model_path) == 1, "Model file required for testing"
        test(args.model_path[0])
