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
from mygym.utils.target_velocity_generator import (  # noqa: E402
    VelocityProfile,
)
from src.tests.test_utils import (  # noqa: E402
    train as _train,
    test as _test,
    load_model,
    get_args,
)
from src.mygym.envs.mujoco import unitree_go2  # noqa: F401, E402


SB3_ALGO = "PPO"
LOGGING_KEYS = [
    ("reward/run", "reward_run"),
    ("reward/ctrl", "reward_ctrl"),
    ("reward/balance", "reward_balance"),
    ("reward/smooth", "reward_smooth"),
    ("reward/safety", "reward_safety"),
    ("train/command_x", "command_x"),
    ("train/command_ry", "command_ry"),
    ("train/command_z", "command_z"),
]


def _make_env(n_envs: int, velocity_profile: VelocityProfile) -> VecEnv:
    return make_vec_env(
        "GO2-v1", n_envs=n_envs, env_kwargs={"velocity_profile": velocity_profile}
    )


def train(
    model_name: str,
    use_sym_policy: bool,
    n_envs: int,
    velocity_profile: VelocityProfile,
):
    env = _make_env(n_envs, velocity_profile)
    model = PPO(
        SymActorCriticPolicy if use_sym_policy else NonSymActorCriticPolicy,
        env,
        verbose=True,
        device="cuda",
        policy_kwargs={"env": env.envs[0]},
    )
    _train(
        model,
        SB3_ALGO,
        model_name=model_name,
        n_timesteps=10000,
        max_iters=200,
        logging_keys=LOGGING_KEYS,
    )


def test(model_path: io.BytesIO, n_envs: int, velocity_profile: VelocityProfile):
    env = _make_env(n_envs, velocity_profile)
    model = load_model(env, model_path, SB3_ALGO)
    _test(model, env, fps=env.metadata["render_fps"])


if __name__ == "__main__":
    args = get_args("main_go2")
    if args.mode == "train":
        assert args.model_name, "Must provide model name"
        train(args.model_name, args.use_sym_policy, args.n_envs, args.velocity_profile)
    else:
        assert args.model_path, "Model file required for testing"
        test(args.model_path, args.n_envs, args.velocity_profile)
