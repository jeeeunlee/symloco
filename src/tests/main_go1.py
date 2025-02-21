import os
import sys
from pathlib import Path

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

from test_utils import get_brax_args  # noqa: E402
from src.mygym.utils import playground  # noqa: E402

ENV_NAME = "Go1JoystickFlatTerrain"
MODELS_DIR = Path("models/").resolve()
LOG_DIR = Path("logs/").resolve()


def train(model_name: str, use_sym: bool):
    playground.setup()
    playground.train(
        model_name,
        "Go1JoystickFlatTerrain",
        MODELS_DIR,
        LOG_DIR,
        use_sym=use_sym,
    )


def test(model_name: str, model_path: str):
    playground.test(
        ENV_NAME, Path(model_path).resolve(), LOG_DIR / model_name, deterministic=True
    )


def debug_train():
    train("playground_sym_test", False)


if __name__ == "__main__":
    # debug_train()
    args = get_brax_args("main_go1")
    if args.mode == "train":
        train(args.run_name, args.use_sym_policy)
    elif args.mode == "test":
        test(args.run_name, args.checkpoint_name)
