import os
import functools
import jax
from jax import numpy as jp
import mediapy as media
import mujoco
import numpy as np
from pathlib import Path
from tqdm import tqdm

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import checkpoint

from torch.utils.tensorboard import SummaryWriter
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command

from src.mygym.networks.brax_sym_network import make_sym_ppo_networks, SymFns
from src.mygym.envs.mujoco_playground.go1_sym_config import go1_sym_config


class ProgressLogger:
    def __init__(self, log_path, pbar):
        self._writer = SummaryWriter(log_path)
        self._pbar = pbar
        self._cur_steps = 0

    def __call__(self, num_steps, metrics):
        self._pbar.update(num_steps - self._cur_steps)
        self._cur_steps = num_steps
        for k, v in metrics.items():
            # All keys are prefixed with "eval/"
            self._writer.add_scalar(
                k[5:], float(v), num_steps, walltime=metrics["eval/walltime"]
            )


def setup():
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ["MUJOCO_GL"] = "egl"

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags


def sym_ppo_config(env_name: str):
    if env_name == "Go1JoystickFlatTerrain":
        return go1_sym_config(env_name)
    else:
        assert False, f"Invalid env_name {env_name}"


def train(
    train_name: str,
    env_name: str,
    model_dir: str,
    log_dir: str,
    use_sym: bool = False,
) -> None:
    ckpt_path = Path(model_dir) / train_name
    log_path = Path(log_dir) / train_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    train_cfg = (
        sym_ppo_config(env_name)
        if use_sym
        else locomotion_params.brax_ppo_config(env_name)
    )
    randomizer = registry.get_domain_randomizer(env_name)

    ppo_training_params = dict(train_cfg)
    network_factory = (
        functools.partial(make_sym_ppo_networks, env_name=env_name)
        if use_sym
        else ppo_networks.make_ppo_networks
    )
    if "network_factory" in train_cfg:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            network_factory, **train_cfg.network_factory
        )

    _train = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    with tqdm(total=train_cfg.num_timesteps) as pbar:
        make_inference_fn, params, metrics = _train(
            environment=env,
            eval_env=registry.load(env_name, config=env_cfg),
            progress_fn=ProgressLogger(log_path, pbar),
        )


def test(env_name: str, model_path: str, log_dir: str, deterministic: bool = True):
    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg)

    policy_fn = checkpoint.load_policy(model_path, deterministic=deterministic)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_fn)

    x_vel = 0.0
    y_vel = 1.0
    yaw_vel = 0.0

    rng = jax.random.PRNGKey(0)
    rollout = []
    modify_scene_fns = []

    swing_peak = []
    rewards = []
    linvel = []
    angvel = []
    track = []
    foot_vel = []
    rews = []
    contact = []
    command = jp.array([x_vel, y_vel, yaw_vel])

    state = jit_reset(rng)
    state.info["command"] = command
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        rews.append({k: v for k, v in state.metrics.items() if k.startswith("reward/")})
        rollout.append(state)
        swing_peak.append(state.info["swing_peak"])
        rewards.append(
            {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
        )
        linvel.append(env.get_global_linvel(state.data))
        angvel.append(env.get_gyro(state.data))
        track.append(
            env._reward_tracking_lin_vel(
                state.info["command"], env.get_local_linvel(state.data)
            )
        )

        feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_vel.append(vel_norm)

        contact.append(state.info["last_contact"])

        xyz = np.array(state.data.xpos[env._torso_body_id])
        xyz += np.array([0, 0, 0.2])
        x_axis = state.data.xmat[env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=abs(state.info["command"][0]) / env_cfg.command_config.a[0],
            )
        )

    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
        modify_scene_fns=mod_fns,
    )

    media.write_video(log_dir / "video.mp4", frames, fps=fps)
