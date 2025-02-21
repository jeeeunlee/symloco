from typing import Literal

from ml_collections import config_dict
from mujoco_playground._src import locomotion
from jax import numpy as jp


def restruct_features_fn(feature: jp.ndarray) -> jp.ndarray:
    # assumption: reflective symmetry over the x axis
    linvel = feature[..., :3]
    gyro = feature[..., 3:6]  # check: gyro is rpy
    gravity = feature[..., 6:9]
    joint_angles = feature[..., 9:21]
    joint_vel = feature[..., 21:33]
    last_act = feature[..., 33:45]
    command = feature[..., 45:]  # dim = 3

    lin_transform = jp.array([1, -1, 1])
    rot_transform = jp.array([-1, 1, -1])
    command_transform = jp.array([1, -1, -1])

    def transform_joints(joints: jp.ndarray, side: Literal["L", "R"]) -> jp.ndarray:
        # FR (3), FL (3), RR (3), RL (3); total dim = (n, 12)
        fr = joints[..., :3]
        fl = joints[..., 3:6]
        rr = joints[..., 6:9]
        rl = joints[..., 9:]
        # shape: (n, 6)
        if side == "L":
            return jp.concatenate([fl, rl], axis=-1)
        elif side == "R":
            return -jp.concatenate([fr, rr], axis=-1)
        assert False, f"Invalid side value {side}"

    feature_left = jp.concatenate(  # (n, 30)
        [
            linvel,  # 3
            gyro,  # 3
            gravity,  # 3
            transform_joints(joint_angles, "L"),  # 6
            transform_joints(joint_vel, "L"),  # 6
            transform_joints(last_act, "L"),  # 6
            command,  # 3
        ],
        axis=-1,
    )
    feature_right = jp.concatenate(
        [
            linvel * lin_transform,
            gyro * rot_transform,
            gravity * lin_transform,
            transform_joints(joint_angles, "R"),
            transform_joints(joint_vel, "R"),
            transform_joints(last_act, "R"),
            command * command_transform,
        ],
        axis=-1,
    )
    return jp.stack([feature_left, feature_right], axis=1)  # dim: (n, 2, 30)


def destruct_actions_fn(structured_actions: jp.ndarray) -> jp.ndarray:
    return jp.concatenate(  # (n, 12)
        [structured_actions[:, 0, :], -structured_actions[:, 1, :]], axis=-1
    )


def go1_sym_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned Brax PPO config for the given environment."""
    env_config = locomotion.get_default_config(env_name)

    return config_dict.create(
        num_timesteps=200_000_000,
        num_evals=10,
        num_resets_per_eval=1,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            restructured_feature_dim=30,
            restructured_action_dim=12,
            policy_latent_size=256,
            value_latent_size=256,
            num_policy_layers=3,
            num_value_layers=3,
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )
