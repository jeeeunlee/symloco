import jax
from jax import numpy as jp
from brax.training import types, distribution, networks
from brax.training.agents.ppo.networks import PPONetworks
from typing import Callable
from flax import linen
from src.mygym.envs.mujoco_playground.go1_sym_config import (
    restruct_features_fn,
    destruct_actions_fn,
)

JpCallable = Callable[[jp.ndarray], jp.ndarray]

SymFns = tuple[JpCallable, JpCallable]


class SymAggregator(linen.Module):
    aggregator: JpCallable = jp.sum

    def __call__(self, data: jp.ndarray) -> jp.ndarray:
        return self.aggregator(data, axis=1, keepdims=True).repeat(
            data.shape[1], axis=1
        )


class SymMLP(linen.Module):
    latent_size: int
    num_layers: int
    post_decode: linen.Module
    aggregator: JpCallable = jp.sum
    num_processing_steps: int = 3
    activation: networks.ActivationFn = linen.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(
        self, data: jp.ndarray
    ) -> jp.ndarray:  # n, 2, restructured_features_dim
        mlp_kwargs = {
            "activation": self.activation,
            "kernel_init": self.kernel_init,
            "activate_final": self.activate_final,
            "bias": self.bias,
            "layer_norm": self.layer_norm,
        }
        encoder = networks.MLP((self.latent_size,) * self.num_layers, **mlp_kwargs)
        core = networks.MLP((self.latent_size,) * self.num_layers, **mlp_kwargs)
        decoder = linen.Sequential(
            [
                networks.MLP((self.latent_size,) * self.num_layers, **mlp_kwargs),
                self.post_decode,
            ]
        )

        latent0 = encoder(data)  # n, 2, latent_size
        latent = latent0
        for _ in range(self.num_processing_steps):
            agg = SymAggregator(self.aggregator)(latent)
            latent = core(
                jp.concatenate([latent0, agg], axis=-1)  # n, 2, latent_size * 2
            )  # n, 2, latent_size
        return decoder(latent)  # n, 2, latent_size


class SymLinear(linen.Module):
    out_size: int
    destruct_actions_fn: JpCallable

    @linen.compact
    def __call__(self, data: jp.ndarray) -> jp.ndarray:
        return self.destruct_actions_fn(linen.Dense(self.out_size)(data))


class SymAggLinear(linen.Module):
    out_size: int = 1
    aggregator: JpCallable = jp.sum
    axis: int = 1

    @linen.compact
    def __call__(self, data: jp.ndarray) -> jp.ndarray:
        return linen.Dense(self.out_size)(self.aggregator(data, axis=self.axis))  # n, 1


def get_sym_fns(env_name: str) -> SymFns:
    """
    Get restruct_features_fn and destruct_actions_fn from env_name (workaround since fns cannot be saved in chkpt files)
    """
    if env_name == "Go1JoystickFlatTerrain":
        return restruct_features_fn, destruct_actions_fn
    else:
        assert False, f"SymFns undefined for env {env_name}"


def make_sym_policy_network(
    env_name: str,
    param_size: int,
    obs_size: types.ObservationSize,
    restructured_feature_dim: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    latent_size=256,
    num_layers=1,
    activation: networks.ActivationFn = linen.relu,
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""
    restruct_features_fn, destruct_actions_fn = get_sym_fns(env_name)

    policy_module = SymMLP(
        latent_size,
        num_layers,
        post_decode=SymLinear(param_size, destruct_actions_fn),
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        obs = restruct_features_fn(obs)
        return policy_module.apply(policy_params, obs)

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jp.zeros((1, 2, restructured_feature_dim))
    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_sym_value_network(
    env_name: str,
    obs_size: types.ObservationSize,
    restructured_feature_dim: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    latent_size=256,
    num_layers=1,
    activation: networks.ActivationFn = linen.relu,
    obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    """Creates a value network."""
    restruct_features_fn, _ = get_sym_fns(env_name)
    value_module = SymMLP(
        latent_size,
        num_layers,
        post_decode=SymAggLinear(),
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, value_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        obs = restruct_features_fn(obs)
        return jp.squeeze(value_module.apply(value_params, obs), axis=-1)

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jp.zeros((1, 2, restructured_feature_dim))
    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply
    )


def make_sym_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    env_name: str,  # used for recovering restruct_feature_fn and destruct_actions_fn
    restructured_feature_dim: int,
    restructured_action_dim: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_latent_size: int = 32,
    value_latent_size: int = 256,
    num_policy_layers: int = 2,
    num_value_layers: int = 2,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
) -> PPONetworks:
    """Make PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_sym_policy_network(
        env_name,
        restructured_action_dim,
        observation_size,
        restructured_feature_dim,
        preprocess_observations_fn=preprocess_observations_fn,
        latent_size=policy_latent_size,
        num_layers=num_policy_layers,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = make_sym_value_network(
        env_name,
        observation_size,
        restructured_feature_dim,
        preprocess_observations_fn=preprocess_observations_fn,
        latent_size=value_latent_size,
        num_layers=num_value_layers,
        activation=activation,
        obs_key=value_obs_key,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
