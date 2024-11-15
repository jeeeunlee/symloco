import os
import sys

sys.path.append(os.getcwd())

from typing import Callable, Tuple, Union
from collections import OrderedDict
from functools import partial

from gymnasium import spaces
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.preprocessing import get_action_dim

# default
LATENT_SIZE = 64
NUM_LAYER = 1  # 2
OUTPUT_SIZE = 5


def make_mlp_model(input_dim: int, latent_size: int, num_layer: int, nameadd: str = ""):
    layers_dict = OrderedDict()
    for i in range(num_layer):
        layers_dict["dense" + str(i) + nameadd] = nn.Linear(input_dim, latent_size)
        layers_dict["act" + str(i) + nameadd] = nn.ReLU()
        # input_dim = latent_size
    return nn.Sequential(layers_dict)

class Aggregator():
    """agg"""
    def __init__(self, aggregator=th.sum):
        self._aggregator = aggregator

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        axis_dim = inputs.shape  # n,2,64
        res = self._aggregator(inputs, dim=1, keepdim=True)  # n,1,64
        return res.repeat(1, axis_dim[1], 1)  # n,2,64


class SetEncodeProcessDecode(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_size=LATENT_SIZE,
        num_layer=NUM_LAYER,
        output_size=OUTPUT_SIZE,
        num_processing_steps=3,
    ):
        super(SetEncodeProcessDecode, self).__init__()

        self._encoder = make_mlp_model(
            input_dim, latent_size, num_layer, nameadd="_enc"
        )
        self._core = make_mlp_model(
            2 * latent_size, latent_size, num_layer, nameadd="_core"
        )
        self._decoder = make_mlp_model(
            latent_size, latent_size, num_layer, nameadd="_dec"
        )
        self._aggregator = Aggregator()
        self._num_processing_steps = num_processing_steps

    def forward(self, input_op):
        latent0 = self._encoder(input_op)  # Shape [n,2,12 + 1]
        latent = latent0  # Shape [n,2,64]

        for _ in range(self._num_processing_steps):
            latent = self._aggregator(latent)  # Shape [n,2,64]
            core_input = th.cat([latent0, latent], axis=2)  # Shape [n,2,128]
            latent = self._core(core_input)  # Shape [n,2,64]

        return self._decoder(latent)  # Shape [n,2,64]


class ReflectionNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        env,  #: Union[GymEnv, str]
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        # env
        self.env = env
        # do later
        self.restructured_feature_dim = env.unwrapped.restructured_feature_dim
        self.restructured_action_dim = env.unwrapped.restructured_action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = SetEncodeProcessDecode(
            self.restructured_feature_dim, last_layer_dim_pi
        )
        # Value network
        self.value_net = SetEncodeProcessDecode(
            self.restructured_feature_dim, last_layer_dim_vf
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:  # shape [n,21]
        features_set = self.env.unwrapped.restruct_features_fn(
            features
        )  # shape [n,2,14]
        return self.policy_net(features_set)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:  # shape [n,21]
        features_set = self.env.unwrapped.restruct_features_fn(
            features
        )  # shape [n,2,14]
        return self.value_net(features_set)


class ReflectionLinearNetwork(nn.Module):
    def __init__(self, env: Union[GymEnv, str], latent_dim: int):
        super().__init__()
        self.restruct_features_fn = env.unwrapped.restruct_features_fn
        self.destruct_actions_fn = env.unwrapped.destruct_actions_fn
        # do later
        self.restructured_feature_dim = env.unwrapped.restructured_feature_dim
        self.restructured_action_dim = env.unwrapped.restructured_action_dim

        self.output_net = nn.Linear(latent_dim, self.restructured_action_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        restructured_action = self.output_net(latent)
        return self.destruct_actions_fn(restructured_action)


class ReflectionAggregatorLinearNetwork(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int = 1, aggregator=th.sum, dim=1):
        super().__init__()
        self._aggregator = aggregator
        self._dim = dim
        self.output_net = nn.Linear(latent_dim, output_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        aggregated_latent = self._aggregator(latent, dim=self._dim)
        return self.output_net(aggregated_latent)  # shape [n,1]


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.env = kwargs["env"]
        # additional
        self.restructured_feature_dim = self.env.unwrapped.restructured_feature_dim
        self.restructured_action_dim = self.env.unwrapped.restructured_action_dim

        self.action_dim = get_action_dim(action_space)

        # Disable orthogonal initialization
        new_kwargs = {k: v for k, v in kwargs.items() if k not in ["env"]}
        new_kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **new_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ReflectionNetwork(env=self.env)

    def _build_action_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        replace self.action_dist.proba_distribution_net
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        # mean_actions = nn.Linear(latent_dim, self.self.restructured_action_dim)
        mean_actions = ReflectionLinearNetwork(self.env, latent_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(
            th.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self._build_action_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )

        self.value_net = ReflectionAggregatorLinearNetwork(
            self.mlp_extractor.latent_dim_vf, 1
        )
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)  # Shape [n,2,64]
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)  # shape [n,1]
        distribution = self._get_action_dist_from_latent(latent_pi)  # shape [n,6]
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
