import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '/home/xingru/symloco-main/'))
import numpy as np
from typing import Callable, Tuple
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
import shutil  
import src.mygym.networks


class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int = 18, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)



class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Callable[[float], float], *args, **kwargs):
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(feature_dim=18)
        print(self.mlp_extractor)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        # Restructure the observation into the structured features
        restructured_features = self.restruct_features_fn(obs) 
        restructured_features = restructured_features.view(restructured_features.shape[0], -1).float()  # [batch_size, 18]

        # Process the features
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(restructured_features)
        else:
            pi_features, vf_features = restructured_features, restructured_features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Get actions
        structured_actions = distribution.get_actions(deterministic=deterministic)

        # Compute log probability before reshaping
        log_prob = distribution.log_prob(structured_actions)

        # Now reshape structured_actions to 3D
        if structured_actions.dim() == 2:  # If shape is [batch_size, action_dim]
            batch_size = structured_actions.shape[0]
            action_dim = structured_actions.shape[1]
            # Reshape actions to 3D: [batch_size, num_groups, group_size]
            num_groups = 3
            group_size = action_dim // num_groups
            structured_actions = structured_actions.view(batch_size, num_groups, group_size)

        # Decompose structured actions back to the original action space
        actions = self.destruct_actions_fn(structured_actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        # Compute symmetry loss (Optional: store it as an attribute)
        self.symmetry_loss = self.compute_mirror_symmetry_loss(obs, latent_pi, values, deterministic)

        # Return only three values as required by Stable Baselines 3
        return actions, values, log_prob
   

    def compute_mirror_symmetry_loss(self, obs: th.Tensor, latent_pi, values, deterministic):
        """
        Compute the symmetry loss by creating mirrored observations and comparing mirrored actions/values
        without recursive calls to the forward function.
        """
        mirror_obs = self.get_mirror_observation(obs)
        features = self.extract_features(mirror_obs)
        
        if self.share_features_extractor:
            mirror_latent_pi, mirror_latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            mirror_latent_pi = self.mlp_extractor.forward_actor(pi_features)
            mirror_latent_vf = self.mlp_extractor.forward_critic(vf_features)

        mirror_values = self.value_net(mirror_latent_vf)

        # Policy symmetry loss (Equation 14)
        policy_loss = th.mean((latent_pi - mirror_latent_pi) ** 2)

        # Value symmetry loss (Equation 15)
        value_loss = th.mean((values - mirror_values) ** 2)

        return policy_loss + value_loss

    def get_mirror_observation(self, obs):
        """
        Compute the mirrored version of the observation based on the robot's mirror symmetry.
        """
        mirrored_obs = obs.clone()
        # Example for mirror symmetry: reverse specific joint angles or positional values
        mirrored_obs[:, 1] *= -1  # Placeholder: modify according to the Go1's mirror symmetry
        return mirrored_obs

    def restruct_features_fn(self, feature):
        """
        Restructure the features (observation) into the structured format as defined in environment.
        """
        # Adjust the feature selection to match the desired output size of 18
        rootx = feature[:, [0, 9]]  # Shape [n, 2]
        rootz = feature[:, [1, 10]]  # Shape [n, 2]
        rooty = feature[:, [2, 11]]  # Shape [n, 2]

        bfoot_pos = feature[:, 3:6]  # Shape [n, 3]
        ffoot_pos = feature[:, 6:9]  # Shape [n, 3]
        bfoot_vel = feature[:, 12:15]  # Shape [n, 3]
        ffoot_vel = feature[:, 15:18]  # Shape [n, 3]

        # Concatenate the features to get a feature vector of size [batch_size, 18]
        restructured_features = th.cat([rootx, rootz, rooty, bfoot_pos, ffoot_pos, bfoot_vel, ffoot_vel], dim=1)
        return restructured_features  # Shape [batch_size, 18]


    def destruct_actions_fn(self, structured_actions):
        """
        Decompose the structured actions into the original action space, as defined in your environment.
        """
        # Check the shape of structured_actions
        # print(f"structured_actions shape before destructuring: {structured_actions.shape}")

        if structured_actions.dim() == 3:
            # Reshape actions from [batch_size, num_groups, group_size] to [batch_size, action_dim]
            batch_size, num_groups, group_size = structured_actions.shape
            actions = structured_actions.view(batch_size, num_groups * group_size)
            return actions  # Ensure that actions are always returned
        else:
            # Raise an error if structured_actions is not 3D
            raise ValueError(f"structured_actions should be a 3D tensor, but got shape: {structured_actions.shape}")


def train(env,sb3_algo:str,modelname=''):
    model = PPO(CustomActorCriticPolicy, env , verbose=1,device="cuda",tensorboard_log=log_dir)

    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(f"{model_dir}/{modelname}"): shutil.rmtree(f"{model_dir}/{modelname}")
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{model_dir}/{modelname}/{sb3_algo}_{TIMESTEPS*iters}")

def test(env,sb3_algo:str,path_to_model:str):
    # data = th.load(path_to_model, map_location="cuda")  # Load the data without `weights_only=False`
    model= PPO.load(path_to_model,env=env)
    obs=env.reset()
    done = False
    n_envs = env.num_envs
    extra_steps = [500]*n_envs
    while True:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)        
        env.render("human")
        for i, done in enumerate(dones):
            if done:
                extra_steps[i] -= 1
            if extra_steps[i] < 0:
                break


### #create to hold models and logs
# model_dir = "models"
# log_dir = "logs"
# if __name__== '__main__':
#     env = make_vec_env("simple_cheetah", n_envs=4) #simple_idp
#     # train(env,sb3_algo="PPO",modelname='symmetrytraining')
#     test(env,sb3_algo="PPO",path_to_model='models/symmetrytraining/PPO_1250000.zip')


#upload environment
env = make_vec_env("simple_cheetah", n_envs=4) #simple_idp
model = CustomPPO(CustomActorCriticPolicy, env , verbose=1,device="cuda")
# model.learn(100)

obs = env.reset()
done = False
n_envs = env.num_envs
extra_steps = [500]*n_envs
while True:
    action, states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)        
    # env.render("human")
    for i, done in enumerate(dones):
        if done:
            extra_steps[i] -= 1
        if extra_steps[i] < 0:
            break


# #upload environment
# env = make_vec_env("simple_cheetah", n_envs=4) #simple_idp
# model = CustomPPO(CustomActorCriticPolicy, env , verbose=1,device="cuda")
# while True:
#     iters += 1
#     model.learn(total_timesteps=100, reset_num_timesteps=True)
    
#     # Check if the robot has fallen and handle appropriately
#     for env_idx in range(env.num_envs):
#         if env.get_attr('is_fallen', env_idx):
#             print(f"Environment {env_idx} - Robot has fallen at iteration {iters}")
    
#     model.save(f"{model_dir}/{modelname}/{sb3_algo}_{TIMESTEPS*iters}")
    # for i, done in enumerate(dones):
    #     if done:
    #         extra_steps[i] -= 1
    #         if info[i].get("is_fallen", False):
    #             print(f"Robot {i} has fallen.")
    #             break  # End simulation if robot has fallen
            
    #     if extra_steps[i] < 0:
    #         break