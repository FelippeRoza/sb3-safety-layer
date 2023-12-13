from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy

from . import safety_layer


class SafeActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        environment = None,
        safety_layer = None,
        sl_mode = 'deterministic',
        linear_model = True,
        margin = 0.05,
        prob = 0.9,
    ):
        self.safety_layer = safety_layer
        self.sl_mode = sl_mode
        self.linear_model = linear_model
        self.env = environment
        self.margin = margin
        self.p = prob
        self.solver_interventions = 0
        self.solver_infeasible = 0
        self.applied_p = -1
        self.cost_pred_error = 0.0
        self.correction = 0.0
        self.buffer = {}
    
        super(SafeActorCriticPolicy, self).__init__(observation_space = observation_space,
            action_space = action_space,
            lr_schedule = lr_schedule,
            net_arch = net_arch,
            activation_fn = activation_fn,
            ortho_init = ortho_init,
            use_sde = use_sde,
            log_std_init = log_std_init,
            full_std = full_std,
            use_expln = use_expln,
            squash_output = squash_output,
            features_extractor_class = features_extractor_class,
            features_extractor_kwargs = features_extractor_kwargs,
            share_features_extractor = share_features_extractor,
            normalize_images = normalize_images,
            optimizer_class = optimizer_class,
            optimizer_kwargs = optimizer_kwargs,
            )
        
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: safe action, value and log probability of the action  
        """

        actions, values, log_prob = super(SafeActorCriticPolicy, self).forward(obs, deterministic)
        if self.sl_mode != 'unsafe':
            actions = safety_layer.get_safe_actions(self, obs, actions)
        self.buffer['old_a'] = actions.squeeze(0).detach().cpu().numpy()

        return actions, values, log_prob