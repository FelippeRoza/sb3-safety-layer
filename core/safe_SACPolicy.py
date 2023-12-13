from stable_baselines3.sac.policies import SACPolicy
from typing import Any, Dict, List, Optional, Type, Union
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)

from . import safety_layer


class SafeSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
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
    
        super(SafeSACPolicy, self).__init__(
            observation_space = observation_space,
            action_space = action_space,
            lr_schedule = lr_schedule,
            net_arch = net_arch,
            activation_fn = activation_fn,
            use_sde = use_sde,
            log_std_init = log_std_init,
            use_expln = use_expln,
            clip_mean = clip_mean,
            features_extractor_class = features_extractor_class,
            features_extractor_kwargs = features_extractor_kwargs,
            normalize_images = normalize_images,
            optimizer_class = optimizer_class,
            optimizer_kwargs = optimizer_kwargs,
            n_critics = n_critics,
            share_features_extractor = share_features_extractor,
            )
        
    def _predict(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: safe action, value and log probability of the action  
        """

        action = super(SafeSACPolicy, self)._predict(obs, deterministic)
        if self.sl_mode != 'unsafe':
            action = safety_layer.get_safe_actions(self, obs, action)
        self.buffer['old_a'] = action.squeeze(0).detach().cpu().numpy()

        return action