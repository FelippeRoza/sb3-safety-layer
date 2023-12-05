from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import cvxpy as cp
import scipy as sp
from scipy.stats import norm



from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy



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
        margin = 0.05,
        prob = 0.9,
    ):
        self.safety_layer = safety_layer
        self.sl_mode = sl_mode
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


    def optimize(self, problem, var, og_action):
        """
        input: cvxpy problem, cvxpy optimization variable, rl action
        returns: safe action (np array) if solution is found
        if no solution (safe_action is None) returns the og_action"""
        problem.solve(solver=cp.SCS, max_iters=500)
        safe_action = var.value
        
        if safe_action is None:
            self.solver_infeasible += 1
            return og_action
                    
        return safe_action

    def calculate_probability(self, action, g_mean, g_std, c, margin):

        numerator = - margin - c - g_mean @ action
        denominator = np.sqrt((g_std @ action) ** 2)
        p = sp.stats.norm.cdf(np.divide(numerator, denominator))

        return p
    
    @th.no_grad()
    def get_safe_actions(self, state, act):

        actions = act.squeeze(0).detach().cpu().numpy()
        state = state.squeeze(0).detach().cpu().numpy()

        C = np.array(self.env.calculate_cost())
        g_mean, g_std = self.safety_layer.forward_mean_std(C, state, actions)
        margin = np.repeat(self.margin, C.shape)

        # calculate cost prediction error
        if self.env._elapsed_steps == 0:
            self.cost_pred_error = 0.0
        else:
            self.cost_pred_error = np.linalg.norm(self.buffer['old_c_pred'] - C)    
            self.safety_layer.replay_buffer.add(self.buffer['old_c'], 
                                                np.concatenate([self.buffer['old_s'], self.buffer['old_a']]), 
                                                C, 0, False, False)
        self.buffer = {'old_s': state, 'old_c': C, 'old_c_pred': C + g_mean @ actions }
        
        
        lower_p = self.calculate_probability(actions, g_mean, g_std, C, margin)

        if (lower_p < 0.95).any():

            x = cp.Variable(actions.shape)
            cost = cp.sum_squares((x - actions) * 0.5)

            if self.sl_mode == 'hard':
                prob = cp.Problem(cp.Minimize(cost), [C + g_mean @ x  <= - margin])

            elif self.sl_mode == 'prob':
                prob = cp.Problem(cp.Minimize(cost), [C + norm.ppf(self.p)*cp.abs(g_std @ x) + g_mean @ x  <= - margin])

            else:
                l1_penalty = 1000
                e = cp.Variable(C.shape)

                cost = cp.sum_squares((x - actions) * 0.5 ) + l1_penalty * cp.norm1(e)
                if self.sl_mode == 'soft':
                    prob = cp.Problem(cp.Minimize(cost), [C + g_mean @ x.T  <= - margin + e])

                elif self.sl_mode == 'hybrid':
                    prob = cp.Problem(cp.Minimize(cost), [C + norm.ppf(self.p)*cp.abs(g_std @ x) + g_mean @ x  <= - margin + e])

           
            modified_action = self.optimize(prob, x, actions)
            self.applied_p = min(self.calculate_probability(modified_action, g_mean, g_std, C, margin))
            self.correction = np.linalg.norm(modified_action - actions)
            self.solver_interventions += self.correction > 1e-3

            return th.Tensor(modified_action).unsqueeze(0)
        else:
            self.applied_p = min(lower_p)
            self.correction = 0.0
            return act

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: safe action, value and log probability of the action  
        """

        actions, values, log_prob = super(SafeActorCriticPolicy, self).forward(obs, deterministic)
        if self.sl_mode != 'unsafe':
            actions = self.get_safe_actions(obs, actions)
        self.buffer['old_a'] = actions.squeeze(0).detach().cpu().numpy()

        return actions, values, log_prob