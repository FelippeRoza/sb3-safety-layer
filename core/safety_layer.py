import torch as th
import cvxpy as cp
import gymnasium as gym
import numpy as np
import torch as th
import cvxpy as cp
import scipy as sp
from scipy.stats import norm



def linear_model_optimization(self, actions, C, margin, g_mean, g_std):
        # cvxpy optimization for linearized cost prediction models in the form c_1 = c_0 + g(s)*a
        
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

    prob.solve(solver=cp.SCS, max_iters=500)
    safe_action = x.value        
    if safe_action is None:
        self.solver_infeasible += 1
        return actions
                
    return safe_action

def e2e_model_optimization(self, state, actions, C, margin):
    # optimizations for cost prediction models in the form c_1 = c_0 + g(s,a) 

    def objective_function(x, a_len):
        # find the actions closest to the policy action
        l1_penalty = 1000
        a = x[:a_len]
        eps = x[a_len:]
        return np.linalg.norm(actions - a) + l1_penalty * np.linalg.norm(eps, ord=1)
    
    def constraint_function(x, a_len):
        # guarantee C is always negative (- because scipy.minimize expects C>0)
        a = x[:a_len]
        eps = x[a_len:]
        with th.no_grad():
            c_delta, c_std = self.safety_layer.forward_mean_std(C, state, a)
        return - ( (C + c_delta + margin + c_std + eps).max() )

    initial_guess = np.append(actions, np.zeros(C.shape))
    con = {'type': 'ineq', 'fun':constraint_function, 'args': [actions.shape[0]]} # constraints
    result = sp.optimize.minimize(objective_function, initial_guess, method = 'COBYLA', 
                                    constraints=con, args=actions.shape[0])

    return result.x[:actions.shape[0]]

def calculate_probability(self, action, g_mean, g_std, c, margin):

    if self.linear_model:
        numerator = - margin - c - g_mean @ action
        denominator = np.sqrt((g_std @ action) ** 2)
    else:
        numerator = - margin - c - g_mean
        denominator = np.sqrt((g_std) ** 2)

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
    if self.env._elapsed_steps == 0 or self.buffer.get('old_c_pred') is None:
        self.cost_pred_error = 0.0
    else:
        self.cost_pred_error = np.linalg.norm(self.buffer['old_c_pred'] - C)    
        self.safety_layer.replay_buffer.add(self.buffer['old_c'], 
                                            np.concatenate([self.buffer['old_s'], self.buffer['old_a']]), 
                                            C, 0, False, False)
    if self.linear_model:
        self.buffer = {'old_s': state, 'old_c': C, 'old_c_pred': C + g_mean @ actions }
    else:
        self.buffer = {'old_s': state, 'old_c': C, 'old_c_pred': C + g_mean}
    
    lower_p = calculate_probability(self, actions, g_mean, g_std, C, margin)

    if (lower_p < 0.95).any():

        if self.linear_model:
            modified_action = linear_model_optimization(self, actions, C, margin, g_mean, g_std)
        else:
            modified_action = e2e_model_optimization(self, state, actions, C, margin)

        self.applied_p = min(calculate_probability(self, modified_action, g_mean, g_std, C, margin))
        self.correction = np.linalg.norm(modified_action - actions)
        self.solver_interventions += self.correction > 1e-3

        return th.Tensor(modified_action).unsqueeze(0)
    else:
        self.applied_p = min(lower_p)
        self.correction = 0.0
        return act
    