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

def e2e_model_optimization(sl_mode, sl, prob, linear_model, state, actions, C, margin, obs):
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
        if sl_mode == 'soft':
            with th.no_grad():
                c_delta, c_std = sl.forward_mean_std(C, np.concatenate((state, a)))
            return - ( (C + c_delta + margin + eps).max() )
        elif sl_mode == 'prob':
            with th.no_grad():
                c_next_pred, g_std = sl.predict(C, np.concatenate((state, a)), a, return_std=True)
            p = min(calculate_probability(linear_model, a, c_next_pred, g_std, margin))
            return p - prob
    
    def constraint_function2(x, a_len):
        # bounded actions [-1, 1]
        a = x[:a_len]
        
        return 1 - (abs(a)).max()

    initial_guess = np.append(actions, np.zeros(C.shape))
    con = {'type': 'ineq', 'fun':constraint_function, 'args': [actions.shape[0]]} # constraints
    con2 = {'type': 'ineq', 'fun':constraint_function2, 'args': [actions.shape[0]]} # constraints
    result = sp.optimize.minimize(objective_function, initial_guess, method = 'COBYLA', 
                                    constraints=[con,con2], args=actions.shape[0])

    return result.x[:actions.shape[0]]

def calculate_probability(linear_model, action, c_next_pred, g_std, margin):

    numerator = - margin - c_next_pred
    if linear_model:
        denominator = np.sqrt((g_std @ action) ** 2)
    else:
        denominator = np.sqrt((g_std) ** 2)

    p = sp.stats.norm.cdf(np.divide(numerator, denominator))

    return p

@th.no_grad()
def get_safe_actions(sl, env, state, act):
    margin = 0.3
    sl_mode = 'prob'
    prob = 0.8
    linear_model = sl.linearized

    # actions = act.squeeze(0).detach().cpu().numpy()
    actions = act
    actions = np.clip(actions, env.action_space.low, env.action_space.high)
    obs = state
    # state = state.squeeze(0).detach().cpu().numpy()
    state = state

    C = np.array(env.calculate_cost())
    C_next_pred, g_std = sl.predict(C, np.concatenate((state, actions)), actions, return_std=True)
    margin = np.repeat(margin, C.shape)
    
    # return original action if probabilities are already satisfied
    if sl_mode in ['prob', 'hybrid', 'soft']:
        lower_p = min(calculate_probability(linear_model, actions, C_next_pred, g_std, margin))
        if lower_p >= prob:
            return act, g_std.mean()
    
    if linear_model:
        g_mean, g_std = sl.forward_mean_std(C, np.concatenate((state, actions)))
        modified_action = linear_model_optimization(actions, C, margin, g_mean, g_std)
    else:
        modified_action = e2e_model_optimization(sl_mode, sl, prob, linear_model, state, actions, C, margin, obs)

    c_next_pred, g_std = sl.predict(C, np.concatenate((state, modified_action)), modified_action, return_std=True)
    # self.applied_p = min(calculate_probability(self, modified_action, c_next_pred, g_std, margin))
    correction = np.linalg.norm(modified_action - actions)
    # if correction > 1e-3:
    #     print('here')
    

    return modified_action, g_std.mean()

    