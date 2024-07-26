import gymnasium as gym
import continuousSafetyGym
from core.sl_wrapper import get_safe_actions

def make_wrapped_env(env_id, safety_layer, seed=0):
    def _init():
        env = gym.make(env_id)
        env = SafetyWrappedEnv(env, safety_layer)
        # env.seed(seed)
        return env
    return _init


class SafetyWrappedEnv(gym.Wrapper):
    def __init__(self, env, safety_layer=None):
        super(SafetyWrappedEnv, self).__init__(env)
        self.sl = safety_layer
        self.old_action = None
        self.safe_action = None
        self.total_collisions = 0

    def step(self, action):
        obs = self.env.get_observation()
        self.old_action = action
        if self.sl:
            action, std = get_safe_actions(self.sl, self.env, obs, action)
        self.safe_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_collisions = self.env.total_collisions
        return obs, reward, terminated, truncated, info






