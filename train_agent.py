import argparse
import os
import gymnasium as gym
from stable_baselines3 import DDPG, PPO, A2C, SAC
import continuousSafetyGym
from core.callbacks import TensorboardCallback
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from costDynamicsModel import CDM
import json
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from core.safetyEnvWrapper import SafetyWrappedEnv, make_wrapped_env

import torch
import torch.nn as nn

class SafetyLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SafetyLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")


def main(args):

    data_dir = os.path.join('data', 'experiments', args.env_name)
    log_name = f'{args.method}_{args.sl_method}_lin{args.linear_sl}'
    if args.sl_method in ['prob', 'hybrid']:
        log_name = f'{log_name}_p{args.prob}'
    log_name = f'{log_name}{args.log_name}'
    env = gym.make(args.env_name, render_mode = 'rgb_array')
    
    obs, info = env.reset()
    cost = info['cost']

    if args.sl_method != 'unsafe':
        sl = SafetyLayerNN(input_dim=len(obs)+len(cost)+env.action_space.shape[0], 
                      output_dim=len(cost))
        load_model(sl, f'data/sl_models/{args.env_name}_sl_model.pth')
    else:
        sl = None

    env = SafetyWrappedEnv(env, sl)
    
    if args.method == 'SAC':
        rl_agent = SAC(SACPolicy, env, verbose=1, tensorboard_log=data_dir)
        
    elif args.method == 'PPO':
        rl_agent = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log=data_dir)

    rl_agent.learn(total_timesteps=args.train_steps, log_interval=None, tb_log_name=log_name,
                callback=TensorboardCallback(env, args.log_freq, render_freq=args.render_freq,
                                             sl_retrain_steps=args.sl_retrain_steps))

    with open(os.path.join(rl_agent.logger.dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    rl_agent.save(os.path.join(rl_agent.logger.dir, 'rl_model'), exclude=['policy_kwargs'])
    if args.sl_method != 'unsafe':
        sl.save(rl_agent.logger.dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SB3SL: RL with a Safety Layer.')

    parser.add_argument('--env_name', choices=['ContSafetyBallReach-v1', 'ContSafetyBallReach-v0',
                                               'MultiagentDescentralizedSafe-v0', 'ContSafetyBallGather-v0',
                                               'SpaceshipSafe-v0', 'SafetyPointGoal1Gymnasium-v0'], 
                        default='ContSafetyBallReach-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=['PPO', 'SAC'], default='PPO')
    parser.add_argument('--train_steps', type=float, default=1e6)
    parser.add_argument('--render_freq', type=int, default=0, 
                        help='number of steps between recording a video. Must be >0 to record.')
    parser.add_argument('--log_freq', type=int, default=1000, 
                        help='number of steps between logging to tensorboard.')
    parser.add_argument('--ensemble_size', type=int, default=5)

    parser.add_argument('--sl_buffer_size', type=int, default=1_000_000, help='buffer size of the safety layer.')
    parser.add_argument('--pretrained_sl', action='store_true')
    # parser.add_argument('--pretrained_sl_dir')
    parser.add_argument('--linear_sl', action='store_true')
    parser.add_argument('--sl_retrain_steps', type=int, default=0, help='number of steps to collect samples and retrain the sl models')
    parser.add_argument('--prob', type=float, default=0.8)
    parser.add_argument('--sl_method', choices=['prob', 'hybrid', 'hard', 'soft', 'unsafe'], default='unsafe')
    parser.add_argument('--log_name', default='')
    
    args = parser.parse_args()

    main(args)