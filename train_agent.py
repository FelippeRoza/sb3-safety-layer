import argparse
import os
import gymnasium as gym
from stable_baselines3 import DDPG, PPO, A2C, SAC
import continuousSafetyGym
from core.callbacks import TensorboardCallback
from core.safe_SACPolicy import SafeSACPolicy
from core.safe_ActorCriticPolicy import SafeActorCriticPolicy
from costDynamicsModel import CDM
import json

def main(args):

    data_dir = os.path.join('data', 'experiments', args.env_name)
    log_name = f'{args.method}_{args.sl_method}_lin{args.linear_sl}'
    if args.sl_method in ['prob', 'hybrid']:
        log_name = f'{log_name}_p{args.prob}'
    log_name = f'{log_name}{args.log_name}'

    env = gym.make(args.env_name, render_mode = 'rgb_array')

    sl = CDM(env, buffer_size=args.sl_buffer_size, linearized=args.linear_sl)
    if args.pretrained_sl:
        sl.load(os.path.join('data', 'pretrained_cdm', args.env_name, f'linear_{args.linear_sl}'))
    if args.method == 'SAC':
        rl_agent = SAC(SafeSACPolicy, env, verbose=1, tensorboard_log=data_dir,
                    policy_kwargs={'environment': env, 'safety_layer': sl, 
                                'sl_mode': args.sl_method, 'linear_model': args.linear_sl})
    elif args.method == 'PPO':
        rl_agent = PPO(SafeActorCriticPolicy, env, verbose=1, tensorboard_log=data_dir,
                    policy_kwargs={'environment': env, 'safety_layer': sl, 
                                'sl_mode': args.sl_method, 'linear_model': args.linear_sl})

    rl_agent.learn(total_timesteps=args.train_steps, log_interval=None, tb_log_name=log_name,
                callback=TensorboardCallback(env, args.log_freq, render_freq=args.render_freq,
                                             sl_retrain_steps=args.sl_retrain_steps))

    with open(os.path.join(rl_agent.logger.dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    rl_agent.save(os.path.join(rl_agent.logger.dir, 'rl_model'), exclude=['policy_kwargs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SB3SL: RL with a Safety Layer.')

    parser.add_argument('--env_name', choices=['ContSafetyBallReach-v1', 'ContSafetyBallReach-v0', 
                                               'MultiagentDescentralizedSafe-v0'], 
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
    parser.add_argument('--linear_sl', action='store_true')
    parser.add_argument('--sl_retrain_steps', type=int, default=0, help='number of steps to collect samples and retrain the sl models')
    parser.add_argument('--prob', type=float, default=0.8)
    parser.add_argument('--sl_method', choices=['prob', 'hybrid', 'hard', 'soft', 'unsafe'], default='unsafe')
    parser.add_argument('--log_name', default='')
    
    args = parser.parse_args()

    main(args)