import numpy as np
import torch
from typing import Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.utils import safe_mean
import imageio
import time
from stable_baselines3.common.logger import TensorBoardOutputFormat
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from PIL import Image, ImageDraw, ImageFont

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, log_interval, verbose=0, render_freq = 0, sl_retrain_steps = 0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.thresh_violations = 0
        self.sl_retrain_steps = sl_retrain_steps
        self.start_time = time.time_ns()
        self._render_freq = render_freq
        self.p_list, self.c_list, self.correction, self.c_pred_error, self.std_list = [], [], [], [], []
        self.collision_step = [] #

        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = None
        self.current_lengths = None
    
    def _on_training_start(self):
        self.start_time = time.time()  # Start the timer
        self.current_rewards = np.zeros(self.training_env.num_envs)
        self.current_lengths = np.zeros(self.training_env.num_envs)
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        
        if self.env.sl:
            self.c_pred_error.append(self.env.sl.cost_pred_error)
            if (self.sl_retrain_steps > 0 and self.num_timesteps % self.sl_retrain_steps == 0):
                old_weights = copy.deepcopy(self.env.sl.dynamics_model.state_dict())
                prior_eval = self.env.sl.evaluate()
                self.env.sl.train(n_epochs=5)                
                post_eval = self.env.sl.evaluate()
                if prior_eval.mean() > post_eval.mean(): # model worsened
                    self.env.sl.dynamics_model.load_state_dict(old_weights)
                    self.env.sl.save(os.path.join(self.logger.dir, f'sl_{self.num_timesteps}'))
        else:
            self.c_pred_error.append(-1)
        
        self.current_rewards += self.locals['rewards']
        self.current_lengths += 1
        # Check for done environments and log their rewards
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.current_rewards[i] = 0  # Reset the current reward for the finished episode
                self.current_lengths[i] = 0  # Reset the current length for the finished episode

        # Log
        if (self.num_timesteps % self.log_interval == 0):
            self.logger.record("rollout/ep_rew_mean", safe_mean([self.episode_rewards]))
            self.logger.record("rollout/ep_len_mean", safe_mean([self.episode_lengths]))
            self.logger.record("time/fps", self.log_interval / (time.time() - self.start_time))
            self.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")
            self.current_rewards = np.zeros(self.training_env.num_envs)
            self.current_lengths = np.zeros(self.training_env.num_envs)
            self.start_time = time.time()  # Start the timer
            self.logger.record('safety/violations', self.env.total_collisions)
            self.logger.record('safety/ep_c_pred_mean_error', safe_mean(self.c_pred_error))

            self.logger.dump(self.num_timesteps)

        # Video recording
        if (self._render_freq > 0) and (self.num_timesteps % self._render_freq == 0):
            screens = []
            fig, ax = plt.subplots(figsize=(4, 2.56))
            frame = 1
            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self.env.render()
                # Convert frame to a PIL image
                image = Image.fromarray(screen)
                # Draw text on the image
                draw = ImageDraw.Draw(image)
                
                # draw.text((10, 10), f"Reward {reward}", fill=(0, 0, 0))
                draw.text((10, 20), f"Unsafe action {self.env.old_action}", fill=(0, 0, 0))
                draw.text((10, 30), f"Safe action {self.env.safe_action}", fill=(0, 0, 0))
                correction = np.linalg.norm(self.env.safe_action - self.env.old_action)
                if correction > 0.1:
                    draw.text((10, 40), f"Diff {correction}", fill=(255, 0, 0))

                # clear figure
                ax.clear()
                (ax.set_xlim(-1, 1), ax.set_ylim(-1, 1))

                ax.quiver(0, 0, self.env.safe_action[0], self.env.safe_action[1], angles='xy', scale_units='xy', scale=1, color='r')
                ax.quiver(0, 0, self.env.old_action[0], self.env.old_action[1], angles='xy', scale_units='xy', scale=1, color='b')
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                screens.append(np.hstack((image, data)))

            evaluate_policy(
                self.model,
                self.env,
                callback=grab_screens,
                n_eval_episodes=2,
                deterministic=True,
            )
            plt.close()
            imageio.mimsave(os.path.join(self.logger.dir, f'evaluation_{self.num_timesteps}.mp4'), screens, fps=30)


        return True
