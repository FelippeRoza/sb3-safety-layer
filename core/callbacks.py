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
import os

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
        self.p_list, self.correction, self.c_pred_error = [], [], []
        self.collision_step = [] #
    
    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        
        self.p_list.append(self.model.policy.applied_p)
        self.correction.append(self.model.policy.correction)
        self.c_pred_error.append(self.model.policy.cost_pred_error)

        if self.env.collision:
            self.thresh_violations += 1
            self.collision_step.append(len(self.p_list) - 1)

        if self.model.policy.sl_mode != 'unsafe':
            if (self.sl_retrain_steps > 0 and self.num_timesteps % self.sl_retrain_steps == 0):
                self.model.policy.safety_layer.train(n_epochs=2)
        

        if (self.num_timesteps % self.log_interval == 0):
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
            
            self.logger.record('safety/violations', self.thresh_violations)
            self.logger.record('safety/interventions', self.model.policy.solver_interventions)
            self.logger.record('safety/ep_prob_mean', safe_mean(self.p_list))
            self.logger.record('safety/ep_correction_mean', safe_mean(self.correction))
            self.logger.record('safety/ep_c_pred_mean_error', safe_mean(self.c_pred_error))


            fps = int( self.log_interval / ((time.time_ns() - self.start_time) / 1e9) )
            self.logger.record("time/fps", fps)
            self.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")

            self.p_list, self.correction, self.c_pred_error = [], [], []
            self.start_time = time.time_ns()
            self.logger.dump(self.num_timesteps)


        # Video recording
        if (self._render_freq > 0) and (self.num_timesteps % self._render_freq == 0):
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self.env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self.env,
                callback=grab_screens,
                n_eval_episodes=1,
                deterministic=True,
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=100),
                exclude=("stdout", "log", "json", "csv"),
            )


        return True
