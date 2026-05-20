import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class CircleEventDrivenReward(BaseRewardFunction):
    """Terminal penalty for failed circle tracking episodes."""

    def __init__(self, config):
        super().__init__(config)
        self.bad_done_base = float(getattr(config, 'circle_bad_done_base', 50.0))
        self.bad_done_per_step = float(getattr(config, 'circle_bad_done_per_step', 3.0))
        self.max_steps = int(getattr(config, 'max_steps', 2000))

    def get_reward(self, task, env):
        bad = env.bad_done.float()
        remaining = (self.max_steps - env.step_count).clamp_min(0).float()
        penalty = self.bad_done_base + self.bad_done_per_step * remaining
        return -penalty * bad
