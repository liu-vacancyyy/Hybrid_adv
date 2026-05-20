import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class HoverEventDrivenReward(BaseRewardFunction):
    """Event reward for the hover task.

    On a ``bad_done`` step, returns a large negative reward whose magnitude
    scales with the *remaining* steps in the episode.  This way an early
    crash hurts proportionally more than a late crash, giving PPO a sharp
    "don't terminate early" signal that the original fixed -200 cannot
    provide on a 1500-step episode (where +6.7 * 1500 dwarfs -200).

    No terminal bonus is granted on clean done/timeout.
    Only ``bad_done`` receives a terminal penalty.
    """

    def __init__(self, config):
        super().__init__(config)
        # Per-step penalty magnitude when terminated by bad_done.
        # total_penalty = bad_done_base + bad_done_per_step * remaining_steps
        self.bad_done_base       = float(getattr(config, 'hover_bad_done_base',     50.0))
        self.bad_done_per_step   = float(getattr(config, 'hover_bad_done_per_step',  3.0))
        self.max_steps           = int(getattr(config, 'max_steps', 1500))

    def get_reward(self, task, env):
        bad      = env.bad_done.float()
        # remaining_steps is at least 0.
        remaining = (self.max_steps - env.step_count).clamp_min(0).float()
        penalty   = (self.bad_done_base + self.bad_done_per_step * remaining)
        return -penalty * bad
