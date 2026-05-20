import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class RCHumanEventDrivenReward(BaseRewardFunction):
    """Penalty for safety terminations."""

    def __init__(self, config):
        super().__init__(config)
        self.bad_done_penalty = float(getattr(config, 'rc_human_bad_done_penalty', 200.0))

    def get_reward(self, task, env):
        return -self.bad_done_penalty * env.bad_done
