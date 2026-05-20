import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class SimulinkEventDrivenReward(BaseRewardFunction):
    """
    Terminal reward for Simulink task.
    - Bad_done: fixed penalty
    - Done / Timeout: reward proportional to settle_count
      (the number of steps where theta stays within 2% settling band)
    """
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(config, 'max_steps', 1000)

    def get_reward(self, task, env):
        """
        Terminal reward:
          bad_done  -> -200
          done/timeout -> +200 * (settle_count / max_steps)
            settle_count 越大，说明在稳态误差带内停留越久，奖励越高

        Args:
            task: task instance (SimulinkTask)
            env: environment instance

        Returns:
            (tensor): reward
        """
        # Settle ratio: fraction of episode spent within 2% band
        settle_ratio = task.settle_count / self.max_steps

        # is_done: episode completed normally (max_steps reached)
        # bad_done: episode terminated abnormally (extreme state / divergence)
        bad = env.bad_done.float()

        reward = -200.0 * bad

        return reward
