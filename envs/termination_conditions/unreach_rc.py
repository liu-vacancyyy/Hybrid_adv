import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from termination_condition_base import BaseTerminationCondition
from utils.utils import wrap_PI


class UnreachRC(BaseTerminationCondition):
    """
    UnreachHeading
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config, device):
        super().__init__(config)
        self.device = torch.device(device)
        self.max_check_interval = getattr(config, 'max_check_interval', 1500)
        self.min_check_interval = getattr(config, 'min_check_interval', 300)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target in limited time.

        Args:
            env: environment instance

        Returns:Q
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        check_time = env.step_count
        # 判断时间
        done = check_time >= self.max_check_interval
        bad_done = torch.zeros_like(done)
        exceed_time_limit = torch.zeros_like(done)
        if torch.any(bad_done):
            self.log(f'unreach heading!')
            print(torch.sum(bad_done), 'unreach heading!')
        if torch.any(done):
            self.log(f'reset target!')
            print(torch.sum(done), 'reset target!')
        return bad_done, done, exceed_time_limit, info
