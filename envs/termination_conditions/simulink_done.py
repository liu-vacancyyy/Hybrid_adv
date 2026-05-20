import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class SimulinkDone(BaseTerminationCondition):
    """
    SimulinkDone
    Episode terminates normally (done=True) when max_steps is reached.
    Unlike generic Timeout (which sets exceed_time_limit), this sets done=True
    so that the episode is properly counted in average_episode_rewards.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 1000)

    def get_termination(self, task, env, info={}):
        """
        Return done=True when step_count >= max_steps.

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        done = (env.step_count - self.max_steps) >= 0
        bad_done = torch.zeros_like(done)
        exceed_time_limit = torch.zeros_like(done)
        if torch.any(done):
            self.log(f"simulink episode done (max_steps reached)!")
            print(torch.sum(done).item(), "simulink episode done (max_steps reached)!")
        return bad_done, done, exceed_time_limit, info
