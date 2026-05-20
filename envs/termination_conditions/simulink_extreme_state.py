import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class SimulinkExtremeState(BaseTerminationCondition):
    """
    SimulinkExtremeState
    End up the simulation if the Simulink model is on an extreme state.
    Checks: theta, q, u, w
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_theta = getattr(config, 'max_theta', 3.0)
        self.max_q = getattr(config, 'max_q', 10.0)
        self.max_u = getattr(config, 'max_u', 200.0)
        self.max_w = getattr(config, 'max_w', 50.0)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the Simulink model is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        u = env.model.s[:, 0]
        w = env.model.s[:, 1]
        q = env.model.s[:, 2]
        theta = env.model.s[:, 3]

        mask_theta = torch.abs(theta) > self.max_theta
        mask_q = torch.abs(q) > self.max_q
        mask_u = torch.abs(u) > self.max_u
        mask_w = torch.abs(w) > self.max_w

        bad_done = mask_theta | mask_q | mask_u | mask_w
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'simulink extreme state!')
            print(torch.sum(bad_done), 'simulink extreme state!')
        return bad_done, done, exceed_time_limit, info
