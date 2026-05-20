import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class ExtremeOmega(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_omega = getattr(config, 'max_omega', 2)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        omega1, omega2, omega3 = env.model.get_euler_angular_velocity()
        bad_done = (torch.abs(omega1) > self.max_omega) | (torch.abs(omega2) > self.max_omega) | (torch.abs(omega3) > self.max_omega)
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'extreme omega!')
            if getattr(self.config, 'termination_verbose', True):
                print(torch.sum(bad_done), 'extreme omega!')
        return bad_done, done, exceed_time_limit, info
