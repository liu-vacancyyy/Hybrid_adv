import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class SimulinkDivergence(BaseTerminationCondition):
    """
    SimulinkDivergence
    Terminate episode if the current tracking error exceeds 120% of the initial error.
    This prevents the controller from making the situation worse.
    """

    def __init__(self, config):
        super().__init__(config)
        self.divergence_ratio = getattr(config, 'divergence_ratio', 1.2)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        bad_done if |theta - target| > 120% * |initial_theta - target|

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        theta = env.model.s[:, 3]
        current_error = torch.abs(theta - task.target_theta)
        initial_error = task.step_magnitude  # |initial_theta - target|
        threshold = self.divergence_ratio * initial_error

        bad_done = current_error > threshold
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'simulink divergence!')
            print(torch.sum(bad_done), 'simulink divergence! error exceeded 120% of initial')
        return bad_done, done, exceed_time_limit, info
