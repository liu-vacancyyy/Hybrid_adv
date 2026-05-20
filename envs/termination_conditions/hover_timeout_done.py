import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class HoverTimeoutDone(BaseTerminationCondition):
    """Timeout for hover task: mark as done (not exceed_time_limit)."""

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 500)

    def get_termination(self, task, env, info={}):
        timeout = (env.step_count - self.max_steps) >= 0
        bad_done = torch.zeros_like(timeout)
        done = timeout
        exceed_time_limit = torch.zeros_like(timeout)
        if torch.any(timeout):
            self.log("hover timeout -> done")
        return bad_done, done, exceed_time_limit, info
