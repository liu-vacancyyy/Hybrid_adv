import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition


class BodySideVelocity(BaseTerminationCondition):
    """LearningToFly-style hard guard on body-frame lateral velocity."""

    def __init__(self, config):
        super().__init__(config)
        self.max_y_velocity = float(getattr(
            config,
            'max_body_y_velocity',
            getattr(config, 'max_y_velocity', 2.0),
        ))

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.

        LearningToFly terminates when abs(velocity_body[1]) > 2.  In this
        codebase env.model.get_velocity() returns body-frame (U, V, W), so V
        is the equivalent side velocity.
        """
        _u, v, _w = env.model.get_velocity()
        bad_done = torch.abs(v) > self.max_y_velocity
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log('body side velocity is too high!')
            if getattr(self.config, 'termination_verbose', True):
                print(torch.sum(bad_done), 'body side velocity is too high!')
        return bad_done, done, exceed_time_limit, info
