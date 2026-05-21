import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from termination_condition_base import BaseTerminationCondition
from utils.utils import wrap_PI


class RCHumanTrackingError(BaseTerminationCondition):
    """Terminate rc_human episodes when command tracking is persistently poor."""

    def __init__(self, config):
        super().__init__(config)
        self.enable = bool(getattr(config, 'rc_human_tracking_bad_done_enable', True))
        yaw_deg = float(getattr(config, 'rc_human_bad_done_yaw_error_deg', 20.0))
        self.yaw_error_limit = yaw_deg * torch.pi / 180.0
        self.axis_vel_error_limit = float(
            getattr(config, 'rc_human_bad_done_axis_vel_error', 0.5)
        )
        self.grace_steps = int(getattr(config, 'rc_human_bad_done_grace_steps', 50))
        self.persist_steps = max(
            1, int(getattr(config, 'rc_human_bad_done_persist_steps', 10))
        )
        self.violation_count = None

    def _ensure_state(self, env):
        if (
            self.violation_count is None
            or self.violation_count.numel() != env.n
            or self.violation_count.device != env.device
        ):
            self.violation_count = torch.zeros(
                env.n, dtype=torch.long, device=env.device
            )

    def get_termination(self, task, env, info={}):
        self._ensure_state(env)
        done = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        exceed_time_limit = torch.zeros_like(done)
        if not self.enable:
            bad_done = torch.zeros_like(done)
            return bad_done, done, exceed_time_limit, info

        _roll, _pitch, heading = env.model.get_posture()
        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        local_vx, local_vy = task.ground_to_local_velocity(vx_n, vy_e, heading)

        yaw_error = torch.abs(wrap_PI(heading - task.target_heading))
        axis_vel_error = torch.max(
            torch.stack((
                torch.abs(local_vx - task.target_vx),
                torch.abs(local_vy - task.target_vy),
                torch.abs(vz - task.target_vz),
            ), dim=0),
            dim=0,
        ).values

        active = env.step_count > self.grace_steps
        violation = active & (
            (yaw_error > self.yaw_error_limit)
            | (axis_vel_error > self.axis_vel_error_limit)
        )
        self.violation_count = torch.where(
            violation,
            self.violation_count + 1,
            torch.zeros_like(self.violation_count),
        )
        bad_done = self.violation_count >= self.persist_steps

        if torch.any(bad_done):
            self.log('rc_human tracking error is too high!')
            if getattr(self.config, 'termination_verbose', True):
                print(torch.sum(bad_done), 'rc_human tracking error is too high!')
        return bad_done, done, exceed_time_limit, info
