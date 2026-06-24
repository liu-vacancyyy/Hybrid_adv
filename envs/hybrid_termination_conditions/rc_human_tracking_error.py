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
        self.vxyvz_dynamic_enable = bool(
            getattr(config, 'rc_human_vxyvz_dynamic_bad_done_enable', False)
        )
        self.vxyvz_dynamic_margin = float(
            getattr(config, 'rc_human_vxyvz_dynamic_bad_done_margin', 0.5)
        )
        self.yaw_tracking_enable = bool(
            getattr(config, 'rc_human_yaw_tracking_enable', True)
        )
        yaw_deg = float(getattr(config, 'rc_human_bad_done_yaw_error_deg', 20.0))
        self.yaw_error_limit = yaw_deg * torch.pi / 180.0
        self.axis_vel_error_limit = float(
            getattr(config, 'rc_human_bad_done_axis_vel_error', 0.5)
        )
        self.vx_error_min = float(
            getattr(config, 'rc_human_bad_done_vx_error_min', self.axis_vel_error_limit)
        )
        self.vx_error_frac = float(
            getattr(config, 'rc_human_bad_done_vx_error_frac', 0.0)
        )
        self.vy_error_limit = float(
            getattr(config, 'rc_human_bad_done_vy_error', self.axis_vel_error_limit)
        )
        self.vz_error_limit = float(
            getattr(config, 'rc_human_bad_done_vz_error', self.axis_vel_error_limit)
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
        if (not self.enable) and (not self.vxyvz_dynamic_enable):
            bad_done = torch.zeros_like(done)
            return bad_done, done, exceed_time_limit, info

        _roll, _pitch, heading = env.model.get_posture()
        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()

        yaw_error = torch.abs(wrap_PI(heading - task.target_heading))
        local_vx_error, local_vy_error = task.heading_local_velocity_error(
            vx_n, vy_e, heading
        )
        vx_error = torch.abs(local_vx_error)
        vy_error = torch.abs(local_vy_error)
        vz_error = torch.abs(vz - task.target_vz)
        vx_limit = torch.maximum(
            torch.full_like(vx_error, self.vx_error_min),
            torch.abs(task.target_vx) * self.vx_error_frac,
        )

        active = env.step_count > self.grace_steps
        transient_left = getattr(task, 'command_transient_left', None)
        if transient_left is None:
            velocity_tracking_active = torch.ones_like(active, dtype=torch.bool)
        else:
            velocity_tracking_active = transient_left <= 0
        mode5_state = getattr(task, 'mode5_release_state', None)
        if mode5_state is not None:
            velocity_tracking_active = velocity_tracking_active & (mode5_state != 1)
        if self.enable and self.yaw_tracking_enable:
            yaw_violation = yaw_error > self.yaw_error_limit
        else:
            yaw_violation = torch.zeros_like(yaw_error, dtype=torch.bool)
        if self.enable:
            velocity_violation = velocity_tracking_active & (
                (vx_error > vx_limit)
                | (vy_error > self.vy_error_limit)
                | (vz_error > self.vz_error_limit)
            )
        else:
            velocity_violation = torch.zeros_like(vx_error, dtype=torch.bool)

        if self.vxyvz_dynamic_enable:
            initial_local_vx = getattr(task, 'initial_local_vx', torch.zeros_like(vx_error))
            initial_local_vy = getattr(task, 'initial_local_vy', torch.zeros_like(vy_error))
            initial_vz = getattr(task, 'initial_vz', torch.zeros_like(vz_error))
            dynamic_vx_limit = torch.abs(task.target_vx - initial_local_vx) + self.vxyvz_dynamic_margin
            dynamic_vy_limit = torch.abs(task.target_vy - initial_local_vy) + self.vxyvz_dynamic_margin
            dynamic_vz_limit = torch.abs(task.target_vz - initial_vz) + self.vxyvz_dynamic_margin
            dynamic_velocity_violation = velocity_tracking_active & (
                (vx_error > dynamic_vx_limit)
                | (vy_error > dynamic_vy_limit)
                | (vz_error > dynamic_vz_limit)
            )
        else:
            dynamic_velocity_violation = torch.zeros_like(vx_error, dtype=torch.bool)

        violation = active & (
            yaw_violation | velocity_violation | dynamic_velocity_violation
        )
        self.violation_count = torch.where(
            violation,
            self.violation_count + 1,
            torch.zeros_like(self.violation_count),
        )
        bad_done = self.violation_count >= self.persist_steps

        if torch.any(bad_done):
            message = 'rc_human tracking error is too high!'
            if self.vxyvz_dynamic_enable and not self.enable:
                message = 'rc_human dynamic vxvyvz tracking error is too high!'
            self.log(message)
            if getattr(self.config, 'termination_verbose', True):
                print(torch.sum(bad_done), message)
        return bad_done, done, exceed_time_limit, info
