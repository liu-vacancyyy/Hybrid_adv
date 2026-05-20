"""Circle trajectory PID controller for the HybridModel.

Compared with ``HoverPIDController``, this controller adds trajectory
feedforward:
    - target circle velocity is added to the outer position loop;
    - target centripetal acceleration is added to the acceleration command;
    - optional head motor assist provides forward acceleration when aligned.
"""
import math

import torch

from algorithms.pid.hover_pid import HoverPIDController


class CirclePIDController(HoverPIDController):
    """Position PID tuned for moving circular targets."""

    POS_XY_P = 0.65
    MAX_VEL_XY = 2.0
    VEL_XY_P = 1.0
    VEL_XY_I = 0.12
    VEL_XY_IMAX = 0.40
    MAX_TILT_XY = math.radians(12.0)
    MAX_HORIZ_ACCEL = 1.8

    def __init__(self, *args, use_head_motor=True, head_max_force=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_head_motor = bool(use_head_motor)
        self.head_max_force = float(head_max_force)
        self.target_vn = torch.zeros(self.n, device=self.device)
        self.target_ve = torch.zeros(self.n, device=self.device)
        self.target_an = torch.zeros(self.n, device=self.device)
        self.target_ae = torch.zeros(self.n, device=self.device)

    def set_circle_targets(self, task):
        """Sync position, heading, velocity and acceleration from CircleTask."""
        self.set_targets(
            target_altitude=task.target_altitude,
            target_heading=task.target_heading,
            target_npos=task.target_npos,
            target_epos=task.target_epos,
        )

        radius = float(getattr(task, 'radius', 10.0))
        omega = float(getattr(task, 'omega', 0.0))
        if hasattr(task, 'target_vn') and hasattr(task, 'target_an'):
            self.target_vn = task.target_vn.to(self.device)
            self.target_ve = task.target_ve.to(self.device)
            self.target_an = task.target_an.to(self.device)
            self.target_ae = task.target_ae.to(self.device)
            return

        phase = task.phase.to(self.device)
        self.target_vn = -radius * omega * torch.sin(phase)
        self.target_ve = radius * omega * torch.cos(phase)
        self.target_an = -radius * omega * omega * torch.cos(phase)
        self.target_ae = -radius * omega * omega * torch.sin(phase)

    def _position_loop(self, npos, epos, vn, ve, heading):
        g = 9.807
        err_n = self.target_npos - npos
        err_e = self.target_epos - epos

        target_vn = self.target_vn + self.POS_XY_P * err_n
        target_ve = self.target_ve + self.POS_XY_P * err_e
        target_vn = torch.clamp(target_vn, -self.MAX_VEL_XY, self.MAX_VEL_XY)
        target_ve = torch.clamp(target_ve, -self.MAX_VEL_XY, self.MAX_VEL_XY)

        vel_err_n = target_vn - vn
        vel_err_e = target_ve - ve
        self.vel_int_n = torch.clamp(
            self.vel_int_n + self.VEL_XY_I * vel_err_n * self.dt,
            -self.VEL_XY_IMAX, self.VEL_XY_IMAX)
        self.vel_int_e = torch.clamp(
            self.vel_int_e + self.VEL_XY_I * vel_err_e * self.dt,
            -self.VEL_XY_IMAX, self.VEL_XY_IMAX)

        accel_n = self.target_an + self.VEL_XY_P * vel_err_n + self.vel_int_n
        accel_e = self.target_ae + self.VEL_XY_P * vel_err_e + self.vel_int_e
        accel_norm = torch.sqrt(accel_n * accel_n + accel_e * accel_e).clamp_min(1e-6)
        accel_scale = torch.clamp(self.MAX_HORIZ_ACCEL / accel_norm, max=1.0)
        accel_n = accel_n * accel_scale
        accel_e = accel_e * accel_scale

        cpsi = torch.cos(heading)
        spsi = torch.sin(heading)
        accel_fwd = cpsi * accel_n + spsi * accel_e
        accel_right = -spsi * accel_n + cpsi * accel_e
        target_pitch = torch.clamp(-accel_fwd / g, -self.MAX_TILT_XY, self.MAX_TILT_XY)
        target_roll = torch.clamp(accel_right / g, -self.MAX_TILT_XY, self.MAX_TILT_XY)
        return target_roll, target_pitch

    def _apply_head_motor(self, action, model):
        if not self.use_head_motor:
            return action

        _roll, _pitch, heading = model.get_posture()
        vn, ve = model.get_ground_speed()
        cpsi = torch.cos(heading)
        spsi = torch.sin(heading)
        v_fwd = cpsi * vn + spsi * ve
        target_v_fwd = cpsi * self.target_vn + spsi * self.target_ve
        target_a_fwd = cpsi * self.target_an + spsi * self.target_ae

        speed_deficit = torch.relu(target_v_fwd - v_fwd)
        accel_cmd = torch.relu(target_a_fwd + 0.4 * speed_deficit)
        force = torch.clamp(accel_cmd * 1.779, 0.0, self.head_max_force)

        action = action.clone()
        action[:, 0] = 2.0 * force / model.max_F - 1.0
        return torch.clamp(action, -1.0, 1.0)

    def compute_action(self, model):
        action = super().compute_action(model)
        return self._apply_head_motor(action, model)
