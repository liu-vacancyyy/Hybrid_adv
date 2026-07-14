import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class VelocityHoverReward(BaseRewardFunction):
    """Dense reward for zero-velocity hover command tracking."""

    def __init__(self, config):
        super().__init__(config)
        self.w_alive = float(getattr(config, 'velocity_hover_w_alive', 0.2))
        self.w_vel = float(getattr(config, 'velocity_hover_w_vel', 4.0))
        self.w_yaw = float(getattr(config, 'velocity_hover_w_yaw', 0.5))
        self.w_attitude = float(getattr(config, 'velocity_hover_w_attitude', 1.0))
        self.w_omega = float(getattr(config, 'velocity_hover_w_omega', 0.6))
        self.w_smooth = float(getattr(config, 'velocity_hover_w_smooth', 0.4))

        self.sig_vx = float(getattr(config, 'velocity_hover_sig_vx', 0.25))
        self.sig_vy = float(getattr(config, 'velocity_hover_sig_vy', 0.25))
        self.sig_vz = float(getattr(config, 'velocity_hover_sig_vz', 0.20))
        self.sig_yaw = float(getattr(config, 'velocity_hover_sig_yaw', 0.35))
        self.sig_attitude = float(getattr(config, 'velocity_hover_sig_attitude', 0.18))
        self.sig_omega = float(getattr(config, 'velocity_hover_sig_omega', 0.8))
        self.sig_smooth = float(getattr(config, 'velocity_hover_sig_smooth', 0.20))
        self.smooth_norm = float(getattr(config, 'velocity_hover_smooth_norm', 1500.0))

    def get_reward(self, task, env):
        roll, pitch, heading = env.model.get_posture()
        vx, vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        p, q, r = env.model.get_angular_velocity()

        err_vx = vx - task.target_vx
        err_vy = vy - task.target_vy
        err_vz = vz - task.target_vz
        vel_cost = (
            (err_vx / self.sig_vx) * (err_vx / self.sig_vx)
            + (err_vy / self.sig_vy) * (err_vy / self.sig_vy)
            + (err_vz / self.sig_vz) * (err_vz / self.sig_vz)
        )
        yaw_error = wrap_PI(task.target_heading - heading)
        att_sq = roll * roll + pitch * pitch
        omega_sq = p * p + q * q + r * r

        delta_u = env.model.u[:, :5] - env.model.recent_u[:, :5]
        smooth_sq = torch.sum((delta_u / max(self.smooth_norm, 1e-6)) ** 2, dim=1)

        r_vel = torch.exp(-vel_cost)
        r_yaw = torch.exp(-(yaw_error * yaw_error) / (self.sig_yaw ** 2))
        r_attitude = torch.exp(-att_sq / (self.sig_attitude ** 2))
        r_omega = torch.exp(-omega_sq / (self.sig_omega ** 2))
        r_smooth = torch.exp(-smooth_sq / (self.sig_smooth ** 2))

        return (
            self.w_alive
            + self.w_vel * r_vel
            + self.w_yaw * r_yaw
            + self.w_attitude * r_attitude
            + self.w_omega * r_omega
            + self.w_smooth * r_smooth
        )


class VelocityHoverEventReward(BaseRewardFunction):
    """Large terminal penalty for failed velocity-hover episodes."""

    def __init__(self, config):
        super().__init__(config)
        self.bad_done_base = float(getattr(config, 'velocity_hover_bad_done_base', 50.0))
        self.bad_done_per_step = float(getattr(config, 'velocity_hover_bad_done_per_step', 3.0))
        self.max_steps = int(getattr(config, 'max_steps', 1500))

    def get_reward(self, task, env):
        bad = env.bad_done.float()
        remaining = (self.max_steps - env.step_count).clamp_min(0).float()
        penalty = self.bad_done_base + self.bad_done_per_step * remaining
        return -penalty * bad
