import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class RCHumanReward(BaseRewardFunction):
    """Dense reward for human-like RC command tracking."""

    def __init__(self, config):
        super().__init__(config)
        self.w_alive = float(getattr(config, 'rc_human_w_alive', 0.2))
        self.w_vel = float(getattr(config, 'rc_human_w_vel', 5.0))
        self.w_yaw = float(getattr(config, 'rc_human_w_yaw', 1.5))
        self.w_attitude = float(getattr(config, 'rc_human_w_attitude', 1.2))
        self.w_omega = float(getattr(config, 'rc_human_w_omega', 0.6))
        self.w_smooth = float(getattr(config, 'rc_human_w_smooth', 0.5))

        self.sig_vel = float(getattr(config, 'rc_human_sig_vel', 0.45))
        self.sig_vx = float(getattr(config, 'rc_human_sig_vx', self.sig_vel))
        self.sig_vy = float(getattr(config, 'rc_human_sig_vy', self.sig_vel))
        self.sig_vz = float(getattr(config, 'rc_human_sig_vz', self.sig_vel))
        self.sig_yaw = float(getattr(config, 'rc_human_sig_yaw', 0.22))
        self.sig_attitude = float(getattr(config, 'rc_human_sig_attitude', 0.25))
        self.sig_omega = float(getattr(config, 'rc_human_sig_omega', 1.5))
        self.sig_smooth = float(getattr(config, 'rc_human_sig_smooth', 1.2))

    def get_reward(self, task, env):
        task.sync_command(env)

        roll, pitch, heading = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()

        dvx, dvy = task.ground_to_local_velocity(
            vx_n - task.target_vn, vy_e - task.target_ve, heading
        )
        dvz = vz - task.target_vz
        dyaw = wrap_PI(heading - task.target_heading)

        vel_sq = dvx * dvx + dvy * dvy + dvz * dvz
        vel_norm_sq = (
            (dvx / max(self.sig_vx, 1e-6)) ** 2
            + (dvy / max(self.sig_vy, 1e-6)) ** 2
            + (dvz / max(self.sig_vz, 1e-6)) ** 2
        )
        att_sq = roll * roll + pitch * pitch
        omega_sq = p * p + q * q + r * r
        task.update_episode_metrics(torch.sqrt(vel_sq), dyaw, torch.sqrt(att_sq))

        r_vel = torch.exp(-vel_norm_sq)
        r_yaw = torch.exp(-(dyaw * dyaw) / (self.sig_yaw ** 2))
        r_attitude = torch.exp(-att_sq / (self.sig_attitude ** 2))
        r_omega = torch.exp(-omega_sq / (self.sig_omega ** 2))

        reward = (
            self.w_alive
            + self.w_vel * r_vel
            + self.w_yaw * r_yaw
            + self.w_attitude * r_attitude
            + self.w_omega * r_omega
        )

        if self.w_smooth > 0.0:
            delta_u = env.model.u - env.model.recent_u
            delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
            r_smooth = torch.exp(-delta_u_sq / (self.sig_smooth ** 2))
            reward = reward + self.w_smooth * r_smooth

        return reward
