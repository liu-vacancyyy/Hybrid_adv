import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class RCReward(BaseRewardFunction):
    """Dense shaped reward for RC tracking task."""

    def __init__(self, config):
        super().__init__(config)
        self.w_alive    = float(getattr(config, 'rc_w_alive',    0.2))
        self.w_vx       = float(getattr(config, 'rc_w_vx',       2.0))
        self.w_vz       = float(getattr(config, 'rc_w_vz',       2.0))
        self.w_heading  = float(getattr(config, 'rc_w_heading',  1.5))
        self.w_attitude = float(getattr(config, 'rc_w_attitude', 1.0))
        self.w_omega    = float(getattr(config, 'rc_w_omega',    0.5))
        self.w_smooth   = float(getattr(config, 'rc_w_smooth',   0.3))

        self.sig_vx       = float(getattr(config, 'rc_sig_vx',       1.0))
        self.sig_vz       = float(getattr(config, 'rc_sig_vz',       1.0))
        self.sig_yaw      = float(getattr(config, 'rc_sig_yaw',      0.3))
        self.sig_attitude = float(getattr(config, 'rc_sig_attitude', 0.3))
        self.sig_omega    = float(getattr(config, 'rc_sig_omega',    1.5))
        self.sig_smooth   = float(getattr(config, 'rc_sig_smooth',   5.0))

    def get_reward(self, task, env):
        roll, pitch, heading = env.model.get_posture()
        P, Q, R              = env.model.get_angular_velocity()
        vx, _vy              = env.model.get_ground_speed()
        vz                   = env.model.get_climb_rate()

        d_vx  = vx - task.target_vx
        d_vz  = vz - task.target_vz
        d_yaw = wrap_PI(heading - task.target_heading)

        att_sq   = roll * roll + pitch * pitch
        omega_sq = P * P + Q * Q + R * R

        r_vx       = torch.exp(-(d_vx ** 2)  / (self.sig_vx       ** 2))
        r_vz       = torch.exp(-(d_vz ** 2)  / (self.sig_vz       ** 2))
        r_yaw      = torch.exp(-(d_yaw ** 2) / (self.sig_yaw      ** 2))
        r_attitude = torch.exp(-att_sq       / (self.sig_attitude ** 2))
        r_omega    = torch.exp(-omega_sq     / (self.sig_omega    ** 2))

        delta_u = env.model.u - env.model.recent_u
        delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
        r_smooth = torch.exp(-delta_u_sq / (self.sig_smooth ** 2))

        reward = (
            self.w_alive
            + self.w_vx       * r_vx
            + self.w_vz       * r_vz
            + self.w_heading  * r_yaw
            + self.w_attitude * r_attitude
            + self.w_omega    * r_omega
            + self.w_smooth   * r_smooth
        )
        return reward
