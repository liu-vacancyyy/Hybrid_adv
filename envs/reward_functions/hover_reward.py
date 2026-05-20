import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class HoverReward(BaseRewardFunction):
    """Position-aware hover / goto reward.

    Encourages, in this order:
        - hold target horizontal position (delta_xy -> 0)
        - hold target altitude            (delta_alt -> 0)
        - hold target heading             (delta_yaw -> 0)
        - level attitude                  (roll, pitch -> 0)
        - zero translational velocity     (vx, vy, vz -> 0)
        - zero angular rate               (P, Q, R -> 0)
        - smooth control                  (||u_t - u_{t-1}||)
        - alive bonus

    Each shaped term is exp(-x^2 / sigma^2) in [0, 1].

    Required task attributes:
        target_npos, target_epos, target_altitude, target_heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.w_alive    = getattr(config, 'hover_w_alive',    0.2)
        self.w_xy       = getattr(config, 'hover_w_xy',       2.0)
        self.w_alt      = getattr(config, 'hover_w_alt',      1.0)
        self.w_yaw      = getattr(config, 'hover_w_yaw',      0.5)
        self.w_attitude = getattr(config, 'hover_w_attitude', 1.0)
        self.w_vel      = getattr(config, 'hover_w_vel',      1.0)
        self.w_omega    = getattr(config, 'hover_w_omega',    0.5)
        self.w_smooth   = getattr(config, 'hover_w_smooth',   0.3)

        # exp(-x^2 / sigma^2) scales
        self.sig_xy       = getattr(config, 'hover_sig_xy',       3.0)   # m
        self.sig_alt      = getattr(config, 'hover_sig_alt',      1.0)   # m
        self.sig_yaw      = getattr(config, 'hover_sig_yaw',      0.3)   # rad
        self.sig_attitude = getattr(config, 'hover_sig_attitude', 0.2)   # rad
        self.sig_vel      = getattr(config, 'hover_sig_vel',      1.5)   # m/s
        self.sig_omega    = getattr(config, 'hover_sig_omega',    1.5)   # rad/s
        self.sig_smooth   = getattr(config, 'hover_sig_smooth',   5.0)   # N^2 (sum over 5 motors)

    def get_reward(self, task, env):
        npos, epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        P, Q, R              = env.model.get_angular_velocity()
        vx, vy               = env.model.get_ground_speed()
        vz                   = env.model.get_climb_rate()

        d_n   = task.target_npos     - npos
        d_e   = task.target_epos     - epos
        d_alt = task.target_altitude - altitude
        d_yaw = wrap_PI(task.target_heading - heading)

        xy_sq    = d_n * d_n + d_e * d_e
        att_sq   = roll * roll + pitch * pitch
        vel_sq   = vx * vx + vy * vy + vz * vz
        omega_sq = P * P + Q * Q + R * R

        r_xy       = torch.exp(-xy_sq      / (self.sig_xy       ** 2))
        r_alt      = torch.exp(-(d_alt**2) / (self.sig_alt      ** 2))
        r_yaw      = torch.exp(-(d_yaw**2) / (self.sig_yaw      ** 2))
        r_attitude = torch.exp(-att_sq     / (self.sig_attitude ** 2))
        r_vel      = torch.exp(-vel_sq     / (self.sig_vel      ** 2))
        r_omega    = torch.exp(-omega_sq   / (self.sig_omega    ** 2))

        delta_u = env.model.u - env.model.recent_u
        delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
        r_smooth = torch.exp(-delta_u_sq / (self.sig_smooth ** 2))

        reward = (
            self.w_alive
            + self.w_xy       * r_xy
            + self.w_alt      * r_alt
            + self.w_yaw      * r_yaw
            + self.w_attitude * r_attitude
            + self.w_vel      * r_vel
            + self.w_omega    * r_omega
            + self.w_smooth   * r_smooth
        )
        return reward
