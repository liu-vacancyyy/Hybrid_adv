import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class CircleReward(BaseRewardFunction):
    """Shaped reward for moving circle trajectory tracking.

    The target is not static, so this reward differs from HoverReward by adding
    a target-velocity term.  The policy is encouraged to stay near the moving
    point on the circle while matching its tangent velocity.
    """

    def __init__(self, config):
        super().__init__(config)
        self.w_alive = float(getattr(config, 'circle_w_alive', 0.2))
        self.w_xy = float(getattr(config, 'circle_w_xy', 3.0))
        self.w_alt = float(getattr(config, 'circle_w_alt', 1.2))
        self.w_yaw = float(getattr(config, 'circle_w_yaw', 0.8))
        self.w_vel = float(getattr(config, 'circle_w_vel', 1.2))
        self.w_attitude = float(getattr(config, 'circle_w_attitude', 0.6))
        self.w_omega = float(getattr(config, 'circle_w_omega', 0.4))
        self.w_smooth = float(getattr(config, 'circle_w_smooth', 0.5))

        self.sig_xy = float(getattr(config, 'circle_sig_xy', 2.0))
        self.sig_alt = float(getattr(config, 'circle_sig_alt', 0.6))
        self.sig_yaw = float(getattr(config, 'circle_sig_yaw', 0.35))
        self.sig_vel = float(getattr(config, 'circle_sig_vel', 1.0))
        self.sig_attitude = float(getattr(config, 'circle_sig_attitude', 0.25))
        self.sig_omega = float(getattr(config, 'circle_sig_omega', 1.5))
        self.sig_smooth = float(getattr(config, 'circle_sig_smooth', 1.5))

    def get_reward(self, task, env):
        task.sync_target_to_time(env)
        npos, epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        vx, vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        P, Q, R = env.model.get_angular_velocity()

        d_n = task.target_npos - npos
        d_e = task.target_epos - epos
        d_alt = task.target_altitude - altitude
        d_yaw = wrap_PI(task.target_heading - heading)

        xy_sq = d_n * d_n + d_e * d_e
        vel_n = vx - task.target_vn
        vel_e = vy - task.target_ve
        vel_sq = vel_n * vel_n + vel_e * vel_e + vz * vz
        att_sq = roll * roll + pitch * pitch
        omega_sq = P * P + Q * Q + R * R
        delta_u = env.model.u - env.model.recent_u
        delta_u_sq = torch.sum(delta_u * delta_u, dim=1)

        r_xy = torch.exp(-xy_sq / (self.sig_xy ** 2))
        r_alt = torch.exp(-(d_alt * d_alt) / (self.sig_alt ** 2))
        r_yaw = torch.exp(-(d_yaw * d_yaw) / (self.sig_yaw ** 2))
        r_vel = torch.exp(-vel_sq / (self.sig_vel ** 2))
        r_attitude = torch.exp(-att_sq / (self.sig_attitude ** 2))
        r_omega = torch.exp(-omega_sq / (self.sig_omega ** 2))
        r_smooth = torch.exp(-delta_u_sq / (self.sig_smooth ** 2))

        return (
            self.w_alive
            + self.w_xy * r_xy
            + self.w_alt * r_alt
            + self.w_yaw * r_yaw
            + self.w_vel * r_vel
            + self.w_attitude * r_attitude
            + self.w_omega * r_omega
            + self.w_smooth * r_smooth
        )
