import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class RPYThrottleReward(BaseRewardFunction):
    """Dense reward for roll/pitch/yaw-rate/collective-throttle tracking."""

    def __init__(self, config):
        super().__init__(config)
        self.w_alive = float(getattr(config, 'rpy_throttle_w_alive', 0.2))
        self.w_attitude = float(getattr(config, 'rpy_throttle_w_attitude', 5.0))
        self.w_yaw_rate = float(getattr(config, 'rpy_throttle_w_yaw_rate', 2.0))
        self.w_throttle = float(getattr(config, 'rpy_throttle_w_throttle', 2.0))
        self.w_omega = float(getattr(config, 'rpy_throttle_w_omega', 1.0))
        self.w_smooth = float(getattr(config, 'rpy_throttle_w_smooth', 0.4))
        self.w_velocity = float(getattr(config, 'rpy_throttle_w_velocity', 0.4))
        self.w_overshoot = float(getattr(config, 'rpy_throttle_w_overshoot', 1.0))
        self.w_moving_away = float(getattr(config, 'rpy_throttle_w_moving_away', 0.3))

        self.sig_roll = float(getattr(config, 'rpy_throttle_sig_roll', 0.045))
        self.sig_pitch = float(getattr(config, 'rpy_throttle_sig_pitch', 0.045))
        self.sig_yaw_rate = float(getattr(config, 'rpy_throttle_sig_yaw_rate', 0.12))
        self.sig_throttle = float(getattr(config, 'rpy_throttle_sig_throttle', 0.06))
        self.sig_omega = float(getattr(config, 'rpy_throttle_sig_omega', 0.8))
        self.sig_smooth = float(getattr(config, 'rpy_throttle_sig_smooth', 0.8))
        self.sig_velocity = float(getattr(config, 'rpy_throttle_sig_velocity', 2.5))
        self.sig_overshoot = float(getattr(config, 'rpy_throttle_sig_overshoot', 0.35))

    def get_reward(self, task, env):
        task.sync_command(env)

        roll, pitch, heading = env.model.get_posture()
        roll_dot, pitch_dot, yaw_rate = env.model.get_euler_angular_velocity()
        p, q, r = env.model.get_angular_velocity()
        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        _f0, f1, f2, f3, f4 = env.model.get_F()

        roll_error = wrap_PI(roll - task.target_roll)
        pitch_error = wrap_PI(pitch - task.target_pitch)
        yaw_rate_error = yaw_rate - task.target_yaw_rate

        collective = f1 + f2 + f3 + f4
        hover = torch.clamp(task.hover_collective, min=1e-6)
        throttle_error = (collective - task.target_collective) / hover

        attitude_error = torch.sqrt(roll_error * roll_error + pitch_error * pitch_error)
        overshoot_score = task.compute_overshoot_score(
            roll_error, pitch_error, yaw_rate_error, throttle_error
        )
        task.update_episode_metrics(
            attitude_error,
            torch.abs(yaw_rate_error),
            torch.abs(throttle_error),
            overshoot_score,
        )

        r_attitude = torch.exp(-(
            (roll_error / max(self.sig_roll, 1e-6)) ** 2
            + (pitch_error / max(self.sig_pitch, 1e-6)) ** 2
        ))
        r_yaw_rate = torch.exp(-(
            yaw_rate_error * yaw_rate_error / max(self.sig_yaw_rate ** 2, 1e-6)
        ))
        r_throttle = torch.exp(-(
            throttle_error * throttle_error / max(self.sig_throttle ** 2, 1e-6)
        ))
        omega_sq = p * p + q * q + r * r
        r_omega = torch.exp(-omega_sq / max(self.sig_omega ** 2, 1e-6))
        speed_sq = vx_n * vx_n + vy_e * vy_e + vz * vz
        r_velocity = torch.exp(-speed_sq / max(self.sig_velocity ** 2, 1e-6))
        r_overshoot = torch.exp(-(
            overshoot_score * overshoot_score / max(self.sig_overshoot ** 2, 1e-6)
        ))

        moving_away = (
            torch.relu(roll_error * roll_dot)
            + torch.relu(pitch_error * pitch_dot)
            + torch.relu(yaw_rate_error * yaw_rate)
        )

        reward = (
            self.w_alive
            + self.w_attitude * r_attitude
            + self.w_yaw_rate * r_yaw_rate
            + self.w_throttle * r_throttle
            + self.w_omega * r_omega
            + self.w_velocity * r_velocity
            + self.w_overshoot * r_overshoot
            - self.w_moving_away * moving_away
        )

        if self.w_smooth > 0.0:
            delta_u = env.model.u - env.model.recent_u
            delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
            r_smooth = torch.exp(-delta_u_sq / max(self.sig_smooth ** 2, 1e-6))
            reward = reward + self.w_smooth * r_smooth

        return reward


class RPYThrottleEventDrivenReward(BaseRewardFunction):
    """Penalty for safety terminations."""

    def __init__(self, config):
        super().__init__(config)
        self.bad_done_penalty = float(getattr(config, 'rpy_throttle_bad_done_penalty', 200.0))

    def get_reward(self, task, env):
        return -self.bad_done_penalty * env.bad_done
