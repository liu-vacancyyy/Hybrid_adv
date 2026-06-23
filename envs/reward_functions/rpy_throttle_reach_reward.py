import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class RPYThrottleReachReward(BaseRewardFunction):
    """Dense reward for one-shot roll/pitch/yaw/throttle reach tasks."""

    def __init__(self, config):
        super().__init__(config)
        p = 'rpy_throttle_reach_'
        self.constraints_as_cost = bool(getattr(config, p + 'constraints_as_cost', False))
        self.w_alive = float(getattr(config, p + 'w_alive', 0.05))
        self.w_attitude = float(getattr(config, p + 'w_attitude', 5.0))
        self.w_yaw = float(getattr(config, p + 'w_yaw', 2.0))
        self.w_throttle = float(getattr(config, p + 'w_throttle', 2.0))
        self.w_omega = float(getattr(config, p + 'w_omega', 1.2))
        self.w_smooth = float(getattr(config, p + 'w_smooth', 0.5))
        self.w_overshoot = float(getattr(config, p + 'w_overshoot', 1.8))
        self.w_moving_away = float(getattr(config, p + 'w_moving_away', 0.5))
        self.w_safety = float(getattr(config, p + 'w_safety', 1.5))
        self.w_danger = float(getattr(config, p + 'w_danger', 1.2))
        self.w_override = float(getattr(config, p + 'w_override', 0.08))
        self.w_saturation = float(getattr(config, p + 'w_saturation', 0.25))

        self.sig_roll = float(getattr(config, p + 'sig_roll', 0.045))
        self.sig_pitch = float(getattr(config, p + 'sig_pitch', 0.045))
        self.sig_yaw = float(getattr(config, p + 'sig_yaw', 0.08))
        self.sig_throttle = float(getattr(config, p + 'sig_throttle', 0.055))
        self.sig_omega = float(getattr(config, p + 'sig_omega', 0.7))
        self.sig_smooth = float(getattr(config, p + 'sig_smooth', 0.8))
        self.sig_overshoot = float(getattr(config, p + 'sig_overshoot', 0.25))

    def get_reward(self, task, env):
        task.sync_command(env)

        roll, pitch, yaw = env.model.get_posture()
        roll_dot, pitch_dot, yaw_rate = env.model.get_euler_angular_velocity()
        p, q, r = env.model.get_angular_velocity()
        f0, f1, f2, f3, f4 = env.model.get_F()

        roll_error = wrap_PI(roll - task.target_roll)
        pitch_error = wrap_PI(pitch - task.target_pitch)
        yaw_error = wrap_PI(yaw - task.target_yaw)
        collective = f1 + f2 + f3 + f4
        hover = torch.clamp(task.hover_collective, min=1e-6)
        throttle_error = (collective - task.target_collective) / hover
        prev_throttle_error = task.prev_throttle_error.detach()

        attitude_error = torch.sqrt(
            roll_error * roll_error + pitch_error * pitch_error
        )
        overshoot_score = task.compute_overshoot_score(
            roll_error, pitch_error, yaw_error, throttle_error
        )
        danger_score, safety_score = task.compute_safety_scores(env)
        task.update_constraint_terms(env)
        task.update_episode_metrics(
            attitude_error,
            torch.abs(yaw_error),
            torch.abs(throttle_error),
            overshoot_score,
            danger_score,
        )

        r_attitude = torch.exp(-(
            (roll_error / max(self.sig_roll, 1e-6)) ** 2
            + (pitch_error / max(self.sig_pitch, 1e-6)) ** 2
        ))
        r_yaw = torch.exp(-(
            yaw_error * yaw_error / max(self.sig_yaw ** 2, 1e-6)
        ))
        r_throttle = torch.exp(-(
            throttle_error * throttle_error / max(self.sig_throttle ** 2, 1e-6)
        ))
        omega_sq = p * p + q * q + r * r
        r_omega = torch.exp(-omega_sq / max(self.sig_omega ** 2, 1e-6))
        r_overshoot = torch.exp(-(
            overshoot_score * overshoot_score / max(self.sig_overshoot ** 2, 1e-6)
        ))

        moving_away = (
            torch.relu(roll_error * roll_dot)
            + torch.relu(pitch_error * pitch_dot)
            + torch.relu(yaw_error * yaw_rate)
            + torch.relu(throttle_error * (throttle_error - prev_throttle_error))
        )
        saturation = torch.mean(torch.relu(torch.abs(env.model.u / 7.0) - 0.92), dim=1)

        reward_alive = torch.full_like(r_attitude, self.w_alive)
        reward_attitude = self.w_attitude * r_attitude
        reward_yaw = self.w_yaw * r_yaw
        reward_throttle = self.w_throttle * r_throttle
        reward_omega = self.w_omega * r_omega
        reward_overshoot = self.w_overshoot * r_overshoot
        reward_moving_away = -self.w_moving_away * moving_away
        if self.constraints_as_cost:
            reward_safety = torch.zeros_like(reward_alive)
            reward_danger = torch.zeros_like(reward_alive)
            reward_override = torch.zeros_like(reward_alive)
        else:
            reward_safety = self.w_safety * safety_score
            reward_danger = -self.w_danger * danger_score
            reward_override = -self.w_override * task.safety_override_active.float()
        reward_saturation = -self.w_saturation * saturation

        if self.w_smooth > 0.0:
            delta_u = env.model.u - env.model.recent_u
            delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
            r_smooth = torch.exp(-delta_u_sq / max(self.sig_smooth ** 2, 1e-6))
            reward_smooth = self.w_smooth * r_smooth
        else:
            reward_smooth = torch.zeros_like(reward_alive)
            r_smooth = torch.zeros_like(reward_alive)

        reward = (
            reward_alive
            + reward_attitude
            + reward_yaw
            + reward_throttle
            + reward_omega
            + reward_smooth
            + reward_overshoot
            + reward_moving_away
            + reward_safety
            + reward_danger
            + reward_override
            + reward_saturation
        )

        task.last_reward_terms = {
            'reward/total_mean': reward.detach().mean(),
            'reward/alive_mean': reward_alive.detach().mean(),
            'reward/attitude_mean': reward_attitude.detach().mean(),
            'reward/yaw_mean': reward_yaw.detach().mean(),
            'reward/throttle_mean': reward_throttle.detach().mean(),
            'reward/omega_mean': reward_omega.detach().mean(),
            'reward/smooth_mean': reward_smooth.detach().mean(),
            'reward/overshoot_mean': reward_overshoot.detach().mean(),
            'reward/moving_away_mean': reward_moving_away.detach().mean(),
            'reward/safety_mean': reward_safety.detach().mean(),
            'reward/danger_mean': reward_danger.detach().mean(),
            'reward/override_mean': reward_override.detach().mean(),
            'reward/saturation_mean': reward_saturation.detach().mean(),
            'reward/raw_attitude_score_mean': r_attitude.detach().mean(),
            'reward/raw_yaw_score_mean': r_yaw.detach().mean(),
            'reward/raw_throttle_score_mean': r_throttle.detach().mean(),
            'reward/raw_omega_score_mean': r_omega.detach().mean(),
            'reward/raw_smooth_score_mean': r_smooth.detach().mean(),
            'reward/raw_overshoot_score_mean': r_overshoot.detach().mean(),
            'reward/raw_safety_score_mean': safety_score.detach().mean(),
            'reward/raw_danger_score_mean': danger_score.detach().mean(),
        }
        return reward


class RPYThrottleReachEventReward(BaseRewardFunction):
    """Terminal reward/penalty for reach task success and bad_done."""

    def __init__(self, config):
        super().__init__(config)
        p = 'rpy_throttle_reach_'
        self.constraints_as_cost = bool(getattr(config, p + 'constraints_as_cost', False))
        self.bad_done_penalty = float(getattr(config, p + 'bad_done_penalty', 220.0))
        self.clean_done_bonus = float(getattr(config, p + 'clean_done_bonus', 8.0))
        self.success_bonus = float(getattr(config, p + 'success_bonus', 40.0))

    def get_reward(self, task, env):
        bad_done = env.bad_done.bool()
        clean_done = env.is_done.bool() & (~bad_done)
        success = clean_done & task.episode_success_mask(env)
        reward = torch.zeros(env.n, device=env.device)
        if not self.constraints_as_cost:
            reward = reward - self.bad_done_penalty * bad_done.float()
        reward = reward + self.clean_done_bonus * clean_done.float()
        reward = reward + self.success_bonus * success.float()
        return reward
