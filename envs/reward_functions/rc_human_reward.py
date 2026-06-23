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
        self.w_yaw_rate = float(getattr(config, 'rc_human_w_yaw_rate', 0.0))
        self.w_smooth = float(getattr(config, 'rc_human_w_smooth', 0.5))
        self.yaw_tracking_enable = bool(
            getattr(config, 'rc_human_yaw_tracking_enable', True)
        )
        self.yaw_reward_enable = bool(
            getattr(config, 'rc_human_yaw_reward_enable', self.yaw_tracking_enable)
        )
        self.yaw_reward_mode = str(
            getattr(config, 'rc_human_yaw_reward_mode', 'target')
        ).lower()

        self.sig_vel = float(getattr(config, 'rc_human_sig_vel', 0.45))
        self.sig_vx = float(getattr(config, 'rc_human_sig_vx', self.sig_vel))
        self.sig_vy = float(getattr(config, 'rc_human_sig_vy', self.sig_vel))
        self.sig_vz = float(getattr(config, 'rc_human_sig_vz', self.sig_vel))
        self.sig_yaw = float(getattr(config, 'rc_human_sig_yaw', 0.22))
        self.sig_yaw_delta = float(getattr(config, 'rc_human_sig_yaw_delta', self.sig_yaw))
        self.sig_attitude = float(getattr(config, 'rc_human_sig_attitude', 0.25))
        self.sig_omega = float(getattr(config, 'rc_human_sig_omega', 1.5))
        self.sig_yaw_rate = float(getattr(config, 'rc_human_sig_yaw_rate', self.sig_omega))
        self.sig_smooth = float(getattr(config, 'rc_human_sig_smooth', 1.2))
        self.w_vel_precision = float(getattr(config, 'rc_human_w_vel_precision', 0.0))
        self.w_yaw_precision = float(getattr(config, 'rc_human_w_yaw_precision', 0.0))
        self.rel_error_fixed_scale = float(
            getattr(config, 'rc_human_rel_error_fixed_scale', 0.0)
        )
        self.rel_error_floor = float(getattr(config, 'rc_human_rel_error_floor', self.sig_vel))
        self.w_rel_tracking = float(getattr(config, 'rc_human_w_rel_tracking', 0.0))
        self.w_rel_precision = float(getattr(config, 'rc_human_w_rel_precision', 0.0))
        self.rel_precision_gain = float(getattr(config, 'rc_human_rel_precision_gain', 6.0))
        self.w_overshoot = float(getattr(config, 'rc_human_w_overshoot', 0.0))
        self.overshoot_deadband = float(getattr(config, 'rc_human_overshoot_deadband', 0.05))
        self.w_adaptive_damping = float(getattr(config, 'rc_human_w_adaptive_damping', 0.0))
        self.adaptive_damping_gain = float(getattr(config, 'rc_human_adaptive_damping_gain', 4.0))

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
        dyaw_reward = dyaw
        yaw_sig = self.sig_yaw
        if self.yaw_reward_mode in {'delta', 'step', 'rate'}:
            recent_s = getattr(env.model, 'recent_s', None)
            if recent_s is not None and recent_s.shape[1] > 5:
                dyaw_reward = wrap_PI(heading - recent_s[:, 5])
            yaw_sig = self.sig_yaw_delta

        vel_sq = dvx * dvx + dvy * dvy + dvz * dvz
        vel_norm_sq = (
            (dvx / max(self.sig_vx, 1e-6)) ** 2
            + (dvy / max(self.sig_vy, 1e-6)) ** 2
            + (dvz / max(self.sig_vz, 1e-6)) ** 2
        )
        att_sq = roll * roll + pitch * pitch
        omega_sq = p * p + q * q + r * r
        task.update_episode_metrics(torch.sqrt(vel_sq), dyaw, torch.sqrt(att_sq))

        initial_local_vx = getattr(task, 'initial_local_vx', torch.zeros_like(dvx))
        initial_local_vy = getattr(task, 'initial_local_vy', torch.zeros_like(dvy))
        initial_vz = getattr(task, 'initial_vz', torch.zeros_like(dvz))
        if self.rel_error_fixed_scale > 0.0:
            fixed_scale = max(self.rel_error_fixed_scale, 1e-6)
            scale_vx = torch.full_like(dvx, fixed_scale)
            scale_vy = torch.full_like(dvy, fixed_scale)
            scale_vz = torch.full_like(dvz, fixed_scale)
        else:
            scale_vx = torch.clamp(torch.abs(task.target_vx - initial_local_vx),
                                   min=max(self.rel_error_floor, 1e-6))
            scale_vy = torch.clamp(torch.abs(task.target_vy - initial_local_vy),
                                   min=max(self.rel_error_floor, 1e-6))
            scale_vz = torch.clamp(torch.abs(task.target_vz - initial_vz),
                                   min=max(self.rel_error_floor, 1e-6))
        rel_vx = torch.abs(dvx) / scale_vx
        rel_vy = torch.abs(dvy) / scale_vy
        rel_vz = torch.abs(dvz) / scale_vz
        rel_error = (rel_vx + rel_vy + rel_vz) / 3.0

        r_vel = torch.exp(-vel_norm_sq)
        if self.yaw_reward_enable:
            r_yaw = torch.exp(-(dyaw_reward * dyaw_reward) / (max(yaw_sig, 1e-6) ** 2))
        else:
            r_yaw = torch.zeros_like(r_vel)
        r_attitude = torch.exp(-att_sq / (self.sig_attitude ** 2))
        r_omega = torch.exp(-omega_sq / (self.sig_omega ** 2))
        r_yaw_rate = torch.exp(-(r * r) / (max(self.sig_yaw_rate, 1e-6) ** 2))
        r_rel_precision = torch.exp(-self.rel_precision_gain * rel_error)

        reward_alive = torch.full_like(r_vel, self.w_alive)
        reward_vel = self.w_vel * r_vel
        reward_attitude = self.w_attitude * r_attitude
        reward_omega = self.w_omega * r_omega
        reward_yaw_rate = self.w_yaw_rate * r_yaw_rate
        reward_rel_tracking = -self.w_rel_tracking * rel_error
        reward_rel_precision = self.w_rel_precision * r_rel_precision
        reward = reward_alive + reward_vel + reward_attitude + reward_omega
        reward = reward + reward_yaw_rate
        reward = reward + reward_rel_tracking + reward_rel_precision
        if self.yaw_reward_enable:
            reward_yaw = self.w_yaw * r_yaw
            reward = reward + reward_yaw
        else:
            reward_yaw = torch.zeros_like(reward)

        if self.w_vel_precision > 0.0:
            vel_precision = (
                torch.abs(dvx) / max(self.sig_vx, 1e-6)
                + torch.abs(dvy) / max(self.sig_vy, 1e-6)
                + torch.abs(dvz) / max(self.sig_vz, 1e-6)
            ) / 3.0
            reward_vel_precision = -self.w_vel_precision * vel_precision
            reward = reward + reward_vel_precision
        else:
            vel_precision = torch.zeros_like(reward)
            reward_vel_precision = torch.zeros_like(reward)
        if self.yaw_reward_enable and self.w_yaw_precision > 0.0:
            yaw_precision = torch.abs(dyaw_reward) / max(yaw_sig, 1e-6)
            reward_yaw_precision = -self.w_yaw_precision * yaw_precision
            reward = reward + reward_yaw_precision
        else:
            yaw_precision = torch.zeros_like(reward)
            reward_yaw_precision = torch.zeros_like(reward)

        if self.w_overshoot > 0.0:
            step_vx = task.target_vx - initial_local_vx
            step_vy = task.target_vy - initial_local_vy
            step_vz = task.target_vz - initial_vz
            overshoot_vx = torch.clamp(dvx * torch.sign(step_vx), min=0.0) / scale_vx
            overshoot_vy = torch.clamp(dvy * torch.sign(step_vy), min=0.0) / scale_vy
            overshoot_vz = torch.clamp(dvz * torch.sign(step_vz), min=0.0) / scale_vz
            overshoot_rel = (overshoot_vx + overshoot_vy + overshoot_vz) / 3.0
            overshoot_excess = torch.clamp(overshoot_rel - self.overshoot_deadband, min=0.0)
            reward_overshoot = -self.w_overshoot * overshoot_excess
            reward = reward + reward_overshoot
        else:
            overshoot_rel = torch.zeros_like(reward)
            reward_overshoot = torch.zeros_like(reward)

        if self.w_adaptive_damping > 0.0:
            proximity = torch.exp(-self.adaptive_damping_gain * rel_error)
            reward_adaptive_damping = -self.w_adaptive_damping * omega_sq * proximity
            reward = reward + reward_adaptive_damping
        else:
            proximity = torch.zeros_like(reward)
            reward_adaptive_damping = torch.zeros_like(reward)

        if self.w_smooth > 0.0:
            delta_u = env.model.u - env.model.recent_u
            delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
            r_smooth = torch.exp(-delta_u_sq / (self.sig_smooth ** 2))
            reward_smooth = self.w_smooth * r_smooth
            reward = reward + reward_smooth
        else:
            reward_smooth = torch.zeros_like(reward)

        task.last_reward_terms = {
            'reward/total_mean': reward.detach().mean(),
            'reward/alive_mean': reward_alive.detach().mean(),
            'reward/vel_gaussian_mean': reward_vel.detach().mean(),
            'reward/rel_tracking_mean': reward_rel_tracking.detach().mean(),
            'reward/rel_precision_mean': reward_rel_precision.detach().mean(),
            'reward/overshoot_mean': reward_overshoot.detach().mean(),
            'reward/adaptive_damping_mean': reward_adaptive_damping.detach().mean(),
            'reward/yaw_mean': reward_yaw.detach().mean(),
            'reward/yaw_rate_mean': reward_yaw_rate.detach().mean(),
            'reward/attitude_mean': reward_attitude.detach().mean(),
            'reward/omega_mean': reward_omega.detach().mean(),
            'reward/smooth_mean': reward_smooth.detach().mean(),
            'reward/vel_precision_mean': reward_vel_precision.detach().mean(),
            'reward/yaw_precision_mean': reward_yaw_precision.detach().mean(),
            'reward/raw_rel_error_mean': rel_error.detach().mean(),
            'reward/raw_rel_precision_mean': r_rel_precision.detach().mean(),
            'reward/raw_overshoot_rel_mean': overshoot_rel.detach().mean(),
            'reward/raw_damping_proximity_mean': proximity.detach().mean(),
            'reward/raw_yaw_error_abs_mean': torch.abs(dyaw_reward).detach().mean(),
            'reward/raw_yaw_rate_abs_mean': torch.abs(r).detach().mean(),
        }

        return reward
