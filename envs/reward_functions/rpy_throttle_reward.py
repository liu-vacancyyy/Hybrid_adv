import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


RPY_THROTTLE_REWARD_PRESETS = {
    # Current default.  Keep this close to the first no-forward run so old
    # configs remain comparable.
    'balanced': {
        'w_alive': 0.2,
        'w_attitude': 5.0,
        'w_yaw_rate': 2.0,
        'w_throttle': 2.0,
        'w_omega': 1.0,
        'w_smooth': 0.4,
        'w_velocity': 0.0,
        'w_overshoot': 1.0,
        'w_moving_away': 0.3,
        'sig_roll': 0.045,
        'sig_pitch': 0.045,
        'sig_yaw_rate': 0.12,
        'sig_throttle': 0.06,
        'sig_omega': 0.8,
        'sig_smooth': 0.8,
        'sig_velocity': 2.5,
        'sig_overshoot': 0.35,
    },
    # Faster tracking pressure.  Use this to check whether the policy is
    # under-incentivized on the commanded attitude/yaw/throttle targets.
    'track_strict': {
        'w_alive': 0.2,
        'w_attitude': 7.0,
        'w_yaw_rate': 2.8,
        'w_throttle': 2.8,
        'w_omega': 0.6,
        'w_smooth': 0.2,
        'w_velocity': 0.0,
        'w_overshoot': 0.7,
        'w_moving_away': 0.2,
        'sig_roll': 0.038,
        'sig_pitch': 0.038,
        'sig_yaw_rate': 0.10,
        'sig_throttle': 0.05,
        'sig_omega': 0.9,
        'sig_smooth': 0.9,
        'sig_velocity': 2.5,
        'sig_overshoot': 0.30,
    },
    # Heavier damping and anti-overshoot pressure.  This is usually the safer
    # no-forward variant when attitude targets are reached but ring or overshoot.
    'damped': {
        'w_alive': 0.2,
        'w_attitude': 5.0,
        'w_yaw_rate': 2.0,
        'w_throttle': 2.0,
        'w_omega': 1.6,
        'w_smooth': 0.8,
        'w_velocity': 0.0,
        'w_overshoot': 2.2,
        'w_moving_away': 0.8,
        'sig_roll': 0.050,
        'sig_pitch': 0.050,
        'sig_yaw_rate': 0.13,
        'sig_throttle': 0.07,
        'sig_omega': 0.65,
        'sig_smooth': 0.55,
        'sig_velocity': 2.5,
        'sig_overshoot': 0.22,
    },
    # No-forward motor layout has less authority margin in combined commands.
    # This version makes collective tracking more explicit while still
    # penalizing overshoot.
    'throttle_focus': {
        'w_alive': 0.2,
        'w_attitude': 4.5,
        'w_yaw_rate': 1.8,
        'w_throttle': 3.8,
        'w_omega': 1.2,
        'w_smooth': 0.6,
        'w_velocity': 0.0,
        'w_overshoot': 1.5,
        'w_moving_away': 0.5,
        'sig_roll': 0.045,
        'sig_pitch': 0.045,
        'sig_yaw_rate': 0.12,
        'sig_throttle': 0.045,
        'sig_omega': 0.75,
        'sig_smooth': 0.7,
        'sig_velocity': 2.5,
        'sig_overshoot': 0.28,
    },
}

RPY_THROTTLE_REWARD_ALIASES = {
    'default': 'balanced',
    'base': 'balanced',
    'normal': 'balanced',
    'strict': 'track_strict',
    'tracking': 'track_strict',
    'track': 'track_strict',
    'anti_overshoot': 'damped',
    'damping': 'damped',
    'smooth': 'damped',
    'throttle': 'throttle_focus',
    'collective': 'throttle_focus',
}


def _normalize_reward_variant(name):
    normalized = str(name).strip().lower().replace('-', '_')
    normalized = RPY_THROTTLE_REWARD_ALIASES.get(normalized, normalized)
    if normalized not in RPY_THROTTLE_REWARD_PRESETS:
        valid = ', '.join(sorted(RPY_THROTTLE_REWARD_PRESETS))
        raise ValueError(
            f'Unknown RPY_THROTTLE_REWARD_VARIANT={name!r}; valid variants: {valid}'
        )
    return normalized


class RPYThrottleReward(BaseRewardFunction):
    """Dense reward for roll/pitch/yaw-rate/collective-throttle tracking."""

    preset_name = 'balanced'
    allow_generic_config_override = True

    def __init__(self, config):
        super().__init__(config)
        self.reward_variant = _normalize_reward_variant(self.preset_name)
        preset = RPY_THROTTLE_REWARD_PRESETS[self.reward_variant]

        self.w_alive = self._float_param(config, 'w_alive', preset)
        self.w_attitude = self._float_param(config, 'w_attitude', preset)
        self.w_yaw_rate = self._float_param(config, 'w_yaw_rate', preset)
        self.w_throttle = self._float_param(config, 'w_throttle', preset)
        self.w_omega = self._float_param(config, 'w_omega', preset)
        self.w_smooth = self._float_param(config, 'w_smooth', preset)
        self.w_velocity = self._float_param(config, 'w_velocity', preset)
        self.w_overshoot = self._float_param(config, 'w_overshoot', preset)
        self.w_moving_away = self._float_param(config, 'w_moving_away', preset)

        self.sig_roll = self._float_param(config, 'sig_roll', preset)
        self.sig_pitch = self._float_param(config, 'sig_pitch', preset)
        self.sig_yaw_rate = self._float_param(config, 'sig_yaw_rate', preset)
        self.sig_throttle = self._float_param(config, 'sig_throttle', preset)
        self.sig_omega = self._float_param(config, 'sig_omega', preset)
        self.sig_smooth = self._float_param(config, 'sig_smooth', preset)
        self.sig_velocity = self._float_param(config, 'sig_velocity', preset)
        self.sig_overshoot = self._float_param(config, 'sig_overshoot', preset)

    def _float_param(self, config, name, preset):
        variant_key = f'rpy_throttle_{self.reward_variant}_{name}'
        generic_key = f'rpy_throttle_{name}'
        if hasattr(config, variant_key):
            return float(getattr(config, variant_key))
        if self.allow_generic_config_override and hasattr(config, generic_key):
            return float(getattr(config, generic_key))
        return float(preset[name])

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

        reward_alive = torch.full_like(r_attitude, self.w_alive)
        reward_attitude = self.w_attitude * r_attitude
        reward_yaw_rate = self.w_yaw_rate * r_yaw_rate
        reward_throttle = self.w_throttle * r_throttle
        reward_omega = self.w_omega * r_omega
        reward_velocity = self.w_velocity * r_velocity
        reward_overshoot = self.w_overshoot * r_overshoot
        reward_moving_away = -self.w_moving_away * moving_away

        reward = (
            reward_alive
            + reward_attitude
            + reward_yaw_rate
            + reward_throttle
            + reward_omega
            + reward_velocity
            + reward_overshoot
            + reward_moving_away
        )

        if self.w_smooth > 0.0:
            delta_u = env.model.u - env.model.recent_u
            delta_u_sq = torch.sum(delta_u * delta_u, dim=1)
            r_smooth = torch.exp(-delta_u_sq / max(self.sig_smooth ** 2, 1e-6))
            reward_smooth = self.w_smooth * r_smooth
            reward = reward + reward_smooth
        else:
            reward_smooth = torch.zeros_like(reward)

        task.last_reward_terms = {
            'reward/total_mean': reward.detach().mean(),
            'reward/alive_mean': reward_alive.detach().mean(),
            'reward/attitude_mean': reward_attitude.detach().mean(),
            'reward/yaw_rate_mean': reward_yaw_rate.detach().mean(),
            'reward/throttle_mean': reward_throttle.detach().mean(),
            'reward/omega_mean': reward_omega.detach().mean(),
            'reward/smooth_mean': reward_smooth.detach().mean(),
            'reward/velocity_mean': reward_velocity.detach().mean(),
            'reward/overshoot_mean': reward_overshoot.detach().mean(),
            'reward/moving_away_mean': reward_moving_away.detach().mean(),
            'reward/raw_attitude_score_mean': r_attitude.detach().mean(),
            'reward/raw_yaw_rate_score_mean': r_yaw_rate.detach().mean(),
            'reward/raw_throttle_score_mean': r_throttle.detach().mean(),
            'reward/raw_omega_score_mean': r_omega.detach().mean(),
            'reward/raw_smooth_score_mean': r_smooth.detach().mean() if self.w_smooth > 0.0 else torch.zeros((), device=env.device),
            'reward/raw_overshoot_score_mean': r_overshoot.detach().mean(),
        }

        return reward


class RPYThrottleTrackStrictReward(RPYThrottleReward):
    preset_name = 'track_strict'
    allow_generic_config_override = False


class RPYThrottleDampedReward(RPYThrottleReward):
    preset_name = 'damped'
    allow_generic_config_override = False


class RPYThrottleThrottleFocusReward(RPYThrottleReward):
    preset_name = 'throttle_focus'
    allow_generic_config_override = False


def make_rpy_throttle_reward(config):
    variant = _normalize_reward_variant(os.environ.get(
        'RPY_THROTTLE_REWARD_VARIANT',
        getattr(config, 'rpy_throttle_reward_variant', 'balanced'),
    ))
    reward_classes = {
        'balanced': RPYThrottleReward,
        'track_strict': RPYThrottleTrackStrictReward,
        'damped': RPYThrottleDampedReward,
        'throttle_focus': RPYThrottleThrottleFocusReward,
    }
    return reward_classes[variant](config)


class RPYThrottleEventDrivenReward(BaseRewardFunction):
    """Penalty for safety terminations."""

    def __init__(self, config):
        super().__init__(config)
        self.bad_done_penalty = float(getattr(config, 'rpy_throttle_bad_done_penalty', 200.0))

    def get_reward(self, task, env):
        return -self.bad_done_penalty * env.bad_done
