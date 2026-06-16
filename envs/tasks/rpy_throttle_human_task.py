import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tasks.rc_human_task import RCHumanTask
from reward_functions.rpy_throttle_reward import (
    RPYThrottleEventDrivenReward,
    RPYThrottleReward,
)
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.overload import Overload
from hybrid_termination_conditions.high_speed import HighSpeed
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI


class RPYThrottleHumanTask(RCHumanTask):
    """Human-stick task whose targets are roll/pitch/yaw-rate/throttle.

    The raw stick generator and mode0-5 curriculum mirror RCHumanTask, but the
    four stick axes are interpreted as:

      raw_vx  -> pitch command, positive stick means pitch down / forward
      raw_vy  -> roll command
      raw_vz  -> collective lift-throttle offset around hover
      raw_yaw -> yaw-rate command

    The policy action remains the model's five motor commands.
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'rpy_throttle_human'

        self.roll_limit = math.radians(float(
            getattr(config, 'rpy_throttle_roll_limit_deg', 10.0)
        ))
        self.pitch_limit = math.radians(float(
            getattr(config, 'rpy_throttle_pitch_limit_deg', 10.0)
        ))
        self.yaw_rate_limit = float(
            getattr(config, 'rpy_throttle_yaw_rate_limit', self.yaw_rate_limit)
        )
        self.throttle_delta_easy = float(
            getattr(config, 'rpy_throttle_delta_easy', 0.08)
        )
        self.throttle_delta_medium = float(
            getattr(config, 'rpy_throttle_delta_medium', 0.15)
        )
        self.throttle_delta_hard = float(
            getattr(config, 'rpy_throttle_delta_hard', 0.25)
        )
        self.throttle_min_frac = float(
            getattr(config, 'rpy_throttle_min_collective_frac', 0.75)
        )
        self.throttle_max_frac = float(
            getattr(config, 'rpy_throttle_max_collective_frac', 1.25)
        )
        self.throttle_delta_max = max(
            abs(1.0 - self.throttle_min_frac),
            abs(self.throttle_max_frac - 1.0),
            abs(self.throttle_delta_hard),
            1e-6,
        )
        self.command_rate_limit_frac = max(0.0, float(os.environ.get(
            'RPY_THROTTLE_COMMAND_RATE_LIMIT_FRAC',
            os.environ.get(
                'RC_HUMAN_COMMAND_RATE_LIMIT_FRAC',
                getattr(
                    config,
                    'rpy_throttle_command_rate_limit_frac',
                    getattr(config, 'rc_human_command_rate_limit_frac', 0.0),
                ),
            ),
        )))

        self.levels_per_mode = int(
            getattr(config, 'rpy_throttle_levels_per_mode', self.levels_per_mode)
        )
        self.easy_stick = float(
            getattr(config, 'rpy_throttle_easy_stick', self.easy_stick)
        )
        self.medium_stick = float(
            getattr(config, 'rpy_throttle_medium_stick', self.medium_stick)
        )
        self.hard_stick = float(
            getattr(config, 'rpy_throttle_hard_stick', self.hard_stick)
        )
        self.mode3_reverse_axes = max(1, min(4, int(
            getattr(config, 'rpy_throttle_mode3_reverse_axes', self.mode3_reverse_axes)
        )))

        mode_order = self._parse_mode_order(config)
        self.mode_order = torch.tensor(mode_order, dtype=torch.long, device=self.device)
        mode_slots = int(os.environ.get(
            'RPY_THROTTLE_MAX_MODE_SLOTS',
            getattr(config, 'rpy_throttle_max_mode_slots', int(self.mode_order.numel())),
        ))
        self.active_mode_slots = max(1, min(mode_slots, int(self.mode_order.numel())))
        self.max_curriculum_level = self.levels_per_mode * self.active_mode_slots - 1
        self._normalize_curriculum_mix()

        self.success_attitude_error = float(getattr(
            config, 'rpy_throttle_success_attitude_error', math.radians(3.0)
        ))
        self.success_yaw_rate_error = float(getattr(
            config, 'rpy_throttle_success_yaw_rate_error', 0.12
        ))
        self.success_throttle_error = float(getattr(
            config, 'rpy_throttle_success_throttle_error', 0.08
        ))
        self.success_overshoot = float(getattr(
            config, 'rpy_throttle_success_overshoot', 0.20
        ))

        self.target_roll = torch.zeros(self.n, device=self.device)
        self.target_pitch = torch.zeros(self.n, device=self.device)
        self.target_collective = torch.zeros(self.n, device=self.device)
        self.target_throttle_frac = torch.zeros(self.n, device=self.device)
        self.hover_collective = torch.ones(self.n, device=self.device)

        self.prev_roll_error = torch.zeros(self.n, device=self.device)
        self.prev_pitch_error = torch.zeros(self.n, device=self.device)
        self.prev_yaw_rate_error = torch.zeros(self.n, device=self.device)
        self.prev_throttle_error = torch.zeros(self.n, device=self.device)

        self.episode_attitude_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_yaw_rate_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_throttle_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_overshoot_sum = torch.zeros(self.n, device=self.device)
        self.episode_metric_count = torch.zeros(self.n, device=self.device)
        self.episode_metric_skipped_count = torch.zeros(self.n, device=self.device)

        self.reward_functions = [
            RPYThrottleReward(self.config),
            RPYThrottleEventDrivenReward(self.config),
        ]
        self.termination_conditions = [
            Overload(self.config),
            LowAltitude(self.config),
            HighSpeed(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
            HoverTimeoutDone(self.config),
        ]

    def _parse_mode_order(self, config):
        raw = os.environ.get(
            'RPY_THROTTLE_MODE_ORDER',
            getattr(
                config,
                'rpy_throttle_mode_order',
                getattr(config, 'rc_human_mode_order', '0 1 2 3 4 5'),
            ),
        )
        if isinstance(raw, (list, tuple)):
            values = [int(x) for x in raw]
        else:
            values = [int(x) for x in str(raw).replace(',', ' ').split()]
        if not values:
            raise ValueError('RPY_THROTTLE_MODE_ORDER must contain at least one mode id')
        invalid = [x for x in values if x < 0 or x > 5]
        if invalid:
            raise ValueError(f'Invalid rpy_throttle mode ids: {invalid}')
        if len(set(values)) != len(values):
            raise ValueError(f'RPY_THROTTLE_MODE_ORDER contains duplicates: {values}')
        return values

    def reset(self, env):
        reset = (env.is_done.bool() | env.bad_done.bool()) | env.exceed_time_limit.bool()
        if int(torch.sum(reset).item()) == 0:
            return

        super().reset(env)

        _roll, _pitch, _heading = env.model.get_posture()
        _f0, f1, f2, f3, f4 = env.model.get_F()
        hover = torch.clamp(f1 + f2 + f3 + f4, min=1e-6)

        self.target_roll[reset] = 0.0
        self.target_pitch[reset] = 0.0
        self.target_yaw_rate[reset] = 0.0
        self.hover_collective[reset] = hover[reset]
        self.target_collective[reset] = hover[reset]
        self.target_throttle_frac[reset] = 0.0
        self.vx_forward_limit[reset] = self.throttle_delta_easy

        self.prev_roll_error[reset] = 0.0
        self.prev_pitch_error[reset] = 0.0
        self.prev_yaw_rate_error[reset] = 0.0
        self.prev_throttle_error[reset] = 0.0
        self.episode_attitude_error_sum[reset] = 0.0
        self.episode_yaw_rate_error_sum[reset] = 0.0
        self.episode_throttle_error_sum[reset] = 0.0
        self.episode_overshoot_sum[reset] = 0.0
        self.episode_metric_count[reset] = 0.0
        self.episode_metric_skipped_count[reset] = 0.0

    def _curriculum_amplitude(self, level, mode):
        if not self.curriculum_enable:
            return torch.full((level.numel(),), self.hard_stick, device=self.device)

        sublevel = (level % self.levels_per_mode).float()
        denom = max(float(self.levels_per_mode - 1), 1.0)
        progress = sublevel / denom

        amp = self.easy_stick + progress * (self.hard_stick - self.easy_stick)
        amp[mode == 0] = 0.0
        hard = (mode == 4) | (mode == 5)
        amp[hard] = torch.clamp(amp[hard], min=self.medium_stick, max=self.hard_stick)
        return amp

    def _resample_raw_sticks(self, mask):
        size = int(torch.sum(mask).item())
        if size == 0:
            return
        d = self.device

        idx = torch.where(mask)[0]
        sampled_level = self._sample_command_level(idx)
        mode = self._curriculum_mode_from_level(sampled_level)
        amp = self._curriculum_amplitude(sampled_level, mode)
        signs = torch.sign(torch.rand(size, 4, device=d) - 0.5)
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        values = torch.zeros(size, 4, device=d)

        # Scheme C mode layout:
        #   0: neutral hover trim.
        #   1: roll/pitch only.
        #   2: collective throttle only.
        #   3: yaw-rate only.
        #   4: two-axis combinations.
        #   5: strong three/four-axis combinations.
        attitude = mode == 1
        if torch.any(attitude):
            count = int(attitude.sum().item())
            values[attitude, 0] = (
                torch.rand(count, device=d) * amp[attitude] * signs[attitude, 0]
            )
            values[attitude, 1] = (
                torch.rand(count, device=d) * amp[attitude] * signs[attitude, 1]
            )

        throttle = mode == 2
        if torch.any(throttle):
            values[throttle, 2] = amp[throttle] * signs[throttle, 2]

        yaw = mode == 3
        if torch.any(yaw):
            values[yaw, 3] = amp[yaw] * signs[yaw, 3]

        dual = mode == 4
        if torch.any(dual):
            count = int(dual.sum().item())
            pair_table = torch.tensor(
                [
                    [0, 1],  # pitch + roll
                    [0, 2],  # pitch + throttle
                    [1, 2],  # roll + throttle
                    [0, 3],  # pitch + yaw
                    [1, 3],  # roll + yaw
                    [2, 3],  # throttle + yaw
                ],
                dtype=torch.long,
                device=d,
            )
            pairs = pair_table[torch.randint(0, pair_table.shape[0], (count,), device=d)]
            local_values = torch.zeros(count, 4, device=d)
            magnitudes = (0.45 + 0.55 * torch.rand(count, 2, device=d)) * amp[dual].reshape(-1, 1)
            local_signs = signs[dual].gather(1, pairs)
            local_values.scatter_(1, pairs, magnitudes * local_signs)
            values[dual, :] = local_values

        strong = mode == 5
        if torch.any(strong):
            count = int(strong.sum().item())
            local_values = torch.zeros(count, 4, device=d)
            scores = torch.rand(count, 4, device=d)
            three_axes = torch.rand(count, device=d) < 0.5
            top3 = torch.topk(scores, k=3, dim=1).indices
            top4 = torch.arange(4, device=d).reshape(1, 4).expand(count, 4)
            magnitudes = (0.35 + 0.65 * torch.rand(count, 4, device=d)) * amp[strong].reshape(-1, 1)
            local_values = magnitudes * signs[strong]
            three_axis_values = torch.zeros_like(local_values)
            three_axis_values.scatter_(1, top3, local_values.gather(1, top3))
            values[strong, :] = torch.where(
                three_axes.reshape(-1, 1),
                three_axis_values,
                local_values.gather(1, top4),
            )

        prev_raw = torch.stack((
            self.desired_raw_vx[idx],
            self.desired_raw_vy[idx],
            self.desired_raw_vz[idx],
            self.desired_raw_yaw[idx],
        ), dim=1)

        self.operation_mode[mask] = mode
        self.mode5_release_state[idx] = 0
        self.mode5_hold_elapsed[idx] = 0
        self.mode5_recovery_left[idx] = 0
        self.mode5_pre_release_raw[idx] = 0.0

        delta_raw = torch.max(torch.abs(values - prev_raw), dim=1).values
        reversal = (
            (torch.abs(prev_raw) > self.command_reversal_threshold)
            & (torch.abs(values) > self.command_reversal_threshold)
            & (prev_raw * values < 0.0)
        )
        large_transient = (
            (delta_raw >= self.large_command_transient_delta)
            | torch.any(reversal, dim=1)
        )
        changed = delta_raw > self.command_transient_threshold
        if torch.any(changed):
            grace_steps = torch.full(
                (size,),
                max(self.command_transient_grace_steps, 0),
                dtype=torch.long,
                device=d,
            )
            grace_steps[large_transient] = max(
                self.large_command_transient_grace_steps,
                self.command_transient_grace_steps,
                0,
            )
            self.command_transient_left[idx[changed]] = grace_steps[changed]

        self.desired_raw_vx[mask] = torch.clamp(values[:, 0], -1.0, 1.0)
        self.desired_raw_vy[mask] = torch.clamp(values[:, 1], -1.0, 1.0)
        self.desired_raw_vz[mask] = torch.clamp(values[:, 2], -1.0, 1.0)
        self.desired_raw_yaw[mask] = torch.clamp(values[:, 3], -1.0, 1.0)
        self.vx_forward_limit[mask] = self._curriculum_vx_forward_limit(
            sampled_level, mode
        )
        self.dwell_left[mask] = torch.randint(
            self.dwell_min,
            self.dwell_max + 1,
            (size,),
            device=d,
            dtype=torch.long,
        )

    def _curriculum_vx_forward_limit(self, level, mode):
        if not self.curriculum_enable:
            return torch.full((level.numel(),), self.throttle_delta_hard, device=self.device)

        sublevel = (level % self.levels_per_mode).float()
        denom = max(float(self.levels_per_mode - 1), 1.0)
        progress = torch.clamp(sublevel / denom, 0.0, 1.0)
        first_half = progress <= 0.5
        early = self.throttle_delta_easy + (
            self.throttle_delta_medium - self.throttle_delta_easy
        ) * (progress / 0.5)
        late = self.throttle_delta_medium + (
            self.throttle_delta_hard - self.throttle_delta_medium
        ) * ((progress - 0.5) / 0.5)
        limit = torch.where(first_half, early, late)
        limit = torch.clamp(limit, min=0.0, max=self.throttle_delta_max)
        return torch.where(mode == 0, torch.zeros_like(limit), limit)

    def _update_mode5_release_state(self, env, mask):
        active_hold = mask & (self.mode5_release_state == 1)
        if torch.any(active_hold):
            self.mode5_hold_elapsed[active_hold] += 1
            min_elapsed = self.mode5_hold_elapsed[active_hold] >= self.mode5_hold_min_steps
            timeout = self.mode5_hold_elapsed[active_hold] >= self.mode5_hold_max_steps
            release_now_local = min_elapsed | timeout

            if torch.any(release_now_local):
                active_idx = torch.where(active_hold)[0]
                release_idx = active_idx[release_now_local]
                self.desired_raw_vx[release_idx] = 0.0
                self.desired_raw_vy[release_idx] = 0.0
                self.desired_raw_vz[release_idx] = 0.0
                self.desired_raw_yaw[release_idx] = 0.0
                self.mode5_release_state[release_idx] = 2
                self.mode5_recovery_left[release_idx] = max(
                    self.mode5_release_recovery_steps,
                    1,
                )
                self.dwell_left[release_idx] = max(
                    self.mode5_release_recovery_steps,
                    1,
                )
                self.command_transient_left[release_idx] = max(
                    self.command_transient_grace_steps,
                    self.mode5_release_recovery_steps,
                    0,
                )

        active_released = mask & (self.mode5_release_state == 2)
        if torch.any(active_released):
            self.mode5_recovery_left[active_released] = torch.clamp(
                self.mode5_recovery_left[active_released] - 1,
                min=0,
            )
            recovered = active_released & (self.mode5_recovery_left <= 0)
            if torch.any(recovered):
                self.mode5_release_state[recovered] = 0
                self.mode5_hold_elapsed[recovered] = 0
                self.mode5_pre_release_raw[recovered] = 0.0
                self.dwell_left[recovered] = 0

    def _update_px4_vtol_mc_targets_from_sticks(self, mask, env):
        stick_pitch, stick_roll = self._limit_stick_unit_length_xy(
            self.stick_vx[mask], self.stick_vy[mask]
        )
        self.stick_vx[mask] = stick_pitch
        self.stick_vy[mask] = stick_roll

        self.target_pitch[mask] = -stick_pitch * self.pitch_limit
        self.target_roll[mask] = stick_roll * self.roll_limit
        self.target_yaw_rate[mask] = self.stick_yaw[mask] * self.yaw_rate_limit

        throttle_frac = torch.clamp(
            self.stick_vz[mask] * self.vx_forward_limit[mask],
            self.throttle_min_frac - 1.0,
            self.throttle_max_frac - 1.0,
        )
        target_throttle_frac = throttle_frac.clone()
        target_collective = self.hover_collective[mask] * (1.0 + target_throttle_frac)

        _npos, _epos, altitude = env.model.get_position()
        hover = self.hover_collective[mask]
        altitude_masked = altitude[mask]
        local_guard = (
            ((altitude_masked > self.alt_high) & (target_collective > hover))
            | ((altitude_masked < self.alt_low) & (target_collective < hover))
        )
        if torch.any(local_guard):
            target_collective[local_guard] = hover[local_guard]
            target_throttle_frac[local_guard] = 0.0
            idx = torch.where(mask)[0]
            self.stick_vz[idx[local_guard]] = 0.0

        self.target_throttle_frac[mask] = target_throttle_frac
        self.target_collective[mask] = target_collective

    def compute_overshoot_score(self, roll_error, pitch_error, yaw_rate_error, throttle_error):
        valid = self.command_transient_left <= 0
        crossed_roll = valid & (self.prev_roll_error * roll_error < 0.0)
        crossed_pitch = valid & (self.prev_pitch_error * pitch_error < 0.0)
        crossed_yaw = valid & (self.prev_yaw_rate_error * yaw_rate_error < 0.0)
        crossed_throttle = valid & (self.prev_throttle_error * throttle_error < 0.0)

        roll_score = torch.where(
            crossed_roll,
            torch.abs(roll_error) / max(self.roll_limit, 1e-6),
            torch.zeros_like(roll_error),
        )
        pitch_score = torch.where(
            crossed_pitch,
            torch.abs(pitch_error) / max(self.pitch_limit, 1e-6),
            torch.zeros_like(pitch_error),
        )
        yaw_score = torch.where(
            crossed_yaw,
            torch.abs(yaw_rate_error) / max(self.yaw_rate_limit, 1e-6),
            torch.zeros_like(yaw_rate_error),
        )
        throttle_score = torch.where(
            crossed_throttle,
            torch.abs(throttle_error) / max(self.throttle_delta_max, 1e-6),
            torch.zeros_like(throttle_error),
        )

        self.prev_roll_error = roll_error.detach()
        self.prev_pitch_error = pitch_error.detach()
        self.prev_yaw_rate_error = yaw_rate_error.detach()
        self.prev_throttle_error = throttle_error.detach()

        return torch.sqrt(
            roll_score * roll_score
            + pitch_score * pitch_score
            + yaw_score * yaw_score
            + throttle_score * throttle_score
        )

    def update_episode_metrics(
        self,
        attitude_error,
        yaw_rate_error,
        throttle_error,
        overshoot_score,
    ):
        valid = torch.ones_like(attitude_error, dtype=torch.bool, device=self.device)
        if self.success_ignore_transient:
            valid = valid & (self.command_transient_left <= 0)

        valid_f = valid.detach().float()
        self.episode_attitude_error_sum += attitude_error.detach() * valid_f
        self.episode_yaw_rate_error_sum += yaw_rate_error.detach() * valid_f
        self.episode_throttle_error_sum += throttle_error.detach() * valid_f
        self.episode_overshoot_sum += overshoot_score.detach() * valid_f
        self.episode_metric_count += valid_f
        self.episode_metric_skipped_count += (~valid).detach().float()

    def _update_curriculum_from_last_episode(self, env, reset):
        if not self.curriculum_enable:
            return

        finished = reset & (self.episode_metric_count > 0)
        if int(torch.sum(finished).item()) == 0:
            return

        count = torch.clamp(self.episode_metric_count[finished], min=1.0)
        mean_attitude_error = self.episode_attitude_error_sum[finished] / count
        mean_yaw_rate_error = self.episode_yaw_rate_error_sum[finished] / count
        mean_throttle_error = self.episode_throttle_error_sum[finished] / count
        mean_overshoot = self.episode_overshoot_sum[finished] / count

        clean_done = env.is_done[finished].bool() & (~env.bad_done[finished].bool())
        accurate = (
            (mean_attitude_error < self.success_attitude_error)
            & (mean_yaw_rate_error < self.success_yaw_rate_error)
            & (mean_throttle_error < self.success_throttle_error)
            & (mean_overshoot < self.success_overshoot)
        )
        success = clean_done & accurate
        idx = torch.where(finished)[0]

        level = self.curriculum_level[idx]
        level = torch.where(success, level + 1, level - 1)
        self.curriculum_level[idx] = torch.clamp(level, 0, self.max_curriculum_level)

    def get_training_metrics(self):
        count = torch.clamp(self.episode_metric_count, min=1.0)
        mean_attitude_error = self.episode_attitude_error_sum / count
        mean_yaw_rate_error = self.episode_yaw_rate_error_sum / count
        mean_throttle_error = self.episode_throttle_error_sum / count
        mean_overshoot = self.episode_overshoot_sum / count

        metrics = {
            'rpy_throttle/curriculum_level_mean': self.curriculum_level.float().mean(),
            'rpy_throttle/curriculum_level_max': self.curriculum_level.float().max(),
            'rpy_throttle/curriculum_level_limit': torch.tensor(
                float(self.max_curriculum_level), device=self.device),
            'rpy_throttle/attitude_error_mean': mean_attitude_error.mean(),
            'rpy_throttle/yaw_rate_error_mean': mean_yaw_rate_error.mean(),
            'rpy_throttle/throttle_error_mean': mean_throttle_error.mean(),
            'rpy_throttle/overshoot_mean': mean_overshoot.mean(),
            'rpy_throttle/throttle_delta_limit_mean': self.vx_forward_limit.mean(),
            'rpy_throttle/target_collective_mean': self.target_collective.mean(),
        }
        metric_total = self.episode_metric_count + self.episode_metric_skipped_count
        metrics['rpy_throttle/success_metric_valid_fraction'] = (
            self.episode_metric_count / torch.clamp(metric_total, min=1.0)
        ).mean()
        metrics['rpy_throttle/success_metric_skipped_fraction'] = (
            self.episode_metric_skipped_count / torch.clamp(metric_total, min=1.0)
        ).mean()
        for mode_id in range(6):
            metrics[f'rpy_throttle/mode_{mode_id}_fraction'] = (
                self.operation_mode == mode_id).float().mean()
        metrics['rpy_throttle/command_transient_fraction'] = (
            self.command_transient_left > 0).float().mean()
        metrics['rpy_throttle/command_rate_limit_frac'] = torch.tensor(
            float(self.command_rate_limit_frac), device=self.device)
        metrics['rpy_throttle/command_rate_limited_fraction'] = (
            self.command_rate_limited.float().mean())
        metrics['rpy_throttle/command_raw_delta_mean'] = self.command_raw_delta.mean()
        metrics['rpy_throttle/mode5_hold_fraction'] = (
            self.mode5_release_state == 1).float().mean()
        metrics['rpy_throttle/mode5_released_fraction'] = (
            self.mode5_release_state == 2).float().mean()
        return metrics

    def _build_obs(self, env, add_sensor_noise):
        self.sync_command(env)

        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        eas = env.model.get_EAS()
        alpha_sin, alpha_cos, beta_sin, beta_cos = env.model.get_aero_sincos()
        p, q, r = env.model.get_angular_velocity()
        _roll_dot, _pitch_dot, yaw_rate = env.model.get_euler_angular_velocity()
        f0, f1, f2, f3, f4 = env.model.get_F()
        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()

        if add_sensor_noise:
            altitude = altitude + torch.randn_like(altitude) * self.sensor_pos_std
            roll, pitch, heading, vx_n, vy_e, vz, p, q, r = self._apply_sensor_noise(
                roll, pitch, heading, vx_n, vy_e, vz, p, q, r
            )
            vt = torch.clamp(vt + torch.randn_like(vt) * self.sensor_vel_std, min=0.0)
            eas = torch.clamp(eas + torch.randn_like(eas) * self.sensor_vel_std, min=0.0)

        local_vx, local_vy = self.ground_to_local_velocity(vx_n, vy_e, heading)
        collective = f1 + f2 + f3 + f4
        hover = torch.clamp(self.hover_collective, min=1e-6)
        throttle_error = (collective - self.target_collective) / hover

        obs = torch.hstack((
            (wrap_PI(roll - self.target_roll) / max(self.roll_limit, 1e-6)).reshape(-1, 1),
            (wrap_PI(pitch - self.target_pitch) / max(self.pitch_limit, 1e-6)).reshape(-1, 1),
            ((yaw_rate - self.target_yaw_rate) / max(self.yaw_rate_limit, 1e-6)).reshape(-1, 1),
            (throttle_error / max(self.throttle_delta_max, 1e-6)).reshape(-1, 1),
            (self.target_roll / max(self.roll_limit, 1e-6)).reshape(-1, 1),
            (self.target_pitch / max(self.pitch_limit, 1e-6)).reshape(-1, 1),
            (self.target_yaw_rate / max(self.yaw_rate_limit, 1e-6)).reshape(-1, 1),
            (self.target_throttle_frac / max(self.throttle_delta_max, 1e-6)).reshape(-1, 1),
            altitude.reshape(-1, 1) / 100.0,
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
            torch.sin(heading).reshape(-1, 1),
            torch.cos(heading).reshape(-1, 1),
            eas.reshape(-1, 1) / 10.0,
            vt.reshape(-1, 1) / 10.0,
            local_vx.reshape(-1, 1) / 5.0,
            local_vy.reshape(-1, 1) / 5.0,
            vz.reshape(-1, 1) / 5.0,
            alpha_sin.reshape(-1, 1),
            alpha_cos.reshape(-1, 1),
            beta_sin.reshape(-1, 1),
            beta_cos.reshape(-1, 1),
            p.reshape(-1, 1),
            q.reshape(-1, 1),
            r.reshape(-1, 1),
            f0.reshape(-1, 1) / 7.0,
            f1.reshape(-1, 1) / 7.0,
            f2.reshape(-1, 1) / 7.0,
            f3.reshape(-1, 1) / 7.0,
            f4.reshape(-1, 1) / 7.0,
        ))

        return obs

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)
