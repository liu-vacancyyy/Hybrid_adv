import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.rc_human_reward import RCHumanReward
from reward_functions.rc_human_event_driven_reward import RCHumanEventDrivenReward
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.overload import Overload
from hybrid_termination_conditions.high_speed import HighSpeed
from hybrid_termination_conditions.rc_human_tracking_error import RCHumanTrackingError
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI


class RCHumanTask(BaseTask):
    """Human-like RC task with PX4 VTOL-MC stick processing.

    The command source is synthetic stick data. Raw sticks are sampled as
    piecewise-constant human inputs, then passed through PX4's deadzone/expo
    mapping. Velocity sticks map directly to velocity setpoints, while yaw uses
    PX4's manual yaw-rate first-order filter. VTOL in MC mode uses the same
    multicopter manual task path; for this velocity-tracking task we mirror the
    ManualPosition direct-velocity mapping: horizontal sticks are expo-shaped,
    limited to the unit circle, scaled by forward/back/side velocity limits,
    and rotated from the heading frame into the local N/E velocity setpoint.
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'rc_human'

        self.dt = float(getattr(config, 'dt', 0.02))
        legacy_vx_limit = float(getattr(config, 'rc_human_vx_limit', 1.0))
        self.vx_min = float(getattr(config, 'rc_human_vx_min', -legacy_vx_limit))
        self.vx_max = float(getattr(config, 'rc_human_vx_max', legacy_vx_limit))
        self.vx_limit = max(abs(self.vx_min), abs(self.vx_max), 1e-6)
        self.vy_limit = float(getattr(config, 'rc_human_vy_limit', 1.0))
        self.vz_limit = float(getattr(config, 'rc_human_vz_limit', 1.0))
        self.yaw_rate_limit = float(getattr(config, 'rc_human_yaw_rate_limit', 0.6))
        self.yaw_command_enable = bool(
            getattr(config, 'rc_human_yaw_command_enable', True)
        )
        self.yaw_hold_enable = bool(
            getattr(config, 'rc_human_yaw_hold_enable', True)
        )
        self.yaw_tracking_enable = bool(
            getattr(config, 'rc_human_yaw_tracking_enable', True)
        )
        self.command_axis_count = 4 if self.yaw_command_enable else 3

        self.deadzone = float(getattr(config, 'rc_human_deadzone', 0.10))
        self.expo = float(getattr(config, 'rc_human_expo', 0.60))
        self.velocity_tau = float(getattr(config, 'rc_human_velocity_tau', 0.0))
        self.yaw_tau = float(getattr(config, 'rc_human_yaw_tau', 0.08))
        self.stick_noise_std = float(getattr(config, 'rc_human_stick_noise_std', 0.015))
        self.dwell_min = int(getattr(config, 'rc_human_dwell_min_steps', 75))
        self.dwell_max = int(getattr(config, 'rc_human_dwell_max_steps', 300))
        self.curriculum_enable = bool(getattr(config, 'rc_human_curriculum_enable', True))
        self.vx_forward_curriculum_enable = bool(
            getattr(config, 'rc_human_vx_forward_curriculum_enable', True)
        )
        self.vx_forward_easy_max = float(
            getattr(config, 'rc_human_vx_forward_easy_max', min(self.vx_max, 1.0))
        )
        self.vx_forward_medium_max = float(
            getattr(config, 'rc_human_vx_forward_medium_max', min(self.vx_max, 3.0))
        )
        self.levels_per_mode = int(getattr(config, 'rc_human_levels_per_mode', 20))
        self.mix_current = float(getattr(config, 'rc_human_mix_current', 0.50))
        self.mix_easy = float(getattr(config, 'rc_human_mix_easy_replay', 0.20))
        self.mix_medium = float(getattr(config, 'rc_human_mix_medium_replay', 0.20))
        self.mix_random = float(getattr(config, 'rc_human_mix_random_replay', 0.10))
        self.easy_stick = float(getattr(config, 'rc_human_easy_stick', 0.25))
        self.medium_stick = float(getattr(config, 'rc_human_medium_stick', 0.65))
        self.hard_stick = float(getattr(config, 'rc_human_hard_stick', 1.0))
        self.command_transient_grace_steps = int(
            getattr(config, 'rc_human_command_transient_grace_steps', 60)
        )
        self.command_transient_threshold = float(
            getattr(config, 'rc_human_command_transient_threshold', 0.20)
        )
        self.mode3_reverse_axes = max(
            1, min(4, int(getattr(config, 'rc_human_mode3_reverse_axes', 1)))
        )
        self.large_command_transient_grace_steps = int(
            getattr(
                config,
                'rc_human_large_command_transient_grace_steps',
                getattr(config, 'rc_human_reversal_transient_grace_steps',
                        self.command_transient_grace_steps),
            )
        )
        self.large_command_transient_delta = float(
            getattr(config, 'rc_human_large_command_transient_delta', 1.0)
        )
        self.command_reversal_threshold = float(
            getattr(
                config,
                'rc_human_command_reversal_threshold',
                0.1,
            )
        )
        self.command_rate_limit_frac = max(0.0, float(os.environ.get(
            'RC_HUMAN_COMMAND_RATE_LIMIT_FRAC',
            getattr(config, 'rc_human_command_rate_limit_frac', 0.0),
        )))
        self.success_ignore_transient = bool(
            getattr(config, 'rc_human_success_ignore_transient', True)
        )
        self.mode5_hold_min_steps = int(
            getattr(config, 'rc_human_mode5_hold_min_steps', 75)
        )
        self.mode5_hold_max_steps = int(
            getattr(config, 'rc_human_mode5_hold_max_steps', 250)
        )
        self.mode5_release_recovery_steps = int(
            getattr(config, 'rc_human_mode5_release_recovery_steps', 150)
        )
        self.mode5_release_speed_error = float(
            getattr(config, 'rc_human_mode5_release_speed_error', 0.35)
        )
        self.mode5_release_target_frac = float(
            getattr(config, 'rc_human_mode5_release_target_frac', 0.80)
        )
        self.success_vel_error = float(getattr(config, 'rc_human_success_vel_error', 0.35))
        self.success_yaw_error = float(getattr(config, 'rc_human_success_yaw_error', 0.18))
        self.success_attitude_error = float(getattr(config, 'rc_human_success_attitude_error', 0.22))
        self.alt_high = float(getattr(config, 'rc_human_alt_high', 95.0))
        self.alt_low = float(getattr(config, 'rc_human_alt_low', 5.0))
        self.altitude_aware_vz_enable = bool(
            getattr(config, 'rc_human_altitude_aware_vz_enable', False)
        )
        self.alt_guard_zone = float(getattr(config, 'rc_human_alt_guard_zone', 5.0))
        self.noise_scale = getattr(config, 'noise_scale', 0.01)
        self.enable_sensor_noise = getattr(config, 'enable_sensor_noise', True)
        self.sensor_pos_std = float(getattr(config, 'sensor_pos_std', 1.0))
        self.sensor_vel_std = float(getattr(config, 'sensor_vel_std', 0.05))
        self.sensor_att_std = float(getattr(config, 'sensor_att_std', 0.005))
        self.sensor_omega_std = float(getattr(config, 'sensor_omega_std', 0.0005))

        self.target_vx = torch.zeros(self.n, device=self.device)
        self.target_vy = torch.zeros(self.n, device=self.device)
        self.target_vn = torch.zeros(self.n, device=self.device)
        self.target_ve = torch.zeros(self.n, device=self.device)
        self.target_vz = torch.zeros(self.n, device=self.device)
        self.target_heading = torch.zeros(self.n, device=self.device)
        self.target_yaw_rate = torch.zeros(self.n, device=self.device)
        self.initial_local_vx = torch.zeros(self.n, device=self.device)
        self.initial_local_vy = torch.zeros(self.n, device=self.device)
        self.initial_vz = torch.zeros(self.n, device=self.device)

        self.raw_vx = torch.zeros(self.n, device=self.device)
        self.raw_vy = torch.zeros(self.n, device=self.device)
        self.raw_vz = torch.zeros(self.n, device=self.device)
        self.raw_yaw = torch.zeros(self.n, device=self.device)
        self.desired_raw_vx = torch.zeros(self.n, device=self.device)
        self.desired_raw_vy = torch.zeros(self.n, device=self.device)
        self.desired_raw_vz = torch.zeros(self.n, device=self.device)
        self.desired_raw_yaw = torch.zeros(self.n, device=self.device)
        self.stick_vx = torch.zeros(self.n, device=self.device)
        self.stick_vy = torch.zeros(self.n, device=self.device)
        self.stick_vz = torch.zeros(self.n, device=self.device)
        self.stick_yaw = torch.zeros(self.n, device=self.device)
        self.command_rate_limited = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.command_raw_delta = torch.zeros(self.n, device=self.device)
        self.vx_forward_limit = torch.full(
            (self.n,), self.vx_forward_easy_max, device=self.device
        )
        self.dwell_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.command_transient_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.mode5_release_state = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.mode5_hold_elapsed = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.mode5_recovery_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.mode5_pre_release_raw = torch.zeros(self.n, device=self.device)
        self.last_synced_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.operation_mode = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.curriculum_level = torch.zeros(self.n, dtype=torch.long, device=self.device)
        mode_order = self._parse_mode_order(config)
        self.mode_order = torch.tensor(mode_order, dtype=torch.long, device=self.device)
        mode_slots = int(os.environ.get(
            'RC_HUMAN_MAX_MODE_SLOTS',
            getattr(config, 'rc_human_max_mode_slots', int(self.mode_order.numel())),
        ))
        self.active_mode_slots = max(1, min(mode_slots, int(self.mode_order.numel())))
        self.max_curriculum_level = self.levels_per_mode * self.active_mode_slots - 1
        self._normalize_curriculum_mix()
        self.episode_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_yaw_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_attitude_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_metric_count = torch.zeros(self.n, device=self.device)
        self.episode_metric_skipped_count = torch.zeros(self.n, device=self.device)
        self.last_reward_terms = {}

        self.reward_functions = [
            RCHumanReward(self.config),
            RCHumanEventDrivenReward(self.config),
        ]
        self.termination_conditions = [
            Overload(self.config),
            LowAltitude(self.config),
            HighSpeed(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
        ]
        if (
            bool(getattr(self.config, 'rc_human_tracking_bad_done_enable', True))
            or bool(getattr(self.config, 'rc_human_vxyvz_dynamic_bad_done_enable', False))
        ):
            self.termination_conditions.append(RCHumanTrackingError(self.config))
        self.termination_conditions.append(HoverTimeoutDone(self.config))

    def _parse_mode_order(self, config):
        raw = os.environ.get(
            'RC_HUMAN_MODE_ORDER',
            getattr(config, 'rc_human_mode_order', '0 1 2 5 3 4'),
        )
        if isinstance(raw, (list, tuple)):
            values = [int(x) for x in raw]
        else:
            values = [int(x) for x in str(raw).replace(',', ' ').split()]
        if not values:
            raise ValueError('RC_HUMAN_MODE_ORDER must contain at least one mode id')
        invalid = [x for x in values if x < 0 or x > 5]
        if invalid:
            raise ValueError(f'Invalid rc_human mode ids: {invalid}')
        if len(set(values)) != len(values):
            raise ValueError(f'RC_HUMAN_MODE_ORDER contains duplicates: {values}')
        return values

    def reset(self, env):
        reset = (env.is_done.bool() | env.bad_done.bool()) | env.exceed_time_limit.bool()
        if int(torch.sum(reset).item()) == 0:
            return

        self._update_curriculum_from_last_episode(env, reset)

        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        _roll, _pitch, heading = env.model.get_posture()
        local_vx, local_vy = self.ground_to_local_velocity(vx_n, vy_e, heading)

        self.target_vx[reset] = 0.0
        self.target_vy[reset] = 0.0
        self.target_vz[reset] = 0.0
        self.target_heading[reset] = heading[reset]
        self.target_yaw_rate[reset] = 0.0
        self.target_vn[reset], self.target_ve[reset] = self.local_to_ground_velocity(
            self.target_vx[reset], self.target_vy[reset], self.target_heading[reset]
        )
        self.initial_local_vx[reset] = local_vx[reset]
        self.initial_local_vy[reset] = local_vy[reset]
        self.initial_vz[reset] = vz[reset]

        self.raw_vx[reset] = 0.0
        self.raw_vy[reset] = 0.0
        self.raw_vz[reset] = 0.0
        self.raw_yaw[reset] = 0.0
        self.desired_raw_vx[reset] = 0.0
        self.desired_raw_vy[reset] = 0.0
        self.desired_raw_vz[reset] = 0.0
        self.desired_raw_yaw[reset] = 0.0
        self.stick_vx[reset] = 0.0
        self.stick_vy[reset] = 0.0
        self.stick_vz[reset] = 0.0
        self.stick_yaw[reset] = 0.0
        self.command_rate_limited[reset] = False
        self.command_raw_delta[reset] = 0.0
        self.vx_forward_limit[reset] = self.vx_forward_easy_max
        self.dwell_left[reset] = 0
        self.command_transient_left[reset] = 0
        self.mode5_release_state[reset] = 0
        self.mode5_hold_elapsed[reset] = 0
        self.mode5_recovery_left[reset] = 0
        self.mode5_pre_release_raw[reset] = 0.0
        self.last_synced_step[reset] = -1
        self.episode_count[reset] += 1
        self.episode_error_sum[reset] = 0.0
        self.episode_yaw_error_sum[reset] = 0.0
        self.episode_attitude_error_sum[reset] = 0.0
        self.episode_metric_count[reset] = 0.0
        self.episode_metric_skipped_count[reset] = 0.0

    def step(self, env):
        self.sync_command(env)

    def sync_command(self, env):
        steps = env.step_count.long()
        need_sync = steps > self.last_synced_step
        guard = 0
        while torch.any(need_sync):
            self._advance_command(env, need_sync)
            self.last_synced_step[need_sync] += 1
            guard += 1
            if guard > 4:
                self.last_synced_step[need_sync] = steps[need_sync]
            need_sync = steps > self.last_synced_step

    def ground_to_local_velocity(self, vx_n, vy_e, heading):
        local_vx = vx_n * torch.cos(heading) + vy_e * torch.sin(heading)
        local_vy = -vx_n * torch.sin(heading) + vy_e * torch.cos(heading)
        return local_vx, local_vy

    def local_to_ground_velocity(self, local_vx, local_vy, heading):
        vx_n = local_vx * torch.cos(heading) - local_vy * torch.sin(heading)
        vy_e = local_vx * torch.sin(heading) + local_vy * torch.cos(heading)
        return vx_n, vy_e

    def _resample_raw_sticks(self, mask, env=None):
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
        values = torch.rand(size, 4, device=d) * amp.reshape(-1, 1) * signs

        # 0 hover/release: all sticks centered.
        hover = mode == 0
        values[hover, :] = 0.0

        # 1 continuous correction: all axes move together, from small to hard.
        # 2 step push: one axis jumps to a high value.
        step = mode == 2
        if torch.any(step):
            axis = torch.randint(
                0,
                self.command_axis_count,
                (int(step.sum().item()),),
                device=d,
            )
            values[step, :] = 0.0
            values[step, axis] = amp[step] * signs[step, 0]

        # 3 reversal: flip the currently commanded sticks hard in the opposite direction.
        reverse = mode == 3
        if torch.any(reverse):
            current = torch.stack((
                self.raw_vx[idx[reverse]],
                self.raw_vy[idx[reverse]],
                self.raw_vz[idx[reverse]],
                self.raw_yaw[idx[reverse]],
            ), dim=1)
            fallback = signs[reverse] * amp[reverse].reshape(-1, 1)
            reversed_values = torch.where(
                torch.abs(current) > 0.1,
                -torch.sign(current) * amp[reverse].reshape(-1, 1),
                fallback,
            )
            if self.mode3_reverse_axes >= 4:
                values[reverse, :] = reversed_values
            else:
                count = int(reverse.sum().item())
                scores = torch.where(
                    torch.abs(current) > 0.1,
                    torch.abs(current),
                    torch.rand(count, 4, device=d) * 1e-3,
                )
                if not self.yaw_command_enable:
                    scores[:, 3] = -1.0
                selected = torch.topk(
                    scores,
                    k=min(self.mode3_reverse_axes, self.command_axis_count),
                    dim=1,
                ).indices
                limited = torch.zeros_like(reversed_values)
                limited.scatter_(1, selected, reversed_values.gather(1, selected))
                values[reverse, :] = limited

        # 4 combined input: several axes move together.
        combined = mode == 4
        if torch.any(combined):
            values[combined, :] = (
                0.35 + 0.65 * torch.rand(int(combined.sum().item()), 4, device=d)
            ) * amp[combined].reshape(-1, 1) * signs[combined]

        prev_raw = torch.stack((
            self.desired_raw_vx[idx],
            self.desired_raw_vy[idx],
            self.desired_raw_vz[idx],
            self.desired_raw_yaw[idx],
        ), dim=1)

        # 5 release: hold a level-dependent forward stick until the aircraft is
        # near the requested speed or a timeout is reached.  The actual release
        # to centered sticks is handled by _update_mode5_release_state().
        release = mode == 5
        if torch.any(release):
            release_idx = idx[release]
            pre_release = amp[release]
            self.mode5_release_state[release_idx] = 1
            self.mode5_hold_elapsed[release_idx] = 0
            self.mode5_recovery_left[release_idx] = 0
            self.mode5_pre_release_raw[release_idx] = pre_release
            self.stick_vy[release_idx] = 0.0
            self.stick_vz[release_idx] = 0.0
            self.stick_yaw[release_idx] = 0.0
            values[release, :] = 0.0
            values[release, 0] = pre_release

        if not self.yaw_command_enable:
            values[:, 3] = 0.0

        if env is not None:
            values[:, 2] = self._clamp_raw_vz_for_altitude(
                idx, values[:, 2], env
            )

        self.operation_mode[mask] = mode
        non_release = ~release
        if torch.any(non_release):
            non_release_idx = idx[non_release]
            self.mode5_release_state[non_release_idx] = 0
            self.mode5_hold_elapsed[non_release_idx] = 0
            self.mode5_recovery_left[non_release_idx] = 0
            self.mode5_pre_release_raw[non_release_idx] = 0.0

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
        if not self.yaw_command_enable:
            self.desired_raw_yaw[mask] = 0.0
        self.vx_forward_limit[mask] = self._curriculum_vx_forward_limit(
            sampled_level, mode
        )
        dwell = torch.randint(
            self.dwell_min, self.dwell_max + 1, (size,), device=d, dtype=torch.long
        )
        if torch.any(release):
            dwell[release] = max(
                self.mode5_hold_max_steps + self.mode5_release_recovery_steps,
                1,
            )
        self.dwell_left[mask] = dwell

    def _update_curriculum_from_last_episode(self, env, reset):
        if not self.curriculum_enable:
            return

        finished = reset & (self.episode_metric_count > 0)
        if int(torch.sum(finished).item()) == 0:
            return

        count = torch.clamp(self.episode_metric_count[finished], min=1.0)
        mean_vel_error = self.episode_error_sum[finished] / count
        mean_yaw_error = self.episode_yaw_error_sum[finished] / count
        mean_attitude_error = self.episode_attitude_error_sum[finished] / count

        clean_done = env.is_done[finished].bool() & (~env.bad_done[finished].bool())
        accurate = (
            (mean_vel_error < self.success_vel_error)
            & (mean_attitude_error < self.success_attitude_error)
        )
        if self.yaw_tracking_enable:
            accurate = accurate & (mean_yaw_error < self.success_yaw_error)
        success = clean_done & accurate
        idx = torch.where(finished)[0]

        level = self.curriculum_level[idx]
        level = torch.where(success, level + 1, level - 1)
        self.curriculum_level[idx] = torch.clamp(level, 0, self.max_curriculum_level)

    def _normalize_curriculum_mix(self):
        total = self.mix_current + self.mix_easy + self.mix_medium + self.mix_random
        if total <= 1e-6:
            self.mix_current = 1.0
            self.mix_easy = 0.0
            self.mix_medium = 0.0
            self.mix_random = 0.0
            return
        self.mix_current /= total
        self.mix_easy /= total
        self.mix_medium /= total
        self.mix_random /= total

    def _randint_level(self, low, high, shape):
        low = int(max(0, min(low, self.max_curriculum_level)))
        high = int(max(low, min(high, self.max_curriculum_level)))
        return torch.randint(low, high + 1, shape, device=self.device, dtype=torch.long)

    def _randint_level_between(self, low, high):
        low = torch.clamp(low.long(), 0, self.max_curriculum_level)
        high = torch.clamp(high.long(), 0, self.max_curriculum_level)
        high = torch.maximum(high, low)
        span = (high - low + 1).float()
        return low + torch.floor(torch.rand_like(span) * span).long()

    def _sample_command_level(self, idx):
        if not self.curriculum_enable:
            return torch.full((idx.numel(),), self.max_curriculum_level,
                              device=self.device, dtype=torch.long)

        current = torch.clamp(self.curriculum_level[idx], 0, self.max_curriculum_level)
        sampled = current.clone()
        selector = torch.rand(idx.numel(), device=self.device)

        current_end = self.mix_current
        easy_end = current_end + self.mix_easy
        medium_end = easy_end + self.mix_medium

        easy_mask = (selector >= current_end) & (selector < easy_end)
        medium_mask = (selector >= easy_end) & (selector < medium_end)
        random_mask = selector >= medium_end

        easy_max = max(0, int(round(self.max_curriculum_level * 0.33)))
        medium_min = min(self.max_curriculum_level, easy_max + 1)
        medium_max = max(medium_min, int(round(self.max_curriculum_level * 0.66)))

        if torch.any(easy_mask):
            high = torch.minimum(
                current[easy_mask],
                torch.full_like(current[easy_mask], easy_max),
            )
            sampled[easy_mask] = self._randint_level_between(
                torch.zeros_like(high), high
            )
        if torch.any(medium_mask):
            current_medium = current[medium_mask]
            can_sample_medium = current_medium >= medium_min
            low = torch.where(
                can_sample_medium,
                torch.full_like(current_medium, medium_min),
                torch.zeros_like(current_medium),
            )
            high = torch.where(
                can_sample_medium,
                torch.minimum(
                    current_medium,
                    torch.full_like(current_medium, medium_max),
                ),
                current_medium,
            )
            sampled[medium_mask] = self._randint_level_between(low, high)
        if torch.any(random_mask):
            high = current[random_mask]
            sampled[random_mask] = self._randint_level_between(
                torch.zeros_like(high), high
            )

        return sampled

    def update_episode_metrics(self, vel_error, yaw_error, attitude_error):
        valid = torch.ones_like(vel_error, dtype=torch.bool, device=self.device)
        if self.success_ignore_transient:
            valid = valid & (self.command_transient_left <= 0)

        valid_f = valid.detach().float()
        self.episode_error_sum += vel_error.detach() * valid_f
        self.episode_yaw_error_sum += torch.abs(yaw_error.detach()) * valid_f
        self.episode_attitude_error_sum += attitude_error.detach() * valid_f
        self.episode_metric_count += valid_f
        self.episode_metric_skipped_count += (~valid).detach().float()

    def get_training_metrics(self):
        count = torch.clamp(self.episode_metric_count, min=1.0)
        mean_vel_error = self.episode_error_sum / count
        mean_yaw_error = self.episode_yaw_error_sum / count
        mean_attitude_error = self.episode_attitude_error_sum / count

        metrics = {
            'rc_human/curriculum_level_mean': self.curriculum_level.float().mean(),
            'rc_human/curriculum_level_max': self.curriculum_level.float().max(),
            'rc_human/curriculum_level_limit': torch.tensor(
                float(self.max_curriculum_level), device=self.device),
            'rc_human/vx_forward_limit_mean': self.vx_forward_limit.mean(),
            'rc_human/vx_forward_limit_max': self.vx_forward_limit.max(),
            'rc_human/tracking_error_mean': mean_vel_error.mean(),
            'rc_human/tracking_vel_error_mean': mean_vel_error.mean(),
            'rc_human/tracking_yaw_error_mean': mean_yaw_error.mean(),
            'rc_human/tracking_attitude_error_mean': mean_attitude_error.mean(),
            'rc_human/mix_current': torch.tensor(
                self.mix_current, device=self.device),
            'rc_human/mix_easy_replay': torch.tensor(
                self.mix_easy, device=self.device),
            'rc_human/mix_medium_replay': torch.tensor(
                self.mix_medium, device=self.device),
            'rc_human/mix_random_replay': torch.tensor(
                self.mix_random, device=self.device),
        }
        metric_total = self.episode_metric_count + self.episode_metric_skipped_count
        metrics['rc_human/success_metric_valid_fraction'] = (
            self.episode_metric_count / torch.clamp(metric_total, min=1.0)
        ).mean()
        metrics['rc_human/success_metric_skipped_fraction'] = (
            self.episode_metric_skipped_count / torch.clamp(metric_total, min=1.0)
        ).mean()
        for mode_id in range(6):
            metrics[f'rc_human/mode_{mode_id}_fraction'] = (
                self.operation_mode == mode_id).float().mean()
        metrics['rc_human/command_transient_fraction'] = (
            self.command_transient_left > 0).float().mean()
        metrics['rc_human/command_rate_limit_frac'] = torch.tensor(
            float(self.command_rate_limit_frac), device=self.device)
        metrics['rc_human/command_rate_limited_fraction'] = (
            self.command_rate_limited.float().mean())
        metrics['rc_human/command_raw_delta_mean'] = self.command_raw_delta.mean()
        metrics['rc_human/yaw_command_enable'] = torch.tensor(
            float(self.yaw_command_enable), device=self.device)
        metrics['rc_human/yaw_tracking_enable'] = torch.tensor(
            float(self.yaw_tracking_enable), device=self.device)
        metrics['rc_human/altitude_aware_vz_enable'] = torch.tensor(
            float(self.altitude_aware_vz_enable), device=self.device)
        metrics['rc_human/mode5_hold_fraction'] = (
            self.mode5_release_state == 1).float().mean()
        metrics['rc_human/mode5_released_fraction'] = (
            self.mode5_release_state == 2).float().mean()
        for key, value in self.last_reward_terms.items():
            metrics[key] = value
        return metrics

    def _curriculum_mode(self, idx):
        if not self.curriculum_enable:
            return self.mode_order[self.active_mode_slots - 1].repeat(idx.numel())
        return self._curriculum_mode_from_level(self.curriculum_level[idx])

    def _curriculum_mode_from_level(self, level):
        mode_slot = torch.clamp(level // self.levels_per_mode, 0, self.active_mode_slots - 1)
        return self.mode_order[mode_slot]

    def _curriculum_amplitude(self, level, mode):
        if not self.curriculum_enable:
            return torch.full((level.numel(),), self.hard_stick, device=self.device)

        sublevel = (level % self.levels_per_mode).float()
        denom = max(float(self.levels_per_mode - 1), 1.0)
        progress = sublevel / denom

        amp = self.easy_stick + progress * (self.hard_stick - self.easy_stick)
        hover = mode == 0
        amp[hover] = 0.0
        hard = (mode == 2) | (mode == 3) | (mode == 4)
        amp[hard] = torch.clamp(amp[hard], min=self.medium_stick, max=self.hard_stick)
        return amp

    def _curriculum_vx_forward_limit(self, level, mode):
        if (not self.curriculum_enable) or (not self.vx_forward_curriculum_enable):
            return torch.full((level.numel(),), self.vx_max, device=self.device)

        sublevel = (level % self.levels_per_mode).float()
        denom = max(float(self.levels_per_mode - 1), 1.0)
        progress = torch.clamp(sublevel / denom, 0.0, 1.0)
        first_half = progress <= 0.5
        early = self.vx_forward_easy_max + (
            self.vx_forward_medium_max - self.vx_forward_easy_max
        ) * (progress / 0.5)
        late = self.vx_forward_medium_max + (
            self.vx_max - self.vx_forward_medium_max
        ) * ((progress - 0.5) / 0.5)
        limit = torch.where(first_half, early, late)
        lower = min(max(0.0, self.vx_forward_easy_max), self.vx_max)
        limit = torch.clamp(limit, min=lower, max=self.vx_max)

        hover = mode == 0
        return torch.where(
            hover,
            torch.full_like(limit, self.vx_forward_easy_max),
            limit,
        )

    def _shape_stick_input(self, raw):
        abs_raw = torch.abs(raw)
        denom = max(1.0 - self.deadzone, 1e-6)
        x = torch.where(
            abs_raw <= self.deadzone,
            torch.zeros_like(raw),
            torch.sign(raw) * (abs_raw - self.deadzone) / denom,
        )
        x = (1.0 - self.expo) * x + self.expo * x * x * x
        return x

    def _limit_stick_unit_length_xy(self, stick_x, stick_y):
        norm = torch.sqrt(stick_x * stick_x + stick_y * stick_y)
        scale = torch.clamp(norm, min=1.0)
        return stick_x / scale, stick_y / scale

    def _px4_process_stick(self, raw, state, tau):
        if self.stick_noise_std > 0.0:
            raw = torch.clamp(raw + torch.randn_like(raw) * self.stick_noise_std, -1.0, 1.0)

        x = self._shape_stick_input(raw)
        if tau <= 0.0:
            return x
        alpha = self.dt / max(tau + self.dt, 1e-6)
        return state + alpha * (x - state)

    def _raw_stick_tensor(self, idx):
        return torch.stack((
            self.raw_vx[idx],
            self.raw_vy[idx],
            self.raw_vz[idx],
            self.raw_yaw[idx],
        ), dim=1)

    def _desired_raw_stick_tensor(self, idx):
        return torch.stack((
            self.desired_raw_vx[idx],
            self.desired_raw_vy[idx],
            self.desired_raw_vz[idx],
            self.desired_raw_yaw[idx],
        ), dim=1)

    def _set_raw_stick_tensor(self, idx, values):
        self.raw_vx[idx] = values[:, 0]
        self.raw_vy[idx] = values[:, 1]
        self.raw_vz[idx] = values[:, 2]
        self.raw_yaw[idx] = values[:, 3]

    def _apply_raw_stick_rate_limit(self, mask):
        if int(torch.sum(mask).item()) == 0:
            return

        idx = torch.where(mask)[0]
        desired = self._desired_raw_stick_tensor(idx)
        current = self._raw_stick_tensor(idx)
        delta = desired - current
        limit_frac = float(self.command_rate_limit_frac)
        if limit_frac > 0.0:
            max_delta = torch.full_like(delta, abs(limit_frac))
            clipped_delta = torch.clamp(delta, -max_delta, max_delta)
            limited = current + clipped_delta
            rate_limited = torch.any(torch.abs(delta) > (max_delta + 1e-6), dim=1)
        else:
            clipped_delta = delta
            limited = desired
            rate_limited = torch.zeros(idx.numel(), dtype=torch.bool, device=self.device)

        limited = torch.clamp(limited, -1.0, 1.0)
        self._set_raw_stick_tensor(idx, limited)
        if not self.yaw_command_enable:
            self.raw_yaw[idx] = 0.0
        self.command_rate_limited[idx] = rate_limited
        self.command_raw_delta[idx] = torch.max(torch.abs(clipped_delta), dim=1).values

    def _scale_vx_stick(self, stick, forward_limit):
        return torch.where(
            stick >= 0.0,
            stick * forward_limit,
            stick * abs(self.vx_min),
        )

    def _altitude_raw_vz_bounds(self, idx, env):
        if (not self.altitude_aware_vz_enable) or idx.numel() == 0:
            lower = torch.full((idx.numel(),), -1.0, device=self.device)
            upper = torch.full((idx.numel(),), 1.0, device=self.device)
            return lower, upper

        _npos, _epos, altitude = env.model.get_position()
        altitude = altitude[idx]
        if self.alt_guard_zone <= 0.0:
            lower = torch.where(
                altitude <= self.alt_low,
                torch.zeros_like(altitude),
                -torch.ones_like(altitude),
            )
            upper = torch.where(
                altitude >= self.alt_high,
                torch.zeros_like(altitude),
                torch.ones_like(altitude),
            )
            return lower, upper

        zone = self.alt_guard_zone
        lower_scale = torch.clamp((altitude - self.alt_low) / zone, 0.0, 1.0)
        upper_scale = torch.clamp((self.alt_high - altitude) / zone, 0.0, 1.0)
        lower = -lower_scale
        upper = upper_scale
        return lower, upper

    def _clamp_raw_vz_for_altitude(self, idx, raw_vz, env):
        lower, upper = self._altitude_raw_vz_bounds(idx, env)
        return torch.minimum(torch.maximum(raw_vz, lower), upper)

    def _apply_altitude_raw_vz_guard(self, mask, env):
        if (not self.altitude_aware_vz_enable) or int(torch.sum(mask).item()) == 0:
            return
        idx = torch.where(mask)[0]
        self.desired_raw_vz[idx] = self._clamp_raw_vz_for_altitude(
            idx, self.desired_raw_vz[idx], env
        )
        self.raw_vz[idx] = self._clamp_raw_vz_for_altitude(
            idx, self.raw_vz[idx], env
        )

    def _update_px4_vtol_mc_targets_from_sticks(self, mask, env):
        stick_vx, stick_vy = self._limit_stick_unit_length_xy(
            self.stick_vx[mask], self.stick_vy[mask]
        )
        self.stick_vx[mask] = stick_vx
        self.stick_vy[mask] = stick_vy

        self.target_vx[mask] = self._scale_vx_stick(
            stick_vx, self.vx_forward_limit[mask]
        )
        self.target_vy[mask] = stick_vy * self.vy_limit
        if self.altitude_aware_vz_enable:
            idx = torch.where(mask)[0]
            self.stick_vz[idx] = self._clamp_raw_vz_for_altitude(
                idx, self.stick_vz[idx], env
            )
        self.target_vz[mask] = self.stick_vz[mask] * self.vz_limit
        if self.yaw_command_enable:
            self.target_yaw_rate[mask] = self.stick_yaw[mask] * self.yaw_rate_limit
        else:
            self.stick_yaw[mask] = 0.0
            self.target_yaw_rate[mask] = 0.0

        _npos, _epos, altitude = env.model.get_position()
        _roll, _pitch, _heading = env.model.get_posture()
        high = mask & (altitude > self.alt_high) & (self.target_vz > 0.0)
        low = mask & (altitude < self.alt_low) & (self.target_vz < 0.0)
        self.target_vz[high | low] = 0.0
        self.stick_vz[high | low] = 0.0

        if self.yaw_command_enable or not self.yaw_hold_enable:
            self.target_heading[mask] = wrap_PI(
                self.target_heading[mask] + self.target_yaw_rate[mask] * self.dt
            )
        self.target_vn[mask], self.target_ve[mask] = self.local_to_ground_velocity(
            self.target_vx[mask], self.target_vy[mask], self.target_heading[mask]
        )

    def _apply_px4_vtol_mc_manual_sticks(self, mask, env):
        if int(torch.sum(mask).item()) == 0:
            return

        self.stick_vx[mask] = self._px4_process_stick(
            self.raw_vx[mask], self.stick_vx[mask], self.velocity_tau
        )
        self.stick_vy[mask] = self._px4_process_stick(
            self.raw_vy[mask], self.stick_vy[mask], self.velocity_tau
        )
        self.stick_vz[mask] = self._px4_process_stick(
            self.raw_vz[mask], self.stick_vz[mask], self.velocity_tau
        )
        if self.yaw_command_enable:
            self.stick_yaw[mask] = self._px4_process_stick(
                self.raw_yaw[mask], self.stick_yaw[mask], self.yaw_tau
            )
        else:
            self.raw_yaw[mask] = 0.0
            self.stick_yaw[mask] = 0.0
        self._update_px4_vtol_mc_targets_from_sticks(mask, env)

    def _update_mode5_release_state(self, env, mask):
        active_hold = mask & (self.mode5_release_state == 1)
        if torch.any(active_hold):
            self.mode5_hold_elapsed[active_hold] += 1
            _roll, _pitch, heading = env.model.get_posture()
            vx_n, vy_e = env.model.get_ground_speed()
            local_vx, _local_vy = self.ground_to_local_velocity(vx_n, vy_e, heading)
            shaped = self._shape_stick_input(self.mode5_pre_release_raw[active_hold])
            intended_vx = self._scale_vx_stick(
                shaped, self.vx_forward_limit[active_hold]
            )
            enough_command = self.target_vx[active_hold] >= (
                intended_vx * self.mode5_release_target_frac
            )
            near_speed = torch.abs(
                local_vx[active_hold] - self.target_vx[active_hold]
            ) < self.mode5_release_speed_error
            min_elapsed = self.mode5_hold_elapsed[active_hold] >= self.mode5_hold_min_steps
            timeout = self.mode5_hold_elapsed[active_hold] >= self.mode5_hold_max_steps
            release_now_local = timeout | (min_elapsed & enough_command & near_speed)

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

    def _advance_command(self, env, mask):
        self._update_mode5_release_state(env, mask)
        resample = mask & (self.dwell_left <= 0) & (self.mode5_release_state == 0)
        self._resample_raw_sticks(resample, env)
        self._apply_altitude_raw_vz_guard(mask, env)
        self._apply_raw_stick_rate_limit(mask)
        self._apply_altitude_raw_vz_guard(mask, env)
        self.dwell_left[mask] = torch.clamp(self.dwell_left[mask] - 1, min=0)
        self.command_transient_left[mask] = torch.clamp(
            self.command_transient_left[mask] - 1, min=0)

        self._apply_px4_vtol_mc_manual_sticks(mask, env)

    def _apply_sensor_noise(self, roll, pitch, heading, vx, vy, vz, p, q, r):
        roll = wrap_PI(roll + torch.randn_like(roll) * self.sensor_att_std)
        pitch = wrap_PI(pitch + torch.randn_like(pitch) * self.sensor_att_std)
        heading = wrap_PI(heading + torch.randn_like(heading) * self.sensor_att_std)
        vx = vx + torch.randn_like(vx) * self.sensor_vel_std
        vy = vy + torch.randn_like(vy) * self.sensor_vel_std
        vz = vz + torch.randn_like(vz) * self.sensor_vel_std
        p = p + torch.randn_like(p) * self.sensor_omega_std
        q = q + torch.randn_like(q) * self.sensor_omega_std
        r = r + torch.randn_like(r) * self.sensor_omega_std
        return roll, pitch, heading, vx, vy, vz, p, q, r

    def _build_obs(self, env, add_sensor_noise):
        self.sync_command(env)

        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        eas = env.model.get_EAS()
        alpha_sin, alpha_cos, beta_sin, beta_cos = env.model.get_aero_sincos()
        p, q, r = env.model.get_angular_velocity()
        f1, f2, f3, f4, f5 = env.model.get_F()
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
        err_local_vx, err_local_vy = self.ground_to_local_velocity(
            vx_n - self.target_vn, vy_e - self.target_ve, heading
        )

        norm_dvx = (err_local_vx / max(self.vx_limit, 1e-6)).reshape(-1, 1)
        norm_dvy = (err_local_vy / max(self.vy_limit, 1e-6)).reshape(-1, 1)
        norm_dvz = ((vz - self.target_vz) / max(self.vz_limit, 1e-6)).reshape(-1, 1)
        norm_dyaw = (wrap_PI(heading - self.target_heading) / torch.pi).reshape(-1, 1)

        obs = torch.hstack((
            norm_dvx,
            norm_dvy,
            norm_dvz,
            norm_dyaw,
            (self.target_vx / max(self.vx_limit, 1e-6)).reshape(-1, 1),
            (self.target_vy / max(self.vy_limit, 1e-6)).reshape(-1, 1),
            (self.target_vz / max(self.vz_limit, 1e-6)).reshape(-1, 1),
            (self.target_yaw_rate / max(self.yaw_rate_limit, 1e-6)).reshape(-1, 1),
            altitude.reshape(-1, 1) / 100.0,
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
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
            f1.reshape(-1, 1) / 7.0,
            f2.reshape(-1, 1) / 7.0,
            f3.reshape(-1, 1) / 7.0,
            f4.reshape(-1, 1) / 7.0,
            f5.reshape(-1, 1) / 7.0,
        ))

        return obs

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)
