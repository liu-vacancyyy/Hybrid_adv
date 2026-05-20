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
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI


class RCHumanTask(BaseTask):
    """Human-like RC task with PX4-style stick processing.

    The command source is synthetic stick data. Raw sticks are sampled as
    piecewise-constant human inputs, then passed through deadzone, expo,
    slew-rate limiting and low-pass filtering before becoming velocity and yaw
    commands. Like PX4 ManualPosition, horizontal sticks are first interpreted
    in the heading frame, then rotated into the local N/E velocity setpoint used
    by the controller. LearningToFly limits are kept for the stick-scaled
    velocity magnitudes.
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'rc_human'

        self.dt = float(getattr(config, 'dt', 0.02))
        self.vx_limit = float(getattr(config, 'rc_human_vx_limit', 1.0))
        self.vy_limit = float(getattr(config, 'rc_human_vy_limit', 1.0))
        self.vz_limit = float(getattr(config, 'rc_human_vz_limit', 1.0))
        self.yaw_rate_limit = float(getattr(config, 'rc_human_yaw_rate_limit', 0.6))

        self.deadzone = float(getattr(config, 'rc_human_deadzone', 0.10))
        self.expo = float(getattr(config, 'rc_human_expo', 0.60))
        self.velocity_tau = float(getattr(config, 'rc_human_velocity_tau', 0.55))
        self.yaw_tau = float(getattr(config, 'rc_human_yaw_tau', 0.75))
        self.stick_slew_rate = float(getattr(config, 'rc_human_stick_slew_rate', 2.0))
        self.stick_noise_std = float(getattr(config, 'rc_human_stick_noise_std', 0.015))
        self.dwell_min = int(getattr(config, 'rc_human_dwell_min_steps', 75))
        self.dwell_max = int(getattr(config, 'rc_human_dwell_max_steps', 300))
        self.curriculum_enable = bool(getattr(config, 'rc_human_curriculum_enable', True))
        self.levels_per_mode = int(getattr(config, 'rc_human_levels_per_mode', 20))
        self.mix_current = float(getattr(config, 'rc_human_mix_current', 0.50))
        self.mix_easy = float(getattr(config, 'rc_human_mix_easy_replay', 0.20))
        self.mix_medium = float(getattr(config, 'rc_human_mix_medium_replay', 0.20))
        self.mix_random = float(getattr(config, 'rc_human_mix_random_replay', 0.10))
        self.easy_stick = float(getattr(config, 'rc_human_easy_stick', 0.25))
        self.medium_stick = float(getattr(config, 'rc_human_medium_stick', 0.65))
        self.hard_stick = float(getattr(config, 'rc_human_hard_stick', 1.0))
        self.success_vel_error = float(getattr(config, 'rc_human_success_vel_error', 0.35))
        self.success_yaw_error = float(getattr(config, 'rc_human_success_yaw_error', 0.18))
        self.success_attitude_error = float(getattr(config, 'rc_human_success_attitude_error', 0.22))
        self.alt_high = float(getattr(config, 'rc_human_alt_high', 95.0))
        self.alt_low = float(getattr(config, 'rc_human_alt_low', 5.0))
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

        self.raw_vx = torch.zeros(self.n, device=self.device)
        self.raw_vy = torch.zeros(self.n, device=self.device)
        self.raw_vz = torch.zeros(self.n, device=self.device)
        self.raw_yaw = torch.zeros(self.n, device=self.device)
        self.stick_vx = torch.zeros(self.n, device=self.device)
        self.stick_vy = torch.zeros(self.n, device=self.device)
        self.stick_vz = torch.zeros(self.n, device=self.device)
        self.stick_yaw = torch.zeros(self.n, device=self.device)
        self.dwell_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.last_synced_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.operation_mode = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.curriculum_level = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.mode_order = torch.tensor([0, 1, 2, 5, 3, 4], dtype=torch.long, device=self.device)
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
            HoverTimeoutDone(self.config),
        ]

    def reset(self, env):
        reset = (env.is_done.bool() | env.bad_done.bool()) | env.exceed_time_limit.bool()
        if int(torch.sum(reset).item()) == 0:
            return

        self._update_curriculum_from_last_episode(env, reset)

        vx_n, vy_e = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        _roll, _pitch, heading = env.model.get_posture()
        local_vx, local_vy = self.ground_to_local_velocity(vx_n, vy_e, heading)

        self.target_vx[reset] = torch.clamp(local_vx[reset], -self.vx_limit, self.vx_limit)
        self.target_vy[reset] = torch.clamp(local_vy[reset], -self.vy_limit, self.vy_limit)
        self.target_vn[reset] = vx_n[reset]
        self.target_ve[reset] = vy_e[reset]
        self.target_vz[reset] = torch.clamp(vz[reset], -self.vz_limit, self.vz_limit)
        self.target_heading[reset] = heading[reset]
        self.target_yaw_rate[reset] = 0.0

        self.raw_vx[reset] = 0.0
        self.raw_vy[reset] = 0.0
        self.raw_vz[reset] = 0.0
        self.raw_yaw[reset] = 0.0
        self.stick_vx[reset] = 0.0
        self.stick_vy[reset] = 0.0
        self.stick_vz[reset] = 0.0
        self.stick_yaw[reset] = 0.0
        self.dwell_left[reset] = 0
        self.last_synced_step[reset] = 0
        self.episode_count[reset] += 1
        self.episode_error_sum[reset] = 0.0
        self.episode_yaw_error_sum[reset] = 0.0
        self.episode_attitude_error_sum[reset] = 0.0
        self.episode_metric_count[reset] = 0.0

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
        values = torch.rand(size, 4, device=d) * amp.reshape(-1, 1) * signs

        # 0 hover/release: all sticks centered.
        hover = mode == 0
        values[hover, :] = 0.0

        # 1 small correction: all axes move gently in a small range.
        # 2 step push: one or two axes jump to a high value.
        step = mode == 2
        if torch.any(step):
            axis = torch.randint(0, 4, (int(step.sum().item()),), device=d)
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
            values[reverse, :] = torch.where(
                torch.abs(current) > 0.1,
                -torch.sign(current) * amp[reverse].reshape(-1, 1),
                fallback,
            )

        # 4 combined input: several axes move together.
        combined = mode == 4
        if torch.any(combined):
            values[combined, :] = (
                0.35 + 0.65 * torch.rand(int(combined.sum().item()), 4, device=d)
            ) * amp[combined].reshape(-1, 1) * signs[combined]

        # 5 release: force the raw input back to center from wherever it was.
        release = mode == 5
        values[release, :] = 0.0

        self.operation_mode[mask] = mode
        self.raw_vx[mask] = torch.clamp(values[:, 0], -1.0, 1.0)
        self.raw_vy[mask] = torch.clamp(values[:, 1], -1.0, 1.0)
        self.raw_vz[mask] = torch.clamp(values[:, 2], -1.0, 1.0)
        self.raw_yaw[mask] = torch.clamp(values[:, 3], -1.0, 1.0)
        self.dwell_left[mask] = torch.randint(
            self.dwell_min, self.dwell_max + 1, (size,), device=d, dtype=torch.long
        )

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
            & (mean_yaw_error < self.success_yaw_error)
            & (mean_attitude_error < self.success_attitude_error)
        )
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
            sampled[easy_mask] = self._randint_level(0, easy_max,
                                                     (int(easy_mask.sum().item()),))
        if torch.any(medium_mask):
            sampled[medium_mask] = self._randint_level(medium_min, medium_max,
                                                       (int(medium_mask.sum().item()),))
        if torch.any(random_mask):
            sampled[random_mask] = self._randint_level(0, self.max_curriculum_level,
                                                       (int(random_mask.sum().item()),))

        return sampled

    def update_episode_metrics(self, vel_error, yaw_error, attitude_error):
        self.episode_error_sum += vel_error.detach()
        self.episode_yaw_error_sum += torch.abs(yaw_error.detach())
        self.episode_attitude_error_sum += attitude_error.detach()
        self.episode_metric_count += 1.0

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
            'rc_human/tracking_error_mean': mean_vel_error.mean(),
            'rc_human/tracking_vel_error_mean': mean_vel_error.mean(),
            'rc_human/tracking_yaw_error_mean': mean_yaw_error.mean(),
            'rc_human/tracking_attitude_error_mean': mean_attitude_error.mean(),
        }
        for mode_id in range(int(self.mode_order.numel())):
            metrics[f'rc_human/mode_{mode_id}_fraction'] = (
                self.operation_mode == mode_id).float().mean()
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
        small = mode == 1
        amp[small] = torch.minimum(amp[small], torch.full_like(amp[small], self.medium_stick))
        hover_or_release = (mode == 0) | (mode == 5)
        amp[hover_or_release] = 0.0
        hard = (mode == 2) | (mode == 3) | (mode == 4)
        amp[hard] = torch.clamp(amp[hard], min=self.medium_stick, max=self.hard_stick)
        return amp

    def _px4_process_stick(self, raw, state, tau):
        if self.stick_noise_std > 0.0:
            raw = torch.clamp(raw + torch.randn_like(raw) * self.stick_noise_std, -1.0, 1.0)

        abs_raw = torch.abs(raw)
        denom = max(1.0 - self.deadzone, 1e-6)
        x = torch.where(
            abs_raw <= self.deadzone,
            torch.zeros_like(raw),
            torch.sign(raw) * (abs_raw - self.deadzone) / denom,
        )
        x = (1.0 - self.expo) * x + self.expo * x * x * x

        max_delta = self.stick_slew_rate * self.dt
        x = state + torch.clamp(x - state, -max_delta, max_delta)

        alpha = self.dt / max(tau + self.dt, 1e-6)
        return state + alpha * (x - state)

    def _advance_command(self, env, mask):
        resample = mask & (self.dwell_left <= 0)
        self._resample_raw_sticks(resample)
        self.dwell_left[mask] = torch.clamp(self.dwell_left[mask] - 1, min=0)

        self.stick_vx[mask] = self._px4_process_stick(
            self.raw_vx[mask], self.stick_vx[mask], self.velocity_tau
        )
        self.stick_vy[mask] = self._px4_process_stick(
            self.raw_vy[mask], self.stick_vy[mask], self.velocity_tau
        )
        self.stick_vz[mask] = self._px4_process_stick(
            self.raw_vz[mask], self.stick_vz[mask], self.velocity_tau
        )
        self.stick_yaw[mask] = self._px4_process_stick(
            self.raw_yaw[mask], self.stick_yaw[mask], self.yaw_tau
        )

        self.target_vx[mask] = self.stick_vx[mask] * self.vx_limit
        self.target_vy[mask] = self.stick_vy[mask] * self.vy_limit
        self.target_vz[mask] = self.stick_vz[mask] * self.vz_limit
        self.target_yaw_rate[mask] = self.stick_yaw[mask] * self.yaw_rate_limit

        _npos, _epos, altitude = env.model.get_position()
        _roll, _pitch, heading = env.model.get_posture()
        high = mask & (altitude > self.alt_high) & (self.target_vz > 0.0)
        low = mask & (altitude < self.alt_low) & (self.target_vz < 0.0)
        self.target_vz[high | low] = 0.0
        self.stick_vz[high | low] = 0.0

        self.target_heading[mask] = wrap_PI(
            self.target_heading[mask] + self.target_yaw_rate[mask] * self.dt
        )
        self.target_vn[mask], self.target_ve[mask] = self.local_to_ground_velocity(
            self.target_vx[mask], self.target_vy[mask], self.target_heading[mask]
        )

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
