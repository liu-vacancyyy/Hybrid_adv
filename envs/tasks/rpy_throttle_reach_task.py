import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.rpy_throttle_reach_reward import (
    RPYThrottleReachEventReward,
    RPYThrottleReachReward,
)
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.overload import Overload
from hybrid_termination_conditions.high_speed import HighSpeed
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI


class RPYThrottleReachTask(BaseTask):
    """Reach one commanded roll/pitch/yaw/throttle target from hover or pose pool."""

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'rpy_throttle_reach'
        self.dt = float(getattr(config, 'dt', 0.02))
        self.noise_scale = getattr(config, 'noise_scale', 0.01)
        self.enable_sensor_noise = bool(getattr(config, 'enable_sensor_noise', True))
        self.sensor_pos_std = float(getattr(config, 'sensor_pos_std', 1.0))
        self.sensor_vel_std = float(getattr(config, 'sensor_vel_std', 0.02))
        self.sensor_att_std = float(getattr(config, 'sensor_att_std', 0.005))
        self.sensor_omega_std = float(getattr(config, 'sensor_omega_std', 0.0005))

        self.roll_limit = math.radians(float(
            getattr(config, 'rpy_throttle_reach_roll_limit_deg', 10.0)
        ))
        self.pitch_limit = math.radians(float(
            getattr(config, 'rpy_throttle_reach_pitch_limit_deg', 10.0)
        ))
        self.yaw_limit = math.radians(float(
            getattr(config, 'rpy_throttle_reach_yaw_limit_deg', 45.0)
        ))
        self.throttle_delta_max = float(
            getattr(config, 'rpy_throttle_reach_throttle_delta_max', 0.22)
        )
        self.throttle_min_frac = float(
            getattr(config, 'rpy_throttle_reach_min_collective_frac', 0.80)
        )
        self.throttle_max_frac = float(
            getattr(config, 'rpy_throttle_reach_max_collective_frac', 1.20)
        )
        self.ramp_steps_min = int(getattr(
            config, 'rpy_throttle_reach_ramp_steps_min', 40
        ))
        self.ramp_steps_max = int(getattr(
            config, 'rpy_throttle_reach_ramp_steps_max', 120
        ))
        self.recover_ramp_steps = int(getattr(
            config, 'rpy_throttle_reach_recover_ramp_steps', 80
        ))
        transition_weights = [
            float(getattr(config, 'rpy_throttle_reach_transition_step_weight', 0.20)),
            float(getattr(config, 'rpy_throttle_reach_transition_ramp_weight', 0.30)),
            float(getattr(config, 'rpy_throttle_reach_transition_smooth_weight', 0.35)),
            float(getattr(config, 'rpy_throttle_reach_transition_cosine_weight', 0.15)),
        ]
        self.transition_weights = self._normalize_transition_weights(transition_weights)

        self.levels_per_mode = int(getattr(
            config, 'rpy_throttle_reach_levels_per_mode', 10
        ))
        self.mode_order = torch.arange(10, dtype=torch.long, device=self.device)
        self.active_mode_slots = int(os.environ.get(
            'RPY_THROTTLE_REACH_MAX_MODE_SLOTS',
            getattr(config, 'rpy_throttle_reach_max_mode_slots', 10),
        ))
        self.active_mode_slots = max(1, min(10, self.active_mode_slots))
        self.max_curriculum_level = self.levels_per_mode * self.active_mode_slots - 1
        self.curriculum_enable = bool(getattr(
            config, 'rpy_throttle_reach_curriculum_enable', True
        ))
        self.uniform_mode_when_no_curriculum = bool(getattr(
            config, 'rpy_throttle_reach_uniform_mode_when_no_curriculum', True
        ))
        self.initial_curriculum_level = int(getattr(
            config, 'rpy_throttle_reach_initial_curriculum_level', 10
        ))
        self.mix_current = float(getattr(config, 'rpy_throttle_reach_mix_current', 0.55))
        self.mix_easy = float(getattr(config, 'rpy_throttle_reach_mix_easy_replay', 0.20))
        self.mix_medium = float(getattr(config, 'rpy_throttle_reach_mix_medium_replay', 0.15))
        self.mix_random = float(getattr(config, 'rpy_throttle_reach_mix_random_replay', 0.10))
        self._normalize_curriculum_mix()

        self.success_attitude_error = float(getattr(
            config, 'rpy_throttle_reach_success_attitude_error', math.radians(3.0)
        ))
        self.success_yaw_error = float(getattr(
            config, 'rpy_throttle_reach_success_yaw_error', math.radians(5.0)
        ))
        self.success_throttle_error = float(getattr(
            config, 'rpy_throttle_reach_success_throttle_error', 0.08
        ))
        self.success_overshoot = float(getattr(
            config, 'rpy_throttle_reach_success_overshoot', 0.22
        ))
        self.success_danger = float(getattr(
            config, 'rpy_throttle_reach_success_danger', 0.08
        ))
        self.success_override_fraction = float(getattr(
            config, 'rpy_throttle_reach_success_override_fraction', 0.20
        ))
        self.success_ignore_transient = bool(getattr(
            config, 'rpy_throttle_reach_success_ignore_transient', True
        ))
        self.transient_grace_steps = int(getattr(
            config, 'rpy_throttle_reach_transient_grace_steps', 40
        ))

        self.safety_override_enable = bool(getattr(
            config, 'rpy_throttle_reach_safety_override_enable', True
        ))
        self.safety_speed_enter = float(getattr(
            config, 'rpy_throttle_reach_safety_speed_enter', 8.0
        ))
        self.safety_speed_exit = float(getattr(
            config, 'rpy_throttle_reach_safety_speed_exit', 6.5
        ))
        self.safety_altitude_enter = float(getattr(
            config, 'rpy_throttle_reach_safety_altitude_enter', 1.5
        ))
        self.safety_altitude_exit = float(getattr(
            config, 'rpy_throttle_reach_safety_altitude_exit', 3.0
        ))
        self.safety_angle_enter_deg = float(getattr(
            config, 'rpy_throttle_reach_safety_angle_enter_deg', 18.0
        ))
        self.safety_angle_exit_deg = float(getattr(
            config, 'rpy_throttle_reach_safety_angle_exit_deg', 10.0
        ))
        self.safety_omega_enter = float(getattr(
            config, 'rpy_throttle_reach_safety_omega_enter', 3.0
        ))
        self.safety_omega_exit = float(getattr(
            config, 'rpy_throttle_reach_safety_omega_exit', 1.5
        ))
        self.safety_min_hold_steps = int(getattr(
            config, 'rpy_throttle_reach_safety_min_hold_steps', 50
        ))
        self.safety_recover_steps = int(getattr(
            config, 'rpy_throttle_reach_safety_recover_steps', 40
        ))

        self.pose_pool_capacity = int(getattr(
            config, 'rpy_throttle_reach_pose_pool_capacity', 4096
        ))
        self.pose_pool_min_insert_steps = int(getattr(
            config, 'rpy_throttle_reach_pose_pool_min_insert_steps', 120
        ))
        self.pose_pool_insert_interval = int(getattr(
            config, 'rpy_throttle_reach_pose_pool_insert_interval', 25
        ))
        self.pose_pool_max_insert_per_step = int(getattr(
            config, 'rpy_throttle_reach_pose_pool_max_insert_per_step', 64
        ))
        self.pose_pool_success_only = bool(getattr(
            config, 'rpy_throttle_reach_pose_pool_success_only', False
        ))
        self.pose_pool_roll_pitch_limit = math.radians(float(getattr(
            config, 'rpy_throttle_reach_pose_pool_roll_pitch_limit_deg', 12.0
        )))
        self.pose_pool_omega_limit = float(getattr(
            config, 'rpy_throttle_reach_pose_pool_omega_limit', 1.2
        ))
        self.pose_pool_speed_limit = float(getattr(
            config, 'rpy_throttle_reach_pose_pool_speed_limit', 4.0
        ))
        self.pose_pool_danger_limit = float(getattr(
            config, 'rpy_throttle_reach_pose_pool_danger_limit', 0.05
        ))
        self.pose_pool_s = torch.zeros(
            self.pose_pool_capacity,
            int(getattr(config, 'num_states', 12)),
            device=self.device,
        )
        self.pose_pool_u = torch.zeros(
            self.pose_pool_capacity,
            int(getattr(config, 'num_controls', 5)),
            device=self.device,
        )
        self.pose_pool_score = torch.full(
            (self.pose_pool_capacity,), float('inf'), device=self.device
        )
        self.pose_pool_valid_count = 0
        self.pose_pool_write_ptr = 0
        self.pose_pool_bootstrapped = False
        self.pose_pool_insert_count = 0
        self.pose_pool_ready_insert_count = int(getattr(
            config,
            'rpy_throttle_reach_pose_pool_ready_insert_count',
            max(1, self.pose_pool_capacity // 2),
        ))

        self.target_roll = torch.zeros(self.n, device=self.device)
        self.target_pitch = torch.zeros(self.n, device=self.device)
        self.target_yaw = torch.zeros(self.n, device=self.device)
        self.target_collective = torch.zeros(self.n, device=self.device)
        self.target_throttle_frac = torch.zeros(self.n, device=self.device)
        self.final_roll = torch.zeros(self.n, device=self.device)
        self.final_pitch = torch.zeros(self.n, device=self.device)
        self.final_yaw = torch.zeros(self.n, device=self.device)
        self.final_throttle_frac = torch.zeros(self.n, device=self.device)
        self.start_roll = torch.zeros(self.n, device=self.device)
        self.start_pitch = torch.zeros(self.n, device=self.device)
        self.start_yaw = torch.zeros(self.n, device=self.device)
        self.start_throttle_frac = torch.zeros(self.n, device=self.device)
        self.command_start_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.command_ramp_steps = torch.ones(self.n, dtype=torch.long, device=self.device)
        self.command_profile = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.command_transient_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.last_synced_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.hover_collective = torch.ones(self.n, device=self.device)

        self.prev_roll_error = torch.zeros(self.n, device=self.device)
        self.prev_pitch_error = torch.zeros(self.n, device=self.device)
        self.prev_yaw_error = torch.zeros(self.n, device=self.device)
        self.prev_throttle_error = torch.zeros(self.n, device=self.device)
        self.episode_attitude_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_yaw_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_throttle_error_sum = torch.zeros(self.n, device=self.device)
        self.episode_overshoot_sum = torch.zeros(self.n, device=self.device)
        self.episode_danger_sum = torch.zeros(self.n, device=self.device)
        self.episode_override_count = torch.zeros(self.n, device=self.device)
        self.episode_metric_count = torch.zeros(self.n, device=self.device)
        self.episode_metric_skipped_count = torch.zeros(self.n, device=self.device)
        self.episode_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.last_reward_terms = {}
        self.last_constraint_terms = {}

        self.operation_mode = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.curriculum_level = torch.full(
            (self.n,),
            max(0, min(self.initial_curriculum_level, self.max_curriculum_level)),
            dtype=torch.long,
            device=self.device,
        )
        self.sampled_level = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.safety_override_active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.safety_hold_left = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.safety_recovered_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.safety_yaw_hold = torch.zeros(self.n, device=self.device)
        self.pool_initial_fraction = torch.zeros(self.n, device=self.device)

        self.reward_functions = [
            RPYThrottleReachReward(self.config),
            RPYThrottleReachEventReward(self.config),
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

        self._bootstrap_pose_pool(env)
        self._update_curriculum_from_last_episode(env, reset)

        idx = torch.where(reset)[0]
        sampled_level = self._sample_command_level(idx)
        mode = self._curriculum_mode_from_level(sampled_level)
        self.sampled_level[idx] = sampled_level
        self.operation_mode[idx] = mode

        pool_init = mode >= 5
        hover_idx = idx[~pool_init]
        pool_idx = idx[pool_init]
        if hover_idx.numel() > 0:
            self._apply_hover_initial_state(env, hover_idx)
        if pool_idx.numel() > 0:
            self._apply_pool_initial_state(env, pool_idx, mode[pool_init])

        _f0, f1, f2, f3, f4 = env.model.get_F()
        hover = torch.clamp(f1 + f2 + f3 + f4, min=1e-6)
        self.hover_collective[idx] = hover[idx]

        roll, pitch, yaw = env.model.get_posture()
        collective = f1 + f2 + f3 + f4
        current_throttle_frac = torch.clamp(
            (collective - hover) / torch.clamp(hover, min=1e-6),
            self.throttle_min_frac - 1.0,
            self.throttle_max_frac - 1.0,
        )
        self.start_roll[idx] = roll[idx]
        self.start_pitch[idx] = pitch[idx]
        self.start_yaw[idx] = yaw[idx]
        self.start_throttle_frac[idx] = current_throttle_frac[idx]
        self._sample_final_targets(idx, sampled_level, mode)

        ramp_steps = self._sample_ramp_steps(sampled_level, mode)
        transition_profile = self._sample_transition_profile(idx)
        # BaseEnv resets step_count after task.reset(), so use the known
        # post-reset value instead of the stale terminal step count.
        self.command_start_step[idx] = 0
        self.command_ramp_steps[idx] = ramp_steps
        self.command_profile[idx] = transition_profile
        self.last_synced_step[idx] = 0
        transient_steps = torch.where(
            transition_profile == 0,
            torch.zeros_like(ramp_steps),
            ramp_steps,
        )
        self.command_transient_left[idx] = torch.maximum(
            torch.full_like(self.command_transient_left[idx], self.transient_grace_steps),
            transient_steps,
        )

        self.target_roll[idx] = self.start_roll[idx]
        self.target_pitch[idx] = self.start_pitch[idx]
        self.target_yaw[idx] = self.start_yaw[idx]
        self.target_throttle_frac[idx] = self.start_throttle_frac[idx]
        step_idx = idx[transition_profile == 0]
        if step_idx.numel() > 0:
            self.target_roll[step_idx] = self.final_roll[step_idx]
            self.target_pitch[step_idx] = self.final_pitch[step_idx]
            self.target_yaw[step_idx] = self.final_yaw[step_idx]
            self.target_throttle_frac[step_idx] = self.final_throttle_frac[step_idx]
        self.target_collective[idx] = self.hover_collective[idx] * (
            1.0 + self.target_throttle_frac[idx]
        )

        self.prev_roll_error[idx] = 0.0
        self.prev_pitch_error[idx] = 0.0
        self.prev_yaw_error[idx] = 0.0
        self.prev_throttle_error[idx] = 0.0
        self.episode_attitude_error_sum[idx] = 0.0
        self.episode_yaw_error_sum[idx] = 0.0
        self.episode_throttle_error_sum[idx] = 0.0
        self.episode_overshoot_sum[idx] = 0.0
        self.episode_danger_sum[idx] = 0.0
        self.episode_override_count[idx] = 0.0
        self.episode_metric_count[idx] = 0.0
        self.episode_metric_skipped_count[idx] = 0.0
        self.episode_count[idx] += 1
        self.safety_override_active[idx] = False
        self.safety_hold_left[idx] = 0
        self.safety_recovered_steps[idx] = 0
        self.safety_yaw_hold[idx] = yaw[idx]
        self.pool_initial_fraction[idx] = pool_init.float()

    def step(self, env):
        self.sync_command(env)
        self._maybe_insert_running_pool_samples(env)

    def sync_command(self, env):
        active = ~(env.bad_done.bool() | env.is_done.bool() | env.exceed_time_limit.bool())
        steps = env.step_count.long()
        need_sync = active & (steps > self.last_synced_step)
        if int(torch.sum(need_sync).item()) == 0:
            return
        self._update_safety_override_state(env, need_sync)
        self._update_command_targets(env, need_sync)
        self.command_transient_left[need_sync] = torch.clamp(
            self.command_transient_left[need_sync] - 1, min=0
        )
        self.last_synced_step[need_sync] = steps[need_sync]

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

    def _normalize_transition_weights(self, weights):
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights = torch.clamp(weights, min=0.0)
        total = weights.sum()
        if total <= 1e-6:
            return torch.full((4,), 0.25, dtype=torch.float32, device=self.device)
        return weights / total

    def _sample_transition_profile(self, idx):
        selector = torch.rand(idx.numel(), device=self.device)
        cumulative = torch.cumsum(self.transition_weights, dim=0)
        profile = torch.zeros(idx.numel(), dtype=torch.long, device=self.device)
        profile = torch.where(selector >= cumulative[0], torch.ones_like(profile), profile)
        profile = torch.where(selector >= cumulative[1], torch.full_like(profile, 2), profile)
        profile = torch.where(selector >= cumulative[2], torch.full_like(profile, 3), profile)
        return profile

    def _command_phase(self, raw_phase, profile):
        raw_phase = torch.clamp(raw_phase, 0.0, 1.0)
        linear = raw_phase
        smooth = raw_phase * raw_phase * (3.0 - 2.0 * raw_phase)
        cosine = 0.5 - 0.5 * torch.cos(raw_phase * math.pi)
        phase = torch.where(profile == 0, torch.ones_like(raw_phase), linear)
        phase = torch.where(profile == 2, smooth, phase)
        phase = torch.where(profile == 3, cosine, phase)
        return phase

    def _randint_level_between(self, low, high):
        low = torch.clamp(low.long(), 0, self.max_curriculum_level)
        high = torch.clamp(high.long(), 0, self.max_curriculum_level)
        high = torch.maximum(high, low)
        span = (high - low + 1).float()
        return low + torch.floor(torch.rand_like(span) * span).long()

    def _sample_command_level(self, idx):
        if not self.curriculum_enable:
            if not self.uniform_mode_when_no_curriculum:
                return torch.full((idx.numel(),), self.max_curriculum_level,
                                  device=self.device, dtype=torch.long)
            if idx.numel() == 0:
                return torch.empty(0, device=self.device, dtype=torch.long)
            max_open_slots = self.active_mode_slots
            if not self._pose_pool_ready_for_pool_modes():
                max_open_slots = min(max_open_slots, 5)
            max_open_slots = max(1, max_open_slots)
            mode_slot = torch.randint(
                0, max_open_slots, (idx.numel(),), device=self.device, dtype=torch.long
            )
            sublevel = torch.randint(
                0, self.levels_per_mode, (idx.numel(),), device=self.device, dtype=torch.long
            )
            return mode_slot * self.levels_per_mode + sublevel

        current = torch.clamp(self.curriculum_level[idx], 0, self.max_curriculum_level)
        sampled = current.clone()
        selector = torch.rand(idx.numel(), device=self.device)
        current_end = self.mix_current
        easy_end = current_end + self.mix_easy
        medium_end = easy_end + self.mix_medium

        easy_mask = (selector >= current_end) & (selector < easy_end)
        medium_mask = (selector >= easy_end) & (selector < medium_end)
        random_mask = selector >= medium_end
        easy_max = max(0, int(round(self.max_curriculum_level * 0.30)))
        medium_min = min(self.max_curriculum_level, easy_max + 1)
        medium_max = max(medium_min, int(round(self.max_curriculum_level * 0.65)))

        if torch.any(easy_mask):
            high = torch.minimum(current[easy_mask], torch.full_like(current[easy_mask], easy_max))
            sampled[easy_mask] = self._randint_level_between(torch.zeros_like(high), high)
        if torch.any(medium_mask):
            cur = current[medium_mask]
            can = cur >= medium_min
            low = torch.where(can, torch.full_like(cur, medium_min), torch.zeros_like(cur))
            high = torch.where(can, torch.minimum(cur, torch.full_like(cur, medium_max)), cur)
            sampled[medium_mask] = self._randint_level_between(low, high)
        if torch.any(random_mask):
            high = current[random_mask]
            sampled[random_mask] = self._randint_level_between(torch.zeros_like(high), high)
        return sampled

    def _pose_pool_ready_for_pool_modes(self):
        return (
            self.pose_pool_valid_count > 0
            and self.pose_pool_insert_count >= self.pose_pool_ready_insert_count
        )

    def _curriculum_mode_from_level(self, level):
        mode_slot = torch.clamp(level // self.levels_per_mode, 0, self.active_mode_slots - 1)
        return self.mode_order[mode_slot]

    def _level_progress(self, level):
        sublevel = (level % self.levels_per_mode).float()
        return sublevel / max(float(self.levels_per_mode - 1), 1.0)

    def _target_amplitude(self, level, mode):
        progress = self._level_progress(level)
        difficulty_mode = torch.where(mode >= 5, mode - 5, mode)
        amp = torch.zeros_like(progress)
        amp = torch.where(difficulty_mode == 0, torch.zeros_like(amp), amp)
        amp = torch.where(difficulty_mode == 1, 0.12 + 0.18 * progress, amp)
        amp = torch.where(difficulty_mode == 2, 0.25 + 0.25 * progress, amp)
        amp = torch.where(difficulty_mode == 3, 0.45 + 0.25 * progress, amp)
        amp = torch.where(difficulty_mode == 4, 0.65 + 0.35 * progress, amp)
        return torch.clamp(amp, 0.0, 1.0)

    def _sample_final_targets(self, idx, level, mode):
        size = idx.numel()
        amp = self._target_amplitude(level, mode)
        signs = torch.sign(torch.rand(size, 4, device=self.device) - 0.5)
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        values = torch.rand(size, 4, device=self.device) * amp.reshape(-1, 1) * signs

        difficulty_mode = torch.where(mode >= 5, mode - 5, mode)

        values[difficulty_mode == 0, :] = 0.0
        single = difficulty_mode == 1
        if torch.any(single):
            axis = torch.randint(0, 4, (int(single.sum().item()),), device=self.device)
            local = torch.zeros(int(single.sum().item()), 4, device=self.device)
            local.scatter_(1, axis.reshape(-1, 1), amp[single].reshape(-1, 1) * signs[single, :1])
            values[single] = local
        attitude = difficulty_mode == 2
        if torch.any(attitude):
            values[attitude, 3] *= 0.25
        combined = difficulty_mode == 3
        if torch.any(combined):
            values[combined, :] *= torch.tensor([1.0, 1.0, 0.8, 0.7], device=self.device)
        hard = difficulty_mode == 4
        if torch.any(hard):
            hard_count = int(hard.sum().item())
            hard_values = (
                (0.35 + 0.65 * torch.rand(hard_count, 4, device=self.device))
                * amp[hard].reshape(-1, 1)
                * signs[hard]
            )
            axis_count = torch.where(
                torch.rand(hard_count, device=self.device) < 0.30,
                torch.full((hard_count,), 4, dtype=torch.long, device=self.device),
                torch.randint(2, 4, (hard_count,), device=self.device),
            )
            axis_score = torch.rand(hard_count, 4, device=self.device)
            axis_rank = torch.argsort(axis_score, dim=1)
            axis_ids = torch.arange(4, device=self.device).reshape(1, 4)
            active_axis = axis_ids >= (4 - axis_count).reshape(-1, 1)
            active_mask = torch.zeros_like(hard_values, dtype=torch.bool)
            active_mask.scatter_(1, axis_rank, active_axis)
            values[hard] = torch.where(active_mask, hard_values, torch.zeros_like(hard_values))

        self.final_roll[idx] = torch.clamp(values[:, 0], -1.0, 1.0) * self.roll_limit
        self.final_pitch[idx] = torch.clamp(values[:, 1], -1.0, 1.0) * self.pitch_limit
        yaw_offset = torch.clamp(values[:, 2], -1.0, 1.0) * self.yaw_limit
        self.final_yaw[idx] = wrap_PI(self.start_yaw[idx] + yaw_offset)
        throttle_frac = torch.clamp(
            values[:, 3] * self.throttle_delta_max,
            self.throttle_min_frac - 1.0,
            self.throttle_max_frac - 1.0,
        )
        self.final_throttle_frac[idx] = throttle_frac

    def _sample_ramp_steps(self, level, mode):
        progress = self._level_progress(level)
        base = self.ramp_steps_max - (self.ramp_steps_max - self.ramp_steps_min) * progress
        base = torch.where(mode >= 5, torch.full_like(base, float(self.ramp_steps_min)), base)
        jitter = torch.randint(-8, 9, (level.numel(),), device=self.device)
        return torch.clamp(base.long() + jitter, min=10)

    def _apply_hover_initial_state(self, env, idx):
        env.model.s[idx, 3] = 0.0
        env.model.s[idx, 4] = 0.0
        env.model.s[idx, 5] = 0.0
        env.model.s[idx, 6:12] = 0.0
        env.model.recent_s[idx] = env.model.s[idx]

    def _apply_pool_initial_state(self, env, idx, mode):
        if self.pose_pool_valid_count <= 0:
            self._apply_hover_initial_state(env, idx)
            return
        count = idx.numel()
        pool_idx = self._sample_pose_pool_indices(mode, count)
        env.model.s[idx] = self.pose_pool_s[pool_idx]
        env.model.u[idx] = self.pose_pool_u[pool_idx]
        env.model.u[idx, 0] = 0.0
        env.model.s[idx, 2] = torch.clamp(
            env.model.s[idx, 2],
            min=max(self.safety_altitude_exit, float(getattr(self.config, 'altitude_limit', 0.5)) + 0.5),
            max=float(getattr(self.config, 'max_altitude', 50.0)),
        )
        env.model.recent_s[idx] = env.model.s[idx]
        env.model.recent_u[idx] = env.model.u[idx]
        env.model.filtered_action[idx] = torch.clamp(2.0 * env.model.u[idx] / env.model.max_F - 1.0, -1.0, 1.0)
        env.model.filtered_action[idx, 0] = -1.0
        env.model.filtered_action_valid[idx] = False

    def _sample_pose_pool_indices(self, mode, count):
        valid_count = int(self.pose_pool_valid_count)
        if valid_count <= 0:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        if valid_count < 10:
            return torch.randint(0, valid_count, (count,), device=self.device)

        valid_score = self.pose_pool_score[:valid_count]
        order = torch.argsort(valid_score, descending=False)
        # Keep the first pool modes gentle: mode5 and mode6 both start from
        # the easiest pose-pool band.  Higher pool modes then move gradually
        # toward harder stored states, while avoiding the worst tail.
        pool_mode = torch.clamp(mode - 6, 0, 3)
        band = torch.clamp(
            ((pool_mode.float() + torch.rand(count, device=self.device)) / 5.0)
            * valid_count,
            0,
            valid_count - 1,
        ).long()
        local_jitter = max(valid_count // 20, 1)
        band = torch.clamp(
            band + torch.randint(-local_jitter, local_jitter + 1, (count,), device=self.device),
            0,
            valid_count - 1,
        )
        return order[band]

    def _bootstrap_pose_pool(self, env):
        if self.pose_pool_bootstrapped:
            return
        seed_count = min(self.pose_pool_capacity, max(self.n, 32))
        base_idx = torch.arange(seed_count, device=self.device) % env.n
        self.pose_pool_s[:seed_count] = env.model.s[base_idx]
        self.pose_pool_u[:seed_count] = env.model.u[base_idx]
        self.pose_pool_u[:seed_count, 0] = 0.0
        roll_noise = (torch.rand(seed_count, device=self.device) * 2.0 - 1.0) * math.radians(2.0)
        pitch_noise = (torch.rand(seed_count, device=self.device) * 2.0 - 1.0) * math.radians(2.0)
        self.pose_pool_s[:seed_count, 3] = roll_noise
        self.pose_pool_s[:seed_count, 4] = pitch_noise
        self.pose_pool_s[:seed_count, 5] = 0.0
        self.pose_pool_s[:seed_count, 6:12] = 0.0
        self.pose_pool_score[:seed_count] = 0.0
        self.pose_pool_valid_count = seed_count
        self.pose_pool_write_ptr = seed_count % self.pose_pool_capacity
        self.pose_pool_bootstrapped = True

    def _insert_pose_samples(self, env, idx, score):
        if idx.numel() == 0:
            return
        max_count = min(idx.numel(), self.pose_pool_max_insert_per_step)
        if idx.numel() > max_count:
            perm = torch.randperm(idx.numel(), device=self.device)[:max_count]
            idx = idx[perm]
            score = score[perm]

        for local_i in range(idx.numel()):
            env_i = int(idx[local_i].item())
            new_score = float(score[local_i].item())
            if self.pose_pool_valid_count < self.pose_pool_capacity:
                slot = self.pose_pool_valid_count
                self.pose_pool_valid_count += 1
            else:
                slot = int(torch.randint(0, self.pose_pool_capacity, (1,), device=self.device).item())
                old_score = float(self.pose_pool_score[slot].item())
                if new_score > old_score and torch.rand((), device=self.device).item() > 0.20:
                    continue
            self.pose_pool_s[slot] = env.model.s[env_i].detach()
            self.pose_pool_u[slot] = env.model.u[env_i].detach()
            self.pose_pool_u[slot, 0] = 0.0
            self.pose_pool_score[slot] = new_score
            self.pose_pool_insert_count += 1

    def _maybe_insert_running_pool_samples(self, env):
        if self.pose_pool_success_only:
            return
        active = ~(env.bad_done.bool() | env.is_done.bool() | env.exceed_time_limit.bool())
        interval_ok = (env.step_count % max(self.pose_pool_insert_interval, 1)) == 0
        mature = env.step_count >= self.pose_pool_min_insert_steps
        danger, _safety = self.compute_safety_scores(env)
        roll, pitch, _yaw = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        speed = env.model.get_TAS()
        count = torch.clamp(self.episode_metric_count, min=1.0)
        score = (
            self.episode_attitude_error_sum / count
            + self.episode_yaw_error_sum / count
            + self.episode_throttle_error_sum / count
            + self.episode_overshoot_sum / count
            + danger
        )
        stable = (
            active
            & interval_ok
            & mature
            & (~self.safety_override_active)
            & (danger <= self.pose_pool_danger_limit)
            & (torch.abs(roll) <= self.pose_pool_roll_pitch_limit)
            & (torch.abs(pitch) <= self.pose_pool_roll_pitch_limit)
            & (torch.sqrt(p * p + q * q + r * r) <= self.pose_pool_omega_limit)
            & (speed <= self.pose_pool_speed_limit)
        )
        self._insert_pose_samples(env, torch.where(stable)[0], score[stable])

    def _update_command_targets(self, env, active):
        if torch.any(self.safety_override_active & active):
            override = self.safety_override_active & active
            self.target_roll[override] = 0.0
            self.target_pitch[override] = 0.0
            self.target_yaw[override] = self.safety_yaw_hold[override]
            self.target_throttle_frac[override] = 0.0

        normal = active & (~self.safety_override_active)
        if torch.any(normal):
            elapsed = (env.step_count[normal].long() - self.command_start_step[normal]).float()
            denom = torch.clamp(self.command_ramp_steps[normal].float(), min=1.0)
            raw_phase = torch.clamp(elapsed / denom, 0.0, 1.0)
            phase = self._command_phase(raw_phase, self.command_profile[normal])
            self.target_roll[normal] = self.start_roll[normal] + phase * (
                self.final_roll[normal] - self.start_roll[normal]
            )
            self.target_pitch[normal] = self.start_pitch[normal] + phase * (
                self.final_pitch[normal] - self.start_pitch[normal]
            )
            yaw_delta = wrap_PI(self.final_yaw[normal] - self.start_yaw[normal])
            self.target_yaw[normal] = wrap_PI(self.start_yaw[normal] + phase * yaw_delta)
            self.target_throttle_frac[normal] = self.start_throttle_frac[normal] + phase * (
                self.final_throttle_frac[normal] - self.start_throttle_frac[normal]
            )

        self.target_throttle_frac[active] = torch.clamp(
            self.target_throttle_frac[active],
            self.throttle_min_frac - 1.0,
            self.throttle_max_frac - 1.0,
        )
        self.target_collective[active] = self.hover_collective[active] * (
            1.0 + self.target_throttle_frac[active]
        )

    def _safety_envelope_masks(self, env):
        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, yaw = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        speed = env.model.get_TAS()
        angle_deg = torch.maximum(torch.abs(torch.rad2deg(roll)), torch.abs(torch.rad2deg(pitch)))
        omega_norm = torch.sqrt(p * p + q * q + r * r)
        enter = (
            (speed >= self.safety_speed_enter)
            | (altitude <= self.safety_altitude_enter)
            | (angle_deg >= self.safety_angle_enter_deg)
            | (omega_norm >= self.safety_omega_enter)
        )
        exit_ready = (
            (speed <= self.safety_speed_exit)
            & (altitude >= self.safety_altitude_exit)
            & (angle_deg <= self.safety_angle_exit_deg)
            & (omega_norm <= self.safety_omega_exit)
        )
        return enter, exit_ready, yaw

    def _update_safety_override_state(self, env, mask):
        if (not self.safety_override_enable) or int(torch.sum(mask).item()) == 0:
            return
        enter, exit_ready, yaw = self._safety_envelope_masks(env)
        enter = enter & mask
        new_enter = enter & (~self.safety_override_active)
        if torch.any(new_enter):
            self.safety_override_active[new_enter] = True
            self.safety_yaw_hold[new_enter] = yaw[new_enter]
            self.safety_hold_left[new_enter] = max(self.safety_min_hold_steps, 1)
            self.safety_recovered_steps[new_enter] = 0
            self.command_transient_left[new_enter] = torch.maximum(
                self.command_transient_left[new_enter],
                torch.full_like(self.command_transient_left[new_enter], self.transient_grace_steps),
            )

        active = mask & self.safety_override_active
        if not torch.any(active):
            return
        self.safety_hold_left[active] = torch.clamp(self.safety_hold_left[active] - 1, min=0)
        safe_active = active & exit_ready
        unsafe_active = active & (~exit_ready)
        self.safety_recovered_steps[unsafe_active] = 0
        self.safety_recovered_steps[safe_active] += 1
        can_exit = (
            active
            & (self.safety_hold_left <= 0)
            & (self.safety_recovered_steps >= max(self.safety_recover_steps, 1))
        )
        if torch.any(can_exit):
            roll, pitch, yaw = env.model.get_posture()
            _f0, f1, f2, f3, f4 = env.model.get_F()
            collective = f1 + f2 + f3 + f4
            hover = torch.clamp(self.hover_collective, min=1e-6)
            throttle_frac = torch.clamp(
                (collective - hover) / hover,
                self.throttle_min_frac - 1.0,
                self.throttle_max_frac - 1.0,
            )
            self.safety_override_active[can_exit] = False
            self.safety_hold_left[can_exit] = 0
            self.safety_recovered_steps[can_exit] = 0
            self.start_roll[can_exit] = roll[can_exit]
            self.start_pitch[can_exit] = pitch[can_exit]
            self.start_yaw[can_exit] = yaw[can_exit]
            self.start_throttle_frac[can_exit] = throttle_frac[can_exit]
            self.command_start_step[can_exit] = env.step_count[can_exit].long()
            self.command_ramp_steps[can_exit] = max(self.recover_ramp_steps, 1)
            self.command_profile[can_exit] = 2
            self.command_transient_left[can_exit] = torch.maximum(
                self.command_transient_left[can_exit],
                torch.full_like(self.command_transient_left[can_exit], self.transient_grace_steps),
            )

    def compute_safety_scores(self, env):
        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, _yaw = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        speed = env.model.get_TAS()
        max_velocity = float(getattr(self.config, 'max_velocity', 10.0))
        altitude_limit = float(getattr(self.config, 'altitude_limit', 0.5))
        max_angle = math.radians(min(float(getattr(self.config, 'max_roll', 30.0)), float(getattr(self.config, 'max_pitch', 25.0))))
        max_omega = float(getattr(self.config, 'max_omega_norm', 12.566370614359172))

        angle = torch.maximum(torch.abs(roll), torch.abs(pitch))
        omega_norm = torch.sqrt(p * p + q * q + r * r)
        terms = [
            torch.relu((speed - self.safety_speed_exit) / max(max_velocity - self.safety_speed_exit, 1e-6)),
            torch.relu((self.safety_altitude_exit - altitude) / max(self.safety_altitude_exit - altitude_limit, 1e-6)),
            torch.relu((angle - math.radians(self.safety_angle_exit_deg)) / max(max_angle - math.radians(self.safety_angle_exit_deg), 1e-6)),
            torch.relu((omega_norm - self.safety_omega_exit) / max(max_omega - self.safety_omega_exit, 1e-6)),
        ]
        danger = torch.stack(terms, dim=1).sum(dim=1)
        safety = torch.exp(-danger * danger)
        return danger, safety

    def update_constraint_terms(self, env):
        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, _yaw = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        speed = env.model.get_TAS()
        ax, ay, az = env.model.get_acceleration()

        altitude_limit = float(getattr(self.config, 'altitude_limit', 0.5))
        max_velocity = float(getattr(self.config, 'max_velocity', 10.0))
        max_roll = float(getattr(self.config, 'max_roll', 30.0))
        max_pitch = float(getattr(self.config, 'max_pitch', 25.0))
        max_omega_norm = float(getattr(
            self.config,
            'max_omega_norm',
            getattr(self.config, 'max_omega', 4.0),
        ))
        acceleration_limit = float(getattr(self.config, 'acceleration_limit', 12.0))

        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
        omega_norm = torch.sqrt(p * p + q * q + r * r)
        acceleration = torch.sqrt(ax * ax + ay * ay + az * az)

        low_altitude = altitude < altitude_limit
        high_speed = speed >= max_velocity
        extreme_angle = (torch.abs(roll_deg) > max_roll) | (torch.abs(pitch_deg) > max_pitch)
        extreme_omega = omega_norm > max_omega_norm
        overload = acceleration > acceleration_limit
        hard_failure = (
            low_altitude
            | high_speed
            | extreme_angle
            | extreme_omega
            | overload
        )

        self.last_constraint_terms = {
            'constraint/low_altitude_mean': low_altitude.detach().float().mean(),
            'constraint/high_speed_mean': high_speed.detach().float().mean(),
            'constraint/extreme_angle_mean': extreme_angle.detach().float().mean(),
            'constraint/extreme_omega_mean': extreme_omega.detach().float().mean(),
            'constraint/overload_mean': overload.detach().float().mean(),
            'constraint/hard_failure_mean': hard_failure.detach().float().mean(),
            'constraint/bad_done_cost_mean': env.bad_done.detach().float().mean(),
        }
        return hard_failure.float()

    def compute_overshoot_score(self, roll_error, pitch_error, yaw_error, throttle_error):
        valid = self.command_transient_left <= 0
        if self.safety_override_enable:
            valid = valid & (~self.safety_override_active)
        crossed_roll = valid & (self.prev_roll_error * roll_error < 0.0)
        crossed_pitch = valid & (self.prev_pitch_error * pitch_error < 0.0)
        crossed_yaw = valid & (self.prev_yaw_error * yaw_error < 0.0)
        crossed_throttle = valid & (self.prev_throttle_error * throttle_error < 0.0)
        roll_score = torch.where(crossed_roll, torch.abs(roll_error) / max(self.roll_limit, 1e-6), torch.zeros_like(roll_error))
        pitch_score = torch.where(crossed_pitch, torch.abs(pitch_error) / max(self.pitch_limit, 1e-6), torch.zeros_like(pitch_error))
        yaw_score = torch.where(crossed_yaw, torch.abs(yaw_error) / max(self.yaw_limit, 1e-6), torch.zeros_like(yaw_error))
        throttle_score = torch.where(crossed_throttle, torch.abs(throttle_error) / max(self.throttle_delta_max, 1e-6), torch.zeros_like(throttle_error))
        self.prev_roll_error = roll_error.detach()
        self.prev_pitch_error = pitch_error.detach()
        self.prev_yaw_error = yaw_error.detach()
        self.prev_throttle_error = throttle_error.detach()
        return torch.sqrt(roll_score * roll_score + pitch_score * pitch_score + yaw_score * yaw_score + throttle_score * throttle_score)

    def update_episode_metrics(self, attitude_error, yaw_error, throttle_error, overshoot_score, danger_score):
        valid = torch.ones_like(attitude_error, dtype=torch.bool, device=self.device)
        if self.success_ignore_transient:
            valid = valid & (self.command_transient_left <= 0)
        if self.safety_override_enable:
            valid = valid & (~self.safety_override_active)
        valid_f = valid.detach().float()
        self.episode_attitude_error_sum += attitude_error.detach() * valid_f
        self.episode_yaw_error_sum += yaw_error.detach() * valid_f
        self.episode_throttle_error_sum += throttle_error.detach() * valid_f
        self.episode_overshoot_sum += overshoot_score.detach() * valid_f
        self.episode_danger_sum += danger_score.detach() * valid_f
        self.episode_metric_count += valid_f
        self.episode_metric_skipped_count += (~valid).detach().float()
        self.episode_override_count += self.safety_override_active.detach().float()

    def episode_success_mask(self, env):
        count = torch.clamp(self.episode_metric_count, min=1.0)
        mean_att = self.episode_attitude_error_sum / count
        mean_yaw = self.episode_yaw_error_sum / count
        mean_thr = self.episode_throttle_error_sum / count
        mean_over = self.episode_overshoot_sum / count
        return (
            (mean_att < self.success_attitude_error)
            & (mean_yaw < self.success_yaw_error)
            & (mean_thr < self.success_throttle_error)
            & (mean_over < self.success_overshoot)
        )

    def _update_curriculum_from_last_episode(self, env, reset):
        if not self.curriculum_enable:
            return
        finished = reset & (self.episode_metric_count > 0)
        if int(torch.sum(finished).item()) == 0:
            return
        clean_done = env.is_done[finished].bool() & (~env.bad_done[finished].bool())
        success = clean_done & self.episode_success_mask(env)[finished]
        idx = torch.where(finished)[0]
        level = self.curriculum_level[idx]
        level = torch.where(success, level + 1, level - 1)
        self.curriculum_level[idx] = torch.clamp(level, 0, self.max_curriculum_level)
        if self.pose_pool_success_only and torch.any(success):
            count = torch.clamp(self.episode_metric_count[idx[success]], min=1.0)
            score = (
                self.episode_attitude_error_sum[idx[success]] / count
                + self.episode_yaw_error_sum[idx[success]] / count
                + self.episode_throttle_error_sum[idx[success]] / count
                + self.episode_overshoot_sum[idx[success]] / count
                + self.episode_danger_sum[idx[success]] / count
            )
            self._insert_pose_samples(env, idx[success], score)

    def _apply_sensor_noise(self, roll, pitch, yaw, u, v, w, p, q, r):
        roll = roll + torch.randn_like(roll) * self.sensor_att_std
        pitch = pitch + torch.randn_like(pitch) * self.sensor_att_std
        yaw = wrap_PI(yaw + torch.randn_like(yaw) * self.sensor_att_std)
        u = u + torch.randn_like(u) * self.sensor_vel_std
        v = v + torch.randn_like(v) * self.sensor_vel_std
        w = w + torch.randn_like(w) * self.sensor_vel_std
        p = p + torch.randn_like(p) * self.sensor_omega_std
        q = q + torch.randn_like(q) * self.sensor_omega_std
        r = r + torch.randn_like(r) * self.sensor_omega_std
        return roll, pitch, yaw, u, v, w, p, q, r

    def _build_obs(self, env, add_sensor_noise):
        self.sync_command(env)
        _npos, _epos, altitude = env.model.get_position()
        roll, pitch, yaw = env.model.get_posture()
        p, q, r = env.model.get_angular_velocity()
        u, v, w = env.model.get_velocity()
        vt = env.model.get_TAS()
        alpha_sin, alpha_cos, beta_sin, beta_cos = env.model.get_aero_sincos()
        f0, f1, f2, f3, f4 = env.model.get_F()
        collective = f1 + f2 + f3 + f4
        hover = torch.clamp(self.hover_collective, min=1e-6)
        throttle_frac = (collective - hover) / hover
        throttle_error = (collective - self.target_collective) / hover
        if add_sensor_noise:
            altitude = altitude + torch.randn_like(altitude) * self.sensor_pos_std
            roll, pitch, yaw, u, v, w, p, q, r = self._apply_sensor_noise(
                roll, pitch, yaw, u, v, w, p, q, r
            )
            vt = torch.clamp(vt + torch.randn_like(vt) * self.sensor_vel_std, min=0.0)

        elapsed = (env.step_count.long() - self.command_start_step).float()
        raw_ramp_phase = torch.clamp(
            elapsed / torch.clamp(self.command_ramp_steps.float(), min=1.0),
            0.0,
            1.0,
        )
        ramp_phase = self._command_phase(raw_ramp_phase, self.command_profile)
        level_progress = self._level_progress(self.sampled_level)
        pool_fill = torch.full(
            (self.n, 1),
            min(float(self.pose_pool_valid_count), float(self.pose_pool_capacity))
            / max(float(self.pose_pool_capacity), 1.0),
            device=self.device,
        )
        obs = torch.hstack((
            (wrap_PI(roll - self.target_roll) / max(self.roll_limit, 1e-6)).reshape(-1, 1),
            (wrap_PI(pitch - self.target_pitch) / max(self.pitch_limit, 1e-6)).reshape(-1, 1),
            (wrap_PI(yaw - self.target_yaw) / max(self.yaw_limit, 1e-6)).reshape(-1, 1),
            (throttle_error / max(self.throttle_delta_max, 1e-6)).reshape(-1, 1),
            (self.target_roll / max(self.roll_limit, 1e-6)).reshape(-1, 1),
            (self.target_pitch / max(self.pitch_limit, 1e-6)).reshape(-1, 1),
            (wrap_PI(self.target_yaw - yaw) / max(self.yaw_limit, 1e-6)).reshape(-1, 1),
            (self.target_throttle_frac / max(self.throttle_delta_max, 1e-6)).reshape(-1, 1),
            (self.final_roll / max(self.roll_limit, 1e-6)).reshape(-1, 1),
            (self.final_pitch / max(self.pitch_limit, 1e-6)).reshape(-1, 1),
            (wrap_PI(self.final_yaw - yaw) / max(self.yaw_limit, 1e-6)).reshape(-1, 1),
            (self.final_throttle_frac / max(self.throttle_delta_max, 1e-6)).reshape(-1, 1),
            ramp_phase.reshape(-1, 1),
            self.safety_override_active.float().reshape(-1, 1),
            altitude.reshape(-1, 1) / 100.0,
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
            torch.sin(yaw).reshape(-1, 1),
            torch.cos(yaw).reshape(-1, 1),
            u.reshape(-1, 1) / 5.0,
            v.reshape(-1, 1) / 5.0,
            w.reshape(-1, 1) / 5.0,
            vt.reshape(-1, 1) / 10.0,
            alpha_sin.reshape(-1, 1),
            alpha_cos.reshape(-1, 1),
            beta_sin.reshape(-1, 1),
            beta_cos.reshape(-1, 1),
            p.reshape(-1, 1),
            q.reshape(-1, 1),
            r.reshape(-1, 1),
            throttle_frac.reshape(-1, 1) / max(self.throttle_delta_max, 1e-6),
            f0.reshape(-1, 1) / 7.0,
            f1.reshape(-1, 1) / 7.0,
            f2.reshape(-1, 1) / 7.0,
            f3.reshape(-1, 1) / 7.0,
            f4.reshape(-1, 1) / 7.0,
            (self.operation_mode.float() / 9.0).reshape(-1, 1),
            level_progress.reshape(-1, 1),
            self.pool_initial_fraction.reshape(-1, 1),
            pool_fill,
        ))
        return obs

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)

    def get_training_metrics(self):
        count = torch.clamp(self.episode_metric_count, min=1.0)
        mean_att = self.episode_attitude_error_sum / count
        mean_yaw = self.episode_yaw_error_sum / count
        mean_thr = self.episode_throttle_error_sum / count
        mean_over = self.episode_overshoot_sum / count
        mean_danger = self.episode_danger_sum / count
        metric_total = self.episode_metric_count + self.episode_metric_skipped_count
        metrics = {
            'rpy_throttle_reach/curriculum_level_mean': self.curriculum_level.float().mean(),
            'rpy_throttle_reach/curriculum_level_max': self.curriculum_level.float().max(),
            'rpy_throttle_reach/curriculum_level_limit': torch.tensor(float(self.max_curriculum_level), device=self.device),
            'rpy_throttle_reach/uniform_mode_sampling': torch.tensor(float(not self.curriculum_enable), device=self.device),
            'rpy_throttle_reach/pool_mode_ready': torch.tensor(float(self._pose_pool_ready_for_pool_modes()), device=self.device),
            'rpy_throttle_reach/pool_ready_insert_count': torch.tensor(float(self.pose_pool_ready_insert_count), device=self.device),
            'rpy_throttle_reach/sampled_level_mean': self.sampled_level.float().mean(),
            'rpy_throttle_reach/attitude_error_mean': mean_att.mean(),
            'rpy_throttle_reach/yaw_error_mean': mean_yaw.mean(),
            'rpy_throttle_reach/throttle_error_mean': mean_thr.mean(),
            'rpy_throttle_reach/overshoot_mean': mean_over.mean(),
            'rpy_throttle_reach/danger_mean': mean_danger.mean(),
            'rpy_throttle_reach/safety_override_fraction': self.safety_override_active.float().mean(),
            'rpy_throttle_reach/episode_override_fraction': (self.episode_override_count / torch.clamp(torch.ones_like(self.episode_override_count) * float(getattr(self.config, 'max_steps', 500)), min=1.0)).mean(),
            'rpy_throttle_reach/target_roll_abs_deg_mean': torch.rad2deg(torch.abs(self.target_roll)).mean(),
            'rpy_throttle_reach/target_pitch_abs_deg_mean': torch.rad2deg(torch.abs(self.target_pitch)).mean(),
            'rpy_throttle_reach/target_yaw_abs_deg_mean': torch.rad2deg(torch.abs(wrap_PI(self.target_yaw - self.start_yaw))).mean(),
            'rpy_throttle_reach/target_throttle_frac_abs_mean': torch.abs(self.target_throttle_frac).mean(),
            'rpy_throttle_reach/pool_valid_count': torch.tensor(float(self.pose_pool_valid_count), device=self.device),
            'rpy_throttle_reach/pool_valid_fraction': torch.tensor(float(self.pose_pool_valid_count) / max(float(self.pose_pool_capacity), 1.0), device=self.device),
            'rpy_throttle_reach/pool_insert_count': torch.tensor(float(self.pose_pool_insert_count), device=self.device),
            'rpy_throttle_reach/pool_initial_fraction': self.pool_initial_fraction.mean(),
            'rpy_throttle_reach/success_metric_valid_fraction': (self.episode_metric_count / torch.clamp(metric_total, min=1.0)).mean(),
            'rpy_throttle_reach/success_metric_skipped_fraction': (self.episode_metric_skipped_count / torch.clamp(metric_total, min=1.0)).mean(),
        }
        for mode_id in range(10):
            metrics[f'rpy_throttle_reach/mode_{mode_id}_fraction'] = (
                self.operation_mode == mode_id).float().mean()
        metrics['rpy_throttle_reach/transition_step_fraction'] = (
            self.command_profile == 0).float().mean()
        metrics['rpy_throttle_reach/transition_ramp_fraction'] = (
            self.command_profile == 1).float().mean()
        metrics['rpy_throttle_reach/transition_smooth_fraction'] = (
            self.command_profile == 2).float().mean()
        metrics['rpy_throttle_reach/transition_cosine_fraction'] = (
            self.command_profile == 3).float().mean()
        for key, value in self.last_reward_terms.items():
            metrics[f'rpy_throttle_reach/{key}'] = value
        for key, value in self.last_constraint_terms.items():
            metrics[f'rpy_throttle_reach/{key}'] = value
        return metrics
