import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.extreme_state import ExtremeState
from hybrid_termination_conditions.ground_collision import GroundCollision
from hybrid_termination_conditions.high_speed import HighSpeed
from hybrid_termination_conditions.overload import Overload
from reward_functions.vtol_mission_reward import (
    VTOLMissionEventReward,
    VTOLMissionReward,
)
from termination_conditions.vtol_mission_termination import (
    VTOLMissionBoundary,
    VTOLMissionSuccess,
    VTOLMissionTimeout,
)
from utils.utils import wrap_PI


class VTOLMissionTask(BaseTask):
    """GPU-vectorized standard_vtol takeoff, cruise, and landing mission."""

    TAKEOFF = 0
    ROTOR_CLIMB = 1
    TRANSITION = 2
    FIXED_WING = 3
    BACK_TRANSITION = 4
    VERTICAL_LANDING = 5
    PHASE_COUNT = 6
    OBSERVATION_SIZE = 45
    PHASE_NAMES = (
        'takeoff',
        'rotor_climb',
        'transition',
        'fixed_wing',
        'back_transition',
        'vertical_landing',
    )

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'vtol_mission'
        self.dt = float(getattr(config, 'dt', 0.02))
        self.max_steps = int(getattr(config, 'max_steps', 3000))

        self.start_n = float(getattr(config, 'mission_start_n', 0.0))
        self.start_e = float(getattr(config, 'mission_start_e', 0.0))
        self.landing_n = float(getattr(config, 'mission_landing_n', 200.0))
        self.landing_e = float(getattr(config, 'mission_landing_e', 0.0))
        self.landing_altitude = float(
            getattr(config, 'mission_landing_altitude', 0.095)
        )
        route_n = self.landing_n - self.start_n
        route_e = self.landing_e - self.start_e
        self.route_length = math.hypot(route_n, route_e)
        if self.route_length < 20.0:
            raise ValueError('VTOL mission route must be at least 20 m long')
        self.route_unit_n = route_n / self.route_length
        self.route_unit_e = route_e / self.route_length
        self.route_heading = math.atan2(route_e, route_n)

        self.cruise_altitude = float(
            getattr(config, 'mission_cruise_altitude', 25.0)
        )
        self.landing_hover_altitude = float(
            getattr(config, 'mission_landing_hover_altitude', 8.0)
        )
        self.takeoff_clearance = float(
            getattr(config, 'mission_takeoff_clearance', 1.0)
        )
        self.approach_distance = float(
            getattr(config, 'mission_approach_distance', 60.0)
        )
        self.descent_capture_radius = float(
            getattr(config, 'mission_descent_capture_radius', 15.0)
        )
        if not 5.0 < self.approach_distance < self.route_length:
            raise ValueError('mission_approach_distance must be inside the route')

        self.transition_speed = float(
            getattr(config, 'mission_transition_complete_speed', 11.0)
        )
        self.backtransition_speed = float(
            getattr(config, 'mission_backtransition_complete_speed', 6.0)
        )
        self.cruise_speed = float(
            getattr(config, 'mission_cruise_speed', 14.0)
        )
        self.approach_speed = float(
            getattr(config, 'mission_approach_speed', 5.0)
        )
        self.rotor_climb_speed = float(
            getattr(config, 'mission_rotor_climb_speed', 2.0)
        )
        self.final_heading = float(
            getattr(config, 'mission_landing_heading', self.route_heading)
        )

        dwell = getattr(config, 'mission_phase_min_steps', [10, 25, 40, 40, 25, 0])
        if len(dwell) != self.PHASE_COUNT:
            raise ValueError('mission_phase_min_steps must contain six values')
        self.phase_min_steps = torch.tensor(
            [int(value) for value in dwell], dtype=torch.long, device=self.device
        )

        self.curriculum_enabled = bool(
            getattr(config, 'mission_curriculum_enable', True)
        )
        weights = getattr(
            config,
            'mission_curriculum_phase_weights',
            [0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
        )
        if len(weights) != self.PHASE_COUNT or any(float(w) < 0.0 for w in weights):
            raise ValueError('mission_curriculum_phase_weights must be six non-negative values')
        self.curriculum_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        )
        if float(self.curriculum_weights.sum()) <= 0.0:
            raise ValueError('mission_curriculum_phase_weights must have positive sum')
        self.curriculum_weights /= self.curriculum_weights.sum()
        self.curriculum_cross_track_std = float(
            getattr(config, 'mission_curriculum_cross_track_std', 2.0)
        )
        self.curriculum_heading_std = math.radians(float(
            getattr(config, 'mission_curriculum_heading_std_deg', 5.0)
        ))

        self.distance_norm = max(float(
            getattr(config, 'mission_distance_norm', self.route_length)
        ), 1e-6)
        self.altitude_norm = max(float(
            getattr(config, 'mission_altitude_norm', self.cruise_altitude)
        ), 1e-6)
        self.speed_norm = max(float(
            getattr(config, 'mission_speed_norm', 20.0)
        ), 1e-6)
        self.vertical_speed_norm = max(float(
            getattr(config, 'mission_vertical_speed_norm', 8.0)
        ), 1e-6)
        self.rate_norm = max(float(
            getattr(config, 'mission_rate_norm', 3.0)
        ), 1e-6)
        self.gps_age_norm = max(float(
            getattr(config, 'mission_gps_age_norm', 0.2)
        ), 1e-6)
        self.sensor_noise_enabled = bool(
            getattr(config, 'enable_sensor_noise', True)
        )
        self.sensor_velocity_std = float(
            getattr(config, 'sensor_vel_std', 0.05)
        )
        self.sensor_attitude_std = float(
            getattr(config, 'sensor_att_std', 0.003)
        )
        self.sensor_rate_std = float(
            getattr(config, 'sensor_omega_std', 0.0003394)
        )
        self.gate_vertical_actuators = bool(
            getattr(config, 'mission_gate_vertical_actuators', True)
        )

        self.phase = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.phase_entry_step = torch.zeros_like(self.phase)
        self.phase_advanced = torch.zeros(
            self.n, dtype=torch.bool, device=self.device
        )
        self.start_phase = torch.zeros_like(self.phase)
        self.target_npos = torch.zeros(self.n, device=self.device)
        self.target_epos = torch.zeros(self.n, device=self.device)
        self.target_altitude = torch.zeros(self.n, device=self.device)
        self.target_heading = torch.zeros(self.n, device=self.device)
        self.target_speed = torch.zeros(self.n, device=self.device)
        self.previous_waypoint_distance = torch.zeros(self.n, device=self.device)
        self.metric_waypoint_distance = torch.zeros(self.n, device=self.device)
        self.metric_landing_distance = torch.zeros(self.n, device=self.device)
        self.metric_gps_age = torch.zeros(self.n, device=self.device)
        self.training_success_count = torch.zeros((), device=self.device)
        self.training_failure_count = torch.zeros((), device=self.device)

        if self.num_observation != self.OBSERVATION_SIZE:
            self.num_observation = self.OBSERVATION_SIZE
            self.load_observation_space()
        if self.num_actions != 8:
            self.num_actions = 8
            self.load_action_space()

        self.reward_functions = [
            VTOLMissionReward(config),
            VTOLMissionEventReward(config),
        ]
        self.termination_conditions = [
            GroundCollision(config),
            ExtremeAngle(config),
            ExtremeOmega(config),
            ExtremeState(config),
            HighSpeed(config),
            Overload(config),
            VTOLMissionBoundary(config),
            VTOLMissionSuccess(config),
            VTOLMissionTimeout(config),
        ]

    def _uniform(self, count, low, high):
        return torch.rand(count, device=self.device) * (high - low) + low

    def _sample_start_phases(self, count):
        if not self.curriculum_enabled:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        return torch.multinomial(
            self.curriculum_weights, count, replacement=True
        )

    def reset(self, env):
        reset = (
            env.is_done.bool()
            | env.bad_done.bool()
            | env.exceed_time_limit.bool()
        )
        count = int(reset.sum().item())
        if count == 0:
            return
        if not hasattr(env.model, 'get_gps_state'):
            raise TypeError('VTOLMissionTask requires GazeboModel GPS support')

        sampled_phase = self._sample_start_phases(count)
        self.phase[reset] = sampled_phase
        self.start_phase[reset] = sampled_phase
        self.phase_entry_step[reset] = 0
        self.phase_advanced[reset] = False
        self._initialize_curriculum_states(env, reset)
        self._refresh_guidance()
        self.previous_waypoint_distance[reset] = self._waypoint_distance(env)[reset]

    def _initialize_curriculum_states(self, env, reset):
        model = env.model
        ground = reset & (self.phase == self.TAKEOFF)
        model.s[ground, 0] = self.start_n
        model.s[ground, 1] = self.start_e
        model.s[ground, 5] = self.route_heading
        model.set_initial_actuators(ground, [0.0, 0.0, 0.0, 0.0, 0.0])

        phase_ranges = {
            self.ROTOR_CLIMB: (0.00, 0.05, 2.0, self.cruise_altitude - 1.0, 0.0, 2.0),
            self.TRANSITION: (0.02, 0.20, self.cruise_altitude - 2.0,
                              self.cruise_altitude + 2.0, 4.0, self.transition_speed),
            self.FIXED_WING: (0.20, max(0.21, 1.0 - self.approach_distance / self.route_length),
                              self.cruise_altitude - 2.0, self.cruise_altitude + 2.0,
                              self.transition_speed, self.cruise_speed + 2.0),
            self.BACK_TRANSITION: (max(0.0, 1.0 - self.approach_distance / self.route_length),
                                   max(0.01, 1.0 - self.descent_capture_radius / self.route_length),
                                   self.landing_hover_altitude, self.cruise_altitude,
                                   self.backtransition_speed, self.cruise_speed),
            self.VERTICAL_LANDING: (max(0.0, 1.0 - self.descent_capture_radius / self.route_length),
                                    1.0, self.landing_altitude + 1.0,
                                    self.landing_hover_altitude + 2.0, 0.0,
                                    self.backtransition_speed),
        }

        for phase_id, values in phase_ranges.items():
            mask = reset & (self.phase == phase_id)
            f_low, f_high, alt_low, alt_high, speed_low, speed_high = values
            fraction = self._uniform(self.n, f_low, f_high)
            cross_track = (
                torch.randn(self.n, device=self.device)
                * self.curriculum_cross_track_std
            )
            model.s[mask, 0] = (
                self.start_n
                + fraction[mask] * (self.landing_n - self.start_n)
                - cross_track[mask] * self.route_unit_e
            )
            model.s[mask, 1] = (
                self.start_e
                + fraction[mask] * (self.landing_e - self.start_e)
                + cross_track[mask] * self.route_unit_n
            )
            model.s[mask, 2] = self._uniform(self.n, alt_low, alt_high)[mask]
            model.s[mask, 3] = torch.randn(self.n, device=self.device)[mask] * 0.02
            pitch = torch.randn(self.n, device=self.device) * 0.02
            if phase_id == self.FIXED_WING:
                pitch += 0.07
            model.s[mask, 4] = pitch[mask]
            model.s[mask, 5] = (
                self.route_heading
                + torch.randn(self.n, device=self.device)[mask]
                * self.curriculum_heading_std
            )
            speed = self._uniform(self.n, speed_low, speed_high)
            model.s[mask, 6] = speed[mask]
            model.s[mask, 7] = torch.randn(self.n, device=self.device)[mask] * 0.2
            model.s[mask, 8] = torch.where(
                self.phase[mask] == self.FIXED_WING,
                speed[mask] * torch.tan(pitch[mask]),
                torch.randn(self.n, device=self.device)[mask] * 0.2,
            )
            model.s[mask, 9:12] = (
                torch.randn((self.n, 3), device=self.device)[mask] * 0.02
            )

            hover_omega = torch.sqrt(
                (model.mass_curr * model.dynamics.g / (4.0 * 2.0e-5))
                .clamp_min(0.0)
            )
            motors = torch.zeros((self.n, 5), device=self.device)
            if phase_id == self.ROTOR_CLIMB:
                motors[:, 0:4] = hover_omega.reshape(-1, 1)
            elif phase_id == self.TRANSITION:
                motors[:, 0:4] = 0.85 * hover_omega.reshape(-1, 1)
                motors[:, 4] = 1400.0
            elif phase_id == self.FIXED_WING:
                motors[:, 4] = 1800.0
            elif phase_id == self.BACK_TRANSITION:
                motors[:, 0:4] = 0.70 * hover_omega.reshape(-1, 1)
                motors[:, 4] = 1100.0
            else:
                motors[:, 0:4] = hover_omega.reshape(-1, 1)
            model.set_initial_actuators(mask, motors)

        model.sync_reset_state(reset)

    def _refresh_guidance(self):
        approach_n = self.landing_n - self.route_unit_n * self.approach_distance
        approach_e = self.landing_e - self.route_unit_e * self.approach_distance

        self.target_npos[:] = self.landing_n
        self.target_epos[:] = self.landing_e
        self.target_altitude[:] = self.landing_altitude
        self.target_heading[:] = self.final_heading
        self.target_speed[:] = 0.0

        vertical = (self.phase == self.TAKEOFF) | (self.phase == self.ROTOR_CLIMB)
        self.target_npos[vertical] = self.start_n
        self.target_epos[vertical] = self.start_e
        self.target_altitude[vertical] = self.cruise_altitude
        self.target_heading[vertical] = self.route_heading
        self.target_speed[vertical] = self.rotor_climb_speed

        cruise = (self.phase == self.TRANSITION) | (self.phase == self.FIXED_WING)
        self.target_npos[cruise] = approach_n
        self.target_epos[cruise] = approach_e
        self.target_altitude[cruise] = self.cruise_altitude
        self.target_heading[cruise] = self.route_heading
        self.target_speed[self.phase == self.TRANSITION] = self.transition_speed
        self.target_speed[self.phase == self.FIXED_WING] = self.cruise_speed

        backtransition = self.phase == self.BACK_TRANSITION
        self.target_altitude[backtransition] = self.landing_hover_altitude
        self.target_heading[backtransition] = self.route_heading
        self.target_speed[backtransition] = self.approach_speed

    def update_before_observation(self, env):
        self.phase_advanced.zero_()
        phase_before = self.phase.clone()
        elapsed = env.step_count - self.phase_entry_step
        dwell_ok = elapsed >= self.phase_min_steps[self.phase]
        npos, epos, altitude = env.model.get_position()
        speed = env.model.get_TAS()
        contact = env.model.get_ground_contact_state()
        distance_to_landing = torch.sqrt(
            (npos - self.landing_n) ** 2 + (epos - self.landing_e) ** 2
        )

        conditions = (
            (~contact['on_ground']) & (altitude >= self.takeoff_clearance),
            altitude >= self.cruise_altitude - 1.5,
            (speed >= self.transition_speed) & (altitude >= self.cruise_altitude - 4.0),
            distance_to_landing <= self.approach_distance,
            (distance_to_landing <= self.descent_capture_radius)
            & (altitude <= self.landing_hover_altitude + 2.0)
            & (speed <= self.backtransition_speed),
        )
        for phase_id, condition in enumerate(conditions):
            advance = (phase_before == phase_id) & dwell_ok & condition
            self.phase[advance] = phase_id + 1
            self.phase_entry_step[advance] = env.step_count[advance]
            self.phase_advanced |= advance
        self._refresh_guidance()

    def maybe_override_action(self, env, action):
        if not self.gate_vertical_actuators or action.shape[1] < 8:
            return action
        action = action.clone()
        vertical = (
            (self.phase == self.TAKEOFF)
            | (self.phase == self.ROTOR_CLIMB)
            | (self.phase == self.VERTICAL_LANDING)
        )
        action[vertical, 4] = -1.0
        action[vertical, 5:8] = 0.0
        return action

    def _waypoint_distance(self, env):
        npos, epos, altitude = env.model.get_position()
        return torch.sqrt(
            (self.target_npos - npos) ** 2
            + (self.target_epos - epos) ** 2
            + (self.target_altitude - altitude) ** 2
        )

    def step(self, env):
        self.previous_waypoint_distance = self._waypoint_distance(env).detach()
        self.metric_waypoint_distance = self.previous_waypoint_distance
        npos, epos, _ = env.model.get_position()
        self.metric_landing_distance = torch.sqrt(
            (npos - self.landing_n) ** 2 + (epos - self.landing_e) ** 2
        ).detach()
        self.metric_gps_age = env.model.get_gps_state()['age'].detach()
        self.training_success_count += self.landing_success(env).sum()
        self.training_failure_count += env.bad_done.sum()

    def get_training_metrics(self):
        metrics = {
            'mission/waypoint_distance_mean': self.metric_waypoint_distance.mean(),
            'mission/landing_distance_mean': self.metric_landing_distance.mean(),
            'mission/gps_age_mean': self.metric_gps_age.mean(),
            'mission/success_count': self.training_success_count,
            'mission/failure_count': self.training_failure_count,
        }
        for phase_id, phase_name in enumerate(self.PHASE_NAMES):
            metrics[f'mission/phase_{phase_name}_fraction'] = (
                self.phase == phase_id
            ).float().mean()
            metrics[f'mission/start_{phase_name}_fraction'] = (
                self.start_phase == phase_id
            ).float().mean()
        return metrics

    def landing_success(self, env):
        npos, epos, _ = env.model.get_position()
        roll, pitch, _ = env.model.get_posture()
        speed = env.model.get_TAS()
        climb_rate = env.model.get_climb_rate()
        contact = env.model.get_ground_contact_state()
        distance = torch.sqrt(
            (npos - self.landing_n) ** 2 + (epos - self.landing_e) ** 2
        )
        radius = float(getattr(self.config, 'mission_landing_radius', 3.0))
        max_speed = float(getattr(self.config, 'mission_landing_max_speed', 1.0))
        max_sink = float(getattr(self.config, 'mission_landing_max_sink_speed', 0.6))
        max_tilt = math.radians(float(
            getattr(self.config, 'mission_landing_max_tilt_deg', 12.0)
        ))
        settle_steps = int(getattr(
            self.config, 'mission_landing_settle_steps', 15
        ))
        return (
            (self.phase == self.VERTICAL_LANDING)
            & contact['on_ground']
            & (contact['contact_duration_steps'] >= settle_steps)
            & (distance <= radius)
            & (speed <= max_speed)
            & (climb_rate.abs() <= max_sink)
            & (roll.abs() <= max_tilt)
            & (pitch.abs() <= max_tilt)
        )

    def _build_obs(self, env, clean=False):
        if clean:
            gps_position = env.model.s[:, 0:3]
            gps_age = torch.zeros(self.n, device=self.device)
            gps_valid = torch.ones(self.n, dtype=torch.bool, device=self.device)
        else:
            gps = env.model.get_gps_state()
            gps_position = gps['position']
            gps_age = gps['age']
            gps_valid = gps['valid']

        npos = gps_position[:, 0]
        epos = gps_position[:, 1]
        altitude = gps_position[:, 2]
        roll, pitch, heading = env.model.get_posture()
        speed = env.model.get_TAS()
        vel_n, vel_e, vel_up = env.model.get_world_velocity()
        p, q, r = env.model.get_angular_velocity()
        sa, ca, sb, cb = env.model.get_aero_sincos()

        if self.sensor_noise_enabled and not clean:
            roll = wrap_PI(roll + torch.randn_like(roll) * self.sensor_attitude_std)
            pitch = wrap_PI(pitch + torch.randn_like(pitch) * self.sensor_attitude_std)
            heading = wrap_PI(
                heading + torch.randn_like(heading) * self.sensor_attitude_std
            )
            vel_n = vel_n + torch.randn_like(vel_n) * self.sensor_velocity_std
            vel_e = vel_e + torch.randn_like(vel_e) * self.sensor_velocity_std
            vel_up = vel_up + torch.randn_like(vel_up) * self.sensor_velocity_std
            p = p + torch.randn_like(p) * self.sensor_rate_std
            q = q + torch.randn_like(q) * self.sensor_rate_std
            r = r + torch.randn_like(r) * self.sensor_rate_std
            speed = torch.sqrt(
                (vel_n * vel_n + vel_e * vel_e + vel_up * vel_up).clamp_min(0.0)
            )

        landing_dn = self.landing_n - npos
        landing_de = self.landing_e - epos
        landing_dalt = self.landing_altitude - altitude
        waypoint_dn = self.target_npos - npos
        waypoint_de = self.target_epos - epos
        waypoint_dalt = self.target_altitude - altitude
        along_track = (
            (npos - self.start_n) * self.route_unit_n
            + (epos - self.start_e) * self.route_unit_e
        )
        cross_track = (
            -(npos - self.start_n) * self.route_unit_e
            + (epos - self.start_e) * self.route_unit_n
        )
        heading_error = wrap_PI(self.target_heading - heading)
        phase_one_hot = torch.nn.functional.one_hot(
            self.phase, num_classes=self.PHASE_COUNT
        ).to(dtype=env.model.s.dtype)

        motor = torch.stack(env.model.get_motor_omega(), dim=1)
        motor_scale = env.model.motor_omega_max.reshape(1, 5)
        surfaces = env.model.u[:, 5:8] / env.model.surface_limit.reshape(1, 3)
        contact = env.model.get_ground_contact_state()
        time_remaining = (
            1.0 - env.step_count.to(env.model.s.dtype) / max(self.max_steps, 1)
        ).clamp(0.0, 1.0)

        obs = torch.hstack((
            (landing_dn / self.distance_norm).reshape(-1, 1),
            (landing_de / self.distance_norm).reshape(-1, 1),
            (landing_dalt / self.altitude_norm).reshape(-1, 1),
            (waypoint_dn / self.distance_norm).reshape(-1, 1),
            (waypoint_de / self.distance_norm).reshape(-1, 1),
            (waypoint_dalt / self.altitude_norm).reshape(-1, 1),
            (along_track / self.distance_norm).reshape(-1, 1),
            (cross_track / self.distance_norm).reshape(-1, 1),
            torch.sin(heading_error).reshape(-1, 1),
            torch.cos(heading_error).reshape(-1, 1),
            phase_one_hot,
            (gps_age / self.gps_age_norm).reshape(-1, 1),
            gps_valid.to(env.model.s.dtype).reshape(-1, 1),
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
            (speed / self.speed_norm).reshape(-1, 1),
            (vel_n / self.speed_norm).reshape(-1, 1),
            (vel_e / self.speed_norm).reshape(-1, 1),
            (vel_up / self.vertical_speed_norm).reshape(-1, 1),
            (self.target_speed / self.speed_norm).reshape(-1, 1),
            (p / self.rate_norm).reshape(-1, 1),
            (q / self.rate_norm).reshape(-1, 1),
            (r / self.rate_norm).reshape(-1, 1),
            sa.reshape(-1, 1),
            ca.reshape(-1, 1),
            sb.reshape(-1, 1),
            cb.reshape(-1, 1),
            motor / motor_scale,
            surfaces,
            contact['on_ground'].to(env.model.s.dtype).reshape(-1, 1),
            (contact['contact_count'].to(env.model.s.dtype) / 4.0).reshape(-1, 1),
            time_remaining.reshape(-1, 1),
        ))
        if obs.shape[1] != self.OBSERVATION_SIZE:
            raise RuntimeError(
                f'VTOL mission observation has {obs.shape[1]} values, '
                f'expected {self.OBSERVATION_SIZE}'
            )
        return obs

    def get_obs(self, env):
        return self._build_obs(env, clean=False)

    def get_clean_obs(self, env):
        return self._build_obs(env, clean=True)
