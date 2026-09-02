import os
import sys
import math

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class VTOLMissionReward(BaseRewardFunction):
    """Dense progress, mode-efficiency, and safety reward for a VTOL mission."""

    def __init__(self, config):
        super().__init__(config)
        self.w_progress = float(getattr(config, 'mission_w_progress', 4.0))
        self.w_altitude = float(getattr(config, 'mission_w_altitude', 1.0))
        self.w_speed = float(getattr(config, 'mission_w_speed', 0.8))
        self.w_heading = float(getattr(config, 'mission_w_heading', 0.8))
        self.w_waypoint = float(getattr(config, 'mission_w_waypoint', 0.5))
        self.w_phase = float(getattr(config, 'mission_w_phase_bonus', 30.0))
        self.w_time = float(getattr(config, 'mission_w_time', 0.03))
        self.w_attitude = float(getattr(config, 'mission_w_attitude_safety', 0.8))
        self.w_rate = float(getattr(config, 'mission_w_rate_safety', 0.25))
        self.w_aero = float(getattr(config, 'mission_w_aero_safety', 0.5))
        self.w_sink = float(getattr(config, 'mission_w_sink_safety', 1.5))
        self.w_smooth = float(getattr(config, 'mission_w_action_smooth', 0.15))
        self.w_mode = float(getattr(config, 'mission_w_mode_efficiency', 0.2))

        self.altitude_sigma = max(float(
            getattr(config, 'mission_reward_altitude_sigma', 4.0)
        ), 1e-6)
        self.speed_sigma = max(float(
            getattr(config, 'mission_reward_speed_sigma', 3.0)
        ), 1e-6)
        self.heading_sigma = max(float(
            getattr(config, 'mission_reward_heading_sigma', 0.35)
        ), 1e-6)
        self.waypoint_sigma = max(float(
            getattr(config, 'mission_reward_waypoint_sigma', 12.0)
        ), 1e-6)
        self.safe_tilt = math.radians(float(
            getattr(config, 'mission_reward_safe_tilt_deg', 25.0)
        ))
        self.safe_rate = float(
            getattr(config, 'mission_reward_safe_rate', 1.5)
        )
        self.safe_alpha = math.radians(float(
            getattr(config, 'mission_reward_safe_alpha_deg', 18.0)
        ))
        self.safe_beta = math.radians(float(
            getattr(config, 'mission_reward_safe_beta_deg', 12.0)
        ))
        self.safe_sink = float(
            getattr(config, 'mission_reward_safe_sink_speed', 1.0)
        )
        self.progress_clip = float(
            getattr(config, 'mission_reward_progress_clip', 2.0)
        )

    @staticmethod
    def _excess_square(value, limit):
        return torch.relu(value.abs() - limit) ** 2

    def get_reward(self, task, env):
        npos, epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        speed = env.model.get_TAS()
        climb_rate = env.model.get_climb_rate()
        p, q, r = env.model.get_angular_velocity()
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()

        waypoint_distance = torch.sqrt(
            (task.target_npos - npos) ** 2
            + (task.target_epos - epos) ** 2
            + (task.target_altitude - altitude) ** 2
        )
        raw_progress = task.previous_waypoint_distance - waypoint_distance
        progress = torch.where(
            task.phase_advanced,
            torch.zeros_like(raw_progress),
            raw_progress.clamp(-self.progress_clip, self.progress_clip),
        )

        altitude_error = task.target_altitude - altitude
        speed_error = task.target_speed - speed
        heading_error = wrap_PI(task.target_heading - heading)
        altitude_reward = torch.exp(
            -(altitude_error / self.altitude_sigma) ** 2
        )
        speed_reward = torch.exp(-(speed_error / self.speed_sigma) ** 2)
        heading_reward = torch.exp(
            -(heading_error / self.heading_sigma) ** 2
        )
        waypoint_reward = torch.exp(
            -(waypoint_distance / self.waypoint_sigma) ** 2
        )

        tilt_cost = (
            self._excess_square(roll, self.safe_tilt)
            + self._excess_square(pitch, self.safe_tilt)
        )
        rate_cost = (
            self._excess_square(p, self.safe_rate)
            + self._excess_square(q, self.safe_rate)
            + self._excess_square(r, self.safe_rate)
        )
        aero_cost = (
            self._excess_square(alpha, self.safe_alpha)
            + self._excess_square(beta, self.safe_beta)
        )
        final_approach = task.phase == task.VERTICAL_LANDING
        sink_cost = torch.where(
            final_approach,
            torch.relu(-climb_rate - self.safe_sink) ** 2,
            torch.zeros_like(climb_rate),
        )

        motor_scale = env.model.motor_omega_max.reshape(1, 5)
        motor = env.model.u[:, 0:5] / motor_scale
        recent_motor = env.model.recent_u[:, 0:5] / motor_scale
        surface = env.model.u[:, 5:8] / env.model.surface_limit.reshape(1, 3)
        recent_surface = (
            env.model.recent_u[:, 5:8]
            / env.model.surface_limit.reshape(1, 3)
        )
        smooth_cost = torch.sum((motor - recent_motor) ** 2, dim=1)
        smooth_cost += torch.sum((surface - recent_surface) ** 2, dim=1)

        vertical_mode = (
            (task.phase == task.TAKEOFF)
            | (task.phase == task.ROTOR_CLIMB)
            | (task.phase == task.VERTICAL_LANDING)
        )
        fixed_mode = task.phase == task.FIXED_WING
        mode_cost = torch.where(
            vertical_mode,
            motor[:, 4] ** 2,
            torch.zeros_like(speed),
        )
        mode_cost += torch.where(
            fixed_mode,
            torch.mean(motor[:, 0:4] ** 2, dim=1),
            torch.zeros_like(speed),
        )

        return (
            self.w_progress * progress
            + self.w_altitude * altitude_reward
            + self.w_speed * speed_reward
            + self.w_heading * heading_reward
            + self.w_waypoint * waypoint_reward
            + self.w_phase * task.phase_advanced.to(speed.dtype)
            - self.w_time
            - self.w_attitude * tilt_cost
            - self.w_rate * rate_cost
            - self.w_aero * aero_cost
            - self.w_sink * sink_cost
            - self.w_smooth * smooth_cost
            - self.w_mode * mode_cost
        )


class VTOLMissionEventReward(BaseRewardFunction):
    """Terminal rewards that distinguish safe landing, crash, and timeout."""

    def __init__(self, config):
        super().__init__(config)
        self.success_bonus = float(
            getattr(config, 'mission_success_bonus', 500.0)
        )
        self.failure_penalty = float(
            getattr(config, 'mission_failure_penalty', 250.0)
        )
        self.timeout_penalty = float(
            getattr(config, 'mission_timeout_penalty', 100.0)
        )

    def get_reward(self, task, env):
        success = task.landing_success(env).to(env.model.s.dtype)
        failure = env.bad_done.to(env.model.s.dtype)
        timeout = env.exceed_time_limit.to(env.model.s.dtype)
        return (
            self.success_bonus * success
            - self.failure_penalty * failure
            - self.timeout_penalty * timeout
        )
