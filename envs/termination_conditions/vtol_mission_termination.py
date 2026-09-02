import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hybrid_termination_conditions.termination_condition_base import (
    BaseTerminationCondition,
)


class VTOLMissionBoundary(BaseTerminationCondition):
    """Terminate unsafe route departures and premature ground contact."""

    def __init__(self, config):
        super().__init__(config)
        self.max_cross_track = float(
            getattr(config, 'mission_max_cross_track', 50.0)
        )
        self.route_margin = float(
            getattr(config, 'mission_route_margin', 30.0)
        )
        self.max_altitude = float(
            getattr(config, 'mission_max_altitude', 50.0)
        )
        self.min_cruise_altitude = float(
            getattr(config, 'mission_min_safe_cruise_altitude', 4.0)
        )

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        npos, epos, altitude = env.model.get_position()
        contact = env.model.get_ground_contact_state()
        along_track = (
            (npos - task.start_n) * task.route_unit_n
            + (epos - task.start_e) * task.route_unit_e
        )
        cross_track = (
            -(npos - task.start_n) * task.route_unit_e
            + (epos - task.start_e) * task.route_unit_n
        )

        off_route = (
            (cross_track.abs() > self.max_cross_track)
            | (along_track < -self.route_margin)
            | (along_track > task.route_length + self.route_margin)
        )
        altitude_violation = altitude > self.max_altitude
        cruise_phase = (
            (task.phase == task.TRANSITION)
            | (task.phase == task.FIXED_WING)
            | (task.phase == task.BACK_TRANSITION)
        )
        too_low = cruise_phase & (altitude < self.min_cruise_altitude)
        premature_contact = (
            contact['on_ground']
            & (task.phase >= task.ROTOR_CLIMB)
            & (task.phase <= task.BACK_TRANSITION)
        )
        finite = torch.isfinite(env.model.s).all(dim=1)
        nonfinite = ~finite

        bad_done = (
            off_route
            | altitude_violation
            | too_low
            | premature_contact
            | nonfinite
        )
        done = torch.zeros_like(bad_done)
        timeout = torch.zeros_like(bad_done)
        info['mission_off_route'] = off_route
        info['mission_altitude_violation'] = altitude_violation | too_low
        info['mission_premature_contact'] = premature_contact
        info['mission_nonfinite'] = nonfinite
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log('VTOL mission safety boundary violated')
        return bad_done, done, timeout, info


class VTOLMissionSuccess(BaseTerminationCondition):
    """Complete only after a settled, accurate, low-energy touchdown."""

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        success = task.landing_success(env)
        bad_done = torch.zeros_like(success)
        timeout = torch.zeros_like(success)
        info['mission_success'] = success
        info['mission_phase'] = task.phase
        info['mission_phase_advanced'] = task.phase_advanced
        info['mission_start_phase'] = task.start_phase
        npos, epos, _ = env.model.get_position()
        info['mission_landing_error'] = torch.sqrt(
            (npos - task.landing_n) ** 2 + (epos - task.landing_e) ** 2
        )
        info['mission_elapsed_steps'] = env.step_count.clone()
        info['mission_touchdown_speed'] = (
            env.model.last_touchdown_vertical_speed.clone()
        )
        if getattr(self.config, 'termination_verbose', True) and torch.any(success):
            self.log('VTOL mission completed with a safe landing')
        return bad_done, success, timeout, info


class VTOLMissionTimeout(BaseTerminationCondition):
    """Mark the maximum mission duration as a truncation, not a crash."""

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = int(getattr(config, 'max_steps', 3000))

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        timeout = env.step_count >= self.max_steps
        zeros = torch.zeros_like(timeout)
        info['mission_timeout'] = timeout
        return zeros, zeros.clone(), timeout, info
