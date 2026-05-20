"""Circle trajectory task.

The aircraft must follow a moving target that traces a horizontal circle.

Geometry (set at every reset):
    centre = spawn_position + circle_offset_left * body_left_unit_vector
    body_left_unit_vector (in NED north/east) = (sin(yaw_spawn), -cos(yaw_spawn))
        i.e. 90 deg port of the aircraft's nose.
    radius      = circle_radius
    altitude    = altitude_spawn
    phase(t=0)  = atan2(spawn_e - centre_e, spawn_n - centre_n)
                 -> the moving target starts on top of the aircraft.

Per-step update:
    phase = initial_phase + circle_omega * dt * step_count
    target_npos = centre_n + R * cos(phase)
    target_epos = centre_e + R * sin(phase)
    target_alt  = altitude_spawn
    target_yaw  = wrap(phase + sign(omega) * pi/2)   (tangent direction)

Defaults: centre 10 m to the left, radius 10 m, period -40 s.

Observation layout extends HoverTask:
    0-23. Hover-style target deltas + aircraft state + motor forces
    24-25. target_vn, target_ve normalized by circle_target_vel_norm
    26-27. target_an, target_ae normalized by circle_target_acc_norm
"""
import os
import sys
import math
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.circle_reward import CircleReward
from reward_functions.circle_event_driven_reward import CircleEventDrivenReward
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.high_speed import HighSpeed
from hybrid_termination_conditions.overload import Overload
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from tasks.hover_task import HOVER_POS_NORM, HOVER_ALT_NORM
from utils.utils import wrap_PI


class CircleTask(BaseTask):
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'circle'

        self.dt = float(getattr(config, 'dt', 0.02))
        self.radius        = float(getattr(config, 'circle_radius', 10.0))
        self.offset_left   = float(getattr(config, 'circle_offset_left', 10.0))
        # Period: positive -> CCW in NED, negative -> CW.
        period = float(getattr(config, 'circle_period', 20.0))
        if period == 0.0:
            self.omega = 0.0
        else:
            self.omega = 2.0 * math.pi / period
        # Optional: explicit override
        self.omega = float(getattr(config, 'circle_omega', self.omega))
        self.face_tangent = bool(getattr(config, 'circle_face_tangent', True))

        self.noise_scale = getattr(config, 'noise_scale', 0.01)
        self.pos_norm = float(getattr(config, 'hover_pos_norm', HOVER_POS_NORM))
        self.alt_norm = float(getattr(config, 'hover_alt_norm', HOVER_ALT_NORM))
        self.enable_sensor_noise = getattr(config, 'enable_sensor_noise', True)
        self.sensor_pos_std = float(getattr(config, 'sensor_pos_std', 0.5))
        self.sensor_vel_std = float(getattr(config, 'sensor_vel_std', 0.05))
        self.sensor_att_std = float(getattr(config, 'sensor_att_std', 0.005))
        self.sensor_omega_std = float(getattr(config, 'sensor_omega_std', 0.0005))
        self.include_target_motion = bool(getattr(config, 'circle_include_target_motion', True))
        self.target_vel_norm = float(getattr(config, 'circle_target_vel_norm', 2.0))
        self.target_acc_norm = float(getattr(config, 'circle_target_acc_norm', 2.0))
        expected_obs = 28 if self.include_target_motion else 24
        if self.num_observation != expected_obs:
            self.num_observation = expected_obs
            self.load_observation_space()

        # Per-env state
        self.center_n = torch.zeros(self.n, device=self.device)
        self.center_e = torch.zeros(self.n, device=self.device)
        self.initial_phase = torch.zeros(self.n, device=self.device)
        self.phase    = torch.zeros(self.n, device=self.device)

        # Targets (kept in sync with phase)
        self.target_npos     = torch.zeros(self.n, device=self.device)
        self.target_epos     = torch.zeros(self.n, device=self.device)
        self.target_altitude = torch.zeros(self.n, device=self.device)
        self.target_heading  = torch.zeros(self.n, device=self.device)
        self.target_vn       = torch.zeros(self.n, device=self.device)
        self.target_ve       = torch.zeros(self.n, device=self.device)
        self.target_an       = torch.zeros(self.n, device=self.device)
        self.target_ae       = torch.zeros(self.n, device=self.device)

        self.reward_functions = [
            CircleReward(self.config),
            CircleEventDrivenReward(self.config),
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
            HighSpeed(self.config),
            Overload(self.config),
            HoverTimeoutDone(self.config),
        ]

    # ------------------------------------------------------------------ #
    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = done | bad_done | exceed_time_limit
        size = int(torch.sum(reset))
        if size == 0:
            return
        npos, epos, altitude = env.model.get_position()
        _, _, heading = env.model.get_posture()

        # Centre = spawn + offset_left * body_left_unit_vector  (NED)
        # body_left = ( sin(yaw), -cos(yaw) )  in (north, east)
        h = heading[reset]
        cn = npos[reset] + self.offset_left * torch.sin(h)
        ce = epos[reset] - self.offset_left * torch.cos(h)
        self.center_n[reset] = cn
        self.center_e[reset] = ce
        self.target_altitude[reset] = altitude[reset].clone()

        # Initial phase puts the moving target on top of the aircraft.
        phi0 = torch.atan2(epos[reset] - ce, npos[reset] - cn)
        self.initial_phase[reset] = phi0
        self.phase[reset] = phi0
        self._refresh_target(reset)

    def step(self, env):
        # Target timing is synchronised in get_obs()/reward from env.step_count.
        # BaseEnv computes obs/reward before calling task.step(), so advancing
        # here would make returned observations one step stale for moving targets.
        pass

    def sync_target_to_time(self, env):
        self.phase = wrap_PI(
            self.initial_phase + self.omega * self.dt * env.step_count.float()
        )
        all_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._refresh_target(all_mask)

    # ------------------------------------------------------------------ #
    def _refresh_target(self, mask):
        cn = self.center_n[mask]
        ce = self.center_e[mask]
        ph = self.phase[mask]
        self.target_npos[mask] = cn + self.radius * torch.cos(ph)
        self.target_epos[mask] = ce + self.radius * torch.sin(ph)
        self.target_vn[mask] = -self.radius * self.omega * torch.sin(ph)
        self.target_ve[mask] = self.radius * self.omega * torch.cos(ph)
        self.target_an[mask] = -self.radius * self.omega * self.omega * torch.cos(ph)
        self.target_ae[mask] = -self.radius * self.omega * self.omega * torch.sin(ph)
        if self.face_tangent and self.omega != 0.0:
            tangent = ph + (math.pi / 2.0 if self.omega > 0 else -math.pi / 2.0)
            self.target_heading[mask] = wrap_PI(tangent)
        # else: leave target_heading at 0 (or whatever the user pre-set)

    # ------------------------------------------------------------------ #
    def _apply_sensor_noise(self, roll, pitch, heading, vx, vy, vz, P, Q, R):
        roll = wrap_PI(roll + torch.randn_like(roll) * self.sensor_att_std)
        pitch = wrap_PI(pitch + torch.randn_like(pitch) * self.sensor_att_std)
        heading = wrap_PI(heading + torch.randn_like(heading) * self.sensor_att_std)
        vx = vx + torch.randn_like(vx) * self.sensor_vel_std
        vy = vy + torch.randn_like(vy) * self.sensor_vel_std
        vz = vz + torch.randn_like(vz) * self.sensor_vel_std
        P = P + torch.randn_like(P) * self.sensor_omega_std
        Q = Q + torch.randn_like(Q) * self.sensor_omega_std
        R = R + torch.randn_like(R) * self.sensor_omega_std
        return roll, pitch, heading, vx, vy, vz, P, Q, R

    def _build_obs(self, env, add_sensor_noise):
        self.sync_target_to_time(env)
        npos, epos, altitude   = env.model.get_position()
        roll, pitch, heading   = env.model.get_posture()
        vt                     = env.model.get_vt()
        vx, vy                 = env.model.get_ground_speed()
        vz                     = env.model.get_climb_rate()
        P, Q, R                = env.model.get_angular_velocity()
        sa, ca, sb, cb         = env.model.get_aero_sincos()
        F_head, F_rf, F_lb, F_lf, F_rb = env.model.get_F()

        if add_sensor_noise:
            npos = npos + torch.randn_like(npos) * self.sensor_pos_std
            epos = epos + torch.randn_like(epos) * self.sensor_pos_std
            altitude = altitude + torch.randn_like(altitude) * self.sensor_pos_std
            roll, pitch, heading, vx, vy, vz, P, Q, R = self._apply_sensor_noise(
                roll, pitch, heading, vx, vy, vz, P, Q, R)
            vt = torch.sqrt((vx * vx + vy * vy + vz * vz).clamp_min(0.0))

        delta_n   = (self.target_npos     - npos     ).reshape(-1, 1) / self.pos_norm
        delta_e   = (self.target_epos     - epos     ).reshape(-1, 1) / self.pos_norm
        delta_alt = (self.target_altitude - altitude ).reshape(-1, 1) / self.alt_norm
        delta_yaw = wrap_PI((self.target_heading - heading).reshape(-1, 1)) / torch.pi

        obs = torch.hstack((
            delta_n, delta_e, delta_alt, delta_yaw,
            torch.sin(roll).reshape(-1, 1),  torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1), torch.cos(pitch).reshape(-1, 1),
            (vt / 10.0).reshape(-1, 1),
            (vx / 5.0).reshape(-1, 1),
            (vy / 5.0).reshape(-1, 1),
            (vz / 5.0).reshape(-1, 1),
            P.reshape(-1, 1), Q.reshape(-1, 1), R.reshape(-1, 1),
            sa.reshape(-1, 1), ca.reshape(-1, 1),
            sb.reshape(-1, 1), cb.reshape(-1, 1),
            (F_head / 7.0).reshape(-1, 1),
            (F_rf   / 7.0).reshape(-1, 1),
            (F_lb   / 7.0).reshape(-1, 1),
            (F_lf   / 7.0).reshape(-1, 1),
            (F_rb   / 7.0).reshape(-1, 1),
        ))
        if self.include_target_motion:
            target_motion = torch.hstack((
                (self.target_vn / self.target_vel_norm).reshape(-1, 1),
                (self.target_ve / self.target_vel_norm).reshape(-1, 1),
                (self.target_an / self.target_acc_norm).reshape(-1, 1),
                (self.target_ae / self.target_acc_norm).reshape(-1, 1),
            ))
            obs = torch.hstack((obs, target_motion))
        if self.noise_scale > 0:
            obs = obs + torch.randn_like(obs) * self.noise_scale
        return obs

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)
