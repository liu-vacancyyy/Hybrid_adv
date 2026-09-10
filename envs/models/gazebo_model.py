import math
import os
import sys

import torch
from torchdiffeq import odeint_adjoint as odeint

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_base import BaseModel
from Gazebo.Gazebo_dynamics import GazeboVTOLDynamics


class GazeboModel(BaseModel):
    """BaseModel wrapper for the Gazebo Classic standard_vtol approximation."""

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.num_states = getattr(self.config, 'num_states', 12)
        self.num_controls = getattr(self.config, 'num_controls', 8)
        self.dt = getattr(self.config, 'dt', 0.02)
        self.solver = getattr(self.config, 'solver', 'euler')
        self.airspeed = getattr(self.config, 'airspeed', 0.0)
        self.gps_enabled = bool(getattr(self.config, 'gps_enable', False))
        self.gps_update_rate_hz = float(
            getattr(self.config, 'gps_update_rate_hz', 10.0)
        )
        if self.gps_update_rate_hz <= 0.0:
            raise ValueError('gps_update_rate_hz must be positive')
        self.gps_update_steps = max(
            1, int(round(1.0 / (self.dt * self.gps_update_rate_hz)))
        )
        self.gps_horizontal_std = float(
            getattr(self.config, 'gps_horizontal_std', 0.0)
        )
        self.gps_vertical_std = float(
            getattr(self.config, 'gps_vertical_std', 0.0)
        )
        if self.gps_horizontal_std < 0.0 or self.gps_vertical_std < 0.0:
            raise ValueError('GPS noise standard deviations must be non-negative')

        self.s = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_s = torch.zeros((self.n, self.num_states), device=self.device)
        self.u = torch.zeros((self.n, self.num_controls), device=self.device)
        self.recent_u = torch.zeros((self.n, self.num_controls), device=self.device)

        self.dynamics = GazeboVTOLDynamics(config)
        self.aero_force_scale = torch.ones(self.n, device=self.device)
        self.ground_contact_enabled = self.dynamics.ground_contact_enabled
        default_substeps = 5 if self.ground_contact_enabled else 1
        self.ground_physics_substeps = int(
            getattr(self.config, 'ground_physics_substeps', default_substeps)
        )
        if self.ground_physics_substeps < 1:
            raise ValueError('ground_physics_substeps must be at least 1')
        self.spawn_mode = str(
            getattr(self.config, 'spawn_mode', getattr(self.config, 'gazebo_spawn_mode', 'air'))
        ).lower()
        if self.spawn_mode not in ('air', 'ground', 'mixed'):
            raise ValueError("spawn_mode must be 'air', 'ground', or 'mixed'")
        self.ground_spawn_probability = float(
            getattr(self.config, 'ground_spawn_probability', 0.5)
        )
        if not 0.0 <= self.ground_spawn_probability <= 1.0:
            raise ValueError('ground_spawn_probability must be between 0 and 1')
        self.ground_spawn_clearance = float(
            getattr(self.config, 'ground_spawn_clearance', 0.0)
        )
        self.ground_spawn_static_equilibrium = bool(
            getattr(self.config, 'ground_spawn_static_equilibrium', True)
        )
        self.ground_init_roll_range = float(
            getattr(self.config, 'ground_init_roll_range', 0.0)
        )
        self.ground_init_pitch_range = float(
            getattr(self.config, 'ground_init_pitch_range', 0.0)
        )
        self.ground_init_vel_range = float(
            getattr(self.config, 'ground_init_vel_range', 0.0)
        )
        self.ground_init_omega_range = float(
            getattr(self.config, 'ground_init_omega_range', 0.0)
        )

        self.max_altitude = float(getattr(self.config, 'max_altitude', 50.0))
        self.min_altitude = float(getattr(self.config, 'min_altitude', 4.0))
        self.init_roll_range = float(getattr(self.config, 'init_roll_range', 0.01 * math.pi))
        self.init_pitch_range = float(getattr(self.config, 'init_pitch_range', 0.01 * math.pi))
        self.init_yaw_range = float(getattr(self.config, 'init_yaw_range', math.pi))
        self.init_vel_range = float(getattr(self.config, 'init_vel_range', 0.1))
        self.init_omega_range = float(getattr(self.config, 'init_omega_range', 0.02))

        self.action_is_gazebo_control = bool(getattr(self.config, 'action_is_gazebo_control', False))
        self.enable_action_filter = bool(getattr(self.config, 'enable_action_filter', False))
        self.action_filter_alpha = float(getattr(self.config, 'action_filter_alpha', 1.0))
        self.action_filter_alpha = min(max(self.action_filter_alpha, 0.0), 1.0)
        self.filtered_action = torch.zeros((self.n, self.num_controls), device=self.device)
        self.filtered_action_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        self.time_constant_up = float(getattr(self.config, 'gazebo_motor_tau_up', 0.0125))
        self.time_constant_down = float(getattr(self.config, 'gazebo_motor_tau_down', 0.025))
        self.motor_cmd_scaling = torch.tensor([1500., 1500., 1500., 1500., 5500.], device=self.device)
        self.motor_omega_max = torch.tensor([1500., 1500., 1500., 1500., 3500.], device=self.device)
        # Task-level flight-mode logic can set this per-environment cap.  It is
        # applied after action filtering, where it cannot be bypassed by a
        # stale pusher command from the preceding control step.
        self.pusher_omega_cap = torch.full(
            (self.n,), float('inf'), device=self.device
        )
        # Captures the cap for the current update.  ``pusher_omega_cap`` is
        # cleared by ``_map_action`` so a later phase cannot inherit it, while
        # this cache lets the motor lag filter enforce the same physical cap on
        # the filtered motor state.
        self._mapped_pusher_omega_cap = torch.full(
            (self.n,), float('inf'), device=self.device
        )
        self.surface_limit = torch.tensor([0.53, 0.53, 0.53], device=self.device)
        self.motor_omega = torch.zeros((self.n, 5), device=self.device)

        self.dr_mass = float(getattr(self.config, 'dr_mass', 0.0))
        self.dr_inertia = float(getattr(self.config, 'dr_inertia', 0.0))
        self.mass_curr = torch.ones(self.n, device=self.device) * self.dynamics.nominal_m
        self._Jx_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jx
        self._Jy_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jy
        self._Jz_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jz
        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t, self._Jz_t)
        self.on_ground = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.just_touchdown = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.contact_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.contact_duration_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.max_ground_penetration = torch.zeros(self.n, device=self.device)
        self.min_ground_clearance = torch.zeros(self.n, device=self.device)
        self.ground_normal_force = torch.zeros(self.n, device=self.device)
        self.touchdown_vertical_speed = torch.zeros(self.n, device=self.device)
        self.last_touchdown_vertical_speed = torch.zeros(
            self.n, device=self.device
        )
        self.ground_force_body = torch.zeros((self.n, 3), device=self.device)
        self.ground_moment_body = torch.zeros((self.n, 3), device=self.device)
        self.gps_position = torch.zeros((self.n, 3), device=self.device)
        self.gps_age_steps = torch.zeros(
            self.n, dtype=torch.long, device=self.device
        )
        self.gps_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self._set_hover_controls(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def _u(self, size, half_range):
        if half_range <= 0.0:
            return torch.zeros(size, device=self.device)
        return (torch.rand(size, device=self.device) * 2.0 - 1.0) * half_range

    def _set_hover_controls(self, mask):
        hover_omega = torch.sqrt(
            (self.mass_curr * self.dynamics.g / 4.0 / 2.0e-5).clamp(min=0.0)
        )
        self.motor_omega[mask, :] = 0.0
        self.motor_omega[mask, 0:4] = hover_omega[mask].reshape(-1, 1)
        self.u[mask, :] = 0.0
        self.u[mask, 0:5] = self.motor_omega[mask, :]
        self.recent_u[mask] = self.u[mask]

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = done | bad_done | exceed_time_limit
        size = int(torch.sum(reset).item())
        if size == 0:
            return

        self.s[reset, :] = 0.0
        self.u[reset, :] = 0.0
        self.motor_omega[reset, :] = 0.0

        ground_reset = torch.zeros_like(reset)
        if self.ground_contact_enabled:
            if self.spawn_mode == 'ground':
                ground_reset = reset.clone()
            elif self.spawn_mode == 'mixed':
                ground_reset[reset] = (
                    torch.rand(size, device=self.device) < self.ground_spawn_probability
                )
        air_reset = reset & ~ground_reset

        self.s[reset, 2] = (torch.rand(size, device=self.device)
                            * (self.max_altitude - self.min_altitude) + self.min_altitude)
        self.s[reset, 3] = self._u(size, self.init_roll_range)
        self.s[reset, 4] = self._u(size, self.init_pitch_range)
        self.s[reset, 5] = self._u(size, self.init_yaw_range)
        for k in range(3):
            self.s[reset, 6 + k] = self._u(size, self.init_vel_range)
            self.s[reset, 9 + k] = self._u(size, self.init_omega_range)

        ground_size = int(ground_reset.sum().item())
        if ground_size > 0:
            self.s[ground_reset, 3] = self._u(ground_size, self.ground_init_roll_range)
            self.s[ground_reset, 4] = self._u(ground_size, self.ground_init_pitch_range)
            for k in range(3):
                self.s[ground_reset, 6 + k] = self._u(
                    ground_size, self.ground_init_vel_range
                )
                self.s[ground_reset, 9 + k] = self._u(
                    ground_size, self.ground_init_omega_range
                )

        dm = self.dr_mass
        di = self.dr_inertia
        if dm > 0.0:
            self.mass_curr[reset] = (
                torch.rand(size, device=self.device) * (2.0 * dm) + (1.0 - dm)
            ) * self.dynamics.nominal_m
        else:
            self.mass_curr[reset] = self.dynamics.nominal_m

        if di > 0.0:
            self._Jx_t[reset] = (
                torch.rand(size, device=self.device) * (2.0 * di) + (1.0 - di)
            ) * self.dynamics.nominal_Jx
            self._Jy_t[reset] = (
                torch.rand(size, device=self.device) * (2.0 * di) + (1.0 - di)
            ) * self.dynamics.nominal_Jy
            self._Jz_t[reset] = (
                torch.rand(size, device=self.device) * (2.0 * di) + (1.0 - di)
            ) * self.dynamics.nominal_Jz
        else:
            self._Jx_t[reset] = self.dynamics.nominal_Jx
            self._Jy_t[reset] = self.dynamics.nominal_Jy
            self._Jz_t[reset] = self.dynamics.nominal_Jz

        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t, self._Jz_t)
        self.aero_force_scale[reset] = 1.0
        self._set_hover_controls(air_reset)

        if ground_size > 0:
            clearance = torch.full(
                (ground_size,), self.ground_spawn_clearance, device=self.device
            )
            if self.ground_spawn_static_equilibrium:
                static_penetration = (
                    self.mass_curr[ground_reset] * self.dynamics.g
                    / (
                        self.dynamics.ground_contact.number_of_points
                        * self.dynamics.ground_contact.stiffness
                    )
                )
                clearance = clearance - static_penetration
            self.s[ground_reset, 2] = self.dynamics.ground_contact.resting_cg_altitude(
                self.s[ground_reset], clearance=clearance
            )

        self.sync_reset_state(reset)

    def sync_reset_state(self, reset):
        """Synchronize caches after a task changes reset states for curriculum."""
        self.recent_s[reset] = self.s[reset]
        self.recent_u[reset] = self.u[reset]
        self.filtered_action[reset] = 0.0
        if not self.action_is_gazebo_control:
            self.filtered_action[reset, 0:5] = -1.0
        self._reset_ground_diagnostics(reset)
        self.filtered_action_valid[reset] = (
            self.enable_action_filter & self.on_ground[reset]
        )
        self.pusher_omega_cap[reset] = float('inf')
        self._mapped_pusher_omega_cap[reset] = float('inf')
        self._reset_gps(reset)
        self.aero_force_scale[reset] = 1.0

    def set_initial_actuators(self, mask, motor_omega, surfaces=None):
        """Set batched actuator state without applying motor-filter transients."""
        omega = torch.as_tensor(
            motor_omega, device=self.device, dtype=self.s.dtype
        )
        if omega.ndim == 1:
            omega = omega.reshape(1, 5).expand(self.n, 5)
        if omega.shape != (self.n, 5):
            raise ValueError('motor_omega must have shape [5] or [num_envs, 5]')
        self.motor_omega[mask] = omega[mask]
        self.u[mask, 0:5] = omega[mask]
        if surfaces is None:
            self.u[mask, 5:8] = 0.0
        else:
            surface_values = torch.as_tensor(
                surfaces, device=self.device, dtype=self.s.dtype
            )
            if surface_values.ndim == 1:
                surface_values = surface_values.reshape(1, 3).expand(self.n, 3)
            if surface_values.shape != (self.n, 3):
                raise ValueError('surfaces must have shape [3] or [num_envs, 3]')
            self.u[mask, 5:8] = surface_values[mask]

    def _sample_gps(self, mask):
        position = self.s[:, 0:3].clone()
        if self.gps_enabled:
            horizontal_noise = torch.randn(
                (self.n, 2), device=self.device, dtype=self.s.dtype
            ) * self.gps_horizontal_std
            vertical_noise = torch.randn(
                self.n, device=self.device, dtype=self.s.dtype
            ) * self.gps_vertical_std
            position[:, 0:2] += horizontal_noise
            position[:, 2] += vertical_noise
        self.gps_position[mask] = position[mask]
        self.gps_age_steps[mask] = 0
        self.gps_valid[mask] = True

    def _reset_gps(self, reset):
        self.gps_age_steps[reset] = 0
        self.gps_valid[reset] = False
        self._sample_gps(reset)

    def _update_gps(self):
        if not self.gps_enabled:
            return
        self.gps_age_steps += 1
        update = self.gps_age_steps >= self.gps_update_steps
        self._sample_gps(update)

    def _reset_ground_diagnostics(self, reset):
        self.just_touchdown[reset] = False
        self.touchdown_vertical_speed[reset] = 0.0
        self.last_touchdown_vertical_speed[reset] = 0.0
        self.contact_duration_steps[reset] = 0
        if not self.ground_contact_enabled:
            self.on_ground[reset] = False
            self.contact_count[reset] = 0
            self.max_ground_penetration[reset] = 0.0
            self.min_ground_clearance[reset] = 0.0
            self.ground_normal_force[reset] = 0.0
            self.ground_force_body[reset] = 0.0
            self.ground_moment_body[reset] = 0.0
            return

        force, moment, diagnostics = self.dynamics.get_ground_contact(self.s)
        self.on_ground[reset] = diagnostics['on_ground'][reset]
        self.contact_count[reset] = diagnostics['contact_count'][reset]
        self.contact_duration_steps[reset] = diagnostics['on_ground'][reset].long()
        self.max_ground_penetration[reset] = diagnostics['max_penetration'][reset]
        self.min_ground_clearance[reset] = diagnostics['min_clearance'][reset]
        self.ground_normal_force[reset] = diagnostics['total_normal_force'][reset]
        self.ground_force_body[reset] = force[reset]
        self.ground_moment_body[reset] = moment[reset]

    def _map_action(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        if self.enable_action_filter and self.action_filter_alpha < 1.0:
            alpha = self.action_filter_alpha
            valid = self.filtered_action_valid.unsqueeze(-1)
            action = torch.where(valid, (1.0 - alpha) * self.filtered_action + alpha * action, action)

        # Apply task-provided pusher limits after the command filter.  This is
        # deliberately in the actuator mapper: limiting only the incoming
        # policy action would allow a stale filtered command to exceed the
        # current transition envelope.
        pusher_cap = self.pusher_omega_cap
        self._mapped_pusher_omega_cap.copy_(pusher_cap)
        finite_cap = torch.isfinite(pusher_cap)
        if torch.any(finite_cap):
            if self.action_is_gazebo_control:
                normalized_cap = pusher_cap / self.motor_cmd_scaling[4]
            else:
                normalized_cap = (
                    2.0 * pusher_cap / self.motor_cmd_scaling[4] - 1.0
                )
            normalized_cap = normalized_cap.clamp(-1.0, 1.0)
            action[:, 4] = torch.where(
                finite_cap,
                torch.minimum(action[:, 4], normalized_cap),
                action[:, 4],
            )

        if self.enable_action_filter:
            # Keep the cache synchronized even when alpha=1 disables the
            # interpolation.  This is useful for diagnostics and ensures a
            # subsequent re-enable of filtering cannot resurrect a stale
            # command from before a phase-specific actuator cap.
            self.filtered_action = action.clone()
            self.filtered_action_valid[:] = True

        if self.action_is_gazebo_control:
            motor_cmd = action[:, 0:5].clamp(0.0, 1.0)
            surfaces = action[:, 5:8].clamp(-self.surface_limit, self.surface_limit)
        else:
            motor_cmd = (action[:, 0:5] + 1.0) * 0.5
            surfaces = action[:, 5:8] * self.surface_limit.reshape(1, 3)

        omega_ref = motor_cmd * self.motor_cmd_scaling.reshape(1, 5)
        omega_ref = torch.minimum(omega_ref, self.motor_omega_max.reshape(1, 5))
        omega_ref[:, 4] = torch.minimum(
            omega_ref[:, 4], self.pusher_omega_cap
        )
        # The cap is one-step state supplied by the task.  Clear it after the
        # mapping so a direct model update cannot inherit a previous phase's
        # limit.
        self.pusher_omega_cap.fill_(float('inf'))
        return omega_ref, surfaces

    def _update_motor_filter(self, omega_ref):
        tau = torch.where(omega_ref > self.motor_omega,
                          torch.full_like(omega_ref, self.time_constant_up),
                          torch.full_like(omega_ref, self.time_constant_down))
        alpha = torch.exp(-self.dt / tau)
        self.motor_omega = alpha * self.motor_omega + (1.0 - alpha) * omega_ref
        finite_cap = torch.isfinite(self._mapped_pusher_omega_cap)
        if torch.any(finite_cap):
            self.motor_omega[:, 4] = torch.where(
                finite_cap,
                torch.minimum(
                    self.motor_omega[:, 4],
                    self._mapped_pusher_omega_cap,
                ),
                self.motor_omega[:, 4],
            )
        self._mapped_pusher_omega_cap.fill_(float('inf'))

    def get_extended_state(self):
        return self.dynamics.nlplant(torch.hstack((self.s, self.u)))

    def update(self, action):
        if action.shape[1] < 8:
            pad = torch.zeros((action.shape[0], 8 - action.shape[1]), device=action.device, dtype=action.dtype)
            action = torch.hstack((action, pad))
        omega_ref, surfaces = self._map_action(action[:, :8])
        self._update_motor_filter(omega_ref)

        self.recent_u = self.u.clone()
        self.u[:, 0:5] = self.motor_omega
        self.u[:, 5:8] = surfaces
        self.recent_s = self.s.clone()
        if self.ground_contact_enabled:
            self._update_with_ground_contact()
        else:
            self.s = odeint(self.dynamics,
                            torch.hstack((self.s, self.u)),
                            torch.tensor([0., self.dt], device=self.device),
                            method=self.solver)[1, :, :self.num_states]
        self._update_gps()

    def _update_with_ground_contact(self):
        """Semi-implicit integration with contact evaluated at every substep."""
        substep_dt = self.dt / self.ground_physics_substeps
        previous_on_ground = self.on_ground.clone()
        contact_during_step = torch.zeros_like(self.on_ground)
        max_penetration = torch.zeros_like(self.max_ground_penetration)
        max_normal_force = torch.zeros_like(self.ground_normal_force)
        impact_speed = torch.zeros_like(self.touchdown_vertical_speed)

        for _ in range(self.ground_physics_substeps):
            _, _, diagnostics = self.dynamics.get_ground_contact(self.s)
            contact_during_step |= diagnostics['on_ground']
            max_penetration = torch.maximum(
                max_penetration, diagnostics['max_penetration']
            )
            max_normal_force = torch.maximum(
                max_normal_force, diagnostics['total_normal_force']
            )
            impact_speed = torch.maximum(
                impact_speed, diagnostics['max_downward_speed']
            )

            xdot = self.dynamics.nlplant(torch.hstack((self.s, self.u)))
            next_state = self.s.clone()
            next_state[:, 6:12] = (
                self.s[:, 6:12] + substep_dt * xdot[:, 6:12]
            )
            next_state[:, 0:6] = (
                self.s[:, 0:6]
                + substep_dt * self.dynamics.kinematic_derivatives(next_state)
            )
            self.s = next_state

        force, moment, final_diagnostics = self.dynamics.get_ground_contact(self.s)
        contact_during_step |= final_diagnostics['on_ground']
        max_penetration = torch.maximum(
            max_penetration, final_diagnostics['max_penetration']
        )
        max_normal_force = torch.maximum(
            max_normal_force, final_diagnostics['total_normal_force']
        )
        impact_speed = torch.maximum(
            impact_speed, final_diagnostics['max_downward_speed']
        )

        self.just_touchdown = contact_during_step & ~previous_on_ground
        self.on_ground = final_diagnostics['on_ground']
        self.contact_count = final_diagnostics['contact_count']
        self.contact_duration_steps = torch.where(
            self.on_ground,
            self.contact_duration_steps + 1,
            torch.zeros_like(self.contact_duration_steps),
        )
        self.max_ground_penetration = max_penetration
        self.min_ground_clearance = final_diagnostics['min_clearance']
        self.ground_normal_force = max_normal_force
        self.touchdown_vertical_speed = torch.where(
            self.just_touchdown, impact_speed, torch.zeros_like(impact_speed)
        )
        self.last_touchdown_vertical_speed = torch.where(
            self.just_touchdown,
            impact_speed,
            self.last_touchdown_vertical_speed,
        )
        self.ground_force_body = force
        self.ground_moment_body = moment

    def get_state(self):
        return self.s

    def get_control(self):
        return self.u

    def get_motor_omega(self):
        return self.u[:, 0], self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]

    def get_motor_thrusts(self):
        _ = self.get_extended_state()
        thrust = self.dynamics.get_last_motor_thrust(self.n, self.device)
        return thrust[:, 0], thrust[:, 1], thrust[:, 2], thrust[:, 3], thrust[:, 4]

    def get_F(self):
        return self.get_motor_thrusts()

    def get_force_body(self):
        _ = self.get_extended_state()
        force = self.dynamics.get_last_force_body(self.n, self.device)
        return force[:, 0], force[:, 1], force[:, 2]

    def get_moment_body(self):
        _ = self.get_extended_state()
        moment = self.dynamics.get_last_moment_body(self.n, self.device)
        return moment[:, 0], moment[:, 1], moment[:, 2]

    def get_ground_force_body(self):
        return (
            self.ground_force_body[:, 0],
            self.ground_force_body[:, 1],
            self.ground_force_body[:, 2],
        )

    def get_ground_moment_body(self):
        return (
            self.ground_moment_body[:, 0],
            self.ground_moment_body[:, 1],
            self.ground_moment_body[:, 2],
        )

    def get_ground_contact_state(self):
        return {
            'on_ground': self.on_ground,
            'just_touchdown': self.just_touchdown,
            'contact_count': self.contact_count,
            'contact_duration_steps': self.contact_duration_steps,
            'max_penetration': self.max_ground_penetration,
            'min_clearance': self.min_ground_clearance,
            'normal_force': self.ground_normal_force,
            'touchdown_vertical_speed': self.touchdown_vertical_speed,
            'last_touchdown_vertical_speed': self.last_touchdown_vertical_speed,
            'force_body': self.ground_force_body,
            'moment_body': self.ground_moment_body,
        }

    def get_gps_position(self):
        if not self.gps_enabled:
            return self.get_position()
        return (
            self.gps_position[:, 0],
            self.gps_position[:, 1],
            self.gps_position[:, 2],
        )

    def get_gps_state(self):
        if not self.gps_enabled:
            position = self.s[:, 0:3]
            age = torch.zeros(self.n, device=self.device, dtype=self.s.dtype)
            valid = torch.ones(self.n, device=self.device, dtype=torch.bool)
        else:
            position = self.gps_position
            age = self.gps_age_steps.to(self.s.dtype) * self.dt
            valid = self.gps_valid
        return {
            'position': position,
            'age': age,
            'valid': valid,
            'update_rate_hz': self.gps_update_rate_hz,
        }

    def get_position(self):
        return self.s[:, 0], self.s[:, 1], self.s[:, 2]

    def get_world_velocity(self):
        kinematics = self.dynamics.kinematic_derivatives(self.s)
        return kinematics[:, 0], kinematics[:, 1], kinematics[:, 2]

    def get_ground_speed(self):
        vel_n, vel_e, _ = self.get_world_velocity()
        return vel_n, vel_e

    def get_climb_rate(self):
        _, _, vel_up = self.get_world_velocity()
        return vel_up

    def get_posture(self):
        return self.s[:, 3], self.s[:, 4], self.s[:, 5]

    def get_euler_angular_velocity(self):
        kinematics = self.dynamics.kinematic_derivatives(self.s)
        return kinematics[:, 3], kinematics[:, 4], kinematics[:, 5]

    def get_vt(self):
        U, V, W = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        return torch.sqrt((U * U + V * V + W * W).clamp(min=0.0))

    def get_TAS(self):
        return self.get_vt() + self.airspeed * torch.ones(self.n, device=self.device)

    def get_EAS(self):
        return self.get_TAS() / self.get_EAS2TAS()

    def get_AOA(self):
        U = self.s[:, 6]
        W = self.s[:, 8]
        vt2 = U * U + self.s[:, 7] * self.s[:, 7] + W * W
        alpha = torch.atan2(W, U)
        return torch.where(vt2 > 1e-4, alpha, torch.zeros_like(alpha))

    def get_AOS(self):
        U = self.s[:, 6]
        V = self.s[:, 7]
        W = self.s[:, 8]
        vxz = torch.sqrt((U * U + W * W).clamp(min=0.0))
        vt2 = vxz * vxz + V * V
        beta = torch.atan2(V, vxz)
        return torch.where(vt2 > 1e-4, beta, torch.zeros_like(beta))

    def get_aero_sincos(self):
        U = self.s[:, 6]
        V = self.s[:, 7]
        W = self.s[:, 8]
        vxz2 = U * U + W * W
        vt2 = vxz2 + V * V
        vxz = torch.sqrt(vxz2.clamp(min=0.0))
        inv_vxz = torch.rsqrt(vxz2.clamp(min=1e-6))
        inv_vt = torch.rsqrt(vt2.clamp(min=1e-6))
        aero_on = vt2 > 0.25
        sa = torch.where(aero_on, W * inv_vxz, torch.zeros_like(W))
        ca = torch.where(aero_on, U * inv_vxz, torch.zeros_like(U))
        sb = torch.where(aero_on, V * inv_vt, torch.zeros_like(V))
        cb = torch.where(aero_on, vxz * inv_vt, torch.zeros_like(vxz))
        return sa, ca, sb, cb

    def get_angular_velocity(self):
        return self.s[:, 9], self.s[:, 10], self.s[:, 11]

    def get_thrust(self):
        _ = self.get_extended_state()
        thrust = self.dynamics.get_last_motor_thrust(self.n, self.device)
        return torch.sum(thrust[:, 0:4], dim=1)

    def get_control_surface(self):
        return self.u[:, 5], self.u[:, 6], self.u[:, 7], torch.zeros_like(self.u[:, 7])

    def get_velocity(self):
        return self.s[:, 6], self.s[:, 7], self.s[:, 8]

    def get_acceleration(self):
        xdot = self.get_extended_state()
        return xdot[:, 6], xdot[:, 7], xdot[:, 8]

    def get_G(self):
        nx_cg, ny_cg, nz_cg = self.get_accels()
        return torch.sqrt(nx_cg * nx_cg + ny_cg * ny_cg + nz_cg * nz_cg)

    def get_EAS2TAS(self):
        alt = self.s[:, 2]
        tfac = (1.0 - alt / 44330.0).clamp(min=0.1)
        eas2tas = 1.0 / torch.pow(tfac, 4.255)
        return torch.sqrt(eas2tas)

    def get_accels(self):
        grav = self.dynamics.g
        xdot = self.get_extended_state()
        U, V, W = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        P, Q, R = self.s[:, 9], self.s[:, 10], self.s[:, 11]
        Udot, Vdot, Wdot = xdot[:, 6], xdot[:, 7], xdot[:, 8]
        nx_cg = (Udot + Q * W - R * V) / grav + torch.sin(self.s[:, 4])
        ny_cg = ((Vdot + R * U - P * W) / grav
                 - torch.cos(self.s[:, 4]) * torch.sin(self.s[:, 3]))
        nz_cg = (-(Wdot + P * V - Q * U) / grav
                 + torch.cos(self.s[:, 4]) * torch.cos(self.s[:, 3]))
        return nx_cg, ny_cg, nz_cg
