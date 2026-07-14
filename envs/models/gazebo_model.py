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

        self.s = torch.zeros((self.n, self.num_states), device=self.device)
        self.recent_s = torch.zeros((self.n, self.num_states), device=self.device)
        self.u = torch.zeros((self.n, self.num_controls), device=self.device)
        self.recent_u = torch.zeros((self.n, self.num_controls), device=self.device)

        self.dynamics = GazeboVTOLDynamics(config)

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
        self.surface_limit = torch.tensor([0.53, 0.53, 0.53], device=self.device)
        self.motor_omega = torch.zeros((self.n, 5), device=self.device)

        self.dr_mass = float(getattr(self.config, 'dr_mass', 0.0))
        self.dr_inertia = float(getattr(self.config, 'dr_inertia', 0.0))
        self.mass_curr = torch.ones(self.n, device=self.device) * self.dynamics.nominal_m
        self._Jx_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jx
        self._Jy_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jy
        self._Jz_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jz
        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t, self._Jz_t)
        self._set_hover_controls(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def _u(self, size, half_range):
        if half_range <= 0.0:
            return torch.zeros(size, device=self.device)
        return (torch.rand(size, device=self.device) * 2.0 - 1.0) * half_range

    def _set_hover_controls(self, mask):
        count = int(mask.sum().item())
        if count == 0:
            return
        hover_omega = torch.sqrt((self.mass_curr[mask] * self.dynamics.g / 4.0 / 2.0e-5).clamp(min=0.0))
        self.motor_omega[mask, :] = 0.0
        self.motor_omega[mask, 0:4] = hover_omega.reshape(-1, 1)
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

        self.s[reset, 2] = (torch.rand(size, device=self.device)
                            * (self.max_altitude - self.min_altitude) + self.min_altitude)
        self.s[reset, 3] = self._u(size, self.init_roll_range)
        self.s[reset, 4] = self._u(size, self.init_pitch_range)
        self.s[reset, 5] = self._u(size, self.init_yaw_range)
        for k in range(3):
            self.s[reset, 6 + k] = self._u(size, self.init_vel_range)
            self.s[reset, 9 + k] = self._u(size, self.init_omega_range)

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
        self._set_hover_controls(reset)

        self.recent_s[reset] = self.s[reset]
        self.filtered_action[reset] = 0.0
        self.filtered_action_valid[reset] = False

    def _map_action(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        if self.enable_action_filter and self.action_filter_alpha < 1.0:
            alpha = self.action_filter_alpha
            valid = self.filtered_action_valid.unsqueeze(-1)
            action = torch.where(valid, (1.0 - alpha) * self.filtered_action + alpha * action, action)
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
        return omega_ref, surfaces

    def _update_motor_filter(self, omega_ref):
        tau = torch.where(omega_ref > self.motor_omega,
                          torch.full_like(omega_ref, self.time_constant_up),
                          torch.full_like(omega_ref, self.time_constant_down))
        alpha = torch.exp(-self.dt / tau)
        self.motor_omega = alpha * self.motor_omega + (1.0 - alpha) * omega_ref

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
        self.s = odeint(self.dynamics,
                        torch.hstack((self.s, self.u)),
                        torch.tensor([0., self.dt], device=self.device),
                        method=self.solver)[1, :, :self.num_states]

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

    def get_position(self):
        return self.s[:, 0], self.s[:, 1], self.s[:, 2]

    def get_ground_speed(self):
        es = self.get_extended_state()
        return es[:, 0], es[:, 1]

    def get_climb_rate(self):
        es = self.get_extended_state()
        return es[:, 2]

    def get_posture(self):
        return self.s[:, 3], self.s[:, 4], self.s[:, 5]

    def get_euler_angular_velocity(self):
        es = self.get_extended_state()
        return es[:, 3], es[:, 4], es[:, 5]

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
