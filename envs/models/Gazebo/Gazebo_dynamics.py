import torch
import torch.nn as nn


class GazeboVTOLDynamics(nn.Module):
    """Gazebo Classic standard_vtol dynamics approximation.

    State:
        [north, east, altitude, roll, pitch, yaw, U, V, W, P, Q, R]

    Body axes follow the other Hybrid_adv models: x forward, y right, z down.
    Altitude is positive up. Control passed to nlplant is:
        [omega0, omega1, omega2, omega3, omega4,
         left_elevon, right_elevon, elevator]
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.g = float(getattr(config, 'gazebo_g', 9.807))
        self.mass = float(getattr(config, 'gazebo_mass', 5.04))
        self.nominal_m = self.mass
        self.nominal_Jx = float(getattr(config, 'gazebo_Jx', 0.477708333333))
        self.nominal_Jy = float(getattr(config, 'gazebo_Jy', 0.341666666667))
        self.nominal_Jz = float(getattr(config, 'gazebo_Jz', 0.811041666667))

        self.rho = float(getattr(config, 'gazebo_air_density', 1.2041))
        self.rotor_drag_coefficient = float(getattr(config, 'gazebo_rotor_drag_coefficient', 0.000106428))
        self.rolling_moment_coefficient = float(getattr(config, 'gazebo_rolling_moment_coefficient', 1.0e-6))

        self._mass_t = None
        self._Jx_t = None
        self._Jy_t = None
        self._Jz_t = None

        self._last_force_body = None
        self._last_moment_body = None
        self._last_motor_thrust = None
        self._last_aero_force_body = None
        self._last_aero_moment_body = None

        self.rotor_pos = (
            (0.35, 0.35, -0.07),
            (-0.35, -0.35, -0.07),
            (0.35, -0.35, -0.07),
            (-0.35, 0.35, -0.07),
            (-0.22, 0.0, 0.0),
        )
        self.rotor_axis = (
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
        )
        self.rotor_turning = (1.0, 1.0, -1.0, -1.0, -1.0)
        self.motor_constant = (2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 8.54858e-6)
        self.moment_constant = (0.06, 0.06, 0.06, 0.06, 0.01)
        self.max_relative_airspeed = (25.0, 25.0, 25.0, 25.0, 30.0)

        self.surfaces = (
            {
                'alpha0': 0.05984281113,
                'cla': 4.752798721,
                'cda': 0.6417112299,
                'cma': 0.0,
                'alpha_stall': 0.3391428111,
                'cla_stall': -3.85,
                'cda_stall': -0.9233984055,
                'cma_stall': 0.0,
                'cp': (-0.05, -0.30, -0.05),
                'area': 0.50,
                'forward': (1.0, 0.0, 0.0),
                'upward': (0.0, 0.0, -1.0),
                'control_index': 5,
                'control_joint_rad_to_cl': -1.0,
            },
            {
                'alpha0': 0.05984281113,
                'cla': 4.752798721,
                'cda': 0.6417112299,
                'cma': 0.0,
                'alpha_stall': 0.3391428111,
                'cla_stall': -3.85,
                'cda_stall': -0.9233984055,
                'cma_stall': 0.0,
                'cp': (-0.05, 0.30, -0.05),
                'area': 0.50,
                'forward': (1.0, 0.0, 0.0),
                'upward': (0.0, 0.0, -1.0),
                'control_index': 6,
                'control_joint_rad_to_cl': -1.0,
            },
            {
                'alpha0': -0.2,
                'cla': 4.752798721,
                'cda': 0.6417112299,
                'cma': 0.0,
                'alpha_stall': 0.3391428111,
                'cla_stall': -3.85,
                'cda_stall': -0.9233984055,
                'cma_stall': 0.0,
                'cp': (-0.5, 0.0, 0.0),
                'area': 0.01,
                'forward': (1.0, 0.0, 0.0),
                'upward': (0.0, 0.0, -1.0),
                'control_index': 7,
                'control_joint_rad_to_cl': -12.0,
            },
            {
                'alpha0': 0.0,
                'cla': 4.752798721,
                'cda': 0.6417112299,
                'cma': 0.0,
                'alpha_stall': 0.3391428111,
                'cla_stall': -3.85,
                'cda_stall': -0.9233984055,
                'cma_stall': 0.0,
                'cp': (-0.5, 0.0, -0.05),
                'area': 0.02,
                'forward': (1.0, 0.0, 0.0),
                'upward': (0.0, -1.0, 0.0),
                'control_index': None,
                'control_joint_rad_to_cl': 0.0,
            },
        )

    def set_physics(self, mass_t, Jx_t, Jy_t, Jz_t):
        self._mass_t = mass_t
        self._Jx_t = Jx_t
        self._Jy_t = Jy_t
        self._Jz_t = Jz_t

    def compute_extended_state(self, x):
        return self.nlplant(x)

    def forward(self, t, x):
        return self.compute_extended_state(x)

    @staticmethod
    def _vec(x, values):
        return x.new_tensor(values)

    @staticmethod
    def _normalize(v, eps=1e-6):
        return v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=eps)

    @staticmethod
    def _wrap_half_pi(alpha):
        pi = torch.pi
        return torch.remainder(alpha + 0.5 * pi, pi) - 0.5 * pi

    def _physics_terms(self, x):
        n = x.shape[0]
        if self._mass_t is None:
            mass = torch.full((n,), self.mass, device=x.device, dtype=x.dtype)
            Jx = torch.full((n,), self.nominal_Jx, device=x.device, dtype=x.dtype)
            Jy = torch.full((n,), self.nominal_Jy, device=x.device, dtype=x.dtype)
            Jz = torch.full((n,), self.nominal_Jz, device=x.device, dtype=x.dtype)
        else:
            mass = self._mass_t.to(device=x.device, dtype=x.dtype)
            Jx = self._Jx_t.to(device=x.device, dtype=x.dtype)
            Jy = self._Jy_t.to(device=x.device, dtype=x.dtype)
            Jz = self._Jz_t.to(device=x.device, dtype=x.dtype)
        return mass, Jx, Jy, Jz

    def _motor_forces_moments(self, x, wind_body):
        n = x.shape[0]
        omega = x[:, 12:17]
        v_body = x[:, 6:9]
        pqr = x[:, 9:12]

        rotor_pos = x.new_tensor(self.rotor_pos)
        rotor_axis = x.new_tensor(self.rotor_axis)
        sigma = x.new_tensor(self.rotor_turning)
        motor_constant = x.new_tensor(self.motor_constant)
        moment_constant = x.new_tensor(self.moment_constant)
        max_relative_airspeed = x.new_tensor(self.max_relative_airspeed)

        force_total = torch.zeros((n, 3), device=x.device, dtype=x.dtype)
        moment_total = torch.zeros((n, 3), device=x.device, dtype=x.dtype)
        motor_thrust = torch.zeros((n, 5), device=x.device, dtype=x.dtype)

        for i in range(5):
            r_i = rotor_pos[i].reshape(1, 3).expand(n, 3)
            axis_i = rotor_axis[i].reshape(1, 3).expand(n, 3)
            omega_i = omega[:, i]

            v_i = v_body + torch.cross(pqr, r_i, dim=1) - wind_body
            v_parallel_mag = torch.sum(v_i * axis_i, dim=1).abs()
            scalar = (1.0 - v_parallel_mag / max_relative_airspeed[i]).clamp(0.0, 1.0)

            thrust_i = (motor_constant[i] * omega_i * omega_i.abs()).abs() * scalar
            motor_thrust[:, i] = thrust_i
            force_i = thrust_i.reshape(-1, 1) * axis_i
            moment_arm_i = torch.cross(r_i, force_i, dim=1)
            reaction_i = -sigma[i] * moment_constant[i] * thrust_i.reshape(-1, 1) * axis_i

            v_perp = v_i - torch.sum(v_i * axis_i, dim=1, keepdim=True) * axis_i
            prop_drag_i = -omega_i.abs().reshape(-1, 1) * self.rotor_drag_coefficient * v_perp
            rolling_i = (-omega_i.abs().reshape(-1, 1)
                         * sigma[i]
                         * self.rolling_moment_coefficient
                         * v_perp)

            force_total = force_total + force_i + prop_drag_i
            moment_total = moment_total + moment_arm_i + reaction_i + rolling_i

        return force_total, moment_total, motor_thrust

    def _surface_force_moment(self, x, surface, wind_body):
        n = x.shape[0]
        cp = self._vec(x, surface['cp']).reshape(1, 3).expand(n, 3)
        forward = self._vec(x, surface['forward']).reshape(1, 3).expand(n, 3)
        upward = self._vec(x, surface['upward']).reshape(1, 3).expand(n, 3)

        v = x[:, 6:9] + torch.cross(x[:, 9:12], cp, dim=1) - wind_body
        speed = torch.linalg.norm(v, dim=1)
        active = (speed > 0.01) & (torch.sum(forward * v, dim=1) > 0.0)

        v_hat = v / speed.reshape(-1, 1).clamp(min=1e-6)
        span = self._normalize(torch.cross(forward, upward, dim=1))
        sin_sweep = torch.sum(span * v_hat, dim=1).clamp(-1.0, 1.0)
        cos_sweep = torch.sqrt((1.0 - sin_sweep * sin_sweep).clamp(min=0.0))

        v_ld = v - torch.sum(v * span, dim=1, keepdim=True) * span
        v_ld_norm = torch.linalg.norm(v_ld, dim=1)
        active = active & (v_ld_norm > 1e-6)

        drag_dir = -v_ld / v_ld_norm.reshape(-1, 1).clamp(min=1e-6)
        lift_dir = self._normalize(torch.cross(span, v_ld, dim=1))
        moment_dir = span

        cos_alpha = torch.sum(lift_dir * upward, dim=1).clamp(-1.0, 1.0)
        alpha_unsigned = torch.acos(cos_alpha)
        alpha = torch.where(
            torch.sum(lift_dir * forward, dim=1) >= 0.0,
            surface['alpha0'] + alpha_unsigned,
            surface['alpha0'] - alpha_unsigned,
        )
        alpha = self._wrap_half_pi(alpha)

        q = 0.5 * self.rho * v_ld_norm * v_ld_norm
        alpha_stall = surface['alpha_stall']

        cl_mid = surface['cla'] * alpha * cos_sweep
        cl_hi = (surface['cla'] * alpha_stall
                 + surface['cla_stall'] * (alpha - alpha_stall)) * cos_sweep
        cl_hi = torch.maximum(torch.zeros_like(cl_hi), cl_hi)
        cl_lo = (-surface['cla'] * alpha_stall
                 + surface['cla_stall'] * (alpha + alpha_stall)) * cos_sweep
        cl_lo = torch.minimum(torch.zeros_like(cl_lo), cl_lo)
        cl = torch.where(alpha > alpha_stall, cl_hi, torch.where(alpha < -alpha_stall, cl_lo, cl_mid))

        if surface['control_index'] is not None:
            cl = cl + surface['control_joint_rad_to_cl'] * x[:, surface['control_index']]

        cd_mid = surface['cda'] * alpha * cos_sweep
        cd_hi = (surface['cda'] * alpha_stall
                 + surface['cda_stall'] * (alpha - alpha_stall)) * cos_sweep
        cd_lo = (-surface['cda'] * alpha_stall
                 + surface['cda_stall'] * (alpha + alpha_stall)) * cos_sweep
        cd = torch.where(alpha > alpha_stall, cd_hi, torch.where(alpha < -alpha_stall, cd_lo, cd_mid)).abs()

        cm_mid = surface['cma'] * alpha * cos_sweep
        cm_hi = (surface['cma'] * alpha_stall
                 + surface['cma_stall'] * (alpha - alpha_stall)) * cos_sweep
        cm_hi = torch.maximum(torch.zeros_like(cm_hi), cm_hi)
        cm_lo = (-surface['cma'] * alpha_stall
                 + surface['cma_stall'] * (alpha + alpha_stall)) * cos_sweep
        cm_lo = torch.minimum(torch.zeros_like(cm_lo), cm_lo)
        cm = torch.where(alpha > alpha_stall, cm_hi, torch.where(alpha < -alpha_stall, cm_lo, cm_mid))

        force = (cl * q * surface['area']).reshape(-1, 1) * lift_dir
        force = force + (cd * q * surface['area']).reshape(-1, 1) * drag_dir
        moment = torch.cross(cp, force, dim=1)
        moment = moment + (cm * q * surface['area']).reshape(-1, 1) * moment_dir

        mask = active.reshape(-1, 1).to(dtype=x.dtype)
        return force * mask, moment * mask

    def _aero_forces_moments(self, x, wind_body):
        force_total = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        moment_total = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        for surface in self.surfaces:
            force_i, moment_i = self._surface_force_moment(x, surface, wind_body)
            force_total = force_total + force_i
            moment_total = moment_total + moment_i
        return force_total, moment_total

    def nlplant(self, x):
        xdot = torch.zeros_like(x)

        phi = x[:, 3]
        theta = x[:, 4]
        psi = x[:, 5]
        U = x[:, 6]
        V = x[:, 7]
        W = x[:, 8]
        P = x[:, 9]
        Q = x[:, 10]
        R = x[:, 11]

        st = torch.sin(theta)
        ct_raw = torch.cos(theta)
        ct = torch.where(ct_raw.abs() > 1e-6, ct_raw, torch.full_like(ct_raw, 1e-6))
        tt = torch.tan(theta)
        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        xdot[:, 0] = U * (ct_raw * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi)
        xdot[:, 1] = U * (ct_raw * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi)
        xdot[:, 2] = U * st - V * (sphi * ct_raw) - W * (cphi * ct_raw)
        xdot[:, 3] = P + tt * (Q * sphi + R * cphi)
        xdot[:, 4] = Q * cphi - R * sphi
        xdot[:, 5] = (Q * sphi + R * cphi) / ct

        wind_body = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        motor_force, motor_moment, motor_thrust = self._motor_forces_moments(x, wind_body)
        aero_force, aero_moment = self._aero_forces_moments(x, wind_body)
        force_body = motor_force + aero_force
        moment_body = motor_moment + aero_moment

        mass, Jx, Jy, Jz = self._physics_terms(x)

        xdot[:, 6] = R * V - Q * W - self.g * st + force_body[:, 0] / mass
        xdot[:, 7] = P * W - R * U + self.g * ct_raw * sphi + force_body[:, 1] / mass
        xdot[:, 8] = Q * U - P * V + self.g * ct_raw * cphi + force_body[:, 2] / mass

        xdot[:, 9] = (moment_body[:, 0] - (Jz - Jy) * Q * R) / Jx
        xdot[:, 10] = (moment_body[:, 1] - (Jx - Jz) * P * R) / Jy
        xdot[:, 11] = (moment_body[:, 2] - (Jy - Jx) * P * Q) / Jz

        self._last_force_body = force_body.detach()
        self._last_moment_body = moment_body.detach()
        self._last_motor_thrust = motor_thrust.detach()
        self._last_aero_force_body = aero_force.detach()
        self._last_aero_moment_body = aero_moment.detach()
        return xdot

    def get_last_force_body(self, n, device):
        if self._last_force_body is None:
            return torch.zeros((n, 3), device=device)
        return self._last_force_body.to(device=device)

    def get_last_moment_body(self, n, device):
        if self._last_moment_body is None:
            return torch.zeros((n, 3), device=device)
        return self._last_moment_body.to(device=device)

    def get_last_motor_thrust(self, n, device):
        if self._last_motor_thrust is None:
            return torch.zeros((n, 5), device=device)
        return self._last_motor_thrust.to(device=device)
