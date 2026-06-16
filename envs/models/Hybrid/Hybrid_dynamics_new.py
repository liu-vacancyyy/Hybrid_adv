import torch

from Hybrid.Hybrid_dynamics import HybridDynamics


class HybridDynamicsNew(HybridDynamics):
    """Hybrid dynamics with wind-aware aerodynamics.

    State velocity ``U,V,W`` remains the aircraft ground/inertial velocity
    resolved in the body frame.  Aerodynamic force is computed from relative
    air velocity:

        V_air_body = V_ground_body - R_ned_to_body * wind_ned

    With zero wind this reduces to ``HybridDynamics``.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self._wind_ned_t = None
        self._wind_pqr_body_t = None
        self._last_wind_force_body = None
        self._last_wind_moment_body = None

        self.wind_drag_enable = bool(getattr(config, 'wind_drag_enable', True))
        self.wind_pqr_torque_enable = bool(getattr(config, 'wind_pqr_torque_enable', True))
        self.wind_body_cda = (
            float(getattr(config, 'wind_body_cda_x', 0.10)),
            float(getattr(config, 'wind_body_cda_y', 0.18)),
            float(getattr(config, 'wind_body_cda_z', 0.14)),
        )
        self.wind_cp_body = (
            float(getattr(config, 'wind_cp_x', 0.02)),
            float(getattr(config, 'wind_cp_y', 0.0)),
            float(getattr(config, 'wind_cp_z', -0.06)),
        )
        self.wind_pqr_accel_gain = (
            float(getattr(config, 'wind_pqr_accel_gain_p', 0.8)),
            float(getattr(config, 'wind_pqr_accel_gain_q', 0.8)),
            float(getattr(config, 'wind_pqr_accel_gain_r', 0.6)),
        )
        self.wind_force_clip = float(getattr(config, 'wind_force_clip', 25.0))
        self.wind_moment_clip = float(getattr(config, 'wind_moment_clip', 5.0))

    def set_wind_ned(self, wind_ned):
        """Set wind velocity in NED/world axes, in m/s.

        Positive components mean air mass velocity toward north/east/down.
        Shape can be ``(3,)`` or ``(n, 3)``.
        """
        if wind_ned is None:
            self._wind_ned_t = None
            return
        self._wind_ned_t = wind_ned

    def set_wind_pqr_body(self, wind_pqr_body):
        """Set angular-rate gust in body axes, rad/s."""
        if wind_pqr_body is None:
            self._wind_pqr_body_t = None
            return
        self._wind_pqr_body_t = wind_pqr_body

    @staticmethod
    def _wind_body_from_ned(wind_ned, phi, theta, psi):
        wn = wind_ned[:, 0]
        we = wind_ned[:, 1]
        wd = wind_ned[:, 2]

        st = torch.sin(theta)
        ct = torch.cos(theta)
        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        # HybridDynamics stores altitude as positive-up, while Dryden/PX4-style
        # wind uses NED down-positive. Convert wd to altitude-up before applying
        # the transpose of the body->world kinematic matrix.
        wx_b = ct * cpsi * wn + ct * spsi * we - st * wd
        wy_b = ((sphi * cpsi * st - cphi * spsi) * wn
                + (sphi * spsi * st + cphi * cpsi) * we
                + sphi * ct * wd)
        wz_b = ((cphi * st * cpsi + sphi * spsi) * wn
                + (cphi * st * spsi - sphi * cpsi) * we
                + cphi * ct * wd)
        return wx_b, wy_b, wz_b

    @staticmethod
    def _aero_inputs(U, V, W):
        vt2 = U * U + V * V + W * W
        vxz2 = U * U + W * W
        vxz = torch.sqrt(vxz2.clamp(min=0))
        inv_vxz = torch.rsqrt(vxz2.clamp(min=1e-6))
        inv_vt = torch.rsqrt(vt2.clamp(min=1e-6))
        aero_on = vt2 > 0.25
        sa = torch.where(aero_on, W * inv_vxz, torch.zeros_like(W))
        ca = torch.where(aero_on, U * inv_vxz, torch.zeros_like(U))
        sb = torch.where(aero_on, V * inv_vt, torch.zeros_like(V))
        cb = torch.where(aero_on, vxz * inv_vt, torch.zeros_like(vxz))
        return sa, ca, sb, cb, vt2

    def _wind_ned_for_state(self, x):
        if self._wind_ned_t is None:
            return torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        wind = self._wind_ned_t.to(device=x.device, dtype=x.dtype)
        if wind.ndim == 1:
            wind = wind.reshape(1, 3).expand(x.shape[0], 3)
        return wind

    def _wind_pqr_for_state(self, x):
        if self._wind_pqr_body_t is None:
            return torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        pqr = self._wind_pqr_body_t.to(device=x.device, dtype=x.dtype)
        if pqr.ndim == 1:
            pqr = pqr.reshape(1, 3).expand(x.shape[0], 3)
        return pqr

    @staticmethod
    def _air_density(alt):
        rho0 = 1.226
        return rho0 * torch.pow((1. - alt / 44330).clamp(min=0), 4.255)

    def _body_drag_force(self, U, V, W, alt):
        vel = torch.stack((U, V, W), dim=1)
        cda = vel.new_tensor(self.wind_body_cda).reshape(1, 3)
        rho = self._air_density(alt).reshape(-1, 1)
        return -0.5 * rho * cda * vel * torch.abs(vel)

    def _physics_terms(self, x):
        n = x.shape[0]
        dtype = x.dtype
        device = x.device
        if self._Jx_t is not None:
            Jx = self._Jx_t.to(device=device, dtype=dtype)
            Jy = self._Jy_t.to(device=device, dtype=dtype)
            Jz = self._Jz_t.to(device=device, dtype=dtype)
            Jxz = self._Jxz_t.to(device=device, dtype=dtype)
            denom = self._denom_t.to(device=device, dtype=dtype)
        else:
            Jx = torch.full((n,), self.nominal_Jx, device=device, dtype=dtype)
            Jy = torch.full((n,), self.nominal_Jy, device=device, dtype=dtype)
            Jz = torch.full((n,), self.nominal_Jz, device=device, dtype=dtype)
            Jxz = torch.full((n,), self.nominal_Jxz, device=device, dtype=dtype)
            denom = Jx * Jz - Jxz * Jxz
        return Jx, Jy, Jz, Jxz, denom

    def _pqr_gust_moment(self, wind_pqr, x):
        if not self.wind_pqr_torque_enable:
            return torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
        Jx, Jy, Jz, _Jxz, _denom = self._physics_terms(x)
        gain = x.new_tensor(self.wind_pqr_accel_gain).reshape(1, 3)
        inertia = torch.stack((Jx, Jy, Jz), dim=1)
        return inertia * gain * wind_pqr

    def _add_body_moment(self, xdot, moment, x):
        Jx, Jy, Jz, Jxz, denom = self._physics_terms(x)
        xdot[:, 9] = xdot[:, 9] + (Jz * moment[:, 0] + Jxz * moment[:, 2]) / denom
        xdot[:, 10] = xdot[:, 10] + moment[:, 1] / Jy
        xdot[:, 11] = xdot[:, 11] + (Jxz * moment[:, 0] + Jx * moment[:, 2]) / denom

    def get_last_wind_force_body(self, n, device):
        if self._last_wind_force_body is None:
            return torch.zeros((n, 3), device=device)
        return self._last_wind_force_body.to(device=device)

    def get_last_wind_moment_body(self, n, device):
        if self._last_wind_moment_body is None:
            return torch.zeros((n, 3), device=device)
        return self._last_wind_moment_body.to(device=device)

    def nlplant(self, x):
        # Start from the old no-wind dynamics, then replace only the
        # aerodynamic force contribution in Udot/Vdot/Wdot and add equivalent
        # wind-induced body drag / moments.
        xdot = super().nlplant(x)

        if self._wind_ned_t is None and self._wind_pqr_body_t is None:
            zeros = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
            self._last_wind_force_body = zeros.detach()
            self._last_wind_moment_body = zeros.detach()
            return xdot
        wind_ned = self._wind_ned_for_state(x)
        wind_pqr = self._wind_pqr_for_state(x)

        alt = x[:, 2]
        phi = x[:, 3]
        theta = x[:, 4]
        psi = x[:, 5]
        U = x[:, 6]
        V = x[:, 7]
        W = x[:, 8]

        wx_b, wy_b, wz_b = self._wind_body_from_ned(wind_ned, phi, theta, psi)
        U_air = U - wx_b
        V_air = V - wy_b
        W_air = W - wz_b

        sa_old, ca_old, sb_old, cb_old, vt2_old = self._aero_inputs(U, V, W)
        sa_new, ca_new, sb_new, cb_new, vt2_new = self._aero_inputs(U_air, V_air, W_air)

        F_old = self.wing.compute_aerodynamics(sa_old, ca_old, sb_old, cb_old, vt2_old, alt)
        F_new = self.wing.compute_aerodynamics(sa_new, ca_new, sb_new, cb_new, vt2_new, alt)
        dF = torch.stack((
            F_new[0] - F_old[0],
            F_new[1] - F_old[1],
            F_new[2] - F_old[2],
        ), dim=1)

        dM = torch.zeros_like(dF)
        if self.wind_drag_enable:
            drag_old = self._body_drag_force(U, V, W, alt)
            drag_new = self._body_drag_force(U_air, V_air, W_air, alt)
            dF_drag = drag_new - drag_old
            dF = dF + dF_drag
            cp = x.new_tensor(self.wind_cp_body).reshape(1, 3).expand_as(dF_drag)
            dM = dM + torch.cross(cp, dF_drag, dim=1)

        dM = dM + self._pqr_gust_moment(wind_pqr, x)

        if self.wind_force_clip > 0.0:
            dF = torch.clamp(dF, -self.wind_force_clip, self.wind_force_clip)
        if self.wind_moment_clip > 0.0:
            dM = torch.clamp(dM, -self.wind_moment_clip, self.wind_moment_clip)

        if self._mass_inv_t is not None:
            m_inv = self._mass_inv_t.to(device=x.device, dtype=x.dtype)
        else:
            m_inv = 1.0 / self.m

        xdot[:, 6] = xdot[:, 6] + dF[:, 0] * m_inv
        xdot[:, 7] = xdot[:, 7] + dF[:, 1] * m_inv
        xdot[:, 8] = xdot[:, 8] + dF[:, 2] * m_inv
        self._add_body_moment(xdot, dM, x)

        self._last_wind_force_body = dF.detach()
        self._last_wind_moment_body = dM.detach()
        return xdot
