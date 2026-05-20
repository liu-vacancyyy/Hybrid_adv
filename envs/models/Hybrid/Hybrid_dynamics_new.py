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

    def __init__(self):
        super().__init__()
        self._wind_ned_t = None

    def set_wind_ned(self, wind_ned):
        """Set wind velocity in NED/world axes, in m/s.

        Positive components mean air mass velocity toward north/east/down.
        Shape can be ``(3,)`` or ``(n, 3)``.
        """
        if wind_ned is None:
            self._wind_ned_t = None
            return
        self._wind_ned_t = wind_ned

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

    def nlplant(self, x):
        # Start from the old no-wind dynamics, then replace only the
        # aerodynamic force contribution in Udot/Vdot/Wdot.
        xdot = super().nlplant(x)

        if self._wind_ned_t is None:
            return xdot
        wind_ned = self._wind_ned_for_state(x)

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

        if self._mass_inv_t is not None:
            m_inv = self._mass_inv_t.to(device=x.device, dtype=x.dtype)
        else:
            m_inv = 1.0 / self.m

        xdot[:, 6] = xdot[:, 6] + (F_new[0] - F_old[0]) * m_inv
        xdot[:, 7] = xdot[:, 7] + (F_new[1] - F_old[1]) * m_inv
        xdot[:, 8] = xdot[:, 8] + (F_new[2] - F_old[2]) * m_inv
        return xdot
