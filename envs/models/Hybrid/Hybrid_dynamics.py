import torch
import torch.nn as nn
import numpy as np
import math

class Wing:
    def __init__(self, area, direction, angle0):
        self.area = area
        self.direction = direction
        self.angle0 = angle0

    def compute_aerodynamics(self, sa, ca, sb, cb, vt2, alt):
        """Compute body-frame aerodynamic force components directly from
        velocity-component ratios and speed-squared.  No angle computation;
        forces vanish naturally as vt->0 via the dynamic-pressure term (∝ vt²)
        — no low-speed guard is needed.

        sa, ca  = W/vxz, U/vxz  (sin/cos alpha, already safe-normalised)
        sb, cb  = V/vt,  vxz/vt (sin/cos beta,  already safe-normalised)
        vt2     = U² + V² + W²  (speed squared)
        """
        rho0 = 1.226
        air_rho = rho0 * torch.pow((1. - alt / 44330).clamp(min=0), 4.255)

        Cl = 2.0 * sa * ca          # = sin(2α)  — no trig call
        Cd = 2.0 * sa * sa + 0.1   # matches LearningToFly AerodynamicsDatabase::Cd

        # Dynamic pressure × area — goes to zero as vt→0, so no guard needed
        q = air_rho * vt2 * self.area / 2.0

        F_lift = Cl * q
        F_drag = Cd * q

        # Resolve to body axes (identical projection as before)
        F_aero_x =  F_lift * sa    - F_drag * ca * cb
        F_aero_y =                 - F_drag * sb
        F_aero_z = -F_lift * ca    - F_drag * sa * cb

        return F_aero_x, F_aero_y, F_aero_z

class HybridDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        # Nominal physics (scalars, used as fallback when per-env tensors not set)
        self.m           = 1.779   # kg
        self.nominal_m   = 1.779
        self.nominal_Jx  =  0.194
        self.nominal_Jy  =  0.064
        self.nominal_Jz  =  0.253
        self.nominal_Jxz = -0.007

        area = 0.326
        direction = torch.tensor([])
        wing_angle = 5
        self.wing = Wing(area, direction, wing_angle)

        # Per-env physics tensors (None → use scalar nominals above)
        # Shape (n,) when set via set_physics()
        self._mass_inv_t = None
        self._Jx_t       = None
        self._Jy_t       = None
        self._Jz_t       = None
        self._Jxz_t      = None
        self._denom_t    = None

    def set_physics(self, mass_t, Jx_t, Jy_t, Jz_t, Jxz_t):
        """Store per-env physics tensors (shape (n,)) for vectorised domain
        randomisation.  Call once per episode reset; nlplant() will pick them up.
        """
        self._mass_inv_t = 1.0 / mass_t
        self._Jx_t    = Jx_t
        self._Jy_t    = Jy_t
        self._Jz_t    = Jz_t
        self._Jxz_t   = Jxz_t
        self._denom_t = Jx_t * Jz_t - Jxz_t * Jxz_t
    
    def compute_extended_state(self, x):
        return self.nlplant(x)

    def forward(self, t, x):
        es = self.compute_extended_state(x)
        return es
    
    def nlplant(self, x):
        """
        model state(dim 12):
            0. ego_north_position      (unit: m)
            1. ego_east_position       (unit: m)
            2. ego_altitude            (unit: m)
            3. ego_roll                (unit: rad)
            4. ego_pitch               (unit: rad)
            5. ego_yaw                 (unit: rad)
            6. ego_U  (body-x velocity) (unit: m/s)
            7. ego_V  (body-y velocity) (unit: m/s)
            8. ego_W  (body-z velocity) (unit: m/s)
            9. ego_P                   (unit: rad/s)
            10. ego_Q                  (unit: rad/s)
            11. ego_R                  (unit: rad/s)

        model control(dim 5)
            0. head_rotor_input        (unit: N)
            1. lf_rotor_input          (unit: N)
            2. rf_rotor_input          (unit: N)
            3. lb_rotor_input          (unit: N)
            4. rb_rotor_input          (unit: N)
        """

        rotor_torque_coef = 0.01554
        positions = torch.tensor([[0.292, 0.000, 0.069],   
                                  [0.207, 0.305, -0.003],  
                                  [-0.260, -0.305, -0.003], 
                                  [0.210, -0.305, -0.003],   
                                  [-0.263, 0.305, -0.003]]).to(x.device, dtype=x.dtype)

        xdot = torch.zeros_like(x)
        
        g = 9.807
        pi = torch.pi
        r2d = 180.0/pi

        alt = x[:, 2]
        phi = x[:, 3] #roll
        theta = x[:, 4] #pitch
        psi = x[:, 5] #yaw

        # Body-frame Cartesian velocity (primary state, no singularity)
        U = x[:, 6]
        V = x[:, 7]
        W = x[:, 8]
        vt2   = U*U + V*V + W*W
        vxz2  = U*U + W*W
        vxz   = torch.sqrt(vxz2.clamp(min=0))

        # Trig equivalents as safe velocity-component ratios — no atan2.
        # Below vt_min (0.5 m/s) aerodynamics are physically negligible and
        # direction vectors are noise-dominated, so zero them out explicitly.
        inv_vxz   = torch.rsqrt(vxz2.clamp(min=1e-6))
        inv_vt    = torch.rsqrt(vt2.clamp(min=1e-6))
        _aero_on  = vt2 > 0.25   # vt > 0.5 m/s
        sa = torch.where(_aero_on, W * inv_vxz,  torch.zeros_like(W))    # sin(α)
        ca = torch.where(_aero_on, U * inv_vxz,  torch.zeros_like(U))    # cos(α)
        sb = torch.where(_aero_on, V * inv_vt,   torch.zeros_like(V))    # sin(β)
        cb = torch.where(_aero_on, vxz * inv_vt, torch.zeros_like(vxz))  # cos(β)

        P = x[:, 9]
        Q = x[:, 10]
        R = x[:, 11]

        st = torch.sin(theta)
        ct = torch.cos(theta)
        tt = torch.tan(theta)
        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        head_rotor_in = x[:, 12]
        rf_rotor_in = x[:, 13]
        lb_rotor_in = x[:, 14]
        lf_rotor_in = x[:, 15]
        rb_rotor_in = x[:, 16]

        F_z = lf_rotor_in + rf_rotor_in + lb_rotor_in + rb_rotor_in
        F_x = head_rotor_in

        # U/V/W already extracted from state above

        xdot[:, 0] = U * (ct * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi)
        xdot[:, 1] = U * (ct * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi)
        xdot[:, 2] = U * st - V * (sphi * ct) - W * (cphi * ct)
        xdot[:, 3] = P + tt * (Q * sphi + R * cphi)
        xdot[:, 4] = Q * cphi - R * sphi
        xdot[:, 5] = (Q * sphi + R * cphi) / ct

        F_aero_x, F_aero_y, F_aero_z = self.wing.compute_aerodynamics(
            sa, ca, sb, cb, vt2, alt)

        # Use per-env or nominal scalar physics
        if self._mass_inv_t is not None:
            m_inv = self._mass_inv_t
        else:
            m_inv = 1.0 / self.m

        Udot = R * V - Q * W - g * st + F_x * m_inv + F_aero_x * m_inv
        Vdot = P * W - R * U + g * ct * sphi          + F_aero_y * m_inv
        Wdot = Q * U - P * V + g * ct * cphi - F_z * m_inv + F_aero_z * m_inv
        # print(Q, U, P, V, g * ct * cphi, F_z / self.m, F_lift / self.m)
        # print('udot:',Udot)
        # print('vdot:',Vdot)
        # print('wdot:',Wdot)
        # xdot[6:9] = body-frame velocity derivatives (Udot, Vdot, Wdot)
        # Direct assignment — no back-conversion, no vt singularity
        xdot[:, 6] = Udot
        xdot[:, 7] = Vdot
        xdot[:, 8] = Wdot

        dir_head = torch.tensor([1., 0., 0.]).to(x.device, dtype=x.dtype)
        dir_lift = torch.tensor([0., 0., -1.]).to(x.device, dtype=x.dtype)
        rotor_torque = torch.zeros((sa.shape[0], 3)).to(x.device, dtype=x.dtype)
        # print((head_rotor_in.unsqueeze(-1) * dir_head).shape)
        rotor_torque += torch.cross(positions[0].unsqueeze(0), head_rotor_in.unsqueeze(-1) * dir_head)
        rotor_torque += torch.cross(positions[1].unsqueeze(0), rf_rotor_in.unsqueeze(-1) * dir_lift)
        rotor_torque += torch.cross(positions[2].unsqueeze(0), lb_rotor_in.unsqueeze(-1) * dir_lift)
        rotor_torque += torch.cross(positions[3].unsqueeze(0), lf_rotor_in.unsqueeze(-1) * dir_lift)
        rotor_torque += torch.cross(positions[4].unsqueeze(0), rb_rotor_in.unsqueeze(-1) * dir_lift)
        rotor_torque += 1.0 * rotor_torque_coef * head_rotor_in.unsqueeze(-1) * dir_head
        rotor_torque += -1.0 * rotor_torque_coef * rf_rotor_in.unsqueeze(-1) * dir_lift
        rotor_torque += -1.0 * rotor_torque_coef * lb_rotor_in.unsqueeze(-1) * dir_lift
        rotor_torque += 1.0 * rotor_torque_coef * lf_rotor_in.unsqueeze(-1) * dir_lift
        rotor_torque += 1.0 * rotor_torque_coef * rb_rotor_in.unsqueeze(-1) * dir_lift

        if self._Jx_t is not None:
            Jx    = self._Jx_t
            Jy    = self._Jy_t
            Jz    = self._Jz_t
            Jxz   = self._Jxz_t
            denom = self._denom_t
        else:
            Jx    = self.nominal_Jx
            Jy    = self.nominal_Jy
            Jz    = self.nominal_Jz
            Jxz   = self.nominal_Jxz
            denom = Jx * Jz - Jxz * Jxz
        xdot[:, 9]  = ((Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + Jxz * (Jx - Jy + Jz) * P * Q + Jz * rotor_torque[:, 0] + Jxz * rotor_torque[:, 2]) / denom
        xdot[:, 10] = ((Jz - Jx) * P * R - Jxz * (P * P - R * R) + rotor_torque[:, 1]) / Jy
        xdot[:, 11] = ((Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - Jxz * (Jx - Jy + Jz) * Q * R + Jxz * rotor_torque[:, 0] + Jx * rotor_torque[:, 2]) / denom

        return xdot