import os
import sys
import torch
from torchdiffeq import odeint_adjoint as odeint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_base import BaseModel
from Hybrid.Hybrid_dynamics import HybridDynamics


class HybridModel(BaseModel):
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.num_states = getattr(self.config, 'num_states', 12)
        self.num_controls = getattr(self.config, 'num_controls', 5)
        self.dt = getattr(self.config, 'dt', 0.02)
        self.solver = getattr(self.config, 'solver', 'euler')
        self.airspeed = getattr(self.config, 'airspeed', 0)

        self.s = torch.zeros((self.n, self.num_states), device=self.device)  # state
        self.recent_s = torch.zeros((self.n, self.num_states), device=self.device)  # recent state
        self.u = torch.zeros((self.n, self.num_controls), device=self.device) # control
        self.recent_u = torch.zeros((self.n, self.num_controls), device=self.device)  # recent control

        # init parameters
        self.max_altitude = getattr(self.config, 'max_altitude', 50.0)
        self.min_altitude = getattr(self.config, 'min_altitude', 4.0)
        self.max_vt = 2.5
        self.min_vt = 0
        self.max_F = 7.
        self.motor_constrain = 0.4
        self.enable_action_filter = bool(getattr(self.config, 'enable_action_filter', False))
        self.action_filter_alpha = float(getattr(self.config, 'action_filter_alpha', 1.0))
        self.action_filter_alpha = min(max(self.action_filter_alpha, 0.0), 1.0)
        self.filtered_action = torch.zeros((self.n, self.num_controls), device=self.device)
        self.filtered_action_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        # ---- Domain randomisation parameters ----
        # Mass / inertia: each reset multiplies nominal by U(1-dr, 1+dr).
        self.dr_mass    = getattr(self.config, 'dr_mass',    0.05)
        self.dr_inertia = getattr(self.config, 'dr_inertia', 0.20)

        # Initial-state perturbation: per-axis UNIFORM with half-range r,
        # i.e. value ~ U(-r, +r).  xy is always 0; z ~ U(min_alt, max_alt).
        import math as _math
        self.init_roll_range  = getattr(self.config, 'init_roll_range',  0.01 * _math.pi)
        self.init_pitch_range = getattr(self.config, 'init_pitch_range', 0.01 * _math.pi)
        self.init_yaw_range   = getattr(self.config, 'init_yaw_range',         _math.pi)
        self.init_vel_range   = getattr(self.config, 'init_vel_range',   0.1)
        self.init_omega_range = getattr(self.config, 'init_omega_range', 0.02)

        self.dynamics = HybridDynamics()

        # Per-env physical state (initialised to nominal; updated each reset by DR)
        _dyn = self.dynamics
        self.mass_curr = torch.ones(self.n, device=self.device) * _dyn.nominal_m
        self._Jx_t     = torch.ones(self.n, device=self.device) * _dyn.nominal_Jx
        self._Jy_t     = torch.ones(self.n, device=self.device) * _dyn.nominal_Jy
        self._Jz_t     = torch.ones(self.n, device=self.device) * _dyn.nominal_Jz
        self._Jxz_t    = torch.ones(self.n, device=self.device) * _dyn.nominal_Jxz
        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t, self._Jz_t, self._Jxz_t)

    def _u(self, size, r):
        """Per-axis uniform sample on [-r, +r], shape (size,)."""
        if r <= 0.0:
            return torch.zeros(size, device=self.device)
        return (torch.rand(size, device=self.device) * 2.0 - 1.0) * r

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = done | bad_done | exceed_time_limit
        size = int(torch.sum(reset).item())
        if size == 0:
            return

        # ---- State / control reset ----
        self.s[reset, :] = 0.0
        # Lift rotors: initialize to per-env hover thrust so motor_constrain
        # doesn't leave the aircraft in free-fall during the first ~40 steps.
        # (head rotor stays at 0 — no forward thrust at start)

        # Altitude: uniform in [min_altitude, max_altitude] (positive-down NED)
        self.s[reset, 2] = (torch.rand(size, device=self.device)
                            * (self.max_altitude - self.min_altitude) + self.min_altitude)

        # ---- Domain randomisation – mass & inertia ----
        _dyn = self.dynamics
        dm, di = self.dr_mass, self.dr_inertia
        self.mass_curr[reset] = (
            torch.rand(size, device=self.device) * (2 * dm) + (1 - dm)) * _dyn.nominal_m
        self._Jx_t[reset]  = (
            torch.rand(size, device=self.device) * (2 * di) + (1 - di)) * _dyn.nominal_Jx
        self._Jy_t[reset]  = (
            torch.rand(size, device=self.device) * (2 * di) + (1 - di)) * _dyn.nominal_Jy
        self._Jz_t[reset]  = (
            torch.rand(size, device=self.device) * (2 * di) + (1 - di)) * _dyn.nominal_Jz
        self._Jxz_t[reset] = (
            torch.rand(size, device=self.device) * (2 * di) + (1 - di)) * _dyn.nominal_Jxz
        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t,
                                  self._Jz_t, self._Jxz_t)

        # ---- Initial-state randomisation (per-axis UNIFORM) ----
        # xy already zeroed above; z already set to U(min_alt, max_alt).
        self.s[reset, 3] = self._u(size, self.init_roll_range)
        self.s[reset, 4] = self._u(size, self.init_pitch_range)
        self.s[reset, 5] = self._u(size, self.init_yaw_range)
        for k in range(3):
            self.s[reset, 6 + k] = self._u(size, self.init_vel_range)
            self.s[reset, 9 + k] = self._u(size, self.init_omega_range)

        # ---- Pitch-trim motor split ------------------------------------------
        # Rotor positions (body x): rf=+0.207, lf=+0.210, lb=-0.260, rb=-0.263
        # Pitch torque from lift rotor i = x_i * F_i  (from cross-product: pos×F_z)
        # At equal thrust there is a net nose-down moment that causes large
        # initial pitch oscillation.  Solve for F_front/F_rear that gives zero
        # net pitch AND zero net roll AND total lift = m*g:
        #
        #   (x_rf + x_lf)*F_front + (x_lb + x_rb)*F_rear = 0
        #   2*F_front + 2*F_rear = m*g
        #
        # → F_front/F_rear = -(x_lb+x_rb)/(x_rf+x_lf) = 0.523/0.417 ≈ 1.254
        _x_front =  0.207 + 0.210   # sum of front-rotor x-positions
        _x_rear  =  0.260 + 0.263   # magnitude of sum of rear-rotor x-positions
        _ratio   = _x_rear / _x_front            # F_front / F_rear ≈ 1.254
        F_rear_trim  = self.mass_curr[reset] * 9.807 / (2.0 * (1.0 + _ratio))
        F_front_trim = _ratio * F_rear_trim
        self.u[reset, 0] = 0.0             # head rotor: no forward thrust
        self.u[reset, 1] = F_front_trim    # rf (front-right)
        self.u[reset, 2] = F_rear_trim     # lb (rear-left)
        self.u[reset, 3] = F_front_trim    # lf (front-left)
        self.u[reset, 4] = F_rear_trim     # rb (rear-right)

        self.recent_s[reset] = self.s[reset]
        self.recent_u[reset] = self.u[reset]
        self.filtered_action[reset] = 2.0 * self.u[reset] / self.max_F - 1.0
        self.filtered_action_valid[reset] = False

    def get_extended_state(self):
        x = torch.hstack((self.s, self.u))
        return self.dynamics.nlplant(x)
    
    def update(self, action):
        action = torch.clamp(action, -1, 1)
        if self.enable_action_filter and self.action_filter_alpha < 1.0:
            alpha = self.action_filter_alpha
            valid = self.filtered_action_valid.unsqueeze(-1)
            action = torch.where(valid, (1.0 - alpha) * self.filtered_action + alpha * action, action)
            self.filtered_action = action.clone()
            self.filtered_action_valid[:] = True

        thrust_F = torch.clamp(action, -1, 1)
        thrust_F = thrust_F * self.max_F / 2.0 + self.max_F / 2.0
        thrust_F = torch.clamp(thrust_F, 0., self.max_F)
        # print('recent_u=',self.recent_u[0])
        # NOTE: rate-limit reference must be the PREVIOUS u (self.u), not
        # self.recent_u (which still holds the value from two steps ago).
        self.recent_u = self.u.clone()
        thrust_F = torch.clamp(thrust_F, self.u - self.motor_constrain, self.u + self.motor_constrain)
        self.u = thrust_F
        # print('now_u=',self.u[0])
        self.recent_s = self.s
        self.s = odeint(self.dynamics,
                        torch.hstack((self.s, self.u)),
                        torch.tensor([0., self.dt], device=self.device),
                        method=self.solver)[1, :, :self.num_states]
    
    def get_state(self):
        return self.s
    
    def get_control(self):
        return self.u

    def get_F(self):
        return self.u[:, 0], self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]
    
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
        return torch.sqrt(U*U + V*V + W*W)

    def get_TAS(self):
        return self.get_vt() + self.airspeed * torch.ones(self.n, device=self.device)
    
    def get_EAS(self):
        TAS = self.get_TAS()
        EAS2TAS = self.get_EAS2TAS()
        EAS = TAS / EAS2TAS
        return EAS
    
    @staticmethod
    def _aero_smooth_weight(vt2, vt_lo=0.3, vt_hi=1.0):
        """C1-continuous smoothstep weight in [0, 1] over the speed range
        [vt_lo, vt_hi] m/s.  Eliminates the hard threshold discontinuity.

             vt < vt_lo  → 0   (aerodynamics / angles physically negligible)
          vt_lo…vt_hi    → cubic Hermite ramp  (3t²-2t³, C1 at both ends)
             vt > vt_hi  → 1   (full value)
        """
        vt  = torch.sqrt(vt2.clamp(min=0))
        t   = ((vt - vt_lo) / (vt_hi - vt_lo)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def get_AOA(self):
        # α = atan2(W, U) in the body XZ-plane, blended smoothly to 0 at low speed.
        # Forward-flight guard (U<0 → weight→0) is handled via ca contribution.
        U    = self.s[:, 6]
        W    = self.s[:, 8]
        vxz2 = U * U + W * W
        vt2  = vxz2 + self.s[:, 7] ** 2
        inv_vxz = torch.rsqrt(vxz2.clamp(min=1e-6))
        sa = W * inv_vxz                       # sin(α) ∈ [-1, 1]
        ca = U * inv_vxz                       # cos(α) ∈ [-1, 1]
        # Extra forward-flight smooth weight: ca goes from -1 to +1,
        # remap to [0,1] so backward flight is also faded to zero.
        fwd_w = ((ca + 1.0) * 0.5).clamp(0.0, 1.0)   # 0 when U<<0, 1 when U>>0
        w = self._aero_smooth_weight(vt2) * fwd_w
        return w * torch.atan2(sa, ca)

    def get_AOS(self):
        # β = atan2(V, vt), blended smoothly to 0 at low speed.
        U, V, W = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        vxz2 = U * U + W * W
        vt2  = vxz2 + V * V
        inv_vt = torch.rsqrt(vt2.clamp(min=1e-6))
        vxz    = torch.sqrt(vxz2.clamp(min=0))
        sb = V   * inv_vt
        cb = vxz * inv_vt
        w  = self._aero_smooth_weight(vt2)
        return w * torch.atan2(sb, cb)

    def get_aero_sincos(self):
        """Return (sin_alpha, cos_alpha, sin_beta, cos_beta) computed directly
        from body-frame velocity components — no atan2, no low-speed guard.
        Values vanish smoothly as vt→0, matching the dynamic-pressure weighting
        used in the aerodynamics module.
        """
        U, V, W = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        vxz2    = U * U + W * W
        vt2     = vxz2 + V * V
        inv_vxz = torch.rsqrt(vxz2.clamp(min=1e-6))
        inv_vt  = torch.rsqrt(vt2.clamp(min=1e-6))
        sa = W * inv_vxz                           # sin(alpha)
        ca = U * inv_vxz                           # cos(alpha)
        sb = V * inv_vt                            # sin(beta)
        cb = torch.sqrt(vxz2).clamp(min=0) * inv_vt  # cos(beta)
        return sa, ca, sb, cb
    
    def get_angular_velocity(self):
        return self.s[:, 9], self.s[:, 10], self.s[:, 11]
    
    def get_thrust(self):
        return self.u[:, 0]
    
    def get_control_surface(self):
        return self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]

    
    def get_velocity(self):
        # Body-frame velocity components (U, V, W) directly from state
        return self.s[:, 6], self.s[:, 7], self.s[:, 8]
    
    def get_acceleration(self):
        # Body-frame acceleration (Udot, Vdot, Wdot) is now a direct ODE output;
        # no wind-axis back-conversion, no vt singularity, no conditional branches.
        xdot = self.get_extended_state()
        return xdot[:, 6], xdot[:, 7], xdot[:, 8]
    
    def get_G(self):
        # 根据飞行状态计算过载
        nx_cg, ny_cg, nz_cg = self.get_accels()
        G = torch.sqrt(nx_cg ** 2 + ny_cg ** 2 + nz_cg ** 2)
        return G
    
    def get_EAS2TAS(self):
        # 根据高度计算EAS2TAS
        alt = self.s[:, 2]
        tfac = 1 - alt / 44330
        eas2tas = 1 / torch.pow(tfac, 4.255)
        eas2tas = torch.sqrt(eas2tas)
        return eas2tas
    
    def get_accels(self):
        grav = 9.807
        xdot = self.get_extended_state()
        U, V, W   = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        P, Q, R   = self.s[:, 9], self.s[:, 10], self.s[:, 11]
        Udot, Vdot, Wdot = xdot[:, 6], xdot[:, 7], xdot[:, 8]
        nx_cg = 1.0/grav * (Udot + Q*W - R*V) + torch.sin(self.s[:, 4])
        ny_cg = 1.0/grav * (Vdot + R*U - P*W) - torch.cos(self.s[:, 4]) * torch.sin(self.s[:, 3])
        nz_cg = -1.0/grav * (Wdot + P*V - Q*U) + torch.cos(self.s[:, 4]) * torch.cos(self.s[:, 3])
        return nx_cg, ny_cg, nz_cg
    
if __name__ == "__main__":
    hybrid_uav = HybridDynamics()
    state = torch.zeros(1, 12)
    state[:, 0] = 0
    state[:, 1] = 0
    state[:, 2] = 20
    state[:, 6] = 0.001
    control = torch.ones(1, 5) * 9.8 / 4 * 1.779 + 0.4
    control[:, 0] = 0.
    for i in range(100):
        state = odeint(hybrid_uav, torch.hstack((state, control)),
                        torch.tensor([0., 0.02], device=torch.device('cuda:0')),
                        method='euler')[1, :, :12]
        # estate = hybrid_uav.compute_extended_state(torch.hstack((state, control)))
        # print(estate[:,8])
        print("第{:}次的坐标为({:},{:},{:})，姿态为({:},{:},{:})".format(i, state[:, 0], state[:, 1], state[:, 2],
                                                                        state[:, 3], state[:, 4], state[:, 5]))
