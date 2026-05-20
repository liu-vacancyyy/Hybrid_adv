"""
Hover PID cascade controller for the HybridModel (QuadPlane-like platform).

The PID structure and gains are taken from the LearningToFly project
(SimulationUI/projects/copter_simulation/Controller/PID/*.h), specifically:

- Attitude controller (roll/pitch):        P=1.5,   I=0.900, D=0.036,  IMAX=1.5, FILT_HZ=20.0
- Attitude rate controller (roll/pitch):   P=0.135, I=0.090, D=0.0036, IMAX=0.5, FILT_HZ=20.0
- Attitude rate controller (yaw):          P=1.0,   I=0.018, D=0.0,    IMAX=0.5, FILT_HZ=2.5
- Altitude climb-rate controller:          P=2.5,   I=0.200, D=0.01,   IMAX=0.5, FILT_HZ=2.0
- Throttle hover:                          0.5 (of full-range motor max thrust in LearningToFly)

HybridModel action convention:
    u = [head, rf, lb, lf, rb]   (all in Newtons, clamped to [0, max_F=7N])

Rotor body positions (x forward, y right, z down):
    head : ( 0.292,  0.000,  0.069)   - pusher / front motor (not used in hover)
    rf   : ( 0.207,  0.305, -0.003)   - right-front lift
    lb   : (-0.260, -0.305, -0.003)   - left-back  lift
    lf   : ( 0.210, -0.305, -0.003)   - left-front lift
    rb   : (-0.263,  0.305, -0.003)   - right-back lift

Mixing factors for the four lift rotors (head motor is forced to zero):
    throttle_fac = 1.0
    roll_fac     = -0.5  if y > 0 else +0.5       (positive roll cmd -> right-wing-down)
    pitch_fac    = +0.5  if x > 0 else -0.5       (positive pitch cmd -> nose up)
    yaw_fac      = +0.5 on rf & lb, -0.5 on lf & rb  (per rotor torque sign)
"""
import math
import torch


class PID:
    """Vectorised version of the ArduPilot-style PID used in LearningToFly."""

    def __init__(self, kp, ki, kd, imax, filt_hz, dt, n, device):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.imax = float(abs(imax))
        self.dt = float(dt)
        if filt_hz > 0:
            rc = 1.0 / (2.0 * math.pi * float(filt_hz))
            self.filt_alpha = self.dt / (self.dt + rc)
        else:
            self.filt_alpha = 1.0
        self.n = n
        self.device = device
        self.input = torch.zeros(n, device=device)
        self.derivative = torch.zeros(n, device=device)
        self.integral = torch.zeros(n, device=device)
        self.reset_filter = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, mask=None):
        if mask is None:
            self.integral.zero_()
            self.reset_filter.fill_(True)
        else:
            self.integral[mask] = 0.0
            self.reset_filter[mask] = True

    def set_input_filter_all(self, x):
        # First call after reset: initialise input to x, derivative to 0
        first = self.reset_filter
        self.input = torch.where(first, x, self.input)
        self.derivative = torch.where(first, torch.zeros_like(x), self.derivative)
        delta = self.filt_alpha * (x - self.input)
        new_input = self.input + delta
        new_deriv = delta / self.dt
        self.input = torch.where(first, self.input, new_input)
        self.derivative = torch.where(first, self.derivative, new_deriv)
        self.reset_filter = torch.zeros_like(first)

    def set_input_filter_d(self, x):
        first = self.reset_filter
        # On first call, set input and zero derivative
        self.input = torch.where(first, x, self.input)
        self.derivative = torch.where(first, torch.zeros_like(x), self.derivative)
        raw_d = (x - self.input) / self.dt
        new_deriv = self.derivative + self.filt_alpha * (raw_d - self.derivative)
        self.derivative = torch.where(first, self.derivative, new_deriv)
        self.input = x
        self.reset_filter = torch.zeros_like(first)

    def get_p(self):
        return self.kp * self.input

    def get_i(self):
        if self.ki != 0.0 and self.dt > 0.0:
            self.integral = self.integral + self.input * self.ki * self.dt
            self.integral = torch.clamp(self.integral, -self.imax, self.imax)
            return self.integral
        return torch.zeros_like(self.input)

    def get_d(self):
        return self.kd * self.derivative

    def get_pid(self):
        return self.get_p() + self.get_i() + self.get_d()


def wrap_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class HoverPIDController:
    """
    Cascade PID controller for a quad-plane hovering the HybridModel.

    At every call to ``compute_action`` it reads the current state from
    ``model`` and returns a normalised action tensor of shape ``(n, 5)`` in
    ``[-1, 1]`` that can be fed directly to ``HybridModel.update`` - where
    the action is mapped to motor thrust via  ``F = action * max_F/2 + max_F/2``.

    Targets:
        target_altitude  : metres (absolute)
        target_heading   : rad
        target_roll, pitch = 0 (pure hover)
    """

    # --- PID gains from LearningToFly -----------------------------------
    ATTITUDE_RP = dict(kp=1.5,   ki=0.900, kd=0.036,  imax=1.5, filt_hz=20.0)
    RATE_RP     = dict(kp=0.135, ki=0.090, kd=0.0036, imax=0.5, filt_hz=20.0)
    RATE_YAW    = dict(kp=1.0,   ki=0.018, kd=0.0,    imax=0.5, filt_hz=2.5)
    POS_Z_VEL   = dict(kp=2.5,   ki=0.200, kd=0.010,  imax=0.5, filt_hz=2.0)

    # Outer altitude -> climb-rate loop (P only, matches ArduPilot default)
    ALT_P = 1.0
    MAX_CLIMB_RATE = 2.5  # m/s (LearningToFly z_vel imax = 2)

    # Heading -> yaw-rate loop (simple P)
    YAW_P = 2.0
    MAX_YAW_RATE = math.radians(180.0)  # rad/s

    # Attitude output clamps (rad/s)
    MAX_TILT = math.radians(25.0)       # rad (roll/pitch cmd limit)

    # Horizontal position outer loop
    # pos(m) -> vel(m/s) -> accel(m/s²) -> tilt(rad)
    # Keep tilt limit well below ExtremeAngle.max_pitch (25 deg) so the
    # position command never competes with the altitude-transient pitch swing.
    POS_XY_P    = 0.4   # m/s  per m of position error
    MAX_VEL_XY  = 1.5   # m/s  horizontal velocity limit
    VEL_XY_P    = 1.0   # m/s² per m/s of velocity error
    VEL_XY_I    = 0.15  # m/s² accumulated per (m/s · s)
    VEL_XY_IMAX = 0.4   # m/s² integrator saturation
    MAX_TILT_XY = math.radians(10.0)  # rad — position tilt cap (≪ ExtremeAngle 25°)

    def __init__(self, n, device, dt=0.02, mass=1.779, gravity=9.807,
                 max_thrust_per_motor=7.0):
        self.n = n
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dt = dt
        self.max_thrust = max_thrust_per_motor
        # throttle at which 4 lift rotors exactly balance weight
        self.throttle_hover = mass * gravity / 4.0 / self.max_thrust

        d = self.device
        self.roll_pid = PID(dt=dt, n=n, device=d, **self.ATTITUDE_RP)
        self.pitch_pid = PID(dt=dt, n=n, device=d, **self.ATTITUDE_RP)
        self.roll_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_RP)
        self.pitch_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_RP)
        self.yaw_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_YAW)
        self.z_vel_pid = PID(dt=dt, n=n, device=d, **self.POS_Z_VEL)

        # Rotor mixing (head motor index 0 is forced to zero)
        # Rotor x-positions: rf=+0.207, lf=+0.210, lb=-0.260, rb=-0.263
        # At equal throttle the front rotors (x>0) produce more pitch-up moment
        # than the rear rotors produce pitch-down, so there is a net nose-up
        # trim moment.  Bake the geometry compensation into throttle_fac so that
        # pitch_out≈0 in steady hover:
        #   k_front/k_rear = x_rear_sum/x_front_sum = 0.523/0.417 ≈ 1.254
        #   k_front + k_rear = 2  (total thrust unchanged)
        _x_front = 0.207 + 0.210   # 0.417
        _x_rear  = 0.260 + 0.263   # 0.523
        k_front  = 2.0 * _x_rear  / (_x_front + _x_rear)   # ≈ 1.113
        k_rear   = 2.0 * _x_front / (_x_front + _x_rear)   # ≈ 0.887
        #                           head   rf        lb        lf        rb
        self.throttle_fac = torch.tensor([0.0, k_front, k_rear,  k_front, k_rear],  device=d)
        self.roll_fac     = torch.tensor([0.0, -0.5,    0.5,     0.5,    -0.5],     device=d)
        self.pitch_fac    = torch.tensor([0.0,  0.5,   -0.5,     0.5,    -0.5],     device=d)
        self.yaw_fac      = torch.tensor([0.0,  0.5,    0.5,    -0.5,    -0.5],     device=d)

        # Targets
        self.target_altitude = torch.zeros(n, device=d)
        self.target_heading  = torch.zeros(n, device=d)
        self.target_npos     = torch.zeros(n, device=d)
        self.target_epos     = torch.zeros(n, device=d)

        # Horizontal velocity PI integrators
        self.vel_int_n = torch.zeros(n, device=d)
        self.vel_int_e = torch.zeros(n, device=d)

    def reset(self, mask=None):
        for pid in (self.roll_pid, self.pitch_pid,
                    self.roll_rate_pid, self.pitch_rate_pid,
                    self.yaw_rate_pid, self.z_vel_pid):
            pid.reset(mask)
        if mask is None:
            self.vel_int_n.zero_()
            self.vel_int_e.zero_()
        else:
            self.vel_int_n[mask] = 0.0
            self.vel_int_e[mask] = 0.0

    def set_targets(self, target_altitude, target_heading=None,
                    target_npos=None, target_epos=None):
        if torch.is_tensor(target_altitude):
            self.target_altitude[:] = target_altitude.to(self.device)
        else:
            self.target_altitude.fill_(float(target_altitude))
        if target_heading is not None:
            if torch.is_tensor(target_heading):
                self.target_heading[:] = target_heading.to(self.device)
            else:
                self.target_heading.fill_(float(target_heading))
        if target_npos is not None:
            if torch.is_tensor(target_npos):
                self.target_npos[:] = target_npos.to(self.device)
            else:
                self.target_npos.fill_(float(target_npos))
        if target_epos is not None:
            if torch.is_tensor(target_epos):
                self.target_epos[:] = target_epos.to(self.device)
            else:
                self.target_epos.fill_(float(target_epos))

    # --- cascade layers -------------------------------------------------
    def _position_loop(self, npos, epos, vn, ve, heading):
        """NE position → body-frame target_roll / target_pitch.

        Cascade: position-P → velocity-PI → acceleration → tilt angle.
        All computation in world (NED) horizontal frame; final step rotates
        commands into the body frame using current yaw.
        """
        g = 9.807
        # Layer 1: position error → desired NE velocity
        err_n = self.target_npos - npos
        err_e = self.target_epos - epos
        target_vn = torch.clamp(self.POS_XY_P * err_n, -self.MAX_VEL_XY, self.MAX_VEL_XY)
        target_ve = torch.clamp(self.POS_XY_P * err_e, -self.MAX_VEL_XY, self.MAX_VEL_XY)
        # Layer 2: velocity error → desired NE acceleration  (PI)
        vel_err_n = target_vn - vn
        vel_err_e = target_ve - ve
        self.vel_int_n = torch.clamp(
            self.vel_int_n + self.VEL_XY_I * vel_err_n * self.dt,
            -self.VEL_XY_IMAX, self.VEL_XY_IMAX)
        self.vel_int_e = torch.clamp(
            self.vel_int_e + self.VEL_XY_I * vel_err_e * self.dt,
            -self.VEL_XY_IMAX, self.VEL_XY_IMAX)
        accel_n = self.VEL_XY_P * vel_err_n + self.vel_int_n
        accel_e = self.VEL_XY_P * vel_err_e + self.vel_int_e
        # Layer 3: rotate world-frame acceleration into body-frame tilt
        # R^T (NED→body horizontal): [cos ψ  sin ψ; -sin ψ  cos ψ]
        cpsi = torch.cos(heading)
        spsi = torch.sin(heading)
        accel_fwd   =  cpsi * accel_n + spsi * accel_e   # along body-x
        accel_right = -spsi * accel_n + cpsi * accel_e   # along body-y
        # nose-down (negative pitch) → positive fwd accel → target_pitch = -a_fwd/g
        # right-bank (positive roll) → positive right accel → target_roll  = +a_right/g
        target_pitch = torch.clamp(-accel_fwd   / g, -self.MAX_TILT_XY, self.MAX_TILT_XY)
        target_roll  = torch.clamp( accel_right / g, -self.MAX_TILT_XY, self.MAX_TILT_XY)
        return target_roll, target_pitch

    def _altitude_loop(self, altitude, climb_rate):
        err_alt = self.target_altitude - altitude
        target_climb = torch.clamp(self.ALT_P * err_alt,
                                   -self.MAX_CLIMB_RATE, self.MAX_CLIMB_RATE)
        err_v = target_climb - climb_rate
        self.z_vel_pid.set_input_filter_all(err_v)
        target_climb_acc = self.z_vel_pid.get_pid()
        throttle = self.throttle_hover + target_climb_acc
        return torch.clamp(throttle, 0.0, 1.0), target_climb

    def _attitude_loop(self, roll, pitch, heading, target_roll, target_pitch):
        err_heading = wrap_pi(self.target_heading - heading)
        target_yaw_rate = torch.clamp(self.YAW_P * err_heading,
                                      -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        self.roll_pid.set_input_filter_all(target_roll - roll)
        self.pitch_pid.set_input_filter_all(target_pitch - pitch)
        target_roll_rate = self.roll_pid.get_pid()
        target_pitch_rate = self.pitch_pid.get_pid()
        return target_roll_rate, target_pitch_rate, target_yaw_rate, target_roll, target_pitch

    def _rate_loop(self, P, Q, R, tgt_P, tgt_Q, tgt_R):
        self.roll_rate_pid.set_input_filter_d(tgt_P - P)
        self.pitch_rate_pid.set_input_filter_d(tgt_Q - Q)
        self.yaw_rate_pid.set_input_filter_d(tgt_R - R)
        r_out = torch.clamp(self.roll_rate_pid.get_pid(), -1.0, 1.0)
        p_out = torch.clamp(self.pitch_rate_pid.get_pid(), -1.0, 1.0)
        y_out = torch.clamp(self.yaw_rate_pid.get_pid(), -1.0, 1.0)
        return r_out, p_out, y_out

    def _mix(self, throttle, roll_out, pitch_out, yaw_out):
        # All shapes (n,). Broadcast mix to (n, 5) motors.
        th = throttle.unsqueeze(-1) * self.throttle_fac
        ro = roll_out.unsqueeze(-1)  * self.roll_fac
        pi = pitch_out.unsqueeze(-1) * self.pitch_fac
        ya = yaw_out.unsqueeze(-1)   * self.yaw_fac
        scaled = torch.clamp(th + ro + pi + ya, 0.0, 1.0)
        # Map [0,1] throttle to normalised action in [-1,1] for HybridModel.update
        # HybridModel: F = action * max_F/2 + max_F/2  (with head rotor forced 0)
        #   so  action = 2*scaled - 1  where scaled = F / max_F.
        action = 2.0 * scaled - 1.0
        # Force head rotor (index 0) to zero thrust -> action = -1
        action[:, 0] = -1.0
        return action

    # --- public API -----------------------------------------------------
    def compute_action(self, model):
        """Run one PID cascade step. Returns action tensor of shape (n, 5)."""
        npos, epos, altitude = model.get_position()
        roll, pitch, heading = model.get_posture()
        climb_rate = model.get_climb_rate()
        vn, ve = model.get_ground_speed()
        P, Q, R = model.get_angular_velocity()

        t_roll, t_pitch = self._position_loop(npos, epos, vn, ve, heading)
        throttle, target_climb = self._altitude_loop(altitude, climb_rate)
        tgt_P, tgt_Q, tgt_R, t_roll, t_pitch = self._attitude_loop(
            roll, pitch, heading, t_roll, t_pitch)
        r_out, p_out, y_out = self._rate_loop(P, Q, R, tgt_P, tgt_Q, tgt_R)
        action = self._mix(throttle, r_out, p_out, y_out)

        # Store debug info for logging
        self.debug = {
            'target_climb': target_climb,
            'throttle': throttle,
            'target_roll_rate': tgt_P,
            'target_pitch_rate': tgt_Q,
            'target_yaw_rate': tgt_R,
            'roll_out': r_out,
            'pitch_out': p_out,
            'yaw_out': y_out,
        }
        return action
