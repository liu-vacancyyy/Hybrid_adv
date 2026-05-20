"""LearningToFly-style PID baseline for the RC OU tracking task.

The controller reads ``RCTask`` targets directly:
    target_vx, target_vz, target_heading

and outputs HybridModel-normalized actions:
    [head, rf, lb, lf, rb] in [-1, 1]
"""
import math

import torch

from algorithms.pid.hover_pid import PID, wrap_pi


class RCPIDController:
    """Cascade PID for vx / vz / heading tracking.

    This follows the LearningToFly PID stack:

        command -> roll/pitch/yaw-rate target
                -> attitude PID
                -> attitude-rate PID
                -> motor mixer

    For RC, ``target_vx`` is treated as body/local forward velocity, matching
    LearningToFly's ``calc_local_velocity`` usage.  This matters because a
    world-frame north vx target plus an independent heading target can demand
    backward flight when the vehicle points away from north.  ``target_vz`` uses
    the LearningToFly altitude-hold convention: velocity PID output is a
    normalized throttle correction, not a physical acceleration divided by
    ``g``.  The head motor is enabled as a forward-acceleration assist by
    default, matching the quadplane/hybrid use case; pass ``use_head_motor=False``
    to make it behave like pure copter mode.
    """

    # LearningToFly PID gains:
    # SimulationUI/projects/copter_simulation/Controller/PID/*.h
    ATTITUDE_RP = dict(kp=1.5,   ki=0.900, kd=0.036,  imax=1.5, filt_hz=20.0)
    RATE_RP     = dict(kp=0.135, ki=0.090, kd=0.0036, imax=0.5, filt_hz=20.0)
    RATE_YAW    = dict(kp=1.0,   ki=0.018, kd=0.0,    imax=0.5, filt_hz=2.5)
    VEL_Z       = dict(kp=2.5,   ki=0.200, kd=0.010,  imax=0.5, filt_hz=2.0)

    # LearningToFly has no dedicated XY velocity PID for this RC task, so this
    # layer is a conservative velocity-PI outer loop feeding the same attitude
    # stack.  Units: velocity error (m/s) -> desired horizontal accel (m/s^2).
    VEL_XY      = dict(kp=1.05,  ki=0.16,  kd=0.020,  imax=0.45, filt_hz=2.0)

    YAW_P = 2.0
    MAX_YAW_RATE = 0.8
    MAX_TILT = math.radians(14.0)
    MAX_HORIZ_ACCEL = 2.0
    MAX_Z_THROTTLE_CORR = 0.10
    SIDE_DAMP_P = 0.45

    def __init__(self, n, device, dt=0.02, mass=1.779, gravity=9.807,
                 max_thrust_per_motor=7.0, use_head_motor=True,
                 head_max_scaled=0.45):
        self.n = int(n)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dt = float(dt)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.max_thrust = float(max_thrust_per_motor)
        self.throttle_hover = self.mass * self.gravity / (4.0 * self.max_thrust)
        self.use_head_motor = bool(use_head_motor)
        self.head_max_scaled = float(head_max_scaled)

        d = self.device
        self.vx_pid = PID(dt=dt, n=n, device=d, **self.VEL_XY)
        self.vy_pid = PID(dt=dt, n=n, device=d, **self.VEL_XY)
        self.vz_pid = PID(dt=dt, n=n, device=d, **self.VEL_Z)
        self.roll_pid = PID(dt=dt, n=n, device=d, **self.ATTITUDE_RP)
        self.pitch_pid = PID(dt=dt, n=n, device=d, **self.ATTITUDE_RP)
        self.roll_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_RP)
        self.pitch_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_RP)
        self.yaw_rate_pid = PID(dt=dt, n=n, device=d, **self.RATE_YAW)

        _x_front = 0.207 + 0.210
        _x_rear = 0.260 + 0.263
        k_front = 2.0 * _x_rear / (_x_front + _x_rear)
        k_rear = 2.0 * _x_front / (_x_front + _x_rear)
        self.throttle_fac = torch.tensor([0.0, k_front, k_rear, k_front, k_rear], device=d)
        self.roll_fac = torch.tensor([0.0, -0.5, 0.5, 0.5, -0.5], device=d)
        self.pitch_fac = torch.tensor([0.0, 0.5, -0.5, 0.5, -0.5], device=d)
        self.yaw_fac = torch.tensor([0.0, 0.5, 0.5, -0.5, -0.5], device=d)
        self.debug = {}

    def reset(self, mask=None):
        for pid in (
            self.vx_pid, self.vy_pid, self.vz_pid,
            self.roll_pid, self.pitch_pid,
            self.roll_rate_pid, self.pitch_rate_pid, self.yaw_rate_pid,
        ):
            pid.reset(mask)

    def _hover_throttle(self, model):
        mass = getattr(model, 'mass_curr', None)
        if torch.is_tensor(mass):
            return torch.clamp(mass.to(self.device) * self.gravity / (4.0 * self.max_thrust),
                               0.0, 1.0)
        return torch.full((self.n,), self.throttle_hover, device=self.device)

    def _mass(self, model):
        mass = getattr(model, 'mass_curr', None)
        if torch.is_tensor(mass):
            return mass.to(self.device)
        return torch.full((self.n,), self.mass, device=self.device)

    def _velocity_loop(self, task, model):
        vn, ve = model.get_ground_speed()
        vz = model.get_climb_rate()
        _roll, _pitch, heading = model.get_posture()

        cpsi = torch.cos(heading)
        spsi = torch.sin(heading)
        v_fwd = cpsi * vn + spsi * ve
        v_right = -spsi * vn + cpsi * ve

        err_fwd = task.target_vx - v_fwd
        self.vx_pid.set_input_filter_all(err_fwd)
        accel_fwd = torch.clamp(
            self.vx_pid.get_pid(),
            -self.MAX_HORIZ_ACCEL, self.MAX_HORIZ_ACCEL)
        accel_right = torch.clamp(
            -self.SIDE_DAMP_P * v_right,
            -self.MAX_HORIZ_ACCEL, self.MAX_HORIZ_ACCEL)

        if self.use_head_motor:
            mass = self._mass(model)
            head_scaled = torch.clamp(
                torch.relu(accel_fwd) * mass / self.max_thrust,
                0.0, self.head_max_scaled)
            head_accel = head_scaled * self.max_thrust / mass
        else:
            head_scaled = torch.zeros(self.n, device=self.device)
            head_accel = torch.zeros(self.n, device=self.device)

        residual_fwd_accel = accel_fwd - head_accel
        target_pitch = torch.clamp(-residual_fwd_accel / self.gravity, -self.MAX_TILT, self.MAX_TILT)
        target_roll = torch.clamp(accel_right / self.gravity, -self.MAX_TILT, self.MAX_TILT)

        err_vz = task.target_vz - vz
        self.vz_pid.set_input_filter_all(err_vz)
        # RC task's vz convention is opposite to the LearningToFly climb-rate
        # sign, so positive vz error needs *more* lift.
        z_corr = torch.clamp(
            self.vz_pid.get_pid(),
            -self.MAX_Z_THROTTLE_CORR, self.MAX_Z_THROTTLE_CORR)
        throttle = torch.clamp(
            self._hover_throttle(model) + z_corr,
            0.0, 1.0)

        return target_roll, target_pitch, throttle, head_scaled

    def _attitude_loop(self, task, model, target_roll, target_pitch):
        roll, pitch, heading = model.get_posture()
        P, Q, R = model.get_angular_velocity()

        err_heading = wrap_pi(task.target_heading - heading)
        target_yaw_rate = torch.clamp(
            self.YAW_P * err_heading, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        self.roll_pid.set_input_filter_all(target_roll - roll)
        self.pitch_pid.set_input_filter_all(target_pitch - pitch)
        target_p = self.roll_pid.get_pid()
        target_q = self.pitch_pid.get_pid()

        self.roll_rate_pid.set_input_filter_d(target_p - P)
        self.pitch_rate_pid.set_input_filter_d(target_q - Q)
        self.yaw_rate_pid.set_input_filter_d(target_yaw_rate - R)
        roll_out = torch.clamp(self.roll_rate_pid.get_pid(), -1.0, 1.0)
        pitch_out = torch.clamp(self.pitch_rate_pid.get_pid(), -1.0, 1.0)
        yaw_out = torch.clamp(self.yaw_rate_pid.get_pid(), -1.0, 1.0)
        return roll_out, pitch_out, yaw_out, target_yaw_rate

    def _mix(self, throttle, head_scaled, roll_out, pitch_out, yaw_out):
        th = throttle.unsqueeze(-1) * self.throttle_fac
        ro = roll_out.unsqueeze(-1) * self.roll_fac
        pi = pitch_out.unsqueeze(-1) * self.pitch_fac
        ya = yaw_out.unsqueeze(-1) * self.yaw_fac
        scaled = torch.clamp(th + ro + pi + ya, 0.0, 1.0)
        scaled[:, 0] = head_scaled
        return 2.0 * scaled - 1.0

    def compute_action(self, env):
        task = env.task
        model = env.model
        target_roll, target_pitch, throttle, head_scaled = self._velocity_loop(task, model)
        roll_out, pitch_out, yaw_out, target_yaw_rate = self._attitude_loop(
            task, model, target_roll, target_pitch)
        action = self._mix(throttle, head_scaled, roll_out, pitch_out, yaw_out)
        self.debug = {
            'target_roll': target_roll,
            'target_pitch': target_pitch,
            'target_yaw_rate': target_yaw_rate,
            'throttle': throttle,
            'head_scaled': head_scaled,
            'roll_out': roll_out,
            'pitch_out': pitch_out,
            'yaw_out': yaw_out,
        }
        return action
