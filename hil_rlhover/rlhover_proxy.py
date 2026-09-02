#!/usr/bin/env python3
"""MAVLink HIL proxy with optional ONNX RL hover actuator override.

Topology:

    Gazebo Classic HITL model  <udp>  rlhover_proxy.py  <serial/udp>  PX4 FMU

Gazebo must be launched with HIL enabled and serial disabled. The proxy forwards
all MAVLink traffic both ways. When the configured RC switch is active, the FMU
is armed, altitude is high enough, and state estimates are fresh, the proxy
replaces FMU ``HIL_ACTUATOR_CONTROLS`` messages with the ONNX policy output.
When RLHover is inactive, traffic is passed through unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "onnxruntime is required for RLHover ONNX inference. Install it inside "
        "the active environment with: python -m pip install -r hil_rlhover/requirements.txt"
    ) from exc


DEFAULT_PX4_MAVLINK = "/home/a/PX4-Autopilot/src/modules/mavlink/mavlink"
if DEFAULT_PX4_MAVLINK not in sys.path and Path(DEFAULT_PX4_MAVLINK).exists():
    sys.path.insert(0, DEFAULT_PX4_MAVLINK)

try:
    from pymavlink import mavutil
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "pymavlink is required. Install hil_rlhover/requirements.txt or keep "
        "/home/a/PX4-Autopilot/src/modules/mavlink/mavlink available."
    ) from exc


MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_MODE_FLAG_HIL_ENABLED = 32
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_TYPE_GENERIC = 0
MAV_AUTOPILOT_INVALID = 8
MAV_COMP_ID_ONBOARD_COMPUTER = 191


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dcm_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )


class UdpMavlinkEndpoint:
    def __init__(self, local_port: int, remote_host: str, remote_port: int, source_system: int, source_component: int):
        self.remote = (remote_host, remote_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", local_port))
        self.sock.setblocking(False)
        self.mav = mavutil.mavlink.MAVLink(self, srcSystem=source_system, srcComponent=source_component)
        self.parser = mavutil.mavlink.MAVLink(None)

    def fileno(self) -> int:
        return self.sock.fileno()

    def write(self, data: bytes) -> int:
        self.sock.sendto(data, self.remote)
        return len(data)

    def send_msg(self, msg) -> None:
        self.write(msg.get_msgbuf())

    def recv_messages(self) -> list:
        messages = []
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            if addr:
                self.remote = addr
            for byte in data:
                parsed = self.parser.parse_char(bytes([byte]))
                if parsed is not None:
                    messages.append(parsed)
        return messages

    def heartbeat(self) -> None:
        self.mav.heartbeat_send(MAV_TYPE_GENERIC, MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def hil_actuator_controls(self, controls: list[float], mode: int, flags: int = 0) -> None:
        padded = [0.0] * 16
        for i, value in enumerate(controls[:16]):
            padded[i] = float(value)
        self.mav.hil_actuator_controls_send(
            int(time.time() * 1_000_000),
            padded,
            int(mode),
            int(flags),
        )


@dataclass
class VehicleState:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    att_time: float = 0.0

    x: float = 0.0
    y: float = 0.0
    z_down: float = 0.0
    vx_n: float = 0.0
    vy_e: float = 0.0
    vz_d: float = 0.0
    lpos_time: float = 0.0

    rc_raw: list[int] = field(default_factory=list)
    rc_time: float = 0.0
    armed: bool = False
    last_hil_mode: int = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | MAV_MODE_FLAG_HIL_ENABLED
    last_hil_flags: int = 0
    last_controls: list[float] = field(default_factory=lambda: [0.0] * 8)

    def altitude(self) -> float:
        return -self.z_down

    def state_fresh(self, max_age: float) -> bool:
        now = time.monotonic()
        return (now - self.att_time) <= max_age and (now - self.lpos_time) <= max_age

    def update_from_fcu(self, msg) -> None:
        now = time.monotonic()
        msg_type = msg.get_type()

        if msg_type == "ATTITUDE":
            self.roll = float(msg.roll)
            self.pitch = float(msg.pitch)
            self.yaw = float(msg.yaw)
            self.p = float(msg.rollspeed)
            self.q = float(msg.pitchspeed)
            self.r = float(msg.yawspeed)
            self.att_time = now

        elif msg_type == "LOCAL_POSITION_NED":
            self.x = float(msg.x)
            self.y = float(msg.y)
            self.z_down = float(msg.z)
            self.vx_n = float(msg.vx)
            self.vy_e = float(msg.vy)
            self.vz_d = float(msg.vz)
            self.lpos_time = now

        elif msg_type == "RC_CHANNELS":
            values = []
            for i in range(1, 19):
                values.append(int(getattr(msg, f"chan{i}_raw", 0)))
            self.rc_raw = values
            self.rc_time = now

        elif msg_type == "HEARTBEAT":
            self.armed = bool(int(msg.base_mode) & MAV_MODE_FLAG_SAFETY_ARMED)

        elif msg_type == "HIL_ACTUATOR_CONTROLS":
            self.last_hil_mode = int(msg.mode)
            self.last_hil_flags = int(msg.flags)
            self.last_controls = [float(v) for v in msg.controls[:8]]


class VelocityHoverObservation:
    def __init__(self, vx_norm: float = 1.0, vy_norm: float = 1.0, vz_norm: float = 1.0):
        self.vx_norm = max(vx_norm, 1e-6)
        self.vy_norm = max(vy_norm, 1e-6)
        self.vz_norm = max(vz_norm, 1e-6)
        self.vt_norm = 5.0
        self.force_norm = 15.0
        self.motor_scaling = np.array([1500.0, 1500.0, 1500.0, 1500.0, 5500.0], dtype=np.float64)
        self.motor_max = np.array([1500.0, 1500.0, 1500.0, 1500.0, 3500.0], dtype=np.float64)
        self.thrust_coeff = 2.0e-5

    def _motor_thrust_obs(self, controls: Iterable[float]) -> list[float]:
        cmd = np.array(list(controls)[:5], dtype=np.float64)
        if cmd.size < 5:
            cmd = np.pad(cmd, (0, 5 - cmd.size))
        cmd = np.clip(cmd, 0.0, 1.0)
        omega = np.minimum(cmd * self.motor_scaling, self.motor_max)
        thrust = self.thrust_coeff * omega * omega
        return (thrust / self.force_norm).astype(np.float32).tolist()

    def build(self, state: VehicleState, target_yaw: float) -> np.ndarray:
        vx = state.vx_n
        vy = state.vy_e
        vz_up = -state.vz_d
        vel_ned = np.array([state.vx_n, state.vy_e, state.vz_d], dtype=np.float64)
        vel_body = dcm_body_to_ned(state.roll, state.pitch, state.yaw).T @ vel_ned
        u, v, w = vel_body.tolist()
        vt = float(np.linalg.norm(vel_body))

        if vt > 0.5:
            alpha = math.atan2(w, u)
            beta = math.atan2(v, math.sqrt(u * u + w * w))
        else:
            alpha = 0.0
            beta = 0.0

        motor_obs = self._motor_thrust_obs(state.last_controls)
        delta_yaw = wrap_pi(target_yaw - state.yaw) / math.pi

        obs = [
            vx / self.vx_norm,
            vy / self.vy_norm,
            vz_up / self.vz_norm,
            0.0,
            0.0,
            0.0,
            delta_yaw,
            math.sin(state.roll),
            math.cos(state.roll),
            math.sin(state.pitch),
            math.cos(state.pitch),
            vt / self.vt_norm,
            vx / self.vx_norm,
            vy / self.vy_norm,
            vz_up / self.vz_norm,
            state.p,
            state.q,
            state.r,
            math.sin(alpha),
            math.cos(alpha),
            math.sin(beta),
            math.cos(beta),
            *motor_obs,
        ]
        return np.asarray(obs, dtype=np.float32).reshape(1, -1)


class OnnxHoverPolicy:
    def __init__(self, path: Path, provider: str = "CPUExecutionProvider"):
        providers = [provider]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.rnn = np.zeros((1, 1, 128), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)

    def reset(self) -> None:
        self.rnn.fill(0.0)
        self.masks.fill(1.0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        feed = {"obs": obs.astype(np.float32)}
        if "rnn_states" in self.input_names:
            feed["rnn_states"] = self.rnn
        if "masks" in self.input_names:
            feed["masks"] = self.masks
        outputs = self.session.run(None, feed)
        actions = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if len(outputs) > 1:
            self.rnn = np.asarray(outputs[1], dtype=np.float32)
        return np.clip(actions, -1.0, 1.0)


class ActionMapper:
    def __init__(self):
        self.motor_scaling = np.array([1500.0, 1500.0, 1500.0, 1500.0, 5500.0], dtype=np.float64)
        self.motor_max = np.array([1500.0, 1500.0, 1500.0, 1500.0, 3500.0], dtype=np.float64)

    def to_hil_controls(self, action: np.ndarray) -> list[float]:
        if action.size < 5:
            action = np.pad(action, (0, 5 - action.size))
        motor_cmd = (np.clip(action[:5], -1.0, 1.0) + 1.0) * 0.5
        omega_ref = np.minimum(motor_cmd * self.motor_scaling, self.motor_max)
        controls = (omega_ref / self.motor_scaling).tolist()
        return [float(clamp(v, 0.0, 1.0)) for v in controls] + [0.0, 0.0, 0.0]


def parse_fcu_connection(device: str, baud: int):
    if device.startswith("serial:"):
        return mavutil.mavlink_connection(device[len("serial:") :], baud=baud, autoreconnect=True)
    return mavutil.mavlink_connection(device, baud=baud, autoreconnect=True)


def send_raw_to_fcu(fcu, msg) -> None:
    fcu.write(msg.get_msgbuf())


def request_fcu_streams(fcu, rates_hz: dict[int, float]) -> None:
    target_system = getattr(fcu, "target_system", 1) or 1
    target_component = getattr(fcu, "target_component", 1) or 1
    for msg_id, hz in rates_hz.items():
        interval_us = int(1_000_000 / hz) if hz > 0 else -1
        fcu.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(msg_id),
            float(interval_us),
            0,
            0,
            0,
            0,
            0,
        )


def send_arm_command(fcu) -> None:
    target_system = getattr(fcu, "target_system", 1) or 1
    target_component = getattr(fcu, "target_component", 1) or 1
    fcu.mav.command_long_send(
        target_system,
        target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def send_takeoff_command(fcu, altitude: float) -> None:
    target_system = getattr(fcu, "target_system", 1) or 1
    target_component = getattr(fcu, "target_component", 1) or 1
    fcu.mav.command_long_send(
        target_system,
        target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        float("nan"),  # minimum pitch
        0.0,
        0.0,
        float("nan"),  # yaw unchanged
        float("nan"),  # current latitude
        float("nan"),  # current longitude
        float(altitude),
    )


def rc_switch_active(state: VehicleState, channel: int, threshold: int) -> bool:
    if channel <= 0:
        return False
    index = channel - 1
    if index >= len(state.rc_raw):
        return False
    value = state.rc_raw[index]
    return value >= threshold


def auto_hover_ready(state: VehicleState, args: argparse.Namespace) -> tuple[bool, str]:
    altitude = state.altitude()
    horizontal_speed = math.hypot(state.vx_n, state.vy_e)
    vertical_speed_up = -state.vz_d
    attitude_limit = math.radians(max(args.auto_attitude_deg, 0.0))

    checks = [
        (altitude >= args.auto_altitude, f"alt={altitude:.2f}<{args.auto_altitude:.2f}"),
        (horizontal_speed <= args.auto_vxy, f"vxy={horizontal_speed:.2f}>{args.auto_vxy:.2f}"),
        (abs(vertical_speed_up) <= args.auto_vz, f"vz={vertical_speed_up:+.2f}>{args.auto_vz:.2f}"),
        (abs(state.roll) <= attitude_limit, f"roll={math.degrees(state.roll):+.1f}>{args.auto_attitude_deg:.1f}"),
        (abs(state.pitch) <= attitude_limit, f"pitch={math.degrees(state.pitch):+.1f}>{args.auto_attitude_deg:.1f}"),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="velocity-hover ONNX path")
    parser.add_argument("--fcu", default="serial:/dev/ttyACM0", help="serial:/dev/ttyACM0, udp:..., tcp:...")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--gazebo-host", default="127.0.0.1")
    parser.add_argument("--gazebo-port", type=int, default=14560, help="Gazebo HIL UDP port")
    parser.add_argument("--local-port", type=int, default=14557, help="local UDP port used by proxy")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--rlhover-rc-channel", type=int, default=7, help="1-based RC channel for RLHover")
    parser.add_argument("--rlhover-threshold", type=int, default=1600)
    parser.add_argument("--force-rlhover", action="store_true", help="ignore RC switch; still requires safety gates unless disabled")
    parser.add_argument("--auto-takeoff", action="store_true", help="send PX4 arm/takeoff commands before auto RLHover")
    parser.add_argument("--takeoff-altitude", type=float, default=3.0, help="PX4 takeoff target altitude in meters")
    parser.add_argument("--takeoff-delay", type=float, default=3.0, help="seconds to wait after proxy start before auto arm")
    parser.add_argument("--takeoff-command-interval", type=float, default=2.0, help="retry interval for arm/takeoff commands")
    parser.add_argument("--auto-rlhover", action="store_true", help="auto-engage RLHover after PX4 reaches a stable hover")
    parser.add_argument("--auto-altitude", type=float, default=2.0, help="minimum altitude for auto RLHover engagement")
    parser.add_argument("--auto-vxy", type=float, default=0.6, help="max horizontal speed for auto RLHover engagement")
    parser.add_argument("--auto-vz", type=float, default=0.35, help="max absolute vertical speed for auto RLHover engagement")
    parser.add_argument("--auto-attitude-deg", type=float, default=12.0, help="max absolute roll/pitch for auto RLHover engagement")
    parser.add_argument("--auto-hold-seconds", type=float, default=1.5, help="stable-hover duration required before auto engagement")
    parser.add_argument("--auto-no-latch", action="store_true", help="do not latch auto RLHover after first engagement")
    parser.add_argument("--allow-disarmed", action="store_true", help="allow override while disarmed")
    parser.add_argument("--min-altitude", type=float, default=1.0)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--print-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = OnnxHoverPolicy(args.onnx, args.provider)
    obs_builder = VelocityHoverObservation()
    mapper = ActionMapper()
    state = VehicleState()

    fcu = parse_fcu_connection(args.fcu, args.baud)
    gazebo = UdpMavlinkEndpoint(
        local_port=args.local_port,
        remote_host=args.gazebo_host,
        remote_port=args.gazebo_port,
        source_system=255,
        source_component=MAV_COMP_ID_ONBOARD_COMPUTER,
    )

    print(f"[rlhover] FCU: {args.fcu} baud={args.baud}")
    print(f"[rlhover] Gazebo UDP: local=0.0.0.0:{args.local_port} remote={args.gazebo_host}:{args.gazebo_port}")
    print(f"[rlhover] ONNX: {args.onnx}")
    print(f"[rlhover] RC switch: channel={args.rlhover_rc_channel}, threshold={args.rlhover_threshold}")
    if args.auto_rlhover:
        print(
            "[rlhover] auto mode: "
            f"alt>={args.auto_altitude:.1f}m, vxy<={args.auto_vxy:.2f}m/s, "
            f"|vz|<={args.auto_vz:.2f}m/s, |roll/pitch|<={args.auto_attitude_deg:.1f}deg, "
            f"hold={args.auto_hold_seconds:.1f}s"
        )
    if args.auto_takeoff:
        print(
            "[rlhover] auto takeoff: "
            f"delay={args.takeoff_delay:.1f}s, target_alt={args.takeoff_altitude:.1f}m, "
            f"retry={args.takeoff_command_interval:.1f}s"
        )

    last_heartbeat = 0.0
    last_stream_request = 0.0
    start_time = time.monotonic()
    last_arm_command = 0.0
    last_takeoff_command = 0.0
    last_policy = 0.0
    last_print = 0.0
    active_prev = False
    auto_ready_since: Optional[float] = None
    auto_latched = False
    auto_reason = "disabled"
    target_yaw = 0.0
    last_rl_controls = [0.0] * 8

    period = 1.0 / max(args.rate_hz, 1.0)
    stream_rates = {
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 100.0,
        mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: 100.0,
        mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS: 20.0,
    }

    while True:
        now = time.monotonic()

        if now - last_heartbeat >= 1.0:
            gazebo.heartbeat()
            fcu.mav.heartbeat_send(MAV_TYPE_GENERIC, MAV_AUTOPILOT_INVALID, 0, 0, 0)
            last_heartbeat = now

        if now - last_stream_request >= 2.0:
            request_fcu_streams(fcu, stream_rates)
            last_stream_request = now

        if args.auto_takeoff and now - start_time >= args.takeoff_delay:
            if not state.armed and now - last_arm_command >= args.takeoff_command_interval:
                send_arm_command(fcu)
                last_arm_command = now
                print("[rlhover] auto takeoff: sent arm command")

            elif state.armed and state.altitude() < args.takeoff_altitude * 0.8 \
                    and now - last_takeoff_command >= args.takeoff_command_interval:
                send_takeoff_command(fcu, args.takeoff_altitude)
                last_takeoff_command = now
                print(f"[rlhover] auto takeoff: sent takeoff command alt={args.takeoff_altitude:.1f}m")

        readable = [gazebo.fileno()]
        if hasattr(fcu, "port") and hasattr(fcu.port, "fileno"):
            try:
                readable.append(fcu.port.fileno())
            except Exception:
                pass
        if readable:
            select.select(readable, [], [], 0.002)

        for gz_msg in gazebo.recv_messages():
            send_raw_to_fcu(fcu, gz_msg)

        while True:
            msg = fcu.recv_match(blocking=False)
            if msg is None:
                break

            state.update_from_fcu(msg)
            msg_type = msg.get_type()

            safety_ok = (
                (args.allow_disarmed or state.armed)
                and state.altitude() >= args.min_altitude
                and state.state_fresh(args.max_state_age)
            )
            rc_active = rc_switch_active(state, args.rlhover_rc_channel, args.rlhover_threshold)

            if args.auto_rlhover and safety_ok:
                ready, auto_reason = auto_hover_ready(state, args)

                if ready:
                    if auto_ready_since is None:
                        auto_ready_since = now

                    if now - auto_ready_since >= args.auto_hold_seconds:
                        auto_latched = True

                else:
                    auto_ready_since = None
                    if args.auto_no_latch:
                        auto_latched = False

            elif args.auto_rlhover:
                auto_ready_since = None
                auto_reason = "safety_gate"
                if args.auto_no_latch:
                    auto_latched = False

            switch_active = args.force_rlhover or rc_active or (args.auto_rlhover and auto_latched)
            active = switch_active and safety_ok

            if active and not active_prev:
                target_yaw = state.yaw
                policy.reset()
                print(f"[rlhover] engaged target_yaw={target_yaw:+.3f} alt={state.altitude():.2f}m")
            elif active_prev and not active:
                print("[rlhover] disengaged; forwarding PX4 actuators")
            active_prev = active

            if active and now - last_policy >= period:
                obs = obs_builder.build(state, target_yaw)
                action = policy.step(obs)
                last_rl_controls = mapper.to_hil_controls(action)
                state.last_controls = last_rl_controls
                last_policy = now

            if msg_type == "HIL_ACTUATOR_CONTROLS" and active:
                mode = int(state.last_hil_mode) | MAV_MODE_FLAG_HIL_ENABLED | MAV_MODE_FLAG_SAFETY_ARMED
                gazebo.hil_actuator_controls(last_rl_controls, mode=mode, flags=state.last_hil_flags)
            else:
                gazebo.send_msg(msg)

        if active_prev and now - last_policy >= period:
            obs = obs_builder.build(state, target_yaw)
            action = policy.step(obs)
            last_rl_controls = mapper.to_hil_controls(action)
            state.last_controls = last_rl_controls
            mode = int(state.last_hil_mode) | MAV_MODE_FLAG_HIL_ENABLED | MAV_MODE_FLAG_SAFETY_ARMED
            gazebo.hil_actuator_controls(last_rl_controls, mode=mode, flags=state.last_hil_flags)
            last_policy = now

        if now - last_print >= max(args.print_rate, 0.2):
            rc_value = state.rc_raw[args.rlhover_rc_channel - 1] if 0 < args.rlhover_rc_channel <= len(state.rc_raw) else 0
            auto_wait = 0.0 if auto_ready_since is None else max(0.0, args.auto_hold_seconds - (now - auto_ready_since))
            print(
                "[rlhover] "
                f"active={active_prev} armed={state.armed} alt={state.altitude():.2f} "
                f"vn/e/u=({state.vx_n:+.2f},{state.vy_e:+.2f},{-state.vz_d:+.2f}) "
                f"rc{args.rlhover_rc_channel}={rc_value} "
                f"auto_latched={auto_latched} auto_wait={auto_wait:.1f}s auto={auto_reason} "
                f"controls={','.join(f'{c:.2f}' for c in last_rl_controls[:5])}"
            )
            last_print = now


if __name__ == "__main__":
    main()
