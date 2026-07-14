#!/usr/bin/env python
"""Stress-test PID tracking with large vx/vy/vz/yaw-rate commands."""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from algorithms.pid.rc_pid import RCPIDController  # noqa: E402
from algorithms.pid.hover_pid import wrap_pi       # noqa: E402
from envs.control_env import ControlEnv            # noqa: E402


class YawRateRCPIDController(RCPIDController):
    def _attitude_loop(self, task, model, target_roll, target_pitch):
        roll, pitch, _heading = model.get_posture()
        p, q, r = model.get_angular_velocity()
        target_yaw_rate = torch.clamp(
            task.target_yaw_rate,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        )

        self.roll_pid.set_input_filter_all(target_roll - roll)
        self.pitch_pid.set_input_filter_all(target_pitch - pitch)
        target_p = self.roll_pid.get_pid()
        target_q = self.pitch_pid.get_pid()

        self.roll_rate_pid.set_input_filter_d(target_p - p)
        self.pitch_rate_pid.set_input_filter_d(target_q - q)
        self.yaw_rate_pid.set_input_filter_d(target_yaw_rate - r)
        roll_out = torch.clamp(self.roll_rate_pid.get_pid(), -1.0, 1.0)
        pitch_out = torch.clamp(self.pitch_rate_pid.get_pid(), -1.0, 1.0)
        yaw_out = torch.clamp(self.yaw_rate_pid.get_pid(), -1.0, 1.0)
        return roll_out, pitch_out, yaw_out, target_yaw_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--model-name", default="HYBRID_NEW")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=9731)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--num-envs", type=int, default=3)
    p.add_argument("--scenario", choices=["large", "extreme"], default="large")

    p.add_argument("--xy-kp", type=float, default=1.35)
    p.add_argument("--xy-ki", type=float, default=0.12)
    p.add_argument("--xy-kd", type=float, default=0.020)
    p.add_argument("--xy-imax", type=float, default=0.45)
    p.add_argument("--xy-filt-hz", type=float, default=2.0)
    p.add_argument("--z-kp", type=float, default=2.4)
    p.add_argument("--z-ki", type=float, default=0.18)
    p.add_argument("--z-kd", type=float, default=0.010)
    p.add_argument("--z-imax", type=float, default=0.45)
    p.add_argument("--z-filt-hz", type=float, default=2.0)
    p.add_argument("--max-tilt-deg", type=float, default=16.0)
    p.add_argument("--max-horiz-accel", type=float, default=2.5)
    p.add_argument("--max-z-throttle-corr", type=float, default=0.12)
    p.add_argument("--max-yaw-rate", type=float, default=0.6)
    p.add_argument("--side-damp-p", type=float, default=0.45)
    p.add_argument("--head-max-scaled", type=float, default=0.45)
    return p.parse_args()


def choose_device(args):
    if (not args.no_cuda) and torch.cuda.is_available():
        return torch.device(args.device)
    return torch.device("cpu")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_np(x):
    return x.detach().cpu().float().numpy()


def make_pid(env, args, device):
    return YawRateRCPIDController(
        n=env.n,
        device=device,
        dt=env.model.dt,
        max_thrust_per_motor=env.model.max_F,
        use_head_motor=True,
        head_max_scaled=args.head_max_scaled,
        xy_gains=dict(kp=args.xy_kp, ki=args.xy_ki, kd=args.xy_kd,
                      imax=args.xy_imax, filt_hz=args.xy_filt_hz),
        z_gains=dict(kp=args.z_kp, ki=args.z_ki, kd=args.z_kd,
                     imax=args.z_imax, filt_hz=args.z_filt_hz),
        max_tilt_deg=args.max_tilt_deg,
        max_horiz_accel=args.max_horiz_accel,
        max_z_throttle_corr=args.max_z_throttle_corr,
        max_yaw_rate=args.max_yaw_rate,
        side_damp_p=args.side_damp_p,
    )


def make_command(step, dt, n, scenario):
    t = step * dt
    idx = torch.arange(n, dtype=torch.float32)
    phase = idx * 1.7
    if scenario == "extreme":
        vx_amp, vy_amp, vz_amp, yaw_amp = 1.0, 1.0, 1.0, 0.60
        square_gain = 1.0
    else:
        vx_amp, vy_amp, vz_amp, yaw_amp = 0.85, 0.85, 0.85, 0.55
        square_gain = 0.75

    tt = torch.full((n,), float(t))
    vx = vx_amp * torch.sin(0.95 * tt + phase)
    vy = vy_amp * torch.sin(1.25 * tt + 0.7 * phase + 1.1)
    vz = vz_amp * torch.sin(0.75 * tt + 1.3 * phase - 0.4)
    yaw_rate = yaw_amp * torch.sin(1.55 * tt + 0.5 * phase + 0.6)

    block = torch.floor(tt / 2.0 + idx).long()
    vx = torch.where((block % 4) == 0, torch.full_like(vx, square_gain * vx_amp), vx)
    vy = torch.where((block % 4) == 1, torch.full_like(vy, -square_gain * vy_amp), vy)
    vz = torch.where((block % 4) == 2, torch.full_like(vz, square_gain * vz_amp), vz)
    yaw_rate = torch.where((block % 4) == 3, torch.full_like(yaw_rate, -yaw_amp), yaw_rate)
    return vx, vy, vz, yaw_rate


def apply_command(env, vx, vy, vz, yaw_rate):
    task = env.task
    vx = vx.to(env.device)
    vy = vy.to(env.device)
    vz = vz.to(env.device)
    yaw_rate = yaw_rate.to(env.device)
    task.curriculum_enable = False
    task.yaw_command_enable = True
    task.yaw_hold_enable = False
    task.target_vx[:] = vx
    task.target_vy[:] = vy
    task.target_vz[:] = vz
    task.target_yaw_rate[:] = yaw_rate
    task.target_heading[:] = wrap_pi(task.target_heading + yaw_rate * env.model.dt)
    task.raw_vx[:] = torch.clamp(vx / max(float(task.vx_limit), 1e-6), -1.0, 1.0)
    task.raw_vy[:] = torch.clamp(vy / max(float(task.vy_limit), 1e-6), -1.0, 1.0)
    task.raw_vz[:] = torch.clamp(vz / max(float(task.vz_limit), 1e-6), -1.0, 1.0)
    task.raw_yaw[:] = torch.clamp(yaw_rate / max(float(task.yaw_rate_limit), 1e-6), -1.0, 1.0)
    task.stick_vx[:] = task.raw_vx
    task.stick_vy[:] = task.raw_vy
    task.stick_vz[:] = task.raw_vz
    task.stick_yaw[:] = task.raw_yaw


def collect(env, reward, action, pid, step):
    task = env.task
    roll, pitch, heading = env.model.get_posture()
    _npos, _epos, altitude = env.model.get_position()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    tas = env.model.get_TAS()
    p, q, r = env.model.get_angular_velocity()
    alpha = env.model.get_AOA()
    beta = env.model.get_AOS()
    f1, f2, f3, f4, f5 = env.model.get_F()
    local_vx, local_vy = task.ground_to_local_velocity(vx_n, vy_e, heading)
    err_vx = local_vx - task.target_vx
    err_vy = local_vy - task.target_vy
    err_vz = vz - task.target_vz
    yaw_rate_err = r - task.target_yaw_rate
    vel_err = torch.sqrt(err_vx * err_vx + err_vy * err_vy + err_vz * err_vz)
    dbg = getattr(pid, "debug", {})
    z = torch.zeros_like(vz)
    target_roll = dbg.get("target_roll", z)
    target_pitch = dbg.get("target_pitch", z)
    pid_target_yaw_rate = dbg.get("target_yaw_rate", z)

    arrays = {
        "reward": tensor_np(reward),
        "local_vx": tensor_np(local_vx),
        "target_vx": tensor_np(task.target_vx),
        "local_vy": tensor_np(local_vy),
        "target_vy": tensor_np(task.target_vy),
        "vz": tensor_np(vz),
        "target_vz": tensor_np(task.target_vz),
        "heading": tensor_np(heading),
        "target_heading": tensor_np(task.target_heading),
        "r": tensor_np(r),
        "target_yaw_rate": tensor_np(task.target_yaw_rate),
        "yaw_rate_err": tensor_np(yaw_rate_err),
        "vel_err": tensor_np(vel_err),
        "tas": tensor_np(tas),
        "altitude": tensor_np(altitude),
        "roll": tensor_np(roll),
        "pitch": tensor_np(pitch),
        "p": tensor_np(p),
        "q": tensor_np(q),
        "omega_norm": tensor_np(torch.sqrt(p * p + q * q + r * r)),
        "alpha_deg": np.degrees(tensor_np(alpha)),
        "beta_deg": np.degrees(tensor_np(beta)),
        "pid_target_roll_deg": np.degrees(tensor_np(target_roll)),
        "pid_target_pitch_deg": np.degrees(tensor_np(target_pitch)),
        "pid_target_yaw_rate": tensor_np(pid_target_yaw_rate),
        "f1": tensor_np(f1),
        "f2": tensor_np(f2),
        "f3": tensor_np(f3),
        "f4": tensor_np(f4),
        "f5": tensor_np(f5),
    }
    action_np = tensor_np(action)
    for i in range(action_np.shape[1]):
        arrays[f"action_{i}"] = action_np[:, i]
    rows = []
    for i in range(env.n):
        row = {"env": i, "step": step, "time_s": step * env.model.dt}
        row.update({k: float(v[i]) for k, v in arrays.items()})
        rows.append(row)
    return rows


def summarize(trace_rows, term_types):
    rows = []
    for env_id in sorted(set(int(r["env"]) for r in trace_rows)):
        tr = [r for r in trace_rows if int(r["env"]) == env_id]
        arr = lambda k: np.asarray([r[k] for r in tr], dtype=np.float64)
        rows.append({
            "env": env_id,
            "length": len(tr),
            "term_type": term_types[env_id],
            "vel_rmse": float(np.sqrt(np.mean(arr("vel_err") ** 2))),
            "vel_max": float(np.max(arr("vel_err"))),
            "yaw_rate_mae": float(np.mean(np.abs(arr("yaw_rate_err")))),
            "max_tas": float(np.max(arr("tas"))),
            "max_roll_deg": float(np.degrees(np.max(np.abs(arr("roll"))))),
            "max_pitch_deg": float(np.degrees(np.max(np.abs(arr("pitch"))))),
            "max_omega": float(np.max(arr("omega_norm"))),
            "min_altitude": float(np.min(arr("altitude"))),
        })
    return rows


def save_csv(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_env(trace, out_png, summary):
    t = np.asarray([r["time_s"] for r in trace])
    fig, axes = plt.subplots(5, 2, figsize=(17, 15), sharex=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(t, [r["target_vx"] for r in trace], label="target vx")
    ax.plot(t, [r["target_vy"] for r in trace], label="target vy")
    ax.plot(t, [r["target_vz"] for r in trace], label="target vz")
    ax.plot(t, [r["target_yaw_rate"] for r in trace], label="target yaw rate")
    ax.set_title("Command")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.plot(t, [r["local_vx"] for r in trace], label="vx")
    ax.plot(t, [r["target_vx"] for r in trace], "--", label="target vx")
    ax.plot(t, [r["local_vy"] for r in trace], label="vy")
    ax.plot(t, [r["target_vy"] for r in trace], "--", label="target vy")
    ax.set_title("XY tracking")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.plot(t, [r["vz"] for r in trace], label="vz")
    ax.plot(t, [r["target_vz"] for r in trace], "--", label="target vz")
    ax.plot(t, [r["vel_err"] for r in trace], label="vel err")
    ax.set_title("Z / velocity error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(t, [r["r"] for r in trace], label="yaw rate r")
    ax.plot(t, [r["target_yaw_rate"] for r in trace], "--", label="target yaw rate")
    ax.plot(t, [r["yaw_rate_err"] for r in trace], label="yaw rate err")
    ax.set_title("Yaw-rate tracking")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[4]
    ax.plot(t, [r["tas"] for r in trace], label="TAS")
    ax.axhline(5.0, color="tab:red", ls="--", label="speed bad_done")
    ax.axhline(3.0 ** 0.5, color="tab:orange", ls=":", label="dense start")
    ax.set_title("Speed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[5]
    ax.plot(t, np.degrees([r["roll"] for r in trace]), label="roll")
    ax.plot(t, np.degrees([r["pitch"] for r in trace]), label="pitch")
    ax.axhline(30, color="tab:red", ls="--", label="roll bad")
    ax.axhline(-30, color="tab:red", ls="--")
    ax.axhline(25, color="tab:purple", ls="--", label="pitch bad")
    ax.axhline(-25, color="tab:purple", ls="--")
    ax.axhline(20, color="tab:blue", ls=":", label="roll dense")
    ax.axhline(-20, color="tab:blue", ls=":")
    ax.axhline(15, color="tab:orange", ls=":", label="pitch dense")
    ax.axhline(-15, color="tab:orange", ls=":")
    ax.set_title("Attitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[6]
    ax.plot(t, [r["omega_norm"] for r in trace], label="omega norm")
    ax.plot(t, [r["p"] for r in trace], label="p")
    ax.plot(t, [r["q"] for r in trace], label="q")
    ax.plot(t, [r["r"] for r in trace], label="r")
    ax.axhline(4.0, color="tab:red", ls="--", label="omega bad")
    ax.set_title("Angular rates")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[7]
    ax.plot(t, [r["pid_target_roll_deg"] for r in trace], label="pid target roll")
    ax.plot(t, [r["pid_target_pitch_deg"] for r in trace], label="pid target pitch")
    ax.plot(t, [r["pid_target_yaw_rate"] for r in trace], label="pid target yaw rate")
    ax.set_title("PID targets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[8]
    for key in ["f1", "f2", "f3", "f4", "f5"]:
        ax.plot(t, [r[key] for r in trace], label=key)
    ax.set_title("Motor forces")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    ax = axes[9]
    ax.plot(t, [r["altitude"] for r in trace], label="altitude")
    ax.plot(t, [r["alpha_deg"] for r in trace], label="alpha deg")
    ax.plot(t, [r["beta_deg"] for r in trace], label="beta deg")
    ax.set_title("Altitude / aero")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(
        f"PID four-axis stress env {summary['env']} term={summary['term_type']} "
        f"len={summary['length']} vel_rmse={summary['vel_rmse']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run(args, device):
    env = ControlEnv(
        num_envs=args.num_envs,
        config=args.config_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    env.task.sync_command = lambda _env: None
    env.task.yaw_command_enable = True
    env.task.yaw_hold_enable = False
    env.task.yaw_tracking_enable = True
    pid = make_pid(env, args, device)
    pid.reset()
    env.reset()

    active = torch.ones(env.n, dtype=torch.bool, device=device)
    term_types = ["sequence_end"] * env.n
    all_rows = []
    for step in range(args.max_steps):
        vx, vy, vz, yaw_rate = make_command(step, env.model.dt, env.n, args.scenario)
        apply_command(env, vx, vy, vz, yaw_rate)
        with torch.no_grad():
            action = pid.compute_action(env)
        action[~active] = 0.0
        _obs, reward, done, bad_done, exceed, _info = env.step(action)
        rows = collect(env, reward, action, pid, step)
        for row in rows:
            if active[int(row["env"])].item():
                all_rows.append(row)

        finished = active & (done.bool() | bad_done.bool() | exceed.bool())
        if torch.any(finished):
            done_np = tensor_np(done).astype(bool)
            bad_np = tensor_np(bad_done).astype(bool)
            exceed_np = tensor_np(exceed).astype(bool)
            for i in torch.where(finished)[0].detach().cpu().numpy():
                if bad_np[i]:
                    term_types[i] = "bad_done"
                elif done_np[i]:
                    term_types[i] = "done"
                elif exceed_np[i]:
                    term_types[i] = "timeout"
            active[finished] = False
        if step % 100 == 0:
            print(f"[pid-four-axis] step={step:04d} active={int(active.sum().item())}/{env.n}")
        if not torch.any(active):
            break
    env.close()
    return all_rows, term_types


def main():
    args = parse_args()
    os.environ["RC_HUMAN_YAW_COMMAND_ENABLE"] = "1"
    os.environ["RC_HUMAN_YAW_HOLD_ENABLE"] = "0"
    seed_everything(args.seed)
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, term_types = run(args, device)
    summary = summarize(rows, term_types)
    save_csv(rows, out_dir / "pid_four_axis_stress_traces.csv")
    save_csv(summary, out_dir / "pid_four_axis_stress_summary.csv")
    save_csv([vars(args) | {"actual_device": str(device)}], out_dir / "pid_four_axis_stress_params.csv")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for item in summary:
        trace = [r for r in rows if int(r["env"]) == int(item["env"])]
        plot_env(trace, plot_dir / f"pid_four_axis_env{int(item['env'])}.png", item)
    bad = sum(1 for item in summary if item["term_type"] == "bad_done")
    print(f"[pid-four-axis] bad_done={bad}/{len(summary)}")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
