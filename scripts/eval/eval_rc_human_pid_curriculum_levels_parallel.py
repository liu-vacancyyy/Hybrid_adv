#!/usr/bin/env python
"""Evaluate a LearningToFly-style PID baseline on rc_human curriculum levels."""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from algorithms.pid.rc_pid import RCPIDController  # noqa: E402
from envs.control_env import ControlEnv            # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--disable-tracking-bad-done", action="store_true",
                   help="Remove RCHumanTrackingError termination so tracking metrics are not truncated by speed-error bad_done.")
    p.add_argument("--episodes-per-level", type=int, default=1)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=119)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--config-name", type=str, required=True)
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument("--mode-order", type=str, default="0 1 2 3 4 5")
    p.add_argument("--no-head-motor", action="store_true")
    p.add_argument("--head-max-scaled", type=float, default=0.45)
    p.add_argument("--xy-kp", type=float, default=0.95)
    p.add_argument("--xy-ki", type=float, default=0.10)
    p.add_argument("--xy-kd", type=float, default=0.015)
    p.add_argument("--xy-imax", type=float, default=0.35)
    p.add_argument("--xy-filt-hz", type=float, default=2.0)
    p.add_argument("--z-kp", type=float, default=2.0)
    p.add_argument("--z-ki", type=float, default=0.15)
    p.add_argument("--z-kd", type=float, default=0.008)
    p.add_argument("--z-imax", type=float, default=0.35)
    p.add_argument("--z-filt-hz", type=float, default=2.0)
    p.add_argument("--max-tilt-deg", type=float, default=12.0)
    p.add_argument("--max-horiz-accel", type=float, default=1.8)
    p.add_argument("--max-z-throttle-corr", type=float, default=0.08)
    p.add_argument("--yaw-p", type=float, default=1.6)
    p.add_argument("--max-yaw-rate", type=float, default=0.6)
    p.add_argument("--side-damp-p", type=float, default=0.20)
    p.add_argument("--alt-floor", type=float, default=None)
    p.add_argument("--alt-floor-buffer", type=float, default=0.8)
    p.add_argument("--alt-floor-vz-bias", type=float, default=0.8)
    p.add_argument("--save-per-level-plots", action="store_true")
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


def set_fixed_levels(task, level_tensor):
    task.curriculum_enable = True
    task.curriculum_level[:] = level_tensor.to(task.device).long()
    task.max_curriculum_level = max(
        int(task.max_curriculum_level), int(level_tensor.max().item())
    )
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0
    if hasattr(task, "dwell_left"):
        reset_like = task.dwell_left < 0
        task.dwell_left[reset_like] = 0


def fixed_modes(task, level_tensor):
    levels = level_tensor.to(task.device).long()
    slot = torch.clamp(levels // int(task.levels_per_mode), 0, int(task.active_mode_slots) - 1)
    return task.mode_order[slot].detach().cpu().long().numpy()


def make_pid(env, args, device):
    return RCPIDController(
        n=env.n,
        device=device,
        dt=env.model.dt,
        max_thrust_per_motor=env.model.max_F,
        use_head_motor=not args.no_head_motor,
        head_max_scaled=args.head_max_scaled,
        xy_gains=dict(
            kp=args.xy_kp,
            ki=args.xy_ki,
            kd=args.xy_kd,
            imax=args.xy_imax,
            filt_hz=args.xy_filt_hz,
        ),
        z_gains=dict(
            kp=args.z_kp,
            ki=args.z_ki,
            kd=args.z_kd,
            imax=args.z_imax,
            filt_hz=args.z_filt_hz,
        ),
        max_tilt_deg=args.max_tilt_deg,
        max_horiz_accel=args.max_horiz_accel,
        max_z_throttle_corr=args.max_z_throttle_corr,
        yaw_p=args.yaw_p,
        max_yaw_rate=args.max_yaw_rate,
        side_damp_p=args.side_damp_p,
        alt_floor=args.alt_floor,
        alt_floor_buffer=args.alt_floor_buffer,
        alt_floor_vz_bias=args.alt_floor_vz_bias,
    )


def collect_arrays(env, reward, action, prev_action, pid):
    task = env.task
    roll, pitch, heading = env.model.get_posture()
    _npos, _epos, altitude = env.model.get_position()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    alpha = env.model.get_AOA()
    beta = env.model.get_AOS()
    p, q, r = env.model.get_angular_velocity()
    f1, f2, f3, f4, f5 = env.model.get_F()
    local_vx, local_vy = task.ground_to_local_velocity(vx_n, vy_e, heading)
    err_vx, err_vy = task.heading_local_velocity_error(vx_n, vy_e, heading)
    err_vz = vz - task.target_vz
    yaw_err = torch.atan2(
        torch.sin(heading - task.target_heading),
        torch.cos(heading - task.target_heading),
    )
    vel_err = torch.sqrt(err_vx * err_vx + err_vy * err_vy + err_vz * err_vz)
    att_err = torch.sqrt(roll * roll + pitch * pitch)
    if prev_action is None:
        action_delta = torch.zeros(env.n, device=env.device)
    else:
        action_delta = torch.mean(torch.abs(action - prev_action), dim=1)

    if hasattr(env.model, "get_wind_ned"):
        wind_n, wind_e, wind_d = env.model.get_wind_ned()
    else:
        wind_n = wind_e = wind_d = torch.zeros_like(vz)

    dbg = getattr(pid, "debug", {})
    target_roll = dbg.get("target_roll", torch.zeros_like(vz))
    target_pitch = dbg.get("target_pitch", torch.zeros_like(vz))
    target_yaw_rate = dbg.get("target_yaw_rate", torch.zeros_like(vz))
    throttle = dbg.get("throttle", torch.zeros_like(vz))
    head_scaled = dbg.get("head_scaled", torch.zeros_like(vz))
    alt_floor_margin = dbg.get("alt_floor_margin", torch.zeros_like(vz))

    return {
        "reward": tensor_np(reward),
        "altitude": tensor_np(altitude),
        "local_vx": tensor_np(local_vx),
        "target_vx": tensor_np(task.target_vx),
        "local_vy": tensor_np(local_vy),
        "target_vy": tensor_np(task.target_vy),
        "vz": tensor_np(vz),
        "target_vz": tensor_np(task.target_vz),
        "heading": tensor_np(heading),
        "target_heading": tensor_np(task.target_heading),
        "target_yaw_rate": tensor_np(task.target_yaw_rate),
        "yaw_err": tensor_np(yaw_err),
        "vel_err": tensor_np(vel_err),
        "vx_abs_err": tensor_np(torch.abs(err_vx)),
        "vy_abs_err": tensor_np(torch.abs(err_vy)),
        "vz_abs_err": tensor_np(torch.abs(err_vz)),
        "att_err": tensor_np(att_err),
        "roll": tensor_np(roll),
        "pitch": tensor_np(pitch),
        "p": tensor_np(p),
        "q": tensor_np(q),
        "r": tensor_np(r),
        "omega_norm": tensor_np(torch.sqrt(p * p + q * q + r * r)),
        "alpha_deg": np.degrees(tensor_np(alpha)),
        "beta_deg": np.degrees(tensor_np(beta)),
        "wind_north": tensor_np(wind_n),
        "wind_east": tensor_np(wind_e),
        "wind_down": tensor_np(wind_d),
        "raw_vx": tensor_np(task.raw_vx),
        "raw_vy": tensor_np(task.raw_vy),
        "raw_vz": tensor_np(task.raw_vz),
        "raw_yaw": tensor_np(task.raw_yaw),
        "stick_vx": tensor_np(task.stick_vx),
        "stick_vy": tensor_np(task.stick_vy),
        "stick_vz": tensor_np(task.stick_vz),
        "stick_yaw": tensor_np(task.stick_yaw),
        "vx_forward_limit": tensor_np(task.vx_forward_limit),
        "mode5_release_state": tensor_np(task.mode5_release_state),
        "action_delta": tensor_np(action_delta),
        "f1": tensor_np(f1),
        "f2": tensor_np(f2),
        "f3": tensor_np(f3),
        "f4": tensor_np(f4),
        "f5": tensor_np(f5),
        "pid_target_roll_deg": np.degrees(tensor_np(target_roll)),
        "pid_target_pitch_deg": np.degrees(tensor_np(target_pitch)),
        "pid_target_yaw_rate": tensor_np(target_yaw_rate),
        "pid_throttle": tensor_np(throttle),
        "pid_head_scaled": tensor_np(head_scaled),
        "pid_alt_floor_margin": tensor_np(alt_floor_margin),
    }


def append_trace_rows(traces, arrays, active_np, levels_np, modes_np, episode_np, step, dt):
    idxs = np.where(active_np)[0]
    for i in idxs:
        row = {
            "level": int(levels_np[i]),
            "mode_id": int(modes_np[i]),
            "episode": int(episode_np[i]),
            "step": int(step),
            "time_s": float(step * dt),
        }
        for key, values in arrays.items():
            row[key] = float(values[i])
        traces[i].append(row)


def run_vectorized_eval(args, device):
    seed_everything(args.seed)
    levels = np.arange(args.min_level, args.max_level + 1, dtype=np.int64)
    levels_np = np.repeat(levels, args.episodes_per_level)
    episode_np = np.tile(np.arange(args.episodes_per_level, dtype=np.int64), len(levels))
    level_tensor = torch.as_tensor(levels_np, dtype=torch.long, device=device)

    env = ControlEnv(
        num_envs=len(levels_np),
        config=args.config_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    if args.disable_tracking_bad_done and hasattr(env.task, "termination_conditions"):
        before = len(env.task.termination_conditions)
        env.task.termination_conditions = [
            cond for cond in env.task.termination_conditions
            if cond.__class__.__name__ != "RCHumanTrackingError"
        ]
        removed = before - len(env.task.termination_conditions)
        if removed:
            print(f"[pid-eval] disabled {removed} RCHumanTrackingError termination condition(s)")
    set_fixed_levels(env.task, level_tensor)
    modes_np = fixed_modes(env.task, level_tensor)
    pid = make_pid(env, args, device)
    pid.reset()

    env.reset()
    set_fixed_levels(env.task, level_tensor)
    modes_np = fixed_modes(env.task, level_tensor)

    active = torch.ones(env.n, dtype=torch.bool, device=device)
    term_type = np.full(env.n, "truncated", dtype=object)
    prev_action = None
    traces = [[] for _ in range(env.n)]

    for step in range(args.max_steps):
        active_np = tensor_np(active).astype(bool)
        with torch.no_grad():
            action = pid.compute_action(env)
        obs, reward, done, bad_done, exceed, _info = env.step(action)
        _ = obs
        set_fixed_levels(env.task, level_tensor)
        arrays = collect_arrays(env, reward, action, prev_action, pid)
        append_trace_rows(
            traces,
            arrays,
            active_np,
            levels_np,
            modes_np,
            episode_np,
            step,
            env.model.dt,
        )
        prev_action = action.detach().clone()

        finished = active & (done.bool() | bad_done.bool() | exceed.bool())
        if torch.any(finished):
            finished_idx = torch.where(finished)[0].detach().cpu().numpy()
            done_np = done.detach().cpu().numpy().astype(bool)
            bad_np = bad_done.detach().cpu().numpy().astype(bool)
            exceed_np = exceed.detach().cpu().numpy().astype(bool)
            reset_mask = finished
            pid.reset(mask=reset_mask)
            for i in finished_idx:
                if bad_np[i]:
                    term_type[i] = "bad_done"
                elif done_np[i]:
                    term_type[i] = "done"
                elif exceed_np[i]:
                    term_type[i] = "timeout"
            active[finished] = False

        if step % 100 == 0:
            print(f"[pid-eval] step={step:04d} active={int(active.sum().item())}/{env.n}")
        if not torch.any(active):
            break

    env.close()
    return traces, term_type


def summarize_trace(trace, term_type):
    vel = np.asarray([x["vel_err"] for x in trace], dtype=np.float64)
    vx = np.asarray([x["vx_abs_err"] for x in trace], dtype=np.float64)
    vy = np.asarray([x["vy_abs_err"] for x in trace], dtype=np.float64)
    vz = np.asarray([x["vz_abs_err"] for x in trace], dtype=np.float64)
    yaw = np.abs(np.asarray([x["yaw_err"] for x in trace], dtype=np.float64))
    att = np.asarray([x["att_err"] for x in trace], dtype=np.float64)
    action_delta = np.asarray([x["action_delta"] for x in trace], dtype=np.float64)
    altitude = np.asarray([x["altitude"] for x in trace], dtype=np.float64)
    roll_abs = np.abs(np.asarray([x["roll"] for x in trace], dtype=np.float64))
    pitch_abs = np.abs(np.asarray([x["pitch"] for x in trace], dtype=np.float64))
    omega_norm = np.asarray([x["omega_norm"] for x in trace], dtype=np.float64)
    ret = float(np.sum([x["reward"] for x in trace]))
    first = trace[0]
    return {
        "level": first["level"],
        "mode_id": first["mode_id"],
        "episode": first["episode"],
        "vx_forward_limit": first["vx_forward_limit"],
        "length": len(trace),
        "return": ret,
        "vel_mae": float(np.mean(vel)),
        "vel_rmse": float(np.sqrt(np.mean(vel * vel))),
        "vx_mae": float(np.mean(vx)),
        "vy_mae": float(np.mean(vy)),
        "vz_mae": float(np.mean(vz)),
        "vel_p95": float(np.percentile(vel, 95)),
        "yaw_mae_rad": float(np.mean(yaw)),
        "yaw_mae_deg": float(np.degrees(np.mean(yaw))),
        "att_mae_rad": float(np.mean(att)),
        "att_mae_deg": float(np.degrees(np.mean(att))),
        "max_abs_alpha_deg": float(np.max(np.abs([x["alpha_deg"] for x in trace]))),
        "max_abs_beta_deg": float(np.max(np.abs([x["beta_deg"] for x in trace]))),
        "min_altitude": float(np.min(altitude)),
        "max_abs_roll_deg": float(np.degrees(np.max(roll_abs))),
        "max_abs_pitch_deg": float(np.degrees(np.max(pitch_abs))),
        "max_omega_norm": float(np.max(omega_norm)),
        "action_delta_mean": float(np.mean(action_delta)),
        "term_type": str(term_type),
        "success": int(str(term_type) != "bad_done"),
    }


def grouped_means(summary_rows, key_name):
    rows = []
    for key in sorted(set(r[key_name] for r in summary_rows)):
        items = [r for r in summary_rows if r[key_name] == key]
        row = {key_name: key}
        if key_name == "level":
            row["mode_id"] = items[0]["mode_id"]
            row["vx_forward_limit"] = float(np.mean([r["vx_forward_limit"] for r in items]))
        else:
            row["level_min"] = min(r["level"] for r in items)
            row["level_max"] = max(r["level"] for r in items)
        for name in [
            "return", "vel_mae", "vel_rmse", "vx_mae", "vy_mae", "vz_mae",
            "vel_p95", "yaw_mae_deg", "att_mae_deg", "max_abs_alpha_deg",
            "max_abs_beta_deg", "min_altitude", "max_abs_roll_deg",
            "max_abs_pitch_deg", "max_omega_norm", "action_delta_mean",
            "success", "length",
        ]:
            out_name = "success_rate" if name == "success" else name
            row[out_name] = float(np.mean([r[name] for r in items]))
        rows.append(row)
    return rows


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def shade_modes(ax, level_rows):
    if not level_rows:
        return
    current = level_rows[0]["mode_id"]
    start = level_rows[0]["level"]
    sentinel = {"level": level_rows[-1]["level"] + 1, "mode_id": None}
    for row in level_rows[1:] + [sentinel]:
        mode = row["mode_id"]
        if mode == current:
            continue
        end = row["level"] - 1
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.06)
        ax.text(
            (start + end) * 0.5,
            0.98,
            f"mode {current}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
        current = mode
        start = row["level"]


def plot_level_summary(level_rows, out_png):
    levels = np.asarray([r["level"] for r in level_rows])
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    plots = [
        ("success_rate", "Success rate", "rate"),
        ("vel_mae", "Velocity tracking MAE", "m/s"),
        ("vel_p95", "Velocity tracking p95", "m/s"),
        ("yaw_mae_deg", "Yaw hold MAE", "deg"),
        ("att_mae_deg", "Attitude error MAE", "deg"),
        ("max_abs_alpha_deg", "Max |alpha|", "deg"),
    ]
    for ax, (key, title, ylabel) in zip(axes.reshape(-1), plots):
        shade_modes(ax, level_rows)
        ax.plot(levels, [r[key] for r in level_rows], lw=1.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("curriculum level")
    axes[-1, 1].set_xlabel("curriculum level")
    fig.suptitle("RC human PID baseline across fixed curriculum levels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_one_level(trace, out_png):
    t = np.asarray([x["time_s"] for x in trace])
    level = trace[0]["level"]
    mode = trace[0]["mode_id"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(t, [x["target_vx"] for x in trace], label="target vx")
    ax.plot(t, [x["target_vy"] for x in trace], label="target vy")
    ax.plot(t, [x["target_vz"] for x in trace], label="target vz")
    ax.set_title("Command")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.plot(t, [x["raw_vx"] for x in trace], label="raw vx")
    ax.plot(t, [x["raw_vy"] for x in trace], label="raw vy")
    ax.plot(t, [x["raw_vz"] for x in trace], label="raw vz")
    ax.set_title("RC raw")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.plot(t, [x["local_vx"] for x in trace], label="local vx")
    ax.plot(t, [x["target_vx"] for x in trace], "--", label="target vx")
    ax.plot(t, [x["local_vy"] for x in trace], label="local vy")
    ax.plot(t, [x["target_vy"] for x in trace], "--", label="target vy")
    ax.set_title("Horizontal velocity tracking")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[3]
    ax.plot(t, [x["vz"] for x in trace], label="vz")
    ax.plot(t, [x["target_vz"] for x in trace], "--", label="target vz")
    ax.plot(t, np.degrees([x["yaw_err"] for x in trace]), label="yaw err deg")
    ax.set_title("Vertical/yaw tracking")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[4]
    ax.plot(t, [x["pid_target_roll_deg"] for x in trace], label="pid target roll")
    ax.plot(t, [x["pid_target_pitch_deg"] for x in trace], label="pid target pitch")
    ax.plot(t, [x["pid_target_yaw_rate"] for x in trace], label="pid target yaw rate")
    ax.set_title("PID internal targets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[5]
    ax.plot(t, [x["pid_throttle"] for x in trace], label="pid throttle")
    ax.plot(t, [x["pid_head_scaled"] for x in trace], label="pid head scaled")
    ax.set_title("PID throttle/head")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[6]
    ax.plot(t, [x["alpha_deg"] for x in trace], label="alpha deg")
    ax.plot(t, [x["beta_deg"] for x in trace], label="beta deg")
    ax.plot(t, np.degrees([x["att_err"] for x in trace]), label="att err deg")
    ax.set_title("Alpha/beta and attitude")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[7]
    ax.plot(t, [x["f1"] for x in trace], label="f1")
    ax.plot(t, [x["f2"] for x in trace], label="f2")
    ax.plot(t, [x["f3"] for x in trace], label="f3")
    ax.plot(t, [x["f4"] for x in trace], label="f4")
    ax.plot(t, [x["f5"] for x in trace], label="f5")
    ax.set_title("Motor force")
    ax.set_ylabel("N")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"PID rc_human level {level} mode {mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_all_levels(trace_by_level, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for level, trace in trace_by_level.items():
        plot_one_level(trace, out_dir / f"level_{level:03d}.png")


def main():
    args = parse_args()
    if args.mode_order:
        os.environ["RC_HUMAN_MODE_ORDER"] = args.mode_order
        os.environ["RC_HUMAN_MAX_MODE_SLOTS"] = str(len(args.mode_order.split()))
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces, term_type = run_vectorized_eval(args, device)
    nonempty = [(i, tr) for i, tr in enumerate(traces) if tr]
    summary_rows = [summarize_trace(tr, term_type[i]) for i, tr in nonempty]
    level_rows = grouped_means(summary_rows, "level")
    mode_rows = grouped_means(summary_rows, "mode_id")

    trace_rows = [row for _i, tr in nonempty for row in tr]
    trace_by_level = {}
    for _i, tr in nonempty:
        level = tr[0]["level"]
        episode = tr[0]["episode"]
        if episode == 0:
            trace_by_level[level] = tr

    save_csv(summary_rows, out_dir / "rc_human_pid_level_summary_raw.csv")
    save_csv(level_rows, out_dir / "rc_human_pid_level_summary.csv")
    save_csv(mode_rows, out_dir / "rc_human_pid_mode_summary.csv")
    save_csv(trace_rows, out_dir / "rc_human_pid_level_traces.csv")
    plot_level_summary(level_rows, out_dir / "rc_human_pid_level_tracking.png")
    if args.save_per_level_plots:
        plot_all_levels(trace_by_level, out_dir / "per_level_plots")

    params = vars(args).copy()
    params["actual_device"] = str(device)
    save_csv([params], out_dir / "rc_human_pid_params.csv")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
