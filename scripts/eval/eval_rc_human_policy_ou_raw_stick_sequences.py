#!/usr/bin/env python
"""Evaluate an rc_human policy on fixed OU raw-stick command sequences."""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from algorithms.ppo.ppo_actor import PPOActor  # noqa: E402
from envs.control_env import ControlEnv        # noqa: E402


class ActorArgs:
    def __init__(self, args, device):
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.use_safety_aux = False
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--sequence-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--model-name", default="HYBRID_NEW")
    p.add_argument("--mode-order", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=555)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--save-per-sequence-plots", action="store_true")
    p.add_argument("--top-k-plots", type=int, default=8)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)

    p.add_argument("--hidden-size", default="128 128")
    p.add_argument("--act-hidden-size", default="128 128")
    p.add_argument("--activation-id", type=int, default=1)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument(
        "--no-feature-normalization",
        dest="use_feature_normalization",
        action="store_false",
    )
    p.set_defaults(use_feature_normalization=True)
    p.add_argument(
        "--no-recurrent-policy",
        dest="use_recurrent_policy",
        action="store_false",
    )
    p.set_defaults(use_recurrent_policy=True)
    p.add_argument("--recurrent-hidden-size", type=int, default=128)
    p.add_argument("--recurrent-hidden-layers", type=int, default=1)
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


def load_actor(env, args, device):
    actor = PPOActor(ActorArgs(args, device), env.observation_space, env.action_space, device)
    state = torch.load(args.ckpt_path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def load_sequences(path, max_steps):
    groups = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["level"]), int(row["variant"]))
            groups[key].append(row)

    sequences = []
    for (level, variant), rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["step"]))
        steps = min(len(rows), max_steps)
        raw = np.zeros((steps, 4), dtype=np.float32)
        desired = np.zeros((steps, 4), dtype=np.float32)
        for i in range(steps):
            r = rows[i]
            raw[i] = [
                float(r["raw_vx"]),
                float(r["raw_vy"]),
                float(r["raw_vz"]),
                float(r["raw_yaw"]),
            ]
            desired[i] = [
                float(r.get("desired_raw_vx", r["raw_vx"])),
                float(r.get("desired_raw_vy", r["raw_vy"])),
                float(r.get("desired_raw_vz", r["raw_vz"])),
                float(r.get("desired_raw_yaw", r["raw_yaw"])),
            ]
        sequences.append({
            "level": int(level),
            "variant": int(variant),
            "mode_id": int(rows[0]["mode_id"]),
            "raw": raw,
            "desired": desired,
            "steps": steps,
        })
    if not sequences:
        raise ValueError(f"No sequences loaded from {path}")
    return sequences


def set_fixed_context(task, levels, modes):
    task.curriculum_enable = True
    task.curriculum_level[:] = levels.long()
    task.max_curriculum_level = max(
        int(task.max_curriculum_level),
        int(levels.max().item()),
    )
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0
    task.operation_mode[:] = modes.long()
    task.vx_forward_limit[:] = task._curriculum_vx_forward_limit(levels.long(), modes.long())
    task.dwell_left[:] = 10**9
    task.mode5_release_state[:] = 0
    task.mode5_hold_elapsed[:] = 0
    task.mode5_recovery_left[:] = 0
    task.mode5_pre_release_raw[:] = 0.0


def apply_raw_step(env, raw_step, desired_step, levels, modes):
    task = env.task
    set_fixed_context(task, levels, modes)
    raw = torch.as_tensor(raw_step, dtype=torch.float32, device=env.device).clone()
    desired = torch.as_tensor(desired_step, dtype=torch.float32, device=env.device).clone()
    raw = torch.clamp(raw, -1.0, 1.0)
    desired = torch.clamp(desired, -1.0, 1.0)
    if not task.yaw_command_enable:
        raw[:, 3] = 0.0
        desired[:, 3] = 0.0

    task.raw_vx[:] = raw[:, 0]
    task.raw_vy[:] = raw[:, 1]
    task.raw_vz[:] = raw[:, 2]
    task.raw_yaw[:] = raw[:, 3]
    task.desired_raw_vx[:] = desired[:, 0]
    task.desired_raw_vy[:] = desired[:, 1]
    task.desired_raw_vz[:] = desired[:, 2]
    task.desired_raw_yaw[:] = desired[:, 3]
    task.command_rate_limited[:] = False

    mask = torch.ones(env.n, dtype=torch.bool, device=env.device)
    task._apply_altitude_raw_vz_guard(mask, env)
    task._apply_px4_vtol_mc_manual_sticks(mask, env)


def collect_arrays(env, obs, reward, action, prev_action):
    task = env.task
    roll, pitch, heading = env.model.get_posture()
    p, q, r = env.model.get_angular_velocity()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    npos, epos, altitude = env.model.get_position()
    alpha = env.model.get_AOA()
    beta = env.model.get_AOS()
    tas = env.model.get_TAS()
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
    omega_norm = torch.sqrt(p * p + q * q + r * r)
    if prev_action is None:
        action_delta = torch.zeros(env.n, device=env.device)
    else:
        action_delta = torch.mean(torch.abs(action - prev_action), dim=1)

    if hasattr(env.model, "get_wind_ned"):
        wind_n, wind_e, wind_d = env.model.get_wind_ned()
    else:
        wind_n = wind_e = wind_d = torch.zeros_like(vz)

    obs_np = tensor_np(obs)
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
        "target_yaw_rate": tensor_np(task.target_yaw_rate),
        "yaw_err": tensor_np(yaw_err),
        "vel_err": tensor_np(vel_err),
        "att_err": tensor_np(att_err),
        "roll": tensor_np(roll),
        "pitch": tensor_np(pitch),
        "p": tensor_np(p),
        "q": tensor_np(q),
        "r": tensor_np(r),
        "omega_norm": tensor_np(omega_norm),
        "tas": tensor_np(tas),
        "altitude": tensor_np(altitude),
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
        "action_delta": tensor_np(action_delta),
        "f1": tensor_np(f1),
        "f2": tensor_np(f2),
        "f3": tensor_np(f3),
        "f4": tensor_np(f4),
        "f5": tensor_np(f5),
    }
    action_np = tensor_np(action)
    for i in range(action_np.shape[1]):
        arrays[f"action_{i}"] = action_np[:, i]
    for i in range(min(8, obs_np.shape[1])):
        arrays[f"obs_{i:02d}"] = obs_np[:, i]
    return arrays


def append_trace_rows(traces, arrays, active_np, levels, modes, variants, step, dt):
    for i in np.where(active_np)[0]:
        row = {
            "level": int(levels[i]),
            "mode_id": int(modes[i]),
            "variant": int(variants[i]),
            "step": int(step),
            "time_s": float(step * dt),
        }
        for key, values in arrays.items():
            row[key] = float(values[i])
        traces[i].append(row)


def summarize_trace(trace, term_type):
    vel = np.asarray([x["vel_err"] for x in trace], dtype=np.float64)
    yaw = np.abs(np.asarray([x["yaw_err"] for x in trace], dtype=np.float64))
    att = np.asarray([x["att_err"] for x in trace], dtype=np.float64)
    tas = np.asarray([x["tas"] for x in trace], dtype=np.float64)
    roll = np.abs(np.asarray([x["roll"] for x in trace], dtype=np.float64))
    pitch = np.abs(np.asarray([x["pitch"] for x in trace], dtype=np.float64))
    omega = np.asarray([x["omega_norm"] for x in trace], dtype=np.float64)
    altitude = np.asarray([x["altitude"] for x in trace], dtype=np.float64)
    action_delta = np.asarray([x["action_delta"] for x in trace], dtype=np.float64)
    ret = float(np.sum([x["reward"] for x in trace]))
    first = trace[0]
    return {
        "level": int(first["level"]),
        "mode_id": int(first["mode_id"]),
        "variant": int(first["variant"]),
        "length": len(trace),
        "return": ret,
        "vel_mae": float(np.mean(vel)),
        "vel_rmse": float(np.sqrt(np.mean(vel * vel))),
        "vel_max": float(np.max(vel)),
        "yaw_mae_deg": float(np.degrees(np.mean(yaw))),
        "att_mae_deg": float(np.degrees(np.mean(att))),
        "max_tas": float(np.max(tas)),
        "max_roll_deg": float(np.degrees(np.max(roll))),
        "max_pitch_deg": float(np.degrees(np.max(pitch))),
        "max_omega": float(np.max(omega)),
        "min_altitude": float(np.min(altitude)),
        "action_delta_mean": float(np.mean(action_delta)),
        "term_type": str(term_type),
        "survived": int(str(term_type) != "bad_done"),
    }


def save_csv(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence(trace, out_png, title_suffix=""):
    t = np.asarray([x["time_s"] for x in trace])
    level = int(trace[0]["level"])
    mode = int(trace[0]["mode_id"])
    variant = int(trace[0]["variant"])
    fig, axes = plt.subplots(5, 2, figsize=(17, 15), sharex=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    for key in ["raw_vx", "raw_vy", "raw_vz", "raw_yaw"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("OU raw stick")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for key in ["stick_vx", "stick_vy", "stick_vz", "stick_yaw"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("PX4 shaped stick")
    ax.set_ylim(-1.05, 1.05)
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
    ax.plot(t, [x["vel_err"] for x in trace], label="vel error")
    ax.plot(t, [x["tas"] for x in trace], label="TAS")
    ax.axhline(5.0, color="tab:red", ls="--", lw=1.0, label="speed bad_done 5")
    ax.axhline(3.0 ** 0.5, color="tab:orange", ls=":", lw=1.0, label="speed dense start")
    ax.set_title("Velocity error and speed")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[5]
    ax.plot(t, np.degrees([x["roll"] for x in trace]), label="roll")
    ax.plot(t, np.degrees([x["pitch"] for x in trace]), label="pitch")
    ax.axhline(20, color="tab:blue", ls=":", lw=0.9, label="roll dense 20")
    ax.axhline(-20, color="tab:blue", ls=":", lw=0.9)
    ax.axhline(15, color="tab:orange", ls=":", lw=0.9, label="pitch dense 15")
    ax.axhline(-15, color="tab:orange", ls=":", lw=0.9)
    ax.axhline(30, color="tab:red", ls="--", lw=0.9, label="roll bad 30")
    ax.axhline(-30, color="tab:red", ls="--", lw=0.9)
    ax.axhline(25, color="tab:purple", ls="--", lw=0.9, label="pitch bad 25")
    ax.axhline(-25, color="tab:purple", ls="--", lw=0.9)
    ax.set_title("Attitude")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[6]
    ax.plot(t, [x["omega_norm"] for x in trace], label="omega norm")
    ax.plot(t, [x["p"] for x in trace], label="p")
    ax.plot(t, [x["q"] for x in trace], label="q")
    ax.plot(t, [x["r"] for x in trace], label="r")
    ax.set_title("Angular rates")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[7]
    for key in ["f1", "f2", "f3", "f4", "f5"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Actuator outputs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    ax = axes[8]
    ax.plot(t, [x["altitude"] for x in trace], label="altitude")
    ax.set_title("Altitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[9]
    ax.plot(t, [x["alpha_deg"] for x in trace], label="alpha deg")
    ax.plot(t, [x["beta_deg"] for x in trace], label="beta deg")
    ax.plot(t, np.degrees([x["att_err"] for x in trace]), label="att err deg")
    ax.set_title("Aero/attitude error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Policy on OU raw stick: level {level} mode {mode} variant {variant} {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_overview(summary_rows, out_png):
    rows = sorted(summary_rows, key=lambda r: (r["level"], r["variant"]))
    x = np.arange(len(rows))
    labels = [f"L{r['level']}v{r['variant']}" for r in rows]
    bad = np.asarray([r["term_type"] == "bad_done" for r in rows])
    colors = np.where(bad, "tab:red", "tab:blue")

    fig, axes = plt.subplots(4, 1, figsize=(18, 13), sharex=True)
    ax = axes[0]
    ax.bar(x, [r["vel_rmse"] for r in rows], color=colors)
    ax.set_ylabel("m/s")
    ax.set_title("Velocity RMSE")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, [r["max_tas"] for r in rows], color=colors)
    ax.axhline(5.0, color="tab:red", ls="--", lw=1.1, label="speed bad_done 5")
    ax.axhline(3.0 ** 0.5, color="tab:orange", ls=":", lw=1.1, label="dense start sqrt(3)")
    ax.set_ylabel("m/s")
    ax.set_title("Max TAS")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(x, [r["max_roll_deg"] for r in rows], marker="o", label="max |roll|")
    ax.plot(x, [r["max_pitch_deg"] for r in rows], marker="o", label="max |pitch|")
    ax.axhline(20, color="tab:blue", ls=":", lw=1.0, label="roll dense 20")
    ax.axhline(15, color="tab:orange", ls=":", lw=1.0, label="pitch dense 15")
    ax.axhline(30, color="tab:red", ls="--", lw=1.0, label="roll bad 30")
    ax.axhline(25, color="tab:purple", ls="--", lw=1.0, label="pitch bad 25")
    ax.set_ylabel("deg")
    ax.set_title("Max attitude")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4)

    ax = axes[3]
    ax.bar(x, [r["length"] for r in rows], color=colors)
    ax.set_ylabel("steps")
    ax.set_title("Episode length, red = bad_done")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_worst_grid(traces, summary_rows, out_png, top_k):
    rows = sorted(
        summary_rows,
        key=lambda r: (r["term_type"] != "bad_done", -r["vel_rmse"], -r["max_tas"]),
    )[:top_k]
    if not rows:
        return
    trace_by_key = {
        (int(t[0]["level"]), int(t[0]["variant"])): t
        for t in traces
        if t
    }
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3.0 * len(rows)), squeeze=False)
    for row_idx, row in enumerate(rows):
        trace = trace_by_key[(row["level"], row["variant"])]
        t = np.asarray([x["time_s"] for x in trace])
        prefix = f"L{row['level']} v{row['variant']} {row['term_type']}"

        ax = axes[row_idx, 0]
        ax.plot(t, [x["local_vx"] for x in trace], label="vx")
        ax.plot(t, [x["target_vx"] for x in trace], "--", label="target vx")
        ax.plot(t, [x["local_vy"] for x in trace], label="vy")
        ax.plot(t, [x["target_vy"] for x in trace], "--", label="target vy")
        ax.set_title(f"{prefix}: xy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[row_idx, 1]
        ax.plot(t, [x["vz"] for x in trace], label="vz")
        ax.plot(t, [x["target_vz"] for x in trace], "--", label="target vz")
        ax.plot(t, [x["vel_err"] for x in trace], label="vel err")
        ax.set_title("z/error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 2]
        for key in ["raw_vx", "raw_vy", "raw_vz"]:
            ax.plot(t, [x[key] for x in trace], label=key)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("raw stick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)

        ax = axes[row_idx, 3]
        ax.plot(t, [x["tas"] for x in trace], label="TAS")
        ax.plot(t, np.degrees([x["roll"] for x in trace]), label="roll deg")
        ax.plot(t, np.degrees([x["pitch"] for x in trace]), label="pitch deg")
        ax.axhline(5.0, color="tab:red", ls="--", lw=0.9)
        ax.axhline(-5.0, color="tab:red", ls="--", lw=0.9)
        ax.set_title("speed/attitude")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_eval(args, device, sequences):
    levels_np = np.asarray([s["level"] for s in sequences], dtype=np.int64)
    modes_np = np.asarray([s["mode_id"] for s in sequences], dtype=np.int64)
    variants_np = np.asarray([s["variant"] for s in sequences], dtype=np.int64)
    steps = min(args.max_steps, min(s["steps"] for s in sequences))
    raw_np = np.stack([s["raw"][:steps] for s in sequences], axis=0)
    desired_np = np.stack([s["desired"][:steps] for s in sequences], axis=0)

    level_tensor = torch.as_tensor(levels_np, dtype=torch.long, device=device)
    mode_tensor = torch.as_tensor(modes_np, dtype=torch.long, device=device)

    env = ControlEnv(
        num_envs=len(sequences),
        config=args.config_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    # Disable the task's random command generator; commands are supplied by CSV.
    env.task.sync_command = lambda _env: None
    actor = load_actor(env, args, device)

    env.reset()
    apply_raw_step(env, raw_np[:, 0, :], desired_np[:, 0, :], level_tensor, mode_tensor)
    obs = env.obs()

    active = torch.ones(env.n, dtype=torch.bool, device=device)
    term_type = np.full(env.n, "sequence_end", dtype=object)
    traces = [[] for _ in range(env.n)]
    prev_action = None

    for step in range(steps):
        active_np = tensor_np(active).astype(bool)
        masks = active.float().reshape(-1, 1)
        with torch.no_grad():
            action, _, rnn = actor(
                obs,
                torch.zeros(
                    (env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size),
                    device=device,
                ) if step == 0 else rnn,
                masks,
                deterministic=args.deterministic,
            )
        action = action.detach()
        action[~active] = 0.0

        obs_after, reward, done, bad_done, exceed, _info = env.step(action)
        arrays = collect_arrays(env, obs_after, reward, action, prev_action)
        append_trace_rows(
            traces,
            arrays,
            active_np,
            levels_np,
            modes_np,
            variants_np,
            step,
            env.model.dt,
        )
        prev_action = action.clone()

        finished = active & (done.bool() | bad_done.bool() | exceed.bool())
        if torch.any(finished):
            finished_idx = torch.where(finished)[0].detach().cpu().numpy()
            done_np = done.detach().cpu().numpy().astype(bool)
            bad_np = bad_done.detach().cpu().numpy().astype(bool)
            exceed_np = exceed.detach().cpu().numpy().astype(bool)
            for i in finished_idx:
                if bad_np[i]:
                    term_type[i] = "bad_done"
                elif done_np[i]:
                    term_type[i] = "done"
                elif exceed_np[i]:
                    term_type[i] = "timeout"
            active[finished] = False

        if step % 100 == 0:
            print(f"[ou-policy-eval] step={step:04d} active={int(active.sum().item())}/{env.n}")
        if (not torch.any(active)) or step + 1 >= steps:
            break

        apply_raw_step(
            env,
            raw_np[:, step + 1, :],
            desired_np[:, step + 1, :],
            level_tensor,
            mode_tensor,
        )
        obs = env.obs()

    env.close()
    return traces, term_type


def main():
    args = parse_args()
    if args.mode_order:
        os.environ["RC_HUMAN_MODE_ORDER"] = args.mode_order
    device = choose_device(args)
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(args.sequence_csv, args.max_steps)
    print(f"[ou-policy-eval] loaded {len(sequences)} sequences from {args.sequence_csv}")
    print(f"[ou-policy-eval] device={device} ckpt={args.ckpt_path}")

    traces, term_type = run_eval(args, device, sequences)
    nonempty = [(i, tr) for i, tr in enumerate(traces) if tr]
    summary_rows = [summarize_trace(tr, term_type[i]) for i, tr in nonempty]
    trace_rows = [row for _i, tr in nonempty for row in tr]

    save_csv(summary_rows, out_dir / "policy_ou_sequence_summary.csv")
    save_csv(trace_rows, out_dir / "policy_ou_sequence_traces.csv")
    plot_overview(summary_rows, out_dir / "policy_ou_tracking_overview.png")
    plot_worst_grid(
        [tr for _i, tr in nonempty],
        summary_rows,
        out_dir / "policy_ou_tracking_worst_sequences.png",
        args.top_k_plots,
    )
    if args.save_per_sequence_plots:
        plot_dir = out_dir / "per_sequence_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for _i, trace in nonempty:
            level = int(trace[0]["level"])
            variant = int(trace[0]["variant"])
            plot_sequence(
                trace,
                plot_dir / f"level_{level:03d}_variant_{variant}.png",
            )

    bad = sum(1 for r in summary_rows if r["term_type"] == "bad_done")
    print(f"[ou-policy-eval] bad_done={bad}/{len(summary_rows)}")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
