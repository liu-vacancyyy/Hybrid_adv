#!/usr/bin/env python
"""Check mode4 command feasibility under an ideal rpy+collective tracker.

The rollout is intentionally not a policy evaluation.  It samples reach-task
mode4 commands, forces roll/pitch/yaw and collective thrust to match the target,
and propagates the remaining translational dynamics.  This answers whether the
command distribution itself can drive the aircraft into the hard safety envelope.
"""

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

from envs.control_env import ControlEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenario-name", default="rpy_throttle_reach_no_forward_nowind_damped")
    p.add_argument("--model-name", default="HYBRID_NEW_NO_FORWARD")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=260623)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-level", type=int, default=40)
    p.add_argument("--max-level", type=int, default=49)
    p.add_argument("--save-per-episode-plots", action="store_true", default=True)
    p.add_argument("--no-per-episode-plots", dest="save_per_episode_plots", action="store_false")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def body_to_world_velocity(s):
    phi, theta, psi = s[:, 3], s[:, 4], s[:, 5]
    u, v, w = s[:, 6], s[:, 7], s[:, 8]
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    ss, cs = torch.sin(psi), torch.cos(psi)

    north = u * (ct * cs) + v * (sp * cs * st - cp * ss) + w * (cp * st * cs + sp * ss)
    east = u * (ct * ss) + v * (sp * ss * st + cp * cs) + w * (cp * st * ss - sp * cs)
    alt_up = u * st - v * (sp * ct) - w * (cp * ct)
    return north, east, alt_up


def world_to_body_velocity(north, east, alt_up, phi, theta, psi):
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    ss, cs = torch.sin(psi), torch.cos(psi)

    r00 = ct * cs
    r01 = sp * cs * st - cp * ss
    r02 = cp * st * cs + sp * ss
    r10 = ct * ss
    r11 = sp * ss * st + cp * cs
    r12 = cp * st * ss - sp * cs
    r20 = st
    r21 = -sp * ct
    r22 = -cp * ct

    # R maps body velocity to world velocity, so body = R^T world.
    u = r00 * north + r10 * east + r20 * alt_up
    v = r01 * north + r11 * east + r21 * alt_up
    w = r02 * north + r12 * east + r22 * alt_up
    return u, v, w


def force_attitude_preserve_world_velocity(env, mask, roll, pitch, yaw):
    if int(mask.sum().item()) == 0:
        return
    s = env.model.s
    north, east, alt_up = body_to_world_velocity(s[mask])
    u, v, w = world_to_body_velocity(
        north,
        east,
        alt_up,
        roll[mask],
        pitch[mask],
        yaw[mask],
    )
    s[mask, 3] = roll[mask]
    s[mask, 4] = pitch[mask]
    s[mask, 5] = yaw[mask]
    s[mask, 6] = u
    s[mask, 7] = v
    s[mask, 8] = w
    s[mask, 9:12] = 0.0


def collective_to_trimmed_u(env, collective):
    # Match HybridModel reset trim: front/rear thrust ratio cancels pitch torque.
    x_front = 0.207 + 0.210
    x_rear = 0.260 + 0.263
    ratio = x_rear / x_front
    rear = collective / (2.0 * (1.0 + ratio))
    front = ratio * rear
    u = torch.zeros(env.n, 5, device=env.device)
    u[:, 0] = 0.0
    u[:, 1] = front
    u[:, 2] = rear
    u[:, 3] = front
    u[:, 4] = rear
    return torch.clamp(u, 0.0, float(env.model.max_F))


def set_mode4_levels(env, levels):
    task = env.task
    task.curriculum_enable = True
    task.curriculum_level[:] = levels.to(env.device).long()
    task.max_curriculum_level = max(int(task.max_curriculum_level), int(levels.max().item()))
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0


def safety_masks(env, overload_count):
    task = env.task
    _n, _e, altitude = env.model.get_position()
    roll, pitch, _yaw = env.model.get_posture()
    p, q, r = env.model.get_angular_velocity()
    speed = env.model.get_TAS()
    ax, ay, az = env.model.get_acceleration()

    low_altitude = altitude < float(getattr(task.config, "altitude_limit", 0.5))
    high_speed = speed >= float(getattr(task.config, "max_velocity", 10.0))
    extreme_angle = (
        torch.abs(torch.rad2deg(roll)) > float(getattr(task.config, "max_roll", 30.0))
    ) | (
        torch.abs(torch.rad2deg(pitch)) > float(getattr(task.config, "max_pitch", 25.0))
    )
    max_omega_norm = float(getattr(
        task.config,
        "max_omega_norm",
        getattr(task.config, "max_omega", 4.0),
    ))
    omega_norm = torch.sqrt(p * p + q * q + r * r)
    extreme_omega = omega_norm > max_omega_norm
    accel = torch.sqrt(ax * ax + ay * ay + az * az)
    overload_now = accel > float(getattr(task.config, "acceleration_limit", 12.0))
    overload_count = torch.where(overload_now, overload_count + 1, torch.zeros_like(overload_count))
    overload = overload_count >= max(1, int(getattr(task.config, "overload_bad_done_persist_steps", 1)))
    bad = low_altitude | high_speed | extreme_angle | extreme_omega | overload
    masks = {
        "low_altitude": low_altitude,
        "high_speed": high_speed,
        "extreme_angle": extreme_angle,
        "extreme_omega": extreme_omega,
        "overload": overload,
        "bad_done": bad,
        "speed": speed,
        "altitude": altitude,
        "accel": accel,
        "omega_norm": omega_norm,
    }
    return masks, overload_count


def tensor_np(x):
    return x.detach().cpu().float().numpy()


def run_rollout(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    env = ControlEnv(
        num_envs=args.episodes,
        config=args.scenario_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    levels = torch.randint(args.min_level, args.max_level + 1, (env.n,), device=device)
    set_mode4_levels(env, levels)
    env.reset()

    active = torch.ones(env.n, dtype=torch.bool, device=device)
    bad_once = torch.zeros(env.n, dtype=torch.bool, device=device)
    bad_step = torch.full((env.n,), -1, dtype=torch.long, device=device)
    bad_reason = np.full(env.n, "none", dtype=object)
    overload_count = torch.zeros(env.n, dtype=torch.long, device=device)

    history = []
    reason_order = ["low_altitude", "high_speed", "extreme_angle", "extreme_omega", "overload"]

    for _step in range(args.max_steps):
        env.task.sync_command(env)
        force_attitude_preserve_world_velocity(
            env,
            active,
            env.task.target_roll,
            env.task.target_pitch,
            env.task.target_yaw,
        )
        env.model.u[:] = collective_to_trimmed_u(env, env.task.target_collective)
        env.model.recent_u[:] = env.model.u

        x = torch.hstack((env.model.s, env.model.u))
        xdot = env.model.dynamics.nlplant(x)
        next_s = env.model.s + env.model.dt * xdot[:, :env.model.num_states]
        env.model.recent_s[:] = env.model.s
        env.model.s[active] = next_s[active]
        env.model.s[active, 9:12] = 0.0

        # Enforce exact command tracking at the end of the integration interval.
        force_attitude_preserve_world_velocity(
            env,
            active,
            env.task.target_roll,
            env.task.target_pitch,
            env.task.target_yaw,
        )

        env.step_count += 1
        masks, overload_count = safety_masks(env, overload_count)
        bad = masks["bad_done"] & active
        first_bad = bad & (~bad_once)
        if torch.any(first_bad):
            bad_step[first_bad] = env.step_count[first_bad]
            idxs = torch.where(first_bad)[0].detach().cpu().tolist()
            for i in idxs:
                for reason in reason_order:
                    if bool(masks[reason][i].item()):
                        bad_reason[i] = reason
                        break
        bad_once |= bad
        active &= ~bad

        roll, pitch, yaw = env.model.get_posture()
        f0, f1, f2, f3, f4 = env.model.get_F()
        collective = f1 + f2 + f3 + f4
        hover = torch.clamp(env.task.hover_collective, min=1e-6)
        history.append({
            "step": int(env.step_count[0].item()),
            "level": tensor_np(env.task.sampled_level),
            "profile": tensor_np(env.task.command_profile),
            "target_roll_deg": np.degrees(tensor_np(env.task.target_roll)),
            "target_pitch_deg": np.degrees(tensor_np(env.task.target_pitch)),
            "target_yaw_delta_deg": np.degrees(tensor_np(wrap_pi(env.task.target_yaw - env.task.start_yaw))),
            "target_throttle_frac": tensor_np(env.task.target_throttle_frac),
            "roll_deg": np.degrees(tensor_np(roll)),
            "pitch_deg": np.degrees(tensor_np(pitch)),
            "yaw_delta_deg": np.degrees(tensor_np(wrap_pi(yaw - env.task.start_yaw))),
            "throttle_frac": tensor_np((collective - hover) / hover),
            "speed": tensor_np(masks["speed"]),
            "altitude": tensor_np(masks["altitude"]),
            "accel": tensor_np(masks["accel"]),
            "bad_done": tensor_np(bad_once.float()),
        })

    return env, history, bad_once, bad_step, bad_reason


def save_csvs(out_dir, env, history, bad_once, bad_step, bad_reason):
    out_dir.mkdir(parents=True, exist_ok=True)
    final = history[-1]
    rows = []
    for i in range(env.n):
        rows.append({
            "episode": i,
            "level": int(final["level"][i]),
            "command_profile": int(final["profile"][i]),
            "bad_done": int(bool(bad_once[i].item())),
            "bad_step": int(bad_step[i].item()),
            "bad_reason": str(bad_reason[i]),
            "max_speed": max(float(h["speed"][i]) for h in history),
            "min_altitude": min(float(h["altitude"][i]) for h in history),
            "max_accel": max(float(h["accel"][i]) for h in history),
            "final_target_roll_deg": float(final["target_roll_deg"][i]),
            "final_target_pitch_deg": float(final["target_pitch_deg"][i]),
            "final_target_yaw_delta_deg": float(final["target_yaw_delta_deg"][i]),
            "final_target_throttle_frac": float(final["target_throttle_frac"][i]),
        })
    with (out_dir / "episode_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    trace_fields = [
        "step", "episode", "level", "profile", "target_roll_deg", "target_pitch_deg",
        "target_yaw_delta_deg", "target_throttle_frac", "roll_deg", "pitch_deg",
        "yaw_delta_deg", "throttle_frac", "speed", "altitude", "accel", "bad_done",
    ]
    with (out_dir / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trace_fields)
        writer.writeheader()
        for h in history:
            for i in range(env.n):
                writer.writerow({
                    "step": h["step"],
                    "episode": i,
                    "level": int(h["level"][i]),
                    "profile": int(h["profile"][i]),
                    "target_roll_deg": float(h["target_roll_deg"][i]),
                    "target_pitch_deg": float(h["target_pitch_deg"][i]),
                    "target_yaw_delta_deg": float(h["target_yaw_delta_deg"][i]),
                    "target_throttle_frac": float(h["target_throttle_frac"][i]),
                    "roll_deg": float(h["roll_deg"][i]),
                    "pitch_deg": float(h["pitch_deg"][i]),
                    "yaw_delta_deg": float(h["yaw_delta_deg"][i]),
                    "throttle_frac": float(h["throttle_frac"][i]),
                    "speed": float(h["speed"][i]),
                    "altitude": float(h["altitude"][i]),
                    "accel": float(h["accel"][i]),
                    "bad_done": int(float(h["bad_done"][i]) > 0.5),
                })
    return rows


def plot_summary(out_dir, rows, history):
    bad = np.asarray([r["bad_done"] for r in rows], dtype=float)
    max_speed = np.asarray([r["max_speed"] for r in rows], dtype=float)
    min_alt = np.asarray([r["min_altitude"] for r in rows], dtype=float)
    max_accel = np.asarray([r["max_accel"] for r in rows], dtype=float)
    levels = np.asarray([r["level"] for r in rows], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(["clean", "bad_done"], [int((bad == 0).sum()), int((bad == 1).sum())])
    ax.set_title(f"bad_done probability = {bad.mean():.1%}")
    ax.set_ylabel("episodes")

    ax = axes[0, 1]
    bins = np.arange(39.5, 50.5, 1.0)
    ax.hist([levels[bad == 0], levels[bad == 1]], bins=bins, stacked=True, label=["clean", "bad_done"])
    ax.set_xticks(range(40, 50))
    ax.set_title("mode4 sampled levels")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(levels, max_speed, c=bad, cmap="coolwarm", vmin=0, vmax=1)
    ax.axhline(10.0, color="k", linestyle="--", linewidth=1.0, label="max_velocity")
    ax.set_title("max speed")
    ax.set_xlabel("level")
    ax.set_ylabel("m/s")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(min_alt, max_accel, c=bad, cmap="coolwarm", vmin=0, vmax=1)
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1.0, label="min altitude")
    ax.axhline(12.0, color="0.3", linestyle=":", linewidth=1.0, label="accel limit")
    ax.set_title("safety margins")
    ax.set_xlabel("min altitude")
    ax.set_ylabel("max accel")
    ax.legend()

    fig.savefig(out_dir / "mode4_perfect_command_summary.png", dpi=150)
    plt.close(fig)

    speed = np.stack([h["speed"] for h in history], axis=0)
    altitude = np.stack([h["altitude"] for h in history], axis=0)
    accel = np.stack([h["accel"] for h in history], axis=0)
    steps = np.asarray([h["step"] for h in history])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for arr, ax, title, limit in [
        (speed, axes[0], "speed", 10.0),
        (altitude, axes[1], "altitude", 0.5),
        (accel, axes[2], "acceleration", 12.0),
    ]:
        ax.plot(steps, np.percentile(arr, 50, axis=1), label="p50")
        ax.plot(steps, np.percentile(arr, 90, axis=1), label="p90")
        ax.plot(steps, np.percentile(arr, 100, axis=1), label="max")
        ax.axhline(limit, color="k", linestyle="--", linewidth=1.0)
        ax.set_ylabel(title)
        ax.legend()
    axes[-1].set_xlabel("step")
    fig.savefig(out_dir / "mode4_perfect_command_safety_timeseries.png", dpi=150)
    plt.close(fig)


def plot_episode(out_dir, history, rows):
    per_dir = out_dir / "episode_plots"
    per_dir.mkdir(exist_ok=True)
    steps = np.asarray([h["step"] for h in history])
    n = len(rows)
    for i in range(n):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        axes[0].plot(steps, [h["target_roll_deg"][i] for h in history], label="target roll")
        axes[0].plot(steps, [h["target_pitch_deg"][i] for h in history], label="target pitch")
        axes[0].set_ylabel("deg")
        axes[0].legend()
        axes[1].plot(steps, [h["target_yaw_delta_deg"][i] for h in history], label="target yaw delta")
        axes[1].set_ylabel("deg")
        axes[1].legend()
        axes[2].plot(steps, [h["target_throttle_frac"][i] for h in history], label="target throttle frac")
        axes[2].plot(steps, [h["speed"][i] for h in history], label="speed")
        axes[2].axhline(10.0, color="k", linestyle="--", linewidth=1.0)
        axes[2].legend()
        axes[3].plot(steps, [h["altitude"][i] for h in history], label="altitude")
        axes[3].plot(steps, [h["accel"][i] for h in history], label="accel")
        axes[3].axhline(0.5, color="k", linestyle="--", linewidth=1.0)
        axes[3].axhline(12.0, color="0.3", linestyle=":", linewidth=1.0)
        axes[3].set_xlabel("step")
        axes[3].legend()
        fig.suptitle(
            f"episode {i:03d} level {rows[i]['level']} "
            f"bad={rows[i]['bad_done']} reason={rows[i]['bad_reason']}"
        )
        fig.savefig(per_dir / f"episode_{i:03d}.png", dpi=130)
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    env, history, bad_once, bad_step, bad_reason = run_rollout(args)
    rows = save_csvs(out_dir, env, history, bad_once, bad_step, bad_reason)
    plot_summary(out_dir, rows, history)
    if args.save_per_episode_plots:
        plot_episode(out_dir, history, rows)

    bad_count = int(sum(r["bad_done"] for r in rows))
    reasons = {}
    for r in rows:
        reasons[r["bad_reason"]] = reasons.get(r["bad_reason"], 0) + 1
    print(f"episodes={len(rows)} bad_done={bad_count} probability={bad_count / max(len(rows), 1):.3f}")
    print("reasons=", reasons)
    print("output_dir=", out_dir)


if __name__ == "__main__":
    main()
