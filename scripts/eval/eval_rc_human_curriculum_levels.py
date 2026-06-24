#!/usr/bin/env python
"""Evaluate an rc_human actor over all curriculum levels.

This script fixes ``curriculum_level`` to each requested level, runs one or
more episodes, and plots level-wise tracking performance.  It is intended for
post-training diagnosis of the full 0-119 rc_human curriculum.
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
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--episodes-per-level", type=int, default=1)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=119)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument("--mode-order", type=str, default=None,
                   help="Optional RC_HUMAN_MODE_ORDER override, e.g. '0 1 2 5 3 4'.")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--save-per-level-plots", action="store_true",
                   help="Save one detailed tracking/command/observation/wind figure per level.")

    p.add_argument("--hidden-size", type=str, default="128 128")
    p.add_argument("--act-hidden-size", type=str, default="128 128")
    p.add_argument("--activation-id", type=int, default=1)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--no-feature-normalization",
                   dest="use_feature_normalization", action="store_false")
    p.set_defaults(use_feature_normalization=True)
    p.add_argument("--no-recurrent-policy",
                   dest="use_recurrent_policy", action="store_false")
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


def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


def load_actor(env, args, device):
    actor_args = ActorArgs(args, device)
    actor = PPOActor(actor_args, env.observation_space, env.action_space, device)
    state = torch.load(args.ckpt_path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def set_fixed_level(task, level):
    task.curriculum_enable = True
    task.curriculum_level[:] = int(level)
    task.max_curriculum_level = max(task.max_curriculum_level, int(level))
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0
    if hasattr(task, "dwell_left"):
        task.dwell_left[:] = 0


def mode_for_level(task, level):
    mode_slot = min(int(level) // int(task.levels_per_mode),
                    int(task.active_mode_slots) - 1)
    return int(task.mode_order[mode_slot].item())


def record_step(env, obs, reward, action, prev_action):
    roll, pitch, heading = env.model.get_posture()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    alpha = env.model.get_AOA()
    beta = env.model.get_AOS()
    local_vx, local_vy = env.task.ground_to_local_velocity(vx_n, vy_e, heading)
    err_vx, err_vy = env.task.heading_local_velocity_error(vx_n, vy_e, heading)
    err_vz = vz - env.task.target_vz
    yaw_err = torch.atan2(
        torch.sin(heading - env.task.target_heading),
        torch.cos(heading - env.task.target_heading),
    )
    vel_err = torch.sqrt(err_vx * err_vx + err_vy * err_vy + err_vz * err_vz)
    att_err = torch.sqrt(roll * roll + pitch * pitch)
    action_delta = 0.0
    if prev_action is not None:
        action_delta = float(torch.mean(torch.abs(action - prev_action)).item())

    if hasattr(env.model, "get_wind_ned"):
        wind_n, wind_e, wind_d = env.model.get_wind_ned()
    else:
        wind_n = wind_e = wind_d = torch.zeros_like(vz)

    row = {
        "reward": to_float(reward),
        "local_vx": to_float(local_vx),
        "target_vx": to_float(env.task.target_vx),
        "local_vy": to_float(local_vy),
        "target_vy": to_float(env.task.target_vy),
        "vz": to_float(vz),
        "target_vz": to_float(env.task.target_vz),
        "heading": to_float(heading),
        "target_heading": to_float(env.task.target_heading),
        "target_yaw_rate": to_float(env.task.target_yaw_rate),
        "yaw_err": to_float(yaw_err),
        "vel_err": to_float(vel_err),
        "att_err": to_float(att_err),
        "roll": to_float(roll),
        "pitch": to_float(pitch),
        "alpha": to_float(alpha),
        "beta": to_float(beta),
        "alpha_deg": float(np.degrees(to_float(alpha))),
        "beta_deg": float(np.degrees(to_float(beta))),
        "wind_north": to_float(wind_n),
        "wind_east": to_float(wind_e),
        "wind_down": to_float(wind_d),
        "raw_vx": to_float(env.task.raw_vx),
        "raw_vy": to_float(env.task.raw_vy),
        "raw_vz": to_float(env.task.raw_vz),
        "raw_yaw": to_float(env.task.raw_yaw),
        "stick_vx": to_float(env.task.stick_vx),
        "stick_vy": to_float(env.task.stick_vy),
        "stick_vz": to_float(env.task.stick_vz),
        "stick_yaw": to_float(env.task.stick_yaw),
        "action_delta": action_delta,
    }
    obs_np = obs.detach().cpu().reshape(-1).numpy()
    for i, value in enumerate(obs_np):
        row[f"obs_{i:02d}"] = float(value)
    return row


def run_episode(level, episode_idx, args, device):
    seed = args.seed + int(level) * 1009 + episode_idx
    seed_everything(seed)
    env = ControlEnv(num_envs=1, config="rc_human", model=args.model_name,
                     random_seed=seed, device=device)
    set_fixed_level(env.task, level)
    actor = load_actor(env, args, device)

    obs = env.reset()
    set_fixed_level(env.task, level)
    env.task.sync_command(env)
    mode_id = mode_for_level(env.task, level)
    rnn = torch.zeros((env.n, args.recurrent_hidden_layers,
                       args.recurrent_hidden_size), device=device)
    masks = torch.ones((env.n, 1), device=device)
    trace = []
    prev_action = None
    term_type = "truncated"

    for step in range(args.max_steps):
        with torch.no_grad():
            action, _, rnn = actor(obs, rnn, masks, deterministic=args.deterministic)
        obs, reward, done, bad_done, exceed, _info = env.step(action)
        row = record_step(env, obs, reward, action, prev_action)
        row.update({
            "level": int(level),
            "mode_id": mode_id,
            "episode": int(episode_idx),
            "step": int(step),
            "time_s": float(step * env.model.dt),
            "vx_forward_limit": to_float(getattr(
                env.task, "vx_forward_limit", torch.zeros(1, device=device)
            )),
        })
        trace.append(row)
        prev_action = action.detach().clone()

        if torch.any(done):
            term_type = "done"
            break
        if torch.any(bad_done):
            term_type = "bad_done"
            break
        if torch.any(exceed):
            term_type = "timeout"
            break

    env.close()
    return trace, term_type


def summarize_trace(trace, term_type):
    vel = np.asarray([x["vel_err"] for x in trace])
    yaw = np.abs(np.asarray([x["yaw_err"] for x in trace]))
    att = np.asarray([x["att_err"] for x in trace])
    action_delta = np.asarray([x["action_delta"] for x in trace])
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
        "yaw_mae_rad": float(np.mean(yaw)),
        "yaw_mae_deg": float(np.degrees(np.mean(yaw))),
        "att_mae_rad": float(np.mean(att)),
        "att_mae_deg": float(np.degrees(np.mean(att))),
        "action_delta_mean": float(np.mean(action_delta)),
        "term_type": term_type,
        "success": int(term_type in ("done", "timeout")),
    }


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def level_means(summary_rows):
    levels = sorted(set(r["level"] for r in summary_rows))
    rows = []
    for level in levels:
        items = [r for r in summary_rows if r["level"] == level]
        row = {
            "level": level,
            "mode_id": items[0]["mode_id"],
            "vx_forward_limit": float(np.mean([r["vx_forward_limit"] for r in items])),
            "return": float(np.mean([r["return"] for r in items])),
            "vel_mae": float(np.mean([r["vel_mae"] for r in items])),
            "vel_rmse": float(np.mean([r["vel_rmse"] for r in items])),
            "yaw_mae_deg": float(np.mean([r["yaw_mae_deg"] for r in items])),
            "att_mae_deg": float(np.mean([r["att_mae_deg"] for r in items])),
            "action_delta_mean": float(np.mean([r["action_delta_mean"] for r in items])),
            "success_rate": float(np.mean([r["success"] for r in items])),
            "length": float(np.mean([r["length"] for r in items])),
        }
        rows.append(row)
    return rows


def shade_modes(ax, level_rows):
    current = None
    start = None
    for i, row in enumerate(level_rows + [{"level": level_rows[-1]["level"] + 1,
                                           "mode_id": None}]):
        mode = row["mode_id"]
        if current is None:
            current = mode
            start = row["level"]
            continue
        if mode != current:
            end = row["level"] - 1
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.06)
            ax.text((start + end) * 0.5, 0.98, f"mode {current}",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8)
            current = mode
            start = row["level"]


def plot_level_summary(level_rows, out_png):
    levels = np.asarray([r["level"] for r in level_rows])
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    plots = [
        ("return", "Episode return", "return"),
        ("vel_mae", "Velocity tracking MAE", "m/s"),
        ("yaw_mae_deg", "Yaw tracking MAE", "deg"),
        ("att_mae_deg", "Attitude error MAE", "deg"),
        ("action_delta_mean", "Action delta", "mean |a_t-a_t-1|"),
        ("success_rate", "Success rate", "rate"),
    ]
    for ax, (key, title, ylabel) in zip(axes.reshape(-1), plots):
        shade_modes(ax, level_rows)
        ax.plot(levels, [r[key] for r in level_rows], lw=1.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("curriculum level")
    axes[-1, 1].set_xlabel("curriculum level")
    fig.suptitle("RC human policy tracking performance across 120 curriculum levels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def plot_representative_traces(traces, level_rows, out_png):
    chosen_levels = sorted(set([
        level_rows[0]["level"],
        level_rows[len(level_rows) // 4]["level"],
        level_rows[len(level_rows) // 2]["level"],
        level_rows[(3 * len(level_rows)) // 4]["level"],
        level_rows[-1]["level"],
    ]))
    trace_map = {}
    for tr in traces:
        key = (tr[0]["level"], tr[0]["episode"])
        trace_map.setdefault(key, tr)

    fig, axes = plt.subplots(len(chosen_levels), 4,
                             figsize=(18, 3.2 * len(chosen_levels)),
                             squeeze=False)
    for row_idx, level in enumerate(chosen_levels):
        trace = trace_map[(level, 0)]
        t = np.asarray([x["time_s"] for x in trace])
        mode = trace[0]["mode_id"]

        ax = axes[row_idx, 0]
        ax.plot(t, [x["local_vx"] for x in trace], label="vx")
        ax.plot(t, [x["target_vx"] for x in trace], "--", label="target vx")
        ax.set_title(f"level {level} mode {mode}: vx")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 1]
        ax.plot(t, [x["local_vy"] for x in trace], label="vy")
        ax.plot(t, [x["target_vy"] for x in trace], "--", label="target vy")
        ax.plot(t, [x["vz"] for x in trace], label="vz")
        ax.plot(t, [x["target_vz"] for x in trace], "--", label="target vz")
        ax.set_title("vy/vz tracking")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 2]
        ax.plot(t, [x["vel_err"] for x in trace], label="vel err")
        ax.set_title("velocity error")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 3]
        ax.plot(t, np.degrees([x["yaw_err"] for x in trace]), label="yaw err")
        ax.plot(t, np.degrees([x["att_err"] for x in trace]), label="att err")
        ax.set_title("yaw/attitude error")
        ax.set_ylabel("deg")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


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
    ax.plot(t, [x["target_yaw_rate"] for x in trace], label="target yaw rate")
    ax.set_title("Command")
    ax.set_ylabel("m/s or rad/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.plot(t, [x["raw_vx"] for x in trace], label="raw vx")
    ax.plot(t, [x["stick_vx"] for x in trace], label="stick vx")
    ax.plot(t, [x["raw_yaw"] for x in trace], label="raw yaw")
    ax.plot(t, [x["stick_yaw"] for x in trace], label="stick yaw")
    ax.set_title("RC raw/stick")
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
    for key in ["obs_00", "obs_01", "obs_02", "obs_03"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Observation tracking-error channels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[5]
    for key in ["obs_04", "obs_05", "obs_06", "obs_07"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Observation command channels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[6]
    ax.plot(t, [x["wind_north"] for x in trace], label="wind north")
    ax.plot(t, [x["wind_east"] for x in trace], label="wind east")
    ax.plot(t, [x["wind_down"] for x in trace], label="wind down")
    ax.set_title("Wind disturbance")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[7]
    ax.plot(t, [x["alpha_deg"] for x in trace], label="alpha deg")
    ax.plot(t, [x["beta_deg"] for x in trace], label="beta deg")
    ax.plot(t, np.degrees([x["att_err"] for x in trace]), label="att err deg")
    ax.set_title("Alpha/beta and attitude")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"RC human level {level} mode {mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_all_levels(traces, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        level = trace[0]["level"]
        plot_one_level(trace, out_dir / f"level_{level:03d}.png")


def main():
    args = parse_args()
    if args.mode_order:
        os.environ["RC_HUMAN_MODE_ORDER"] = args.mode_order
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_traces = []
    summary_rows = []
    for level in range(args.min_level, args.max_level + 1):
        print(f"[eval] level={level}")
        for ep in range(args.episodes_per_level):
            trace, term_type = run_episode(level, ep, args, device)
            all_traces.append(trace)
            row = summarize_trace(trace, term_type)
            summary_rows.append(row)
            print(
                f"  ep={ep} mode={row['mode_id']} len={row['length']:4d} "
                f"ret={row['return']:8.1f} vel={row['vel_mae']:.4f} "
                f"yaw={row['yaw_mae_deg']:.3f}deg att={row['att_mae_deg']:.3f}deg "
                f"term={term_type}"
            )

    trace_rows = [x for tr in all_traces for x in tr]
    level_rows = level_means(summary_rows)
    save_csv(summary_rows, out_dir / "rc_human_curriculum_level_summary_raw.csv")
    save_csv(level_rows, out_dir / "rc_human_curriculum_level_summary.csv")
    save_csv(trace_rows, out_dir / "rc_human_curriculum_level_traces.csv")
    plot_level_summary(level_rows, out_dir / "rc_human_curriculum_level_tracking.png")
    plot_representative_traces(
        all_traces, level_rows, out_dir / "rc_human_curriculum_representative_traces.png"
    )
    if args.save_per_level_plots:
        plot_all_levels(all_traces, out_dir / "per_level_plots")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
