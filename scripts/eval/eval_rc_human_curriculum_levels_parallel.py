#!/usr/bin/env python
"""Vectorized rc_human curriculum evaluation.

Runs one trained actor on a fixed set of curriculum levels in a single
vectorized ControlEnv.  This is much faster than creating one env per level and
is intended for the full 0-119 curriculum diagnosis.
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
    p.add_argument("--config-name", type=str, default="rc_human")
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument(
        "--mode-order",
        type=str,
        default=None,
        help="Optional RC_HUMAN_MODE_ORDER override, e.g. '0 1 2 5 3 4'.",
    )
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument(
        "--save-per-level-plots",
        action="store_true",
        help="Save one detailed tracking/command/observation/wind figure per level.",
    )

    p.add_argument("--hidden-size", type=str, default="128 128")
    p.add_argument("--act-hidden-size", type=str, default="128 128")
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
        # Only force command sampling at the very beginning or after an env reset.
        reset_like = task.dwell_left < 0
        task.dwell_left[reset_like] = 0


def fixed_modes(task, level_tensor):
    levels = level_tensor.to(task.device).long()
    slot = torch.clamp(levels // int(task.levels_per_mode), 0, int(task.active_mode_slots) - 1)
    return task.mode_order[slot].detach().cpu().long().numpy()


def collect_arrays(env, obs, reward, action, prev_action):
    task = env.task
    roll, pitch, heading = env.model.get_posture()
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
        "command_transient_left": tensor_np(task.command_transient_left),
        "action_delta": tensor_np(action_delta),
        "f1": tensor_np(f1),
        "f2": tensor_np(f2),
        "f3": tensor_np(f3),
        "f4": tensor_np(f4),
        "f5": tensor_np(f5),
    }
    for i in range(min(8, obs_np.shape[1])):
        arrays[f"obs_{i:02d}"] = obs_np[:, i]
    return arrays


def append_trace_rows(
    traces,
    arrays,
    active_np,
    levels_np,
    modes_np,
    episode_np,
    step,
    dt,
):
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
    set_fixed_levels(env.task, level_tensor)
    modes_np = fixed_modes(env.task, level_tensor)
    actor = load_actor(env, args, device)

    obs = env.reset()
    set_fixed_levels(env.task, level_tensor)
    modes_np = fixed_modes(env.task, level_tensor)

    rnn = torch.zeros(
        (env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size),
        device=device,
    )
    active = torch.ones(env.n, dtype=torch.bool, device=device)
    term_type = np.full(env.n, "truncated", dtype=object)
    prev_action = None
    traces = [[] for _ in range(env.n)]

    for step in range(args.max_steps):
        masks = active.float().reshape(-1, 1)
        active_np = tensor_np(active).astype(bool)
        with torch.no_grad():
            action, _, rnn = actor(obs, rnn, masks, deterministic=args.deterministic)

        obs, reward, done, bad_done, exceed, _info = env.step(action)
        set_fixed_levels(env.task, level_tensor)
        arrays = collect_arrays(env, obs, reward, action, prev_action)
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
            for i in finished_idx:
                if bad_np[i]:
                    term_type[i] = "bad_done"
                elif done_np[i]:
                    term_type[i] = "done"
                elif exceed_np[i]:
                    term_type[i] = "timeout"
            active[finished] = False

        if step % 100 == 0:
            print(f"[eval] step={step:04d} active={int(active.sum().item())}/{env.n}")
        if not torch.any(active):
            break

    env.close()
    return traces, term_type


def summarize_trace(trace, term_type):
    vel = np.asarray([x["vel_err"] for x in trace], dtype=np.float64)
    yaw = np.abs(np.asarray([x["yaw_err"] for x in trace], dtype=np.float64))
    att = np.asarray([x["att_err"] for x in trace], dtype=np.float64)
    action_delta = np.asarray([x["action_delta"] for x in trace], dtype=np.float64)
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
        "term_type": str(term_type),
        "success": int(str(term_type) != "bad_done"),
    }


def level_means(summary_rows):
    rows = []
    for level in sorted(set(r["level"] for r in summary_rows)):
        items = [r for r in summary_rows if r["level"] == level]
        rows.append({
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
        })
    return rows


def mode_means(level_rows):
    rows = []
    for mode in sorted(set(r["mode_id"] for r in level_rows)):
        items = [r for r in level_rows if r["mode_id"] == mode]
        rows.append({
            "mode_id": mode,
            "level_min": min(r["level"] for r in items),
            "level_max": max(r["level"] for r in items),
            "return": float(np.mean([r["return"] for r in items])),
            "vel_mae": float(np.mean([r["vel_mae"] for r in items])),
            "yaw_mae_deg": float(np.mean([r["yaw_mae_deg"] for r in items])),
            "att_mae_deg": float(np.mean([r["att_mae_deg"] for r in items])),
            "success_rate": float(np.mean([r["success_rate"] for r in items])),
            "length": float(np.mean([r["length"] for r in items])),
        })
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
    fig.suptitle("RC human policy performance across fixed curriculum levels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def representative_levels(level_rows):
    chosen = []
    current = None
    start = None
    last = None
    for row in level_rows:
        mode = row["mode_id"]
        if current is None:
            current = mode
            start = row["level"]
        elif mode != current:
            chosen.extend([start, last])
            current = mode
            start = row["level"]
        last = row["level"]
    chosen.extend([start, last])
    return sorted(set(chosen))


def plot_representative_traces(trace_by_level, level_rows, out_png):
    chosen = representative_levels(level_rows)
    fig, axes = plt.subplots(len(chosen), 4, figsize=(18, 3.0 * len(chosen)), squeeze=False)
    for row_idx, level in enumerate(chosen):
        trace = trace_by_level[level]
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


def plot_all_levels(trace_by_level, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for level, trace in trace_by_level.items():
        plot_one_level(trace, out_dir / f"level_{level:03d}.png")


def main():
    args = parse_args()
    if args.mode_order:
        os.environ["RC_HUMAN_MODE_ORDER"] = args.mode_order
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces, term_type = run_vectorized_eval(args, device)
    nonempty = [(i, tr) for i, tr in enumerate(traces) if tr]
    summary_rows = [summarize_trace(tr, term_type[i]) for i, tr in nonempty]
    level_rows = level_means(summary_rows)
    mode_rows = mode_means(level_rows)

    trace_rows = [row for _i, tr in nonempty for row in tr]
    trace_by_level = {}
    for _i, tr in nonempty:
        level = tr[0]["level"]
        episode = tr[0]["episode"]
        if episode == 0:
            trace_by_level[level] = tr

    save_csv(summary_rows, out_dir / "rc_human_curriculum_level_summary_raw.csv")
    save_csv(level_rows, out_dir / "rc_human_curriculum_level_summary.csv")
    save_csv(mode_rows, out_dir / "rc_human_curriculum_mode_summary.csv")
    save_csv(trace_rows, out_dir / "rc_human_curriculum_level_traces.csv")

    plot_level_summary(level_rows, out_dir / "rc_human_curriculum_level_tracking.png")
    plot_representative_traces(
        trace_by_level,
        level_rows,
        out_dir / "rc_human_curriculum_representative_traces.png",
    )
    if args.save_per_level_plots:
        plot_all_levels(trace_by_level, out_dir / "per_level_plots")

    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
