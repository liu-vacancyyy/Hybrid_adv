#!/usr/bin/env python
"""Evaluate a trained PPO actor on rc_human command modes.

The default checkpoint is the episode_740 actor from the first2 curriculum run.
By default this script evaluates the first two curriculum modes, i.e. internal
mode IDs 0 and 1.  Pass ``--internal-mode-ids 1 2`` if you want to evaluate the
literal code modes 1 and 2 instead.
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


DEFAULT_CKPT = (
    ROOT / "scripts" / "runs"
    / "2026-05-19_22-58-25_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_first2"
    / "episode_740" / "actor_latest.ckpt"
)


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
    p.add_argument("--ckpt-path", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=740)
    p.add_argument("--episodes-per-mode", type=int, default=24)
    p.add_argument("--plot-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "renders" / "result" / "rc_human_eval_ep740"))
    p.add_argument("--internal-mode-ids", nargs="+", type=int, default=[0, 1],
                   help="Internal rc_human mode IDs to evaluate.")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--show", action="store_true")

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
    torch.cuda.manual_seed_all(seed)


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar(x):
    return float(to_np(x).reshape(-1)[0])


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def load_actor(env, args, device):
    actor_args = ActorArgs(args, device)
    actor = PPOActor(actor_args, env.observation_space, env.action_space, device)
    state = torch.load(args.ckpt_path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def set_fixed_mode(task, internal_mode_id):
    task.curriculum_enable = False
    task.mode_order[:] = internal_mode_id
    task.active_mode_slots = 1
    task.max_curriculum_level = task.levels_per_mode - 1
    task.curriculum_level[:] = task.max_curriculum_level
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0


def record_step(env, reward, action, prev_action):
    roll, pitch, heading = env.model.get_posture()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    local_vx, local_vy = env.task.ground_to_local_velocity(vx_n, vy_e, heading)
    err_vx, err_vy = env.task.ground_to_local_velocity(
        vx_n - env.task.target_vn, vy_e - env.task.target_ve, heading
    )
    err_vz = vz - env.task.target_vz
    yaw_err = torch.atan2(
        torch.sin(heading - env.task.target_heading),
        torch.cos(heading - env.task.target_heading),
    )
    vel_err = torch.sqrt(err_vx * err_vx + err_vy * err_vy + err_vz * err_vz)
    att_err = torch.sqrt(roll * roll + pitch * pitch)

    delta = 0.0
    if prev_action is not None:
        delta = float(torch.mean(torch.abs(action - prev_action)).item())

    return {
        "reward": scalar(reward),
        "local_vx": scalar(local_vx),
        "target_vx": scalar(env.task.target_vx),
        "local_vy": scalar(local_vy),
        "target_vy": scalar(env.task.target_vy),
        "vz": scalar(vz),
        "target_vz": scalar(env.task.target_vz),
        "heading": scalar(heading),
        "target_heading": scalar(env.task.target_heading),
        "yaw_err": scalar(yaw_err),
        "vel_err": scalar(vel_err),
        "att_err": scalar(att_err),
        "roll": scalar(roll),
        "pitch": scalar(pitch),
        "action_l1": float(torch.mean(torch.sum(torch.abs(action), dim=1)).item()),
        "action_delta": delta,
    }


def run_episode(mode_id, ep_idx, args, device):
    seed_everything(args.seed + mode_id * 10000 + ep_idx)
    env = ControlEnv(num_envs=1, config="rc_human", model=args.model_name,
                     random_seed=args.seed + mode_id * 10000 + ep_idx, device=device)
    set_fixed_mode(env.task, mode_id)
    actor = load_actor(env, args, device)

    obs = env.reset()
    env.task.dwell_left[:] = 0
    env.task.sync_command(env)
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
        row = record_step(env, reward, action, prev_action)
        row["step"] = step
        row["time_s"] = step * env.model.dt
        row["mode_id"] = mode_id
        row["episode"] = ep_idx
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
    ret = float(np.sum([x["reward"] for x in trace]))
    return {
        "mode_id": trace[0]["mode_id"],
        "episode": trace[0]["episode"],
        "length": len(trace),
        "return": ret,
        "vel_mae": float(np.mean(vel)),
        "vel_rmse": float(np.sqrt(np.mean(vel * vel))),
        "yaw_mae_rad": float(np.mean(yaw)),
        "yaw_mae_deg": float(np.degrees(np.mean(yaw))),
        "att_mae_rad": float(np.mean(att)),
        "att_mae_deg": float(np.degrees(np.mean(att))),
        "action_delta_mean": float(np.mean([x["action_delta"] for x in trace])),
        "term_type": term_type,
        "success": int(term_type == "done"),
    }


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_representative(summary_rows, plot_n):
    sorted_rows = sorted(summary_rows, key=lambda r: r["vel_mae"])
    if len(sorted_rows) <= plot_n:
        return sorted_rows
    qs = np.linspace(0, len(sorted_rows) - 1, plot_n).round().astype(int)
    chosen = []
    seen = set()
    for i in qs:
        row = sorted_rows[int(i)]
        key = (row["mode_id"], row["episode"])
        if key not in seen:
            chosen.append(row)
            seen.add(key)
    return chosen


def plot_mode_summary(summary_rows, out_png):
    modes = sorted(set(r["mode_id"] for r in summary_rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = [f"mode {m}" for m in modes]

    for ax, key, title, ylabel in [
        (axes[0, 0], "return", "Episode return", "return"),
        (axes[0, 1], "vel_mae", "Velocity tracking error", "m/s"),
        (axes[1, 0], "yaw_mae_deg", "Yaw tracking error", "deg"),
        (axes[1, 1], "att_mae_deg", "Attitude error", "deg"),
    ]:
        data = [[r[key] for r in summary_rows if r["mode_id"] == m] for m in modes]
        ax.boxplot(data, labels=labels)
        means = [np.mean(x) for x in data]
        ax.plot(np.arange(1, len(means) + 1), means, "o-", label="mean")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("RC human policy evaluation by command mode")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def plot_representative(traces, chosen, out_png):
    fig, axes = plt.subplots(len(chosen), 4, figsize=(18, 3.1 * len(chosen)),
                             squeeze=False)
    trace_map = {(tr[0]["mode_id"], tr[0]["episode"]): tr for tr in traces}

    for row_i, item in enumerate(chosen):
        trace = trace_map[(item["mode_id"], item["episode"])]
        t = np.asarray([x["time_s"] for x in trace])
        mode_id = item["mode_id"]
        ep = item["episode"]

        ax = axes[row_i, 0]
        ax.plot(t, [x["local_vx"] for x in trace], label="vx")
        ax.plot(t, [x["target_vx"] for x in trace], "--", label="target vx")
        ax.plot(t, [x["local_vy"] for x in trace], label="vy")
        ax.plot(t, [x["target_vy"] for x in trace], "--", label="target vy")
        ax.set_title(f"mode {mode_id} ep {ep}: horizontal velocity")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_i, 1]
        ax.plot(t, [x["vz"] for x in trace], label="vz")
        ax.plot(t, [x["target_vz"] for x in trace], "--", label="target vz")
        ax.set_title("Vertical velocity")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_i, 2]
        heading = np.unwrap([x["heading"] for x in trace])
        target_heading = np.unwrap([x["target_heading"] for x in trace])
        ax.plot(t, np.degrees(heading), label="heading")
        ax.plot(t, np.degrees(target_heading), "--", label="target")
        ax.set_title("Heading")
        ax.set_ylabel("deg")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[row_i, 3]
        ax.plot(t, [x["vel_err"] for x in trace], label="vel err")
        ax.plot(t, np.degrees([x["yaw_err"] for x in trace]), label="yaw err deg")
        ax.plot(t, np.degrees([x["att_err"] for x in trace]), label="att err deg")
        ax.set_title("Errors")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def main():
    args = parse_args()
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_traces = []
    summary_rows = []
    for mode_id in args.internal_mode_ids:
        print(f"[eval] mode={mode_id}")
        for ep in range(args.episodes_per_mode):
            trace, term_type = run_episode(mode_id, ep, args, device)
            all_traces.append(trace)
            row = summarize_trace(trace, term_type)
            summary_rows.append(row)
            print(
                f"  ep={ep:02d} len={row['length']:4d} return={row['return']:8.1f} "
                f"vel_mae={row['vel_mae']:.4f} yaw_deg={row['yaw_mae_deg']:.3f} "
                f"att_deg={row['att_mae_deg']:.3f} term={term_type}"
            )

    trace_rows = [x for tr in all_traces for x in tr]
    save_csv(summary_rows, out_dir / "rc_human_ep740_mode_summary.csv")
    save_csv(trace_rows, out_dir / "rc_human_ep740_mode_traces.csv")
    plot_mode_summary(summary_rows, out_dir / "rc_human_ep740_mode_summary.png")

    chosen = pick_representative(summary_rows, args.plot_episodes)
    plot_representative(all_traces, chosen,
                        out_dir / "rc_human_ep740_representative_episodes.png")

    print("\n[summary]")
    for mode_id in args.internal_mode_ids:
        rows = [r for r in summary_rows if r["mode_id"] == mode_id]
        print(
            f"mode={mode_id} n={len(rows)} "
            f"return_mean={np.mean([r['return'] for r in rows]):.1f} "
            f"vel_mae={np.mean([r['vel_mae'] for r in rows]):.4f} "
            f"yaw_mae_deg={np.mean([r['yaw_mae_deg'] for r in rows]):.3f} "
            f"att_mae_deg={np.mean([r['att_mae_deg'] for r in rows]):.3f} "
            f"success_rate={np.mean([r['success'] for r in rows]):.3f}"
        )
    print(f"[saved] {out_dir}")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
