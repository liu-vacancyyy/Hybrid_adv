#!/usr/bin/env python
"""Evaluate one PPO/RL actor on the CircleTask for one full episode.

Default checkpoint resolution searches the newest
``scripts/runs/*_Control_circle_*/*/actor_latest.ckpt``.

Example:
    python scripts/eval/eval_circle_policy_metrics.py --device cuda:0
    python scripts/eval/eval_circle_policy_metrics.py \
        --ckpt-path scripts/runs/<run>/episode_100/actor_latest.ckpt
"""
import argparse
import contextlib
import csv
import datetime
import io
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
DEFAULT_CKPT = ROOT / "scripts" / "runs" / "actor_latest.ckpt"

from algorithms.ppo.ppo_actor import PPOActor  # noqa: E402
from envs.control_env import ControlEnv        # noqa: E402


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar(x):
    return float(to_np(x).reshape(-1)[0])


class ActorArgs:
    """Network args matching scripts/train_circle_rl.sh defaults."""

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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default=str(DEFAULT_CKPT),
                        help="Path to actor_latest.ckpt. Defaults to scripts/runs/actor_latest.ckpt.")
    parser.add_argument("--runs-root", type=str, default=str(ROOT / "scripts" / "runs"))
    parser.add_argument("--list-candidates", action="store_true",
                        help="Print circle actor_latest.ckpt candidates under scripts/runs and exit.")
    parser.add_argument("--scenario-name", type=str, default="circle")
    parser.add_argument("--model-name", type=str, default="HYBRID")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 means env.config.max_steps.")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "renders" / "result" / "circle_rl_eval"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true",
                        help="Show the plot window after saving. Requires a working display.")
    parser.add_argument("--verbose-env", action="store_true",
                        help="Show verbose prints from env.step.")

    parser.add_argument("--hidden-size", type=str, default="128 128")
    parser.add_argument("--act-hidden-size", type=str, default="128 128")
    parser.add_argument("--activation-id", type=int, default=1)
    parser.add_argument("--gain", type=float, default=0.01)
    parser.add_argument("--no-feature-normalization",
                        dest="use_feature_normalization",
                        action="store_false")
    parser.set_defaults(use_feature_normalization=True)
    parser.add_argument("--no-recurrent-policy",
                        dest="use_recurrent_policy",
                        action="store_false")
    parser.set_defaults(use_recurrent_policy=True)
    parser.add_argument("--recurrent-hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-hidden-layers", type=int, default=1)
    return parser.parse_args(argv)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(args):
    if (not args.no_cuda) and torch.cuda.is_available():
        return torch.device(args.device)
    return torch.device("cpu")


def resolve_checkpoint(args):
    if args.ckpt_path:
        ckpt = Path(args.ckpt_path)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if ckpt.exists():
            return ckpt
        if ckpt != DEFAULT_CKPT:
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        print(f"[warn] default checkpoint not found: {ckpt}; searching circle actors under scripts/runs")

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    candidates = list(runs_root.glob("*_Control_circle_*/*/actor_latest.ckpt"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if args.list_candidates:
        print(f"Circle actor candidates under {runs_root}:")
        for p in candidates:
            print(p)
        raise SystemExit(0)
    if not candidates:
        raise FileNotFoundError(
            f"No circle actor_latest.ckpt found under {runs_root}. "
            "Expected scripts/runs/*_Control_circle_*/*/actor_latest.ckpt, "
            "or pass --ckpt-path explicitly."
        )
    return candidates[0]


def infer_checkpoint_obs_dim(state_dict):
    for key in ("base.feature_norm.weight", "base.mlp.fc.0.weight"):
        value = state_dict.get(key)
        if value is None:
            continue
        if key.endswith("weight") and value.ndim == 1:
            return int(value.shape[0])
        if value.ndim == 2:
            return int(value.shape[1])
    return None


def adapt_env_obs_space_for_actor(env, actor_obs_dim):
    """Support old 24-dim circle actors after CircleTask moved to 28 dims."""
    if actor_obs_dim is None or actor_obs_dim == env.num_observation:
        return actor_obs_dim or env.num_observation
    if getattr(env.task, "task_name", "") == "circle" and actor_obs_dim == 24:
        env.task.include_target_motion = False
        env.task.num_observation = 24
        env.task.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        return 24
    raise ValueError(
        f"checkpoint expects obs_dim={actor_obs_dim}, but env provides "
        f"obs_dim={env.num_observation}. For circle only old 24-dim actors "
        "can be auto-adapted."
    )


def make_buffer():
    return {
        "step": [], "time_s": [], "reward": [],
        "npos": [], "epos": [], "altitude": [],
        "target_npos": [], "target_epos": [], "target_altitude": [],
        "xy_err": [], "pos_err": [],
        "vx": [], "vy": [], "vz": [],
        "target_vn": [], "target_ve": [],
        "vel_err": [], "roll_rad": [], "pitch_rad": [],
        "heading_rad": [], "target_heading_rad": [], "heading_err_rad": [],
        "f_head": [], "f_lift_mean": [], "f_lift_spread": [],
        "action_l1": [], "action_delta_l1": [],
    }


def record(buf, env, step, reward=None, action=None, prev_action=None):
    npos, epos, altitude = env.model.get_position()
    roll, pitch, heading = env.model.get_posture()
    vx, vy = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    u = env.model.get_control()

    dn = env.task.target_npos - npos
    de = env.task.target_epos - epos
    da = env.task.target_altitude - altitude
    dvn = vx - env.task.target_vn
    dve = vy - env.task.target_ve
    hdg_err = torch.atan2(
        torch.sin(env.task.target_heading - heading),
        torch.cos(env.task.target_heading - heading),
    )

    xy_err = torch.sqrt(dn * dn + de * de)
    pos_err = torch.sqrt(dn * dn + de * de + da * da)
    vel_err = torch.sqrt(dvn * dvn + dve * dve + vz * vz)
    lift = to_np(u).reshape(-1)[1:]

    buf["step"].append(step)
    buf["time_s"].append(step * env.model.dt)
    buf["reward"].append(0.0 if reward is None else scalar(reward))
    buf["npos"].append(scalar(npos))
    buf["epos"].append(scalar(epos))
    buf["altitude"].append(scalar(altitude))
    buf["target_npos"].append(scalar(env.task.target_npos))
    buf["target_epos"].append(scalar(env.task.target_epos))
    buf["target_altitude"].append(scalar(env.task.target_altitude))
    buf["xy_err"].append(scalar(xy_err))
    buf["pos_err"].append(scalar(pos_err))
    buf["vx"].append(scalar(vx))
    buf["vy"].append(scalar(vy))
    buf["vz"].append(scalar(vz))
    buf["target_vn"].append(scalar(env.task.target_vn))
    buf["target_ve"].append(scalar(env.task.target_ve))
    buf["vel_err"].append(scalar(vel_err))
    buf["roll_rad"].append(scalar(roll))
    buf["pitch_rad"].append(scalar(pitch))
    buf["heading_rad"].append(scalar(heading))
    buf["target_heading_rad"].append(scalar(env.task.target_heading))
    buf["heading_err_rad"].append(scalar(hdg_err))
    buf["f_head"].append(float(to_np(u).reshape(-1)[0]))
    buf["f_lift_mean"].append(float(np.mean(lift)))
    buf["f_lift_spread"].append(float(np.max(lift) - np.min(lift)))
    if action is None:
        buf["action_l1"].append(0.0)
        buf["action_delta_l1"].append(0.0)
    else:
        buf["action_l1"].append(float(torch.mean(torch.abs(action)).item()))
        if prev_action is None:
            buf["action_delta_l1"].append(0.0)
        else:
            buf["action_delta_l1"].append(float(torch.mean(torch.abs(action - prev_action)).item()))


def run_episode(env, actor, args, device):
    obs = env.reset()
    rnn = torch.zeros((env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size),
                      device=device)
    masks = torch.ones((env.n, 1), device=device)
    max_steps = args.max_steps if args.max_steps > 0 else int(getattr(env.config, "max_steps", 2000))

    buf = make_buffer()
    env.task.sync_target_to_time(env)
    record(buf, env, step=0)

    total_reward = 0.0
    prev_action = None
    term_type = "truncated"
    steps = 0

    for step in range(1, max_steps + 1):
        with torch.no_grad():
            action, _, rnn = actor(obs, rnn, masks, deterministic=True)

        if args.verbose_env:
            obs, reward, done, bad_done, exceed, _info = env.step(action)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, done, bad_done, exceed, _info = env.step(action)
        total_reward += scalar(reward)
        steps = step

        reset_mask = done | bad_done | exceed
        masks = 1.0 - reset_mask.float().reshape(-1, 1)

        env.task.sync_target_to_time(env)
        record(buf, env, step=step, reward=reward,
               action=action.detach(), prev_action=prev_action)
        prev_action = action.detach().clone()

        if bool(torch.any(reset_mask).item()):
            if bool(torch.any(bad_done).item()):
                term_type = "bad_done"
            elif bool(torch.any(done).item()):
                term_type = "done"
            else:
                term_type = "exceed_time_limit"
            break

    return buf, {
        "steps": steps,
        "total_reward": total_reward,
        "term_type": term_type,
    }


def save_csv(buf, out_csv):
    keys = list(buf.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*(buf[k] for k in keys)):
            writer.writerow(row)


def save_npz(buf, out_npz):
    np.savez(out_npz, **{k: np.asarray(v) for k, v in buf.items()})


def plot_eval(buf, out_png, title, show=False):
    t = np.asarray(buf["time_s"])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    ax = axes[0, 0]
    ax.plot(buf["target_epos"], buf["target_npos"], "k--", label="target")
    ax.plot(buf["epos"], buf["npos"], label="RL")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Horizontal trajectory")
    ax.set_xlabel("East position (m)")
    ax.set_ylabel("North position (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, buf["xy_err"], label="xy")
    ax.plot(t, buf["pos_err"], label="3-D")
    ax.set_title("Tracking error (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(t, buf["altitude"], label="altitude")
    ax.plot(t, buf["target_altitude"], "--", label="target")
    ax.set_title("Altitude (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, buf["vx"], label="vn")
    ax.plot(t, buf["vy"], label="ve")
    ax.plot(t, buf["vz"], label="vz")
    ax.plot(t, buf["target_vn"], "--", label="target vn")
    ax.plot(t, buf["target_ve"], "--", label="target ve")
    ax.set_title("Velocity (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, np.degrees(buf["roll_rad"]), label="roll")
    ax.plot(t, np.degrees(buf["pitch_rad"]), label="pitch")
    ax.set_title("Roll / pitch (deg)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(t, np.degrees(np.unwrap(buf["heading_rad"])), label="heading")
    ax.plot(t, np.degrees(np.unwrap(buf["target_heading_rad"])), "--", label="target")
    ax.set_title("Heading (deg)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(t, buf["f_head"], label="F_head")
    ax.plot(t, buf["f_lift_mean"], label="F_1-4 mean")
    ax.plot(t, buf["f_lift_spread"], label="F_1-4 spread")
    ax.set_title("Motor force (N)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(t, buf["reward"])
    ax.set_title("Per-step reward")
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.plot(t, np.cumsum(buf["reward"]), label="return")
    ax2 = ax.twinx()
    ax2.plot(t, buf["action_delta_l1"], color="#d62728", alpha=0.7, label="|delta action|")
    ax.set_title("Return / action delta")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[plot] saved: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory(buf, out_png, title, show=False):
    """Save a dedicated reference-vs-run trajectory figure."""
    t = np.asarray(buf["time_s"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(buf["target_epos"], buf["target_npos"], "k--", lw=2.0, label="reference")
    ax.plot(buf["epos"], buf["npos"], color="#1f77b4", lw=2.0, label="RL trajectory")
    ax.scatter(buf["target_epos"][0], buf["target_npos"][0], c="k", s=35, marker="o", label="ref start")
    ax.scatter(buf["epos"][0], buf["npos"][0], c="#1f77b4", s=35, marker="o", label="run start")
    ax.scatter(buf["target_epos"][-1], buf["target_npos"][-1], c="k", s=45, marker="x", label="ref end")
    ax.scatter(buf["epos"][-1], buf["npos"][-1], c="#d62728", s=45, marker="x", label="run end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Horizontal trajectory")
    ax.set_xlabel("East position (m)")
    ax.set_ylabel("North position (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, buf["xy_err"], label="XY error")
    ax.plot(t, buf["pos_err"], label="3-D error")
    ax.set_title("Tracking error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, buf["npos"], label="north")
    ax.plot(t, buf["target_npos"], "--", label="north reference")
    ax.plot(t, buf["epos"], label="east")
    ax.plot(t, buf["target_epos"], "--", label="east reference")
    ax.set_title("Horizontal position vs time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, buf["altitude"], label="altitude")
    ax.plot(t, buf["target_altitude"], "--", label="altitude reference")
    ax.set_title("Altitude vs time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[plot] saved: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def summarize(buf, meta):
    xy = np.asarray(buf["xy_err"], dtype=np.float64)
    pos = np.asarray(buf["pos_err"], dtype=np.float64)
    vel = np.asarray(buf["vel_err"], dtype=np.float64)
    hdg = np.asarray(buf["heading_err_rad"], dtype=np.float64)
    return {
        "steps": meta["steps"],
        "term_type": meta["term_type"],
        "total_reward": meta["total_reward"],
        "mean_xy_err": float(np.mean(xy)),
        "rmse_xy_err": float(np.sqrt(np.mean(xy * xy))),
        "final_xy_err": float(xy[-1]),
        "max_xy_err": float(np.max(xy)),
        "mean_pos_err": float(np.mean(pos)),
        "rmse_pos_err": float(np.sqrt(np.mean(pos * pos))),
        "mean_vel_err": float(np.mean(vel)),
        "rmse_heading_deg": float(np.degrees(np.sqrt(np.mean(hdg * hdg)))),
        "mean_action_delta_l1": float(np.mean(buf["action_delta_l1"])),
    }


def main(argv):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = choose_device(args)
    ckpt_path = resolve_checkpoint(args)

    env = ControlEnv(num_envs=1,
                     config=args.scenario_name,
                     model=args.model_name,
                     random_seed=args.seed,
                     device=str(device))

    state_dict = torch.load(str(ckpt_path), map_location=device)
    actor_obs_dim = infer_checkpoint_obs_dim(state_dict)
    actual_obs_dim = adapt_env_obs_space_for_actor(env, actor_obs_dim)
    if actor_obs_dim is not None and actor_obs_dim != 28:
        print(
            f"[compat] checkpoint obs_dim={actor_obs_dim}; "
            f"running env observation in {actual_obs_dim}-dim compatibility mode"
        )

    actor_args = ActorArgs(args, device)
    actor = PPOActor(actor_args, env.observation_space, env.action_space, device=device)
    actor.eval()
    actor.load_state_dict(state_dict)

    buf, meta = run_episode(env, actor, args, device)
    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir / f"seed{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "circle_rl_episode.csv"
    npz_path = out_dir / "circle_rl_episode.npz"
    png_path = out_dir / "circle_rl_summary.png"
    traj_png_path = out_dir / "circle_rl_trajectory.png"
    save_csv(buf, csv_path)
    save_npz(buf, npz_path)
    if not args.no_plot:
        plot_eval(buf, png_path, title="Circle task - PPO/RL actor", show=args.show)
        plot_trajectory(buf, traj_png_path,
                        title="Circle task - RL trajectory vs reference",
                        show=args.show)

    summary = summarize(buf, meta)
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("=" * 72)
    print("Circle RL policy evaluation complete")
    print("=" * 72)
    print(f"checkpoint: {ckpt_path}")
    print(f"seed: {args.seed}, model: {args.model_name}, steps: {summary['steps']}")
    print(f"termination: {summary['term_type']}")
    print(f"total_reward: {summary['total_reward']:.3f}")
    print(f"mean_xy_err: {summary['mean_xy_err']:.3f} m")
    print(f"rmse_xy_err: {summary['rmse_xy_err']:.3f} m")
    print(f"final_xy_err: {summary['final_xy_err']:.3f} m")
    print(f"mean_vel_err: {summary['mean_vel_err']:.3f} m/s")
    print(f"rmse_heading: {summary['rmse_heading_deg']:.2f} deg")
    print(f"output_dir: {out_dir}")
    if not args.no_plot:
        print(f"plot: {png_path}")
        print(f"trajectory_plot: {traj_png_path}")
    else:
        print("plot: disabled by --no-plot")


if __name__ == "__main__":
    main(sys.argv[1:])
