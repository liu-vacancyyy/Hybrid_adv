#!/usr/bin/env python
import argparse
import csv
import datetime
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from config import get_config
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def parse_args(argv):
    parser = get_config()
    group = parser.add_argument_group("RC Eval")
    group.add_argument("--env-name", type=str, default="Control")
    group.add_argument("--scenario-name", type=str, default="rc")
    group.add_argument("--model-name", type=str, default="HYBRID")
    group.add_argument("--ckpt-path", type=str,
                       default="/home/a/tensor_rc/episode_70/actor_latest.ckpt")
    group.add_argument("--runs-root", type=str, default="scripts/runs")
    group.add_argument("--episodes", type=int, default=50)
    group.add_argument("--max-steps", type=int, default=1200)
    group.add_argument("--plot-episodes", type=int, default=3)
    group.add_argument("--train-seed", type=int, default=5,
                       help="训练时使用的seed，用于与评估seed做差异检查")
    group.add_argument("--eval-seed", type=int, default=13,
                       help="评估seed，默认与train_rc.sh不同")
    group.add_argument("--output-dir", type=str, default="renders/result/rc_eval")
    all_args = parser.parse_known_args(argv)[0]
    return all_args


def resolve_checkpoint(args):
    if args.ckpt_path:
        p = Path(args.ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint不存在: {p}")
        return p

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        runs_root = root / runs_root

    if not runs_root.exists():
        raise FileNotFoundError(f"runs目录不存在: {runs_root}")

    candidates = list(runs_root.glob("*_Control_rc_HYBRID_ppo_*/episode_*/actor_latest.ckpt"))
    if not candidates:
        raise FileNotFoundError("未找到RC checkpoint，请通过 --ckpt-path 指定")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def choose_device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device(args.device)
    return torch.device("cpu")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_eval(env, actor, args, device):
    rows = []
    trace_bank = []

    for ep in range(args.episodes):
        obs = env.reset()
        rnn = torch.zeros((env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size), device=device)
        masks = torch.ones((env.n, 1), device=device)

        ep_reward = 0.0
        ep_action_effort = 0.0
        ep_action_delta = 0.0
        prev_action = None

        err_vx, err_vz, err_heading = [], [], []
        vx_hist, vx_tgt_hist = [], []
        vz_hist, vz_tgt_hist = [], []
        hdg_hist, hdg_tgt_hist = [], []
        reward_hist = []

        term_type = "truncated"
        ep_len = 0

        for _ in range(args.max_steps):
            vx, _ = env.model.get_ground_speed()
            vz = env.model.get_climb_rate()
            _, _, heading = env.model.get_posture()
            tvx = env.task.target_vx
            tvz = env.task.target_vz
            thdg = env.task.target_heading

            dvx = to_scalar(vx - tvx)
            dvz = to_scalar(vz - tvz)
            dhdg = wrap_pi_np(to_scalar(heading - thdg))

            err_vx.append(dvx)
            err_vz.append(dvz)
            err_heading.append(dhdg)
            vx_hist.append(to_scalar(vx))
            vx_tgt_hist.append(to_scalar(tvx))
            vz_hist.append(to_scalar(vz))
            vz_tgt_hist.append(to_scalar(tvz))
            hdg_hist.append(to_scalar(heading))
            hdg_tgt_hist.append(to_scalar(thdg))

            with torch.no_grad():
                action, _, rnn = actor(obs, rnn, masks, deterministic=True)

            ep_action_effort += float(torch.mean(torch.sum(torch.abs(action), dim=1)).item())
            if prev_action is not None:
                ep_action_delta += float(torch.mean(torch.abs(action - prev_action)).item())
            prev_action = action.detach().clone()

            obs, reward, done, bad_done, exceed_tl, _ = env.step(action)

            r = float(torch.mean(reward).item())
            ep_reward += r
            reward_hist.append(r)
            ep_len += 1

            done_flag = bool(torch.any(done).item())
            bad_flag = bool(torch.any(bad_done).item())
            timeout_flag = bool(torch.any(exceed_tl).item())
            if done_flag or bad_flag or timeout_flag:
                if done_flag:
                    term_type = "done"
                elif bad_flag:
                    term_type = "bad_done"
                else:
                    term_type = "timeout"
                break

        err_vx_np = np.asarray(err_vx, dtype=np.float64)
        err_vz_np = np.asarray(err_vz, dtype=np.float64)
        err_heading_np = np.asarray(err_heading, dtype=np.float64)

        rows.append({
            "episode": ep,
            "length": ep_len,
            "return": ep_reward,
            "mae_vx": float(np.mean(np.abs(err_vx_np))),
            "mae_vz": float(np.mean(np.abs(err_vz_np))),
            "mae_heading_rad": float(np.mean(np.abs(err_heading_np))),
            "rmse_vx": float(np.sqrt(np.mean(err_vx_np ** 2))),
            "rmse_vz": float(np.sqrt(np.mean(err_vz_np ** 2))),
            "rmse_heading_rad": float(np.sqrt(np.mean(err_heading_np ** 2))),
            "max_abs_heading_err_rad": float(np.max(np.abs(err_heading_np))) if err_heading_np.size else 0.0,
            "avg_action_effort": ep_action_effort / max(ep_len, 1),
            "avg_action_delta": ep_action_delta / max(ep_len - 1, 1),
            "term_type": term_type,
            "success": 1 if term_type == "done" else 0,
        })

        if ep < args.plot_episodes:
            trace_bank.append({
                "episode": ep,
                "vx": np.asarray(vx_hist),
                "vx_tgt": np.asarray(vx_tgt_hist),
                "vz": np.asarray(vz_hist),
                "vz_tgt": np.asarray(vz_tgt_hist),
                "hdg": np.asarray(hdg_hist),
                "hdg_tgt": np.asarray(hdg_tgt_hist),
                "reward": np.asarray(reward_hist),
            })

    return rows, trace_bank


def save_csv(rows, out_csv):
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, trace_bank, out_png):
    returns = np.asarray([r["return"] for r in rows], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float64)
    mae_vx = np.asarray([r["mae_vx"] for r in rows], dtype=np.float64)
    mae_vz = np.asarray([r["mae_vz"] for r in rows], dtype=np.float64)
    mae_hdg = np.asarray([r["mae_heading_rad"] for r in rows], dtype=np.float64)
    success = np.asarray([r["success"] for r in rows], dtype=np.float64)
    action_delta = np.asarray([r["avg_action_delta"] for r in rows], dtype=np.float64)
    term_types = [r["term_type"] for r in rows]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(returns, lw=1.5, color="#1f77b4", label="episode return")
    if len(returns) >= 5:
        k = 5
        roll = np.convolve(returns, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k - 1, len(returns)), roll, lw=2.0, color="#d62728", label="5-ep moving avg")
    ax.set_title("Episode Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels = ["done", "bad_done", "timeout", "truncated"]
    counts = [term_types.count(x) for x in labels]
    ax.bar(labels, counts, color=["#2ca02c", "#d62728", "#9467bd", "#7f7f7f"])
    ax.set_title("Termination Type Count")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    cum_success = np.cumsum(success) / (np.arange(len(success)) + 1.0)
    ax.plot(cum_success, lw=2.0, color="#2ca02c")
    ax.set_title("Cumulative Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.boxplot([mae_vx, mae_vz, mae_hdg], labels=["MAE vx", "MAE vz", "MAE hdg(rad)"])
    ax.set_title("Tracking Error Distribution")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2, 0]
    sc = ax.scatter(lengths, returns, c=action_delta, cmap="viridis", s=28, alpha=0.8)
    ax.set_title("Return vs Episode Length")
    ax.set_xlabel("Length")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Avg |Δaction|")

    ax = axes[2, 1]
    if trace_bank:
        tr = trace_bank[0]
        t = np.arange(len(tr["reward"]))
        hdg_err = wrap_pi_np(tr["hdg"] - tr["hdg_tgt"])
        ax.plot(t, tr["vx"] - tr["vx_tgt"], lw=1.2, label="vx error")
        ax.plot(t, tr["vz"] - tr["vz_tgt"], lw=1.2, label="vz error")
        ax.plot(t, hdg_err, lw=1.2, label="heading error")
        ax.set_title(f"Error Trace (Episode {tr['episode']})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Error")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tracking(trace_bank, out_png):
    if not trace_bank:
        return

    n = len(trace_bank)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.0 * n), squeeze=False)

    for i, tr in enumerate(trace_bank):
        t = np.arange(len(tr["vx"]))

        ax = axes[i, 0]
        ax.plot(t, tr["vx"], lw=1.4, label="vx")
        ax.plot(t, tr["vx_tgt"], "--", lw=1.2, label="vx target")
        ax.set_title(f"Episode {tr['episode']} - vx")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.plot(t, tr["vz"], lw=1.4, label="vz")
        ax.plot(t, tr["vz_tgt"], "--", lw=1.2, label="vz target")
        ax.set_title(f"Episode {tr['episode']} - vz")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[i, 2]
        hdg = np.unwrap(tr["hdg"])
        hdg_t = np.unwrap(tr["hdg_tgt"])
        ax.plot(t, hdg, lw=1.4, label="heading")
        ax.plot(t, hdg_t, "--", lw=1.2, label="heading target")
        ax.set_title(f"Episode {tr['episode']} - heading")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    args = parse_args(argv)
    if args.eval_seed == args.train_seed:
        raise ValueError(
            f"eval_seed({args.eval_seed}) 与 train_seed({args.train_seed}) 相同，请改成不同seed")

    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / f"seed{args.eval_seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(args)
    device = choose_device(args)
    seed_everything(args.eval_seed)

    env = ControlEnv(num_envs=1,
                     config=args.scenario_name,
                     model=args.model_name,
                     random_seed=args.eval_seed,
                     device=str(device))

    actor = PPOActor(args, env.observation_space, env.action_space, device=device)
    actor.eval()
    state_dict = torch.load(str(ckpt_path), map_location=device)
    actor.load_state_dict(state_dict)

    rows, trace_bank = run_eval(env, actor, args, device)
    env.close()

    csv_path = out_dir / "episode_metrics.csv"
    summary_png = out_dir / "rc_eval_summary.png"
    tracking_png = out_dir / "rc_tracking_examples.png"
    npz_path = out_dir / "rc_eval_raw.npz"

    save_csv(rows, csv_path)
    plot_summary(rows, trace_bank, summary_png)
    plot_tracking(trace_bank, tracking_png)

    np.savez(npz_path,
             returns=np.asarray([r["return"] for r in rows]),
             lengths=np.asarray([r["length"] for r in rows]),
             mae_vx=np.asarray([r["mae_vx"] for r in rows]),
             mae_vz=np.asarray([r["mae_vz"] for r in rows]),
             mae_heading_rad=np.asarray([r["mae_heading_rad"] for r in rows]),
             success=np.asarray([r["success"] for r in rows]),
             avg_action_delta=np.asarray([r["avg_action_delta"] for r in rows]))

    avg_return = float(np.mean([r["return"] for r in rows]))
    suc_rate = float(np.mean([r["success"] for r in rows]))
    avg_mae_vx = float(np.mean([r["mae_vx"] for r in rows]))
    avg_mae_vz = float(np.mean([r["mae_vz"] for r in rows]))
    avg_mae_h = float(np.mean([r["mae_heading_rad"] for r in rows]))

    print("=" * 72)
    print("RC策略评估完成")
    print("=" * 72)
    print(f"checkpoint: {ckpt_path}")
    print(f"train_seed: {args.train_seed}, eval_seed: {args.eval_seed}")
    print(f"episodes: {args.episodes}")
    print(f"avg_return: {avg_return:.4f}")
    print(f"success_rate: {suc_rate:.4f}")
    print(f"avg_mae_vx: {avg_mae_vx:.4f} m/s")
    print(f"avg_mae_vz: {avg_mae_vz:.4f} m/s")
    print(f"avg_mae_heading: {avg_mae_h:.4f} rad")
    print(f"结果目录: {out_dir}")
    print(f"- {csv_path.name}")
    print(f"- {summary_png.name}")
    print(f"- {tracking_png.name}")
    print(f"- {npz_path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
