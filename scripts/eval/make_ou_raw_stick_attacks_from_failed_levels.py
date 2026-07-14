#!/usr/bin/env python
"""Generate OU raw-stick attack traces from failed rc_human eval levels."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_mode_order(raw):
    return [int(v) for v in raw.replace(",", " ").split() if v.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode-order", default="0 1 2 5 4 3")
    p.add_argument("--levels-per-mode", type=int, default=20)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--variants-per-level", type=int, default=3)
    p.add_argument("--seed", type=int, default=97230)
    p.add_argument("--ou-theta", type=float, default=2.4)
    p.add_argument("--ou-sigma-base", type=float, default=0.16)
    p.add_argument("--ou-sigma-hard-bonus", type=float, default=0.08)
    p.add_argument("--dwell-min-steps", type=int, default=75)
    p.add_argument("--dwell-max-steps", type=int, default=180)
    p.add_argument("--rate-limit-frac", type=float, default=0.05)
    p.add_argument("--easy-stick", type=float, default=0.25)
    p.add_argument("--medium-stick", type=float, default=0.65)
    p.add_argument("--hard-stick", type=float, default=1.0)
    return p.parse_args()


def mode_from_level(level, mode_order, levels_per_mode):
    return mode_order[min(level // levels_per_mode, len(mode_order) - 1)]


def amp_from_level(level, mode, args):
    sublevel = level % args.levels_per_mode
    progress = sublevel / max(args.levels_per_mode - 1, 1)
    amp = args.easy_stick + progress * (args.hard_stick - args.easy_stick)
    if mode in (2, 3, 4):
        amp = min(max(amp, args.medium_stick), args.hard_stick)
    if mode == 0:
        amp = 0.0
    return float(amp)


def load_failures(eval_dir):
    by_level = defaultdict(lambda: {"rows": [], "bad": 0, "n": 0, "mode": None})
    failed_episodes = defaultdict(set)
    summary_path = eval_dir / "rc_human_curriculum_level_summary_raw.csv"
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            level = int(row["level"])
            ep = int(row["episode"])
            item = by_level[level]
            item["rows"].append(row)
            item["n"] += 1
            item["mode"] = int(row["mode_id"])
            if row["term_type"] == "bad_done":
                item["bad"] += 1
                failed_episodes[level].add(ep)
    failed_levels = [level for level, item in by_level.items() if item["bad"] > 0]
    failed_levels.sort(key=lambda lv: (-by_level[lv]["bad"], by_level[lv]["mode"], lv))
    return by_level, failed_episodes, failed_levels


def load_failed_trace_rows(eval_dir, failed_episodes):
    trace_rows = defaultdict(list)
    trace_path = eval_dir / "rc_human_curriculum_level_traces.csv"
    with trace_path.open(newline="") as f:
        for row in csv.DictReader(f):
            level = int(row["level"])
            ep = int(row["episode"])
            if level in failed_episodes and ep in failed_episodes[level]:
                trace_rows[(level, ep)].append(row)
    return trace_rows


def failed_anchor_vectors(level, amp, failed_episodes, trace_rows):
    anchors = []
    for ep in sorted(failed_episodes[level]):
        rows = trace_rows.get((level, ep), [])
        if not rows:
            continue
        max_err = max(rows, key=lambda r: float(r.get("vel_err", 0.0)))
        candidates = [rows[len(rows) // 3], rows[(2 * len(rows)) // 3], max_err, rows[-1]]
        for row in candidates:
            v = np.array([
                float(row["raw_vx"]),
                float(row["raw_vy"]),
                float(row["raw_vz"]),
                0.0,
            ])
            if np.max(np.abs(v[:3])) > 0.05:
                anchors.append(np.clip(v, -amp, amp))
    return anchors


def random_mu_for_mode(rng, mode, amp, prev_mu, anchors):
    mu = np.zeros(4)
    if mode == 1:
        if anchors and rng.random() < 0.4:
            base = anchors[int(rng.integers(0, len(anchors)))]
            mu[:3] = np.clip(base[:3] + rng.normal(0.0, 0.25 * amp, size=3), -amp, amp)
        else:
            mu[:3] = rng.uniform(-amp, amp, size=3)
    elif mode == 2:
        if anchors and rng.random() < 0.55:
            base = anchors[int(rng.integers(0, len(anchors)))]
            axis = int(np.argmax(np.abs(base[:3])))
            sign = 1.0 if base[axis] >= 0.0 else -1.0
        else:
            axis = int((np.argmax(np.abs(prev_mu[:3])) + rng.choice([1, 2])) % 3)
            sign = rng.choice([-1.0, 1.0])
        mu[axis] = sign * amp
    elif mode == 3:
        if np.max(np.abs(prev_mu[:3])) > 0.05:
            axis = int(np.argmax(np.abs(prev_mu[:3])))
            sign = -1.0 if prev_mu[axis] >= 0.0 else 1.0
        elif anchors:
            base = anchors[int(rng.integers(0, len(anchors)))]
            axis = int(np.argmax(np.abs(base[:3])))
            sign = -1.0 if base[axis] >= 0.0 else 1.0
        else:
            axis = int(rng.integers(0, 3))
            sign = rng.choice([-1.0, 1.0])
        mu[axis] = sign * amp
    elif mode == 4:
        if anchors and rng.random() < 0.45:
            base = anchors[int(rng.integers(0, len(anchors)))]
            signs = np.where(base[:3] >= 0.0, 1.0, -1.0)
            mag = np.maximum(np.abs(base[:3]), rng.uniform(0.35 * amp, amp, size=3))
            mu[:3] = np.clip(signs * mag, -amp, amp)
        else:
            mu[:3] = rng.choice([-1.0, 1.0], size=3) * rng.uniform(0.35 * amp, amp, size=3)
    elif mode == 5:
        mu[0] = amp
    return np.clip(mu, -amp, amp)


def make_schedule(rng, mode, amp, anchors, args):
    mu = np.zeros((args.steps, 4))
    prev = np.zeros(4)
    t = 0
    while t < args.steps:
        dwell = int(rng.integers(args.dwell_min_steps, args.dwell_max_steps + 1))
        nxt = min(args.steps, t + dwell)
        prev = random_mu_for_mode(rng, mode, amp, prev, anchors)
        mu[t:nxt] = prev
        t = nxt
    mu[:, 3] = 0.0
    return mu


def generate_ou(level, mode, amp, variant, anchors, args):
    rng = np.random.default_rng(args.seed + level * 101 + variant * 1009)
    mu = make_schedule(rng, mode, amp, anchors, args)
    desired = np.zeros((args.steps, 4))
    raw = np.zeros((args.steps, 4))
    x = np.zeros(4)
    sigma = args.ou_sigma_base + args.ou_sigma_hard_bonus * min(max(amp, 0.0), 1.0)
    for i in range(args.steps):
        noise = rng.normal(0.0, 1.0, size=4)
        noise[3] = 0.0
        x = x + args.ou_theta * (mu[i] - x) * args.dt + sigma * math.sqrt(args.dt) * noise
        desired[i] = np.clip(x, -amp, amp)
        desired[i, 3] = 0.0
        prev_raw = raw[i - 1] if i else np.zeros(4)
        delta = np.clip(desired[i] - prev_raw, -args.rate_limit_frac, args.rate_limit_frac)
        raw[i] = np.clip(prev_raw + delta, -1.0, 1.0)
        raw[i, 3] = 0.0
    return mu, desired, raw


def write_outputs(out_dir, summary_rows, sequence_rows, params):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ou_raw_stick_attack_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "ou_raw_stick_attack_sequences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    (out_dir / "ou_raw_stick_attack_params.json").write_text(
        json.dumps(params, indent=2) + "\n"
    )


def plot_level(out_dir, level, item, generated, amp, variants):
    colors = ["tab:blue", "tab:orange", "tab:green"]
    names = ["vx", "vy", "vz"]
    t = np.arange(generated[0][2].shape[0]) * 0.02
    fig, axes = plt.subplots(variants, 1, figsize=(14, 2.9 * variants), sharex=True)
    axes = np.atleast_1d(axes)
    for variant, ax in enumerate(axes):
        mu, _desired, raw = generated[variant]
        for idx, name in enumerate(names):
            ax.plot(t, raw[:, idx], color=colors[idx], lw=1.25, label=f"raw_{name}")
            ax.plot(t, mu[:, idx], color=colors[idx], lw=0.75, ls="--", alpha=0.45)
        ax.axhline(amp, color="0.2", lw=0.7, ls=":")
        ax.axhline(-amp, color="0.2", lw=0.7, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"variant {variant}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time s")
    fig.suptitle(
        f"OU raw_stick attack, level {level:03d}, mode {item['mode']}, "
        f"amp={amp:.3f}, source bad_done={item['bad']}/{item['n']}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"level_{level:03d}_mode{item['mode']}_ou_raw_stick_attack.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    mode_order = parse_mode_order(args.mode_order)
    by_level, failed_episodes, failed_levels = load_failures(eval_dir)
    trace_rows = load_failed_trace_rows(eval_dir, failed_episodes)

    summary_rows = []
    sequence_rows = []
    all_generated = {}
    for level in failed_levels:
        item = by_level[level]
        mode = int(item["mode"])
        amp = amp_from_level(level, mode, args)
        anchors = failed_anchor_vectors(level, amp, failed_episodes, trace_rows)
        generated = []
        rows = item["rows"]
        summary_rows.append({
            "level": level,
            "mode_id": mode,
            "bad_done_count": item["bad"],
            "episodes": item["n"],
            "amp": amp,
            "source_failed_episodes": " ".join(str(e) for e in sorted(failed_episodes[level])),
            "source_vel_mae_mean": sum(float(r["vel_mae"]) for r in rows) / len(rows),
            "source_att_mae_deg_mean": sum(float(r["att_mae_deg"]) for r in rows) / len(rows),
            "ou_theta": args.ou_theta,
            "ou_sigma": args.ou_sigma_base + args.ou_sigma_hard_bonus * min(max(amp, 0.0), 1.0),
            "dwell_min_steps": args.dwell_min_steps,
            "dwell_max_steps": args.dwell_max_steps,
            "rate_limit_frac": args.rate_limit_frac,
        })
        for variant in range(args.variants_per_level):
            mu, desired, raw = generate_ou(level, mode, amp, variant, anchors, args)
            generated.append((mu, desired, raw))
            for step in range(args.steps):
                sequence_rows.append({
                    "level": level,
                    "mode_id": mode,
                    "variant": variant,
                    "step": step,
                    "time_s": step * args.dt,
                    "amp": amp,
                    "raw_vx": raw[step, 0],
                    "raw_vy": raw[step, 1],
                    "raw_vz": raw[step, 2],
                    "raw_yaw": raw[step, 3],
                    "desired_raw_vx": desired[step, 0],
                    "desired_raw_vy": desired[step, 1],
                    "desired_raw_vz": desired[step, 2],
                    "desired_raw_yaw": desired[step, 3],
                    "ou_mu_vx": mu[step, 0],
                    "ou_mu_vy": mu[step, 1],
                    "ou_mu_vz": mu[step, 2],
                    "ou_mu_yaw": mu[step, 3],
                })
        all_generated[level] = generated

    params = vars(args).copy()
    params["eval_dir"] = str(eval_dir)
    params["failed_levels"] = failed_levels
    params["mode_order"] = mode_order
    params["yaw_command_enable"] = False
    params["note"] = "Synthetic OU desired raw sticks, rate-limited into raw_stick attacks."
    write_outputs(out_dir, summary_rows, sequence_rows, params)
    for level in failed_levels:
        mode = int(by_level[level]["mode"])
        plot_level(out_dir, level, by_level[level], all_generated[level],
                   amp_from_level(level, mode, args), args.variants_per_level)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
