#!/usr/bin/env python
"""Build RC command datasets for RCTask.

Inputs can be extracted PX4 command NPZ files from ``extract_px4_rc_data.py``.
The script writes:
  - a cleaned PX4-reference dataset
  - a human-like synthetic dataset with piecewise stick targets + smoothing
  - a mixed dataset suitable for RC command replay experiments
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str,
                   default="renders/result/px4_rc_extract/07_46_00_rc_commands.npz")
    p.add_argument("--out-dir", type=str, default="data/rc_commands")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--vx-limit", type=float, default=2.5)
    p.add_argument("--vz-limit", type=float, default=1.5)
    p.add_argument("--yaw-rate-limit", type=float, default=0.6)
    p.add_argument("--synthetic-episodes", type=int, default=128)
    p.add_argument("--synthetic-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def interp_finite(t_src, x_src, t_dst):
    finite = np.isfinite(t_src) & np.isfinite(x_src)
    if finite.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=np.float64)
    return np.interp(t_dst, t_src[finite], x_src[finite])


def save_dataset(path_npz, path_csv, data):
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path_npz, **data)

    keys = list(data.keys())
    arrays = [np.asarray(data[k]) for k in keys]
    flat = {}
    for k, a in zip(keys, arrays):
        flat[k] = a.reshape(-1)
    n = len(next(iter(flat.values())))
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([flat[k][i] for k in keys])


def build_px4_reference(input_npz, dt, vx_limit, vz_limit, yaw_rate_limit):
    src = np.load(input_npz)
    t = src["t"].astype(np.float64)
    t_new = np.arange(0.0, np.nanmax(t) + 1e-9, dt)

    vx = interp_finite(t, src["cmd_vx"], t_new)
    vz = interp_finite(t, src["cmd_vz"], t_new)
    heading = interp_finite(t, np.unwrap(src["cmd_heading"]), t_new)
    heading = wrap_pi(heading)

    vx = np.clip(vx, -vx_limit, vx_limit)
    vz = np.clip(vz, -vz_limit, vz_limit)
    yaw_rate = np.gradient(np.unwrap(heading), dt)
    yaw_rate = np.clip(yaw_rate, -yaw_rate_limit, yaw_rate_limit)

    valid = np.isfinite(vx) & np.isfinite(vz) & np.isfinite(heading)
    return {
        "t": t_new[valid],
        "vx_cmd": vx[valid],
        "vz_cmd": vz[valid],
        "heading_cmd": heading[valid],
        "yaw_rate_cmd": yaw_rate[valid],
        "source_id": np.zeros(int(valid.sum()), dtype=np.int64),
    }


def build_synthetic(n_ep, n_steps, dt, vx_limit, vz_limit, yaw_rate_limit, seed):
    rng = np.random.default_rng(seed)
    total = n_ep * n_steps
    t = np.tile(np.arange(n_steps, dtype=np.float64) * dt, n_ep)
    vx = np.zeros(total, dtype=np.float64)
    vz = np.zeros(total, dtype=np.float64)
    yaw_rate = np.zeros(total, dtype=np.float64)
    heading = np.zeros(total, dtype=np.float64)
    episode = np.repeat(np.arange(n_ep, dtype=np.int64), n_steps)

    # Smooth stick-like behaviour: choose a target every 1.5-5 s, low-pass it.
    tau_v = 0.55
    tau_y = 0.75
    alpha_v = dt / (tau_v + dt)
    alpha_y = dt / (tau_y + dt)

    for ep in range(n_ep):
        s = ep * n_steps
        e = s + n_steps
        cur_vx = rng.normal(0.0, 0.08)
        cur_vz = rng.normal(0.0, 0.05)
        cur_yaw = rng.normal(0.0, 0.03)
        hdg = rng.uniform(-np.pi, np.pi)
        k = s
        while k < e:
            dwell = int(rng.integers(int(1.5 / dt), int(5.0 / dt) + 1))
            target_vx = rng.uniform(-0.85 * vx_limit, 0.85 * vx_limit)
            target_vz = rng.uniform(-0.70 * vz_limit, 0.70 * vz_limit)
            target_yaw = rng.uniform(-0.75 * yaw_rate_limit, 0.75 * yaw_rate_limit)
            for _ in range(dwell):
                if k >= e:
                    break
                cur_vx += alpha_v * (target_vx - cur_vx) + rng.normal(0.0, 0.018)
                cur_vz += alpha_v * (target_vz - cur_vz) + rng.normal(0.0, 0.012)
                cur_yaw += alpha_y * (target_yaw - cur_yaw) + rng.normal(0.0, 0.006)
                cur_vx = np.clip(cur_vx, -vx_limit, vx_limit)
                cur_vz = np.clip(cur_vz, -vz_limit, vz_limit)
                cur_yaw = np.clip(cur_yaw, -yaw_rate_limit, yaw_rate_limit)
                hdg = wrap_pi(hdg + cur_yaw * dt)
                vx[k] = cur_vx
                vz[k] = cur_vz
                yaw_rate[k] = cur_yaw
                heading[k] = hdg
                k += 1

    return {
        "t": t,
        "vx_cmd": vx,
        "vz_cmd": vz,
        "heading_cmd": heading,
        "yaw_rate_cmd": yaw_rate,
        "episode": episode,
        "source_id": np.ones(total, dtype=np.int64),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    px4 = build_px4_reference(
        Path(args.input), args.dt, args.vx_limit, args.vz_limit, args.yaw_rate_limit
    )
    syn = build_synthetic(
        args.synthetic_episodes,
        args.synthetic_steps,
        args.dt,
        args.vx_limit,
        args.vz_limit,
        args.yaw_rate_limit,
        args.seed,
    )

    save_dataset(out_dir / "rc_px4_reference.npz", out_dir / "rc_px4_reference.csv", px4)
    save_dataset(out_dir / "rc_humanlike_synthetic.npz", out_dir / "rc_humanlike_synthetic.csv", syn)

    mixed = {
        "t": np.concatenate([px4["t"], syn["t"]]),
        "vx_cmd": np.concatenate([px4["vx_cmd"], syn["vx_cmd"]]),
        "vz_cmd": np.concatenate([px4["vz_cmd"], syn["vz_cmd"]]),
        "heading_cmd": np.concatenate([px4["heading_cmd"], syn["heading_cmd"]]),
        "yaw_rate_cmd": np.concatenate([px4["yaw_rate_cmd"], syn["yaw_rate_cmd"]]),
        "source_id": np.concatenate([px4["source_id"], syn["source_id"]]),
    }
    save_dataset(out_dir / "rc_mixed_commands.npz", out_dir / "rc_mixed_commands.csv", mixed)

    print(f"[done] out_dir: {out_dir}")
    for name, data in [("px4", px4), ("synthetic", syn), ("mixed", mixed)]:
        print(
            f"{name}: n={len(data['vx_cmd'])} "
            f"vx=[{np.min(data['vx_cmd']):.3f},{np.max(data['vx_cmd']):.3f}] "
            f"vz=[{np.min(data['vz_cmd']):.3f},{np.max(data['vz_cmd']):.3f}] "
            f"yaw_rate=[{np.min(data['yaw_rate_cmd']):.3f},{np.max(data['yaw_rate_cmd']):.3f}]"
        )


if __name__ == "__main__":
    main()
