#!/usr/bin/env python
"""Extract RC-like command data from a PX4 ULog.

The output is aligned to ``vehicle_local_position.timestamp`` when available.
For the RC task in this repo, the most useful columns are:

    t, cmd_vx, cmd_vz, cmd_heading

Command priority:
    1. trajectory_setpoint.velocity[0/2], trajectory_setpoint.yaw
    2. vehicle_local_position_setpoint.vx/vz/yaw
    3. manual_control_setpoint mapped to vx/vz/heading_rate

PX4 local velocity is NED.  ``cmd_vz`` is PX4/NED z velocity, so positive
means downward, matching the repository's positive-down altitude convention.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from pyulog import ULog


DEFAULT_ULG = (
    "/home/a/PX4-Autopilot/build/px4_sitl_default/rootfs/log/"
    "2026-03-31/07_46_00.ulg"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ulg", type=str, default=DEFAULT_ULG)
    p.add_argument("--out-dir", type=str, default="renders/result/px4_rc_extract")
    p.add_argument("--max-nearest-dt", type=float, default=0.25,
                   help="Drop nearest-neighbour topic values farther than this many seconds.")
    p.add_argument("--manual-vx-scale", type=float, default=2.5,
                   help="Fallback pitch stick -> vx command scale.")
    p.add_argument("--manual-vz-scale", type=float, default=1.5,
                   help="Fallback throttle stick -> vz command scale.")
    p.add_argument("--manual-yaw-rate-scale", type=float, default=0.6,
                   help="Fallback yaw stick -> heading-rate scale in rad/s.")
    p.add_argument("--min-manual-samples", type=int, default=20,
                   help="Manual stick topic is ignored below this sample count.")
    return p.parse_args()


def find_dataset(ulog, name):
    for ds in ulog.data_list:
        if ds.name == name:
            return ds
    return None


def time_s(ds):
    if ds is None or "timestamp" not in ds.data:
        return None
    return np.asarray(ds.data["timestamp"], dtype=np.float64) * 1e-6


def values(ds, key):
    if ds is None or key not in ds.data:
        return None
    return np.asarray(ds.data[key], dtype=np.float64)


def nearest_to(base_t, src_t, src_v, max_dt):
    out = np.full(base_t.shape, np.nan, dtype=np.float64)
    if src_t is None or src_v is None or len(src_t) == 0:
        return out
    idx = np.searchsorted(src_t, base_t)
    left = np.clip(idx - 1, 0, len(src_t) - 1)
    right = np.clip(idx, 0, len(src_t) - 1)
    choose_right = np.abs(src_t[right] - base_t) < np.abs(src_t[left] - base_t)
    nearest = np.where(choose_right, right, left)
    dt = np.abs(src_t[nearest] - base_t)
    ok = dt <= max_dt
    out[ok] = src_v[nearest[ok]]
    return out


def finite_count(x):
    return int(np.isfinite(x).sum())


def main():
    args = parse_args()
    ulg_path = Path(args.ulg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ulog = ULog(
        str(ulg_path),
        [
            "manual_control_setpoint",
            "input_rc",
            "vehicle_local_position",
            "trajectory_setpoint",
            "vehicle_local_position_setpoint",
        ],
    )

    manual = find_dataset(ulog, "manual_control_setpoint")
    input_rc = find_dataset(ulog, "input_rc")
    lpos = find_dataset(ulog, "vehicle_local_position")
    traj = find_dataset(ulog, "trajectory_setpoint")
    lpos_sp = find_dataset(ulog, "vehicle_local_position_setpoint")

    base_t = time_s(lpos)
    if base_t is None:
        base_t = time_s(traj)
    if base_t is None:
        base_t = time_s(lpos_sp)
    if base_t is None:
        base_t = time_s(manual)
    if base_t is None:
        raise RuntimeError("No usable timestamped PX4 dataset found.")
    base_t = base_t - base_t[0]

    lpos_t = time_s(lpos)
    if lpos_t is not None:
        lpos_t = lpos_t - lpos_t[0]
    traj_t = time_s(traj)
    if traj_t is not None:
        traj_t = traj_t - traj_t[0]
    lpos_sp_t = time_s(lpos_sp)
    if lpos_sp_t is not None:
        lpos_sp_t = lpos_sp_t - lpos_sp_t[0]
    manual_t = time_s(manual)
    if manual_t is not None:
        manual_t = manual_t - manual_t[0]
    if manual_t is not None and len(manual_t) < args.min_manual_samples:
        manual = None
        manual_t = None
    input_rc_t = time_s(input_rc)
    if input_rc_t is not None:
        input_rc_t = input_rc_t - input_rc_t[0]

    actual_vx = nearest_to(base_t, lpos_t, values(lpos, "vx"), args.max_nearest_dt)
    actual_vy = nearest_to(base_t, lpos_t, values(lpos, "vy"), args.max_nearest_dt)
    actual_vz = nearest_to(base_t, lpos_t, values(lpos, "vz"), args.max_nearest_dt)
    actual_heading = nearest_to(base_t, lpos_t, values(lpos, "heading"), args.max_nearest_dt)

    traj_vx = nearest_to(base_t, traj_t, values(traj, "velocity[0]"), args.max_nearest_dt)
    traj_vz = nearest_to(base_t, traj_t, values(traj, "velocity[2]"), args.max_nearest_dt)
    traj_heading = nearest_to(base_t, traj_t, values(traj, "yaw"), args.max_nearest_dt)

    sp_vx = nearest_to(base_t, lpos_sp_t, values(lpos_sp, "vx"), args.max_nearest_dt)
    sp_vz = nearest_to(base_t, lpos_sp_t, values(lpos_sp, "vz"), args.max_nearest_dt)
    sp_heading = nearest_to(base_t, lpos_sp_t, values(lpos_sp, "yaw"), args.max_nearest_dt)

    manual_pitch = nearest_to(base_t, manual_t, values(manual, "pitch"), args.max_nearest_dt)
    manual_yaw = nearest_to(base_t, manual_t, values(manual, "yaw"), args.max_nearest_dt)
    manual_throttle = nearest_to(base_t, manual_t, values(manual, "throttle"), args.max_nearest_dt)

    # Fallback manual mapping. PX4 throttle is usually [0, 1]; centre it at 0.5.
    manual_vx = manual_pitch * args.manual_vx_scale
    manual_vz = -(manual_throttle - 0.5) * 2.0 * args.manual_vz_scale
    manual_heading = np.full_like(base_t, np.nan)
    if np.isfinite(actual_heading).any():
        manual_heading = np.copy(actual_heading)
        dt = np.diff(base_t, prepend=base_t[0])
        yaw_rate = np.nan_to_num(manual_yaw, nan=0.0) * args.manual_yaw_rate_scale
        manual_heading = actual_heading[0] + np.cumsum(yaw_rate * dt)
        manual_heading = (manual_heading + np.pi) % (2.0 * np.pi) - np.pi

    cmd_vx = np.where(np.isfinite(traj_vx), traj_vx, np.where(np.isfinite(sp_vx), sp_vx, manual_vx))
    cmd_vz = np.where(np.isfinite(traj_vz), traj_vz, np.where(np.isfinite(sp_vz), sp_vz, manual_vz))
    cmd_heading = np.where(
        np.isfinite(traj_heading),
        traj_heading,
        np.where(np.isfinite(sp_heading), sp_heading, manual_heading),
    )

    rc_cols = {}
    for ch in range(8):
        key = f"values[{ch}]"
        rc_cols[f"rc_ch{ch + 1}"] = nearest_to(base_t, input_rc_t, values(input_rc, key), args.max_nearest_dt)

    data = {
        "t": base_t,
        "cmd_vx": cmd_vx,
        "cmd_vz": cmd_vz,
        "cmd_heading": cmd_heading,
        "actual_vx": actual_vx,
        "actual_vy": actual_vy,
        "actual_vz": actual_vz,
        "actual_heading": actual_heading,
        "traj_vx": traj_vx,
        "traj_vz": traj_vz,
        "traj_heading": traj_heading,
        "lpossp_vx": sp_vx,
        "lpossp_vz": sp_vz,
        "lpossp_heading": sp_heading,
        "manual_pitch": manual_pitch,
        "manual_yaw": manual_yaw,
        "manual_throttle": manual_throttle,
        **rc_cols,
    }

    csv_path = out_dir / f"{ulg_path.stem}_rc_commands.csv"
    npz_path = out_dir / f"{ulg_path.stem}_rc_commands.npz"
    keys = list(data.keys())
    rows = zip(*(data[k] for k in keys))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
    np.savez(npz_path, **data)

    print(f"[done] ulg: {ulg_path}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] npz: {npz_path}")
    print(f"[done] samples: {len(base_t)}")
    print("[topics]")
    for name, ds in [
        ("manual_control_setpoint", manual),
        ("input_rc", input_rc),
        ("vehicle_local_position", lpos),
        ("trajectory_setpoint", traj),
        ("vehicle_local_position_setpoint", lpos_sp),
    ]:
        n = 0 if ds is None else len(ds.data["timestamp"])
        print(f"  {name}: {n}")
    print("[usable command samples]")
    print(f"  cmd_vx: {finite_count(cmd_vx)}")
    print(f"  cmd_vz: {finite_count(cmd_vz)}")
    print(f"  cmd_heading: {finite_count(cmd_heading)}")


if __name__ == "__main__":
    main()
