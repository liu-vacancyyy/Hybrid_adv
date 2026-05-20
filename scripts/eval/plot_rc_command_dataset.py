#!/usr/bin/env python
"""Plot generated RC command datasets."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/rc_commands/rc_mixed_commands.npz")
    p.add_argument("--out-dir", type=str, default="renders/result/rc_command_dataset")
    p.add_argument("--synthetic-steps", type=int, default=2000)
    p.add_argument("--episodes-to-plot", type=int, default=4)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def plot_overview(data, out_png):
    t = data["t"]
    vx = data["vx_cmd"]
    vz = data["vz_cmd"]
    heading = data["heading_cmd"]
    yaw_rate = data["yaw_rate_cmd"]
    source = data["source_id"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    for src, label, color in [(0, "PX4 reference", "#1f77b4"), (1, "synthetic", "#2ca02c")]:
        mask = source == src
        if not np.any(mask):
            continue
        x = np.arange(mask.sum())
        axes[0].plot(x, vx[mask], lw=0.8, color=color, alpha=0.8, label=label)
        axes[1].plot(x, vz[mask], lw=0.8, color=color, alpha=0.8, label=label)
        axes[2].plot(x, np.degrees(heading[mask]), lw=0.8, color=color, alpha=0.8, label=label)
        axes[3].plot(x, np.degrees(yaw_rate[mask]), lw=0.8, color=color, alpha=0.8, label=label)

    axes[0].set_ylabel("vx cmd (m/s)")
    axes[1].set_ylabel("vz cmd (m/s)")
    axes[2].set_ylabel("heading (deg)")
    axes[3].set_ylabel("yaw rate (deg/s)")
    axes[3].set_xlabel("Sample index within source")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Generated RC command dataset overview")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[plot] saved: {out_png}")
    return fig


def plot_synthetic_examples(data, out_png, synthetic_steps, episodes_to_plot):
    source = data["source_id"]
    mask = source == 1
    vx = data["vx_cmd"][mask]
    vz = data["vz_cmd"][mask]
    heading = data["heading_cmd"][mask]
    yaw_rate = data["yaw_rate_cmd"][mask]

    n_ep = len(vx) // synthetic_steps
    n_plot = min(episodes_to_plot, n_ep)
    if n_plot <= 0:
        return None

    fig, axes = plt.subplots(n_plot, 4, figsize=(16, 3.0 * n_plot), squeeze=False)
    t = np.arange(synthetic_steps) * 0.02

    for i in range(n_plot):
        s = i * synthetic_steps
        e = s + synthetic_steps
        axes[i, 0].plot(t, vx[s:e])
        axes[i, 0].set_title(f"Episode {i} vx")
        axes[i, 1].plot(t, vz[s:e])
        axes[i, 1].set_title(f"Episode {i} vz")
        axes[i, 2].plot(t, np.degrees(heading[s:e]))
        axes[i, 2].set_title(f"Episode {i} heading")
        axes[i, 3].plot(t, np.degrees(yaw_rate[s:e]))
        axes[i, 3].set_title(f"Episode {i} yaw rate")
        for ax in axes[i]:
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.3)

    fig.suptitle("Human-like synthetic RC command examples")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[plot] saved: {out_png}")
    return fig


def plot_histograms(data, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = [
        ("vx_cmd", "vx cmd (m/s)"),
        ("vz_cmd", "vz cmd (m/s)"),
        ("heading_cmd", "heading cmd (rad)"),
        ("yaw_rate_cmd", "yaw rate cmd (rad/s)"),
    ]
    for ax, (key, title) in zip(axes.reshape(-1), fields):
        x = data[key]
        ax.hist(x[np.isfinite(x)], bins=80, alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("RC command distributions")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[plot] saved: {out_png}")
    return fig


def main():
    args = parse_args()
    data = np.load(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [
        plot_overview(data, out_dir / "rc_command_overview.png"),
        plot_synthetic_examples(
            data,
            out_dir / "rc_command_synthetic_examples.png",
            args.synthetic_steps,
            args.episodes_to_plot,
        ),
        plot_histograms(data, out_dir / "rc_command_histograms.png"),
    ]
    if args.show:
        plt.show()
    for fig in figs:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
