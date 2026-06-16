#!/usr/bin/env python
"""Render rc_human victim policy under a trained adversary.

The default setup uses the current best victim policy (episode_880) and the
episode_75 adversary trained with randomized commands.  Command attack is kept
disabled by default, so rc_human generates modes 0-1 commands normally while
the adversary supplies observation perturbations and wind.
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.adversarial.rc_human_adv_env import RCHumanAdversarialEnv  # noqa: E402
from algorithms.ppo.ppo_actor import PPOActor                              # noqa: E402
from envs.control_env import ControlEnv                                     # noqa: E402
from envs.utils.utils import wrap_PI                                        # noqa: E402


DEFAULT_VICTIM = (
    ROOT / "scripts" / "runs"
    / "2026-05-20_23-15-17_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_first2"
    / "episode_880" / "actor_latest.ckpt"
)
DEFAULT_ADV = (
    ROOT / "scripts" / "runs"
    / "2026-05-21_20-39-18_Control_rc_human_HYBRID_NEW_ppo_rc_human_adv_modes0_1_random_command_ep880"
    / "episode_75" / "adv_actor_latest.ckpt"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--victim-ckpt", type=str, default=str(DEFAULT_VICTIM))
    p.add_argument("--adv-ckpt", type=str, default=str(DEFAULT_ADV))
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument("--mode-order", type=str, default="0 1")
    p.add_argument("--seed", type=int, default=880)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "renders" / "result" / "rc_human_ep880_adv75_random_command"))
    p.add_argument("--tracks-dir", type=str,
                   default=str(ROOT / "renders" / "tracks" / "rc_human_ep880_adv75_random_command"))

    p.add_argument("--hidden-size", type=str, default="128 128")
    p.add_argument("--act-hidden-size", type=str, default="128 128")
    p.add_argument("--activation-id", type=int, default=1)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--recurrent-hidden-size", type=int, default=128)
    p.add_argument("--recurrent-hidden-layers", type=int, default=1)
    p.add_argument("--deterministic-victim", action="store_true", default=True)
    p.add_argument("--deterministic-adv", action="store_true", default=True)

    p.add_argument("--adv-command-frac", type=float, default=0.18)
    p.add_argument("--adv-obs-frac", type=float, default=0.6)
    p.add_argument("--adv-wind-frac", type=float, default=0.5)
    p.add_argument("--adv-command-alpha", type=float, default=0.20)
    p.add_argument("--adv-obs-alpha", type=float, default=0.25)
    p.add_argument("--adv-wind-alpha", type=float, default=0.15)
    p.add_argument("--adv-obs-default-scale", type=float, default=0.02)
    p.add_argument("--adv-obs-max-scale", type=float, default=0.10)
    p.add_argument("--adv-obs-energy-window", type=int, default=50)
    p.add_argument("--adv-obs-energy-budget", type=float, default=50.0)
    p.add_argument("--adv-obs-energy-penalty", type=float, default=0.02)
    p.add_argument("--adv-alive-penalty", type=float, default=0.01)
    p.add_argument("--adv-policy-reward-weight", type=float, default=0.15)
    p.add_argument("--adv-w-vel-error", type=float, default=4.0)
    p.add_argument("--adv-w-axis-vel-error", type=float, default=2.0)
    p.add_argument("--adv-w-yaw-error", type=float, default=2.0)
    p.add_argument("--adv-axis-vel-margin", type=float, default=0.25)
    p.add_argument("--adv-yaw-margin-deg", type=float, default=6.0)
    p.add_argument("--adv-w-vel-bad-margin", type=float, default=8.0)
    p.add_argument("--adv-w-yaw-bad-margin", type=float, default=4.0)
    p.add_argument("--adv-w-attitude", type=float, default=5.0)
    p.add_argument("--adv-w-omega", type=float, default=0.8)
    p.add_argument("--adv-w-force-margin", type=float, default=0.2)
    p.add_argument("--adv-bad-done-bonus", type=float, default=50.0)
    p.add_argument("--adv-linf-penalty", type=float, default=0.0)
    p.add_argument("--adv-smooth-penalty", type=float, default=0.0)
    p.add_argument("--adv-command-smooth-penalty", type=float, default=0.02)
    p.add_argument("--adv-obs-smooth-penalty", type=float, default=0.02)
    p.add_argument("--adv-wind-smooth-penalty", type=float, default=0.03)
    p.add_argument("--adv-command-target-rms-min", type=float, default=0.25)
    p.add_argument("--adv-command-target-rms-max", type=float, default=0.75)
    p.add_argument("--adv-obs-target-rms-min", type=float, default=0.45)
    p.add_argument("--adv-obs-target-rms-max", type=float, default=1.10)
    p.add_argument("--adv-wind-target-rms-min", type=float, default=0.35)
    p.add_argument("--adv-wind-target-rms-max", type=float, default=0.90)
    p.add_argument("--adv-command-range-penalty", type=float, default=0.40)
    p.add_argument("--adv-obs-range-penalty", type=float, default=0.50)
    p.add_argument("--adv-wind-range-penalty", type=float, default=0.50)
    p.add_argument("--adv-saturation-penalty", type=float, default=0.20)
    p.add_argument("--adv-raw-excess-penalty", type=float, default=0.20)
    p.add_argument("--adv-attitude-safe-rad", type=float, default=0.20)
    p.add_argument("--adv-omega-safe-rad", type=float, default=1.2)
    return p.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def actor_args(args, device):
    return SimpleNamespace(
        gain=args.gain,
        hidden_size=args.hidden_size,
        act_hidden_size=args.act_hidden_size,
        activation_id=args.activation_id,
        use_feature_normalization=True,
        use_recurrent_policy=True,
        recurrent_hidden_size=args.recurrent_hidden_size,
        recurrent_hidden_layers=args.recurrent_hidden_layers,
        tpdv=dict(dtype=torch.float32, device=device),
        use_prior=False,
    )


def adv_env_args(args):
    return SimpleNamespace(
        adv_command_frac=args.adv_command_frac,
        adv_obs_frac=args.adv_obs_frac,
        adv_wind_frac=args.adv_wind_frac,
        adv_use_random_command=True,
        adv_command_random_base=False,
        adv_obs_default_scale=args.adv_obs_default_scale,
        adv_obs_max_scale=args.adv_obs_max_scale,
        adv_command_alpha=args.adv_command_alpha,
        adv_obs_alpha=args.adv_obs_alpha,
        adv_wind_alpha=args.adv_wind_alpha,
        adv_obs_energy_window=args.adv_obs_energy_window,
        adv_obs_energy_budget=args.adv_obs_energy_budget,
        adv_obs_energy_penalty=args.adv_obs_energy_penalty,
        adv_alive_penalty=args.adv_alive_penalty,
        adv_policy_reward_weight=args.adv_policy_reward_weight,
        adv_w_vel_error=args.adv_w_vel_error,
        adv_w_axis_vel_error=args.adv_w_axis_vel_error,
        adv_w_yaw_error=args.adv_w_yaw_error,
        adv_axis_vel_margin=args.adv_axis_vel_margin,
        adv_yaw_margin_deg=args.adv_yaw_margin_deg,
        adv_w_vel_bad_margin=args.adv_w_vel_bad_margin,
        adv_w_yaw_bad_margin=args.adv_w_yaw_bad_margin,
        adv_w_attitude=args.adv_w_attitude,
        adv_w_omega=args.adv_w_omega,
        adv_w_force_margin=args.adv_w_force_margin,
        adv_bad_done_bonus=args.adv_bad_done_bonus,
        adv_linf_penalty=args.adv_linf_penalty,
        adv_smooth_penalty=args.adv_smooth_penalty,
        adv_command_smooth_penalty=args.adv_command_smooth_penalty,
        adv_obs_smooth_penalty=args.adv_obs_smooth_penalty,
        adv_wind_smooth_penalty=args.adv_wind_smooth_penalty,
        adv_command_target_rms_min=args.adv_command_target_rms_min,
        adv_command_target_rms_max=args.adv_command_target_rms_max,
        adv_obs_target_rms_min=args.adv_obs_target_rms_min,
        adv_obs_target_rms_max=args.adv_obs_target_rms_max,
        adv_wind_target_rms_min=args.adv_wind_target_rms_min,
        adv_wind_target_rms_max=args.adv_wind_target_rms_max,
        adv_command_range_penalty=args.adv_command_range_penalty,
        adv_obs_range_penalty=args.adv_obs_range_penalty,
        adv_wind_range_penalty=args.adv_wind_range_penalty,
        adv_saturation_penalty=args.adv_saturation_penalty,
        adv_raw_excess_penalty=args.adv_raw_excess_penalty,
        adv_attitude_safe_rad=args.adv_attitude_safe_rad,
        adv_omega_safe_rad=args.adv_omega_safe_rad,
        victim_recurrent_hidden_size=args.recurrent_hidden_size,
        victim_recurrent_hidden_layers=args.recurrent_hidden_layers,
        victim_deterministic=args.deterministic_victim,
    )


def load_actor(path, obs_space, action_space, args, device):
    actor = PPOActor(actor_args(args, device), obs_space, action_space, device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        state = state.get("policy", state.get("state_dict", state))
    actor.load_state_dict(state)
    actor.eval()
    return actor


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def record(base_env, adv_env, step, reward, adv_reward, adv_action):
    task = base_env.task
    model = base_env.model
    roll, pitch, heading = model.get_posture()
    vx_n, vy_e = model.get_ground_speed()
    vz = model.get_climb_rate()
    local_vx, local_vy = task.ground_to_local_velocity(vx_n, vy_e, heading)
    yaw_err = wrap_PI(heading - task.target_heading)
    local_vx_error, local_vy_error = task.ground_to_local_velocity(
        vx_n - task.target_vn, vy_e - task.target_ve, heading
    )
    axis_err = torch.stack((
        torch.abs(local_vx_error),
        torch.abs(local_vy_error),
        torch.abs(vz - task.target_vz),
    ), dim=0)
    p, q, r = model.get_angular_velocity()
    alpha = model.get_AOA()
    beta = model.get_AOS()
    wind_n, wind_e, wind_d = model.get_wind_ned() if hasattr(model, "get_wind_ned") else (
        torch.zeros_like(vz), torch.zeros_like(vz), torch.zeros_like(vz)
    )
    wind_mag = torch.sqrt(wind_n * wind_n + wind_e * wind_e + wind_d * wind_d)
    info = adv_env.last_info
    return {
        "step": step,
        "time_s": step * model.dt,
        "policy_reward": scalar(reward),
        "adv_reward": scalar(adv_reward),
        "local_vx": scalar(local_vx),
        "target_vx": scalar(task.target_vx),
        "local_vy": scalar(local_vy),
        "target_vy": scalar(task.target_vy),
        "vz": scalar(vz),
        "target_vz": scalar(task.target_vz),
        "heading": scalar(heading),
        "target_heading": scalar(task.target_heading),
        "yaw_err_rad": scalar(yaw_err),
        "yaw_err_deg": float(np.degrees(abs(scalar(yaw_err)))),
        "axis_vel_err_max": scalar(torch.max(axis_err, dim=0).values),
        "vel_err": scalar(torch.sqrt(torch.sum(axis_err * axis_err, dim=0))),
        "roll_deg": float(np.degrees(scalar(roll))),
        "pitch_deg": float(np.degrees(scalar(pitch))),
        "alpha_rad": scalar(alpha),
        "beta_rad": scalar(beta),
        "alpha_deg": float(np.degrees(scalar(alpha))),
        "beta_deg": float(np.degrees(scalar(beta))),
        "omega": scalar(torch.sqrt(p * p + q * q + r * r)),
        "wind_n": scalar(wind_n),
        "wind_e": scalar(wind_e),
        "wind_d": scalar(wind_d),
        "wind_mag": scalar(wind_mag),
        "adv_linf": float(info.get("adv/linf", 0.0)),
        "adv_obs_abs": float(info.get("adv/obs_abs", 0.0)),
        "adv_wind_abs": float(info.get("adv/wind_abs", 0.0)),
        "adv_obs_energy_ratio": float(info.get("adv/obs_energy_ratio", 0.0)),
        "adv_action_linf": float(torch.max(torch.abs(adv_action)).detach().cpu().item()),
    }


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(rows, out_png):
    t = np.asarray([r["time_s"] for r in rows])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    axes[0, 0].plot(t, [r["local_vx"] for r in rows], label="vx")
    axes[0, 0].plot(t, [r["target_vx"] for r in rows], "--", label="target vx")
    axes[0, 0].plot(t, [r["local_vy"] for r in rows], label="vy")
    axes[0, 0].plot(t, [r["target_vy"] for r in rows], "--", label="target vy")
    axes[0, 0].set_title("Horizontal velocity")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, [r["vz"] for r in rows], label="vz")
    axes[0, 1].plot(t, [r["target_vz"] for r in rows], "--", label="target vz")
    axes[0, 1].set_title("Vertical velocity")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, np.degrees(np.unwrap([r["heading"] for r in rows])), label="heading")
    axes[1, 0].plot(t, np.degrees(np.unwrap([r["target_heading"] for r in rows])), "--", label="target")
    axes[1, 0].set_title("Heading")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, [r["vel_err"] for r in rows], label="vel err")
    axes[1, 1].plot(t, [r["axis_vel_err_max"] for r in rows], label="max axis err")
    axes[1, 1].plot(t, [r["yaw_err_deg"] for r in rows], label="yaw err deg")
    axes[1, 1].set_title("Tracking errors")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(t, [r["wind_n"] for r in rows], label="wind N")
    axes[2, 0].plot(t, [r["wind_e"] for r in rows], label="wind E")
    axes[2, 0].plot(t, [r["wind_d"] for r in rows], label="wind D")
    axes[2, 0].plot(t, [r["wind_mag"] for r in rows], label="|wind|")
    axes[2, 0].set_title("Adversarial wind")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(t, [r["alpha_deg"] for r in rows], label="alpha deg")
    axes[2, 1].plot(t, [r["beta_deg"] for r in rows], label="beta deg")
    axes[2, 1].plot(t, [r["adv_linf"] for r in rows], label="adv linf")
    axes[2, 1].set_title("Alpha/beta and adversary scale")
    axes[2, 1].set_ylabel("deg / normalized")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    mode_ids = [x for x in args.mode_order.replace(",", " ").split() if x]
    os.environ["RC_HUMAN_MODE_ORDER"] = " ".join(mode_ids)
    os.environ["RC_HUMAN_MAX_MODE_SLOTS"] = str(len(mode_ids))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    tracks_dir = Path(args.tracks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    base_env = ControlEnv(num_envs=1, config="rc_human", model=args.model_name,
                          random_seed=args.seed, device=device)
    victim = load_actor(args.victim_ckpt, base_env.observation_space,
                        base_env.action_space, args, device)
    adv_env = RCHumanAdversarialEnv(base_env, victim, adv_env_args(args), device)
    adv_actor = load_actor(args.adv_ckpt, adv_env.observation_space,
                           adv_env.action_space, args, device)

    # Network construction consumes torch RNG.  Reseed immediately before the
    # environment reset so command randomization is comparable across renders.
    seed_everything(args.seed)
    obs = adv_env.reset()
    adv_rnn = torch.zeros((base_env.n, args.recurrent_hidden_layers,
                           args.recurrent_hidden_size), device=device)
    adv_masks = torch.ones((base_env.n, 1), device=device)

    prefix = tracks_dir / "RCHumanPPO_ep880_adv_ep75_random_command_"
    if not args.no_render:
        base_env.render(count=0, filename=str(prefix))

    rows = []
    term_type = "truncated"
    policy_return = 0.0
    adv_return = 0.0
    for step in range(args.steps):
        with torch.no_grad():
            adv_action, _, adv_rnn = adv_actor(
                obs, adv_rnn, adv_masks, deterministic=args.deterministic_adv
            )
        obs, adv_reward, done, bad_done, exceed, _info = adv_env.step(adv_action)
        policy_reward = torch.tensor(
            [[adv_env.last_info.get("adv/policy_reward", 0.0)]],
            dtype=torch.float32,
            device=device,
        )
        policy_return += scalar(policy_reward)
        adv_return += scalar(adv_reward)
        rows.append(record(base_env, adv_env, step, policy_reward, adv_reward, adv_action))

        if not args.no_render:
            base_env.render(count=1, filename=str(prefix))

        if bool(torch.any(done).item()):
            term_type = "done"
            break
        if bool(torch.any(bad_done).item()):
            term_type = "bad_done"
            break
        if bool(torch.any(exceed).item()):
            term_type = "timeout"
            break

    summary = {
        "victim_ckpt": args.victim_ckpt,
        "adv_ckpt": args.adv_ckpt,
        "seed": args.seed,
        "mode_order": " ".join(mode_ids),
        "length": len(rows),
        "term_type": term_type,
        "policy_return": policy_return,
        "adv_return": adv_return,
        "vel_mae": float(np.mean([r["vel_err"] for r in rows])),
        "axis_vel_err_max_mean": float(np.mean([r["axis_vel_err_max"] for r in rows])),
        "axis_vel_err_max_peak": float(np.max([r["axis_vel_err_max"] for r in rows])),
        "yaw_mae_deg": float(np.mean([r["yaw_err_deg"] for r in rows])),
        "yaw_peak_deg": float(np.max([r["yaw_err_deg"] for r in rows])),
        "wind_mag_mean": float(np.mean([r["wind_mag"] for r in rows])),
        "wind_mag_peak": float(np.max([r["wind_mag"] for r in rows])),
        "adv_obs_abs_mean": float(np.mean([r["adv_obs_abs"] for r in rows])),
        "track_file": str(prefix) + "0.txt.acmi",
    }

    save_csv(rows, out_dir / "rc_human_ep880_adv75_trace.csv")
    save_csv([summary], out_dir / "rc_human_ep880_adv75_summary.csv")
    plot_trace(rows, out_dir / "rc_human_ep880_adv75_trace.png")
    np.savez(out_dir / "rc_human_ep880_adv75_raw.npz", **{
        key: np.asarray([r[key] for r in rows])
        for key in rows[0]
        if isinstance(rows[0][key], (int, float, np.floating))
    })

    print("RC human adversarial render complete")
    for key in [
        "length", "term_type", "policy_return", "adv_return", "vel_mae",
        "axis_vel_err_max_mean", "axis_vel_err_max_peak", "yaw_mae_deg",
        "yaw_peak_deg", "wind_mag_mean", "wind_mag_peak", "adv_obs_abs_mean",
    ]:
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"output_dir: {out_dir}")
    print(f"track_file: {summary['track_file']}")

    adv_env.close()


if __name__ == "__main__":
    main()
