#!/usr/bin/env python
"""Vectorized evaluation for the rpy_throttle_human task."""
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
    def __init__(self, args, device, use_safety_aux=False):
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.use_safety_aux = use_safety_aux
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenario-name", default="rpy_throttle_human_no_forward_nowind")
    p.add_argument("--model-name", default="HYBRID_NEW_NO_FORWARD")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--episodes-per-level", type=int, default=1)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=119)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--pool-warmup-steps", type=int, default=0,
                   help="For rpy_throttle_reach, run mode0-4 first to fill the in-process pose pool.")
    p.add_argument("--mode-order", default="0 1 2 3 4 5")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--save-per-level-plots", action="store_true")
    p.add_argument("--hidden-size", default="128 128")
    p.add_argument("--act-hidden-size", default="128 128")
    p.add_argument("--activation-id", type=int, default=1)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--no-feature-normalization", dest="use_feature_normalization",
                   action="store_false")
    p.set_defaults(use_feature_normalization=True)
    p.add_argument("--no-recurrent-policy", dest="use_recurrent_policy",
                   action="store_false")
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


def wrap_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def load_actor(env, args, device):
    state = torch.load(args.ckpt_path, map_location=device)
    use_safety_aux = any(k.startswith("safety_out.") for k in state.keys())
    actor = PPOActor(
        ActorArgs(args, device, use_safety_aux=use_safety_aux),
        env.observation_space,
        env.action_space,
        device,
    )
    missing, unexpected = actor.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_actor] missing={missing} unexpected={unexpected}")
    actor.eval()
    return actor


def set_fixed_levels(task, levels):
    task.curriculum_enable = True
    task.curriculum_level[:] = levels.to(task.device).long()
    task.max_curriculum_level = max(
        int(task.max_curriculum_level), int(levels.max().item())
    )
    task.mix_current = 1.0
    task.mix_easy = 0.0
    task.mix_medium = 0.0
    task.mix_random = 0.0


def is_reach_task(task):
    return getattr(task, "task_name", "") == "rpy_throttle_reach"


def resample_current_commands(task, env):
    if not hasattr(task, "_resample_raw_sticks"):
        task.sync_command(env)
        return
    mask = torch.ones(task.n, dtype=torch.bool, device=task.device)
    task._resample_raw_sticks(mask)
    task._update_px4_vtol_mc_targets_from_sticks(mask, env)
    task.sync_command(env)


def fixed_modes(task, levels):
    levels = levels.to(task.device).long()
    slots = torch.clamp(levels // int(task.levels_per_mode), 0,
                        int(task.active_mode_slots) - 1)
    return task.mode_order[slots].detach().cpu().long().numpy()


def apply_action_constraints(action, model_name):
    if model_name == "HYBRID_NEW_NO_FORWARD":
        action = action.clone()
        action[:, 0] = -1.0
    return action


def warmup_pose_pool(env, actor, args, device):
    if (not is_reach_task(env.task)) or args.pool_warmup_steps <= 0:
        return
    print(f"[warmup] filling reach pose pool for {args.pool_warmup_steps} steps")
    env.task.curriculum_enable = False
    env.task.uniform_mode_when_no_curriculum = True
    obs = env.reset()
    rnn = torch.zeros(
        (env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size),
        device=device,
    )
    masks = torch.ones(env.n, 1, device=device)
    for step in range(args.pool_warmup_steps):
        with torch.no_grad():
            action, _, rnn = actor(obs, rnn, masks, deterministic=args.deterministic)
        action = apply_action_constraints(action, args.model_name)
        obs, _reward, _done, _bad_done, _exceed, _info = env.step(action)
        if step % 250 == 0:
            print(
                "[warmup] "
                f"step={step:04d} pool_valid={int(env.task.pose_pool_valid_count)} "
                f"pool_insert={int(env.task.pose_pool_insert_count)}"
            )
    print(
        "[warmup] done "
        f"pool_valid={int(env.task.pose_pool_valid_count)} "
        f"pool_insert={int(env.task.pose_pool_insert_count)}"
    )
    env.is_done[:] = True
    env.bad_done[:] = False
    env.exceed_time_limit[:] = False


def collect_arrays(env, obs, reward, action, prev_action):
    task = env.task
    roll, pitch, yaw = env.model.get_posture()
    roll_dot, pitch_dot, yaw_rate = env.model.get_euler_angular_velocity()
    p, q, r = env.model.get_angular_velocity()
    _npos, _epos, altitude = env.model.get_position()
    vt = env.model.get_TAS()
    body_u, body_v, body_w = env.model.get_velocity()
    vx_n, vy_e = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    alpha = env.model.get_AOA()
    beta = env.model.get_AOS()
    f0, f1, f2, f3, f4 = env.model.get_F()
    collective = f1 + f2 + f3 + f4
    hover = torch.clamp(task.hover_collective, min=1e-6)

    roll_error = wrap_pi(roll - task.target_roll)
    pitch_error = wrap_pi(pitch - task.target_pitch)
    if is_reach_task(task):
        target_yaw = task.target_yaw
        yaw_error = wrap_pi(yaw - target_yaw)
        target_yaw_rate = torch.zeros_like(yaw_rate)
        yaw_rate_error = yaw_error
        yaw_metric_name = "yaw_error"
    else:
        target_yaw = yaw
        yaw_error = yaw_rate - task.target_yaw_rate
        target_yaw_rate = task.target_yaw_rate
        yaw_rate_error = yaw_error
        yaw_metric_name = "yaw_rate_error"
    throttle_frac = (collective - hover) / hover
    throttle_error = (collective - task.target_collective) / hover
    attitude_error = torch.sqrt(roll_error * roll_error + pitch_error * pitch_error)
    overshoot = task.compute_overshoot_score(
        roll_error, pitch_error, yaw_error, throttle_error
    )
    moving_away = (
        torch.relu(roll_error * roll_dot)
        + torch.relu(pitch_error * pitch_dot)
        + torch.relu(yaw_error * yaw_rate)
    )
    if prev_action is None:
        action_delta = torch.zeros(env.n, device=env.device)
    else:
        action_delta = torch.mean(torch.abs(action - prev_action), dim=1)

    final_roll = getattr(task, "final_roll", task.target_roll)
    final_pitch = getattr(task, "final_pitch", task.target_pitch)
    final_yaw = getattr(task, "final_yaw", target_yaw)
    final_throttle_frac = getattr(task, "final_throttle_frac", task.target_throttle_frac)
    start_roll = getattr(task, "start_roll", roll)
    start_pitch = getattr(task, "start_pitch", pitch)
    start_yaw = getattr(task, "start_yaw", yaw)
    start_throttle_frac = getattr(task, "start_throttle_frac", throttle_frac)
    raw_pitch = getattr(task, "raw_vx", final_pitch)
    raw_roll = getattr(task, "raw_vy", final_roll)
    raw_throttle = getattr(task, "raw_vz", final_throttle_frac)
    raw_yaw = getattr(task, "raw_yaw", wrap_pi(final_yaw - start_yaw))
    stick_pitch = getattr(task, "stick_vx", task.target_pitch)
    stick_roll = getattr(task, "stick_vy", task.target_roll)
    stick_throttle = getattr(task, "stick_vz", task.target_throttle_frac)
    stick_yaw = getattr(task, "stick_yaw", wrap_pi(target_yaw - start_yaw))
    command_profile = getattr(task, "command_profile", torch.zeros(env.n, device=env.device))
    if hasattr(task, "command_ramp_steps"):
        elapsed = (env.step_count.long() - task.command_start_step).float()
        raw_phase = torch.clamp(
            elapsed / torch.clamp(task.command_ramp_steps.float(), min=1.0),
            0.0,
            1.0,
        )
        ramp_phase = task._command_phase(raw_phase, command_profile)
    else:
        ramp_phase = torch.ones(env.n, device=env.device)

    obs_np = tensor_np(obs)
    arrays = {
        "reward": tensor_np(reward),
        "roll_deg": np.degrees(tensor_np(roll)),
        "target_roll_deg": np.degrees(tensor_np(task.target_roll)),
        "pitch_deg": np.degrees(tensor_np(pitch)),
        "target_pitch_deg": np.degrees(tensor_np(task.target_pitch)),
        "yaw_deg": np.degrees(tensor_np(yaw)),
        "target_yaw_deg": np.degrees(tensor_np(target_yaw)),
        "yaw_rate": tensor_np(yaw_rate),
        "target_yaw_rate": tensor_np(target_yaw_rate),
        "throttle_frac": tensor_np(throttle_frac),
        "target_throttle_frac": tensor_np(task.target_throttle_frac),
        "collective": tensor_np(collective),
        "target_collective": tensor_np(task.target_collective),
        "hover_collective": tensor_np(hover),
        "roll_error_deg": np.degrees(tensor_np(roll_error)),
        "pitch_error_deg": np.degrees(tensor_np(pitch_error)),
        "attitude_error_deg": np.degrees(tensor_np(attitude_error)),
        "yaw_error_deg": np.degrees(tensor_np(yaw_error)),
        "yaw_rate_error": tensor_np(yaw_rate_error),
        "yaw_metric": tensor_np(yaw_error),
        "throttle_error": tensor_np(throttle_error),
        "overshoot": tensor_np(overshoot),
        "moving_away": tensor_np(moving_away),
        "altitude": tensor_np(altitude),
        "vt": tensor_np(vt),
        "body_u": tensor_np(body_u),
        "body_v": tensor_np(body_v),
        "body_w": tensor_np(body_w),
        "vx_n": tensor_np(vx_n),
        "vy_e": tensor_np(vy_e),
        "vz": tensor_np(vz),
        "alpha_deg": np.degrees(tensor_np(alpha)),
        "beta_deg": np.degrees(tensor_np(beta)),
        "p": tensor_np(p),
        "q": tensor_np(q),
        "r": tensor_np(r),
        "yaw_metric_name_id": np.full(env.n, 1.0 if yaw_metric_name == "yaw_error" else 0.0),
        "raw_pitch": tensor_np(raw_pitch),
        "raw_roll": tensor_np(raw_roll),
        "raw_throttle": tensor_np(raw_throttle),
        "raw_yaw": tensor_np(raw_yaw),
        "stick_pitch": tensor_np(stick_pitch),
        "stick_roll": tensor_np(stick_roll),
        "stick_throttle": tensor_np(stick_throttle),
        "stick_yaw": tensor_np(stick_yaw),
        "final_roll_deg": np.degrees(tensor_np(final_roll)),
        "final_pitch_deg": np.degrees(tensor_np(final_pitch)),
        "final_yaw_deg": np.degrees(tensor_np(final_yaw)),
        "final_throttle_frac": tensor_np(final_throttle_frac),
        "start_roll_deg": np.degrees(tensor_np(start_roll)),
        "start_pitch_deg": np.degrees(tensor_np(start_pitch)),
        "start_yaw_deg": np.degrees(tensor_np(start_yaw)),
        "start_throttle_frac": tensor_np(start_throttle_frac),
        "command_profile": tensor_np(command_profile.float()),
        "ramp_phase": tensor_np(ramp_phase),
        "safety_override": tensor_np(
            getattr(
                task,
                "safety_override_active",
                torch.zeros(env.n, dtype=torch.bool, device=env.device),
            ).float()
        ),
        "action_delta": tensor_np(action_delta),
        "action_0": tensor_np(action[:, 0]),
        "action_1": tensor_np(action[:, 1]),
        "action_2": tensor_np(action[:, 2]),
        "action_3": tensor_np(action[:, 3]),
        "action_4": tensor_np(action[:, 4]),
        "f0": tensor_np(f0),
        "f1": tensor_np(f1),
        "f2": tensor_np(f2),
        "f3": tensor_np(f3),
        "f4": tensor_np(f4),
    }
    for i in range(min(12, obs_np.shape[1])):
        arrays[f"obs_{i:02d}"] = obs_np[:, i]
    return arrays


def infer_bad_reasons(last_row, config):
    reasons = []
    if last_row["altitude"] < float(getattr(config, "altitude_limit", 0.2)):
        reasons.append("low_altitude")
    if abs(last_row["roll_deg"]) > float(getattr(config, "max_roll", 30)):
        reasons.append("max_roll")
    if abs(last_row["pitch_deg"]) > float(getattr(config, "max_pitch", 25)):
        reasons.append("max_pitch")
    max_omega_norm = float(getattr(
        config, "max_omega_norm", getattr(config, "max_omega", 4)
    ))
    omega_norm = (
        last_row["p"] ** 2 + last_row["q"] ** 2 + last_row["r"] ** 2
    ) ** 0.5
    if omega_norm > max_omega_norm:
        reasons.append("max_omega_norm")
    if abs(last_row["vt"]) >= float(getattr(config, "max_velocity", 10)):
        reasons.append("high_speed")
    return "+".join(reasons) if reasons else "unknown"


def append_trace_rows(traces, arrays, active_np, levels_np, modes_np, episode_np, step, dt):
    for i in np.where(active_np)[0]:
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


def run_eval(args, device):
    seed_everything(args.seed)
    os.environ["RPY_THROTTLE_MODE_ORDER"] = args.mode_order
    os.environ["RPY_THROTTLE_MAX_MODE_SLOTS"] = str(len(args.mode_order.split()))
    os.environ["RPY_THROTTLE_REACH_MAX_MODE_SLOTS"] = str(len(args.mode_order.split()))

    levels = np.arange(args.min_level, args.max_level + 1, dtype=np.int64)
    levels_np = np.repeat(levels, args.episodes_per_level)
    episode_np = np.tile(np.arange(args.episodes_per_level, dtype=np.int64), len(levels))
    level_tensor = torch.as_tensor(levels_np, dtype=torch.long, device=device)

    env = ControlEnv(
        num_envs=len(levels_np),
        config=args.scenario_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    actor = load_actor(env, args, device)
    warmup_pose_pool(env, actor, args, device)
    set_fixed_levels(env.task, level_tensor)
    obs = env.reset()
    set_fixed_levels(env.task, level_tensor)
    resample_current_commands(env.task, env)
    modes_np = fixed_modes(env.task, level_tensor)

    rnn = torch.zeros(
        (env.n, args.recurrent_hidden_layers, args.recurrent_hidden_size),
        device=device,
    )
    active = torch.ones(env.n, dtype=torch.bool, device=device)
    term_type = np.full(env.n, "truncated", dtype=object)
    bad_reason = np.full(env.n, "", dtype=object)
    prev_action = None
    traces = [[] for _ in range(env.n)]

    for step in range(args.max_steps):
        masks = active.float().reshape(-1, 1)
        active_np = tensor_np(active).astype(bool)
        with torch.no_grad():
            action, _, rnn = actor(obs, rnn, masks, deterministic=args.deterministic)
        action = apply_action_constraints(action, args.model_name)
        obs, reward, done, bad_done, exceed, _info = env.step(action)
        set_fixed_levels(env.task, level_tensor)
        arrays = collect_arrays(env, obs, reward, action, prev_action)
        append_trace_rows(
            traces, arrays, active_np, levels_np, modes_np, episode_np, step, env.model.dt
        )
        prev_action = action.detach().clone()

        finished = active & (done.bool() | bad_done.bool() | exceed.bool())
        if torch.any(finished):
            idxs = torch.where(finished)[0].detach().cpu().numpy()
            done_np = done.detach().cpu().numpy().astype(bool)
            bad_np = bad_done.detach().cpu().numpy().astype(bool)
            exceed_np = exceed.detach().cpu().numpy().astype(bool)
            for i in idxs:
                if bad_np[i]:
                    term_type[i] = "bad_done"
                    if traces[i]:
                        bad_reason[i] = infer_bad_reasons(traces[i][-1], env.config)
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
    return traces, term_type, bad_reason


def summarize_trace(trace, term_type, bad_reason):
    arr = lambda key: np.asarray([x[key] for x in trace], dtype=np.float64)
    ret = float(np.sum(arr("reward")))
    first = trace[0]
    return {
        "level": first["level"],
        "mode_id": first["mode_id"],
        "episode": first["episode"],
        "length": len(trace),
        "return": ret,
        "attitude_error_deg": float(np.mean(np.abs(arr("attitude_error_deg")))),
        "roll_error_deg": float(np.mean(np.abs(arr("roll_error_deg")))),
        "pitch_error_deg": float(np.mean(np.abs(arr("pitch_error_deg")))),
        "yaw_error_deg": float(np.mean(np.abs(arr("yaw_error_deg")))),
        "yaw_rate_error": float(np.mean(np.abs(arr("yaw_rate_error")))),
        "throttle_error": float(np.mean(np.abs(arr("throttle_error")))),
        "overshoot": float(np.mean(arr("overshoot"))),
        "moving_away": float(np.mean(arr("moving_away"))),
        "safety_override_fraction": float(np.mean(arr("safety_override"))),
        "action_delta_mean": float(np.mean(arr("action_delta"))),
        "term_type": str(term_type),
        "bad_reason": str(bad_reason),
        "success": int(str(term_type) != "bad_done"),
    }


def aggregate_rows(rows, key):
    out = []
    for value in sorted(set(r[key] for r in rows)):
        items = [r for r in rows if r[key] == value]
        row = {key: value}
        if key == "level":
            row["mode_id"] = items[0]["mode_id"]
        else:
            row["level_min"] = min(r["level"] for r in items)
            row["level_max"] = max(r["level"] for r in items)
        for metric in [
            "return", "attitude_error_deg", "roll_error_deg", "pitch_error_deg",
            "yaw_error_deg", "yaw_rate_error", "throttle_error", "overshoot", "moving_away",
            "safety_override_fraction", "action_delta_mean", "length", "success",
        ]:
            row[metric] = float(np.mean([r[metric] for r in items]))
        row["bad_done_rate"] = 1.0 - row["success"]
        out.append(row)
    return out


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def shade_modes(ax, rows):
    if not rows or "level" not in rows[0]:
        return
    current = rows[0]["mode_id"]
    start = rows[0]["level"]
    sentinel = {"level": rows[-1]["level"] + 1, "mode_id": None}
    for row in rows[1:] + [sentinel]:
        if row["mode_id"] == current:
            continue
        end = row["level"] - 1
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.06)
        ax.text((start + end) * 0.5, 0.97, f"mode {current}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8)
        current = row["mode_id"]
        start = row["level"]


def plot_summary(level_rows, out_png):
    levels = np.asarray([r["level"] for r in level_rows])
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    plots = [
        ("return", "Return", ""),
        ("bad_done_rate", "Bad done rate", "rate"),
        ("attitude_error_deg", "Attitude error", "deg"),
        ("yaw_error_deg", "Yaw error", "deg"),
        ("throttle_error", "Throttle error", "frac"),
        ("overshoot", "Overshoot score", ""),
        ("moving_away", "Moving-away penalty source", ""),
        ("safety_override_fraction", "Safety override", "rate"),
        ("action_delta_mean", "Action delta", "mean |a_t-a_t-1|"),
        ("length", "Episode length", "steps"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.reshape(-1), plots):
        shade_modes(ax, level_rows)
        ax.plot(levels, [r[metric] for r in level_rows], lw=1.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("curriculum level")
    axes[-1, 1].set_xlabel("curriculum level")
    fig.suptitle("RPY + throttle no-forward evaluation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def representative_levels(level_rows):
    chosen = []
    for mode in sorted(set(r["mode_id"] for r in level_rows)):
        rows = [r for r in level_rows if r["mode_id"] == mode]
        chosen.extend([rows[0]["level"], rows[-1]["level"]])
    return sorted(set(chosen))


def plot_trace(trace, out_png):
    t = np.asarray([x["time_s"] for x in trace])
    level = trace[0]["level"]
    mode = trace[0]["mode_id"]
    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(t, [x["roll_deg"] for x in trace], label="roll")
    ax.plot(t, [x["target_roll_deg"] for x in trace], "--", label="target roll")
    ax.plot(t, [x["pitch_deg"] for x in trace], label="pitch")
    ax.plot(t, [x["target_pitch_deg"] for x in trace], "--", label="target pitch")
    ax.set_title("Attitude tracking")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    if trace[0].get("yaw_metric_name_id", 0.0) > 0.5:
        ax.plot(t, [x["yaw_deg"] for x in trace], label="yaw")
        ax.plot(t, [x["target_yaw_deg"] for x in trace], "--", label="target yaw")
        ax.set_title("Yaw tracking")
        ax.set_ylabel("deg")
    else:
        ax.plot(t, [x["yaw_rate"] for x in trace], label="yaw rate")
        ax.plot(t, [x["target_yaw_rate"] for x in trace], "--", label="target")
        ax.set_title("Yaw-rate tracking")
        ax.set_ylabel("rad/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t, [x["throttle_frac"] for x in trace], label="throttle frac")
    ax.plot(t, [x["target_throttle_frac"] for x in trace], "--", label="target")
    ax.set_title("Throttle fraction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(t, [x["collective"] for x in trace], label="collective")
    ax.plot(t, [x["target_collective"] for x in trace], "--", label="target")
    ax.plot(t, [x["hover_collective"] for x in trace], ":", label="hover")
    ax.set_title("Collective thrust")
    ax.set_ylabel("N")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[4]
    ax.plot(t, [x["attitude_error_deg"] for x in trace], label="att err")
    if trace[0].get("yaw_metric_name_id", 0.0) > 0.5:
        ax.plot(t, [abs(x["yaw_error_deg"]) for x in trace], label="|yaw err| deg")
    else:
        ax.plot(t, [abs(x["yaw_rate_error"]) for x in trace], label="|yaw-rate err|")
    ax.plot(t, [abs(x["throttle_error"]) for x in trace], label="|throttle err|")
    ax.set_title("Errors")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[5]
    ax.plot(t, [x["overshoot"] for x in trace], label="overshoot")
    ax.plot(t, [x["moving_away"] for x in trace], label="moving away")
    ax.set_title("Overshoot / moving away")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[6]
    for key in ["raw_pitch", "raw_roll", "raw_throttle", "raw_yaw"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Raw commands")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[7]
    for key in ["action_0", "action_1", "action_2", "action_3", "action_4"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Policy actions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    ax = axes[8]
    for key in ["f0", "f1", "f2", "f3", "f4"]:
        ax.plot(t, [x[key] for x in trace], label=key)
    ax.set_title("Motor thrust")
    ax.set_ylabel("N")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    ax = axes[9]
    ax.plot(t, [x["altitude"] for x in trace], label="altitude")
    ax.plot(t, [x["vt"] for x in trace], label="TAS")
    ax.plot(t, [x["alpha_deg"] for x in trace], label="alpha")
    ax.plot(t, [x["beta_deg"] for x in trace], label="beta")
    if "safety_override" in trace[0]:
        ax.fill_between(
            t,
            0,
            1,
            where=np.asarray([x["safety_override"] > 0.5 for x in trace]),
            transform=ax.get_xaxis_transform(),
            color="tab:red",
            alpha=0.12,
            label="safety override",
        )
    ax.set_title("State / envelope")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"level {level}, mode {mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = choose_device(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    traces, term_type, bad_reason = run_eval(args, device)

    summary = [
        summarize_trace(trace, term, reason)
        for trace, term, reason in zip(traces, term_type, bad_reason)
        if trace
    ]
    level_rows = aggregate_rows(summary, "level")
    mode_rows = aggregate_rows(summary, "mode_id")

    save_csv(summary, out_dir / "episode_summary.csv")
    save_csv(level_rows, out_dir / "level_summary.csv")
    save_csv(mode_rows, out_dir / "mode_summary.csv")
    plot_summary(level_rows, out_dir / "level_summary.png")

    trace_by_level = {}
    for trace in traces:
        if trace and trace[0]["episode"] == 0:
            trace_by_level.setdefault(trace[0]["level"], trace)

    rep_dir = out_dir / "representative_traces"
    rep_dir.mkdir(exist_ok=True)
    for level in representative_levels(level_rows):
        plot_trace(trace_by_level[level], rep_dir / f"level_{level:03d}_mode_{trace_by_level[level][0]['mode_id']}.png")

    if args.save_per_level_plots:
        all_dir = out_dir / "level_traces"
        all_dir.mkdir(exist_ok=True)
        for level, trace in trace_by_level.items():
            plot_trace(trace, all_dir / f"level_{level:03d}_mode_{trace[0]['mode_id']}.png")

    bad_reasons = {}
    for row in summary:
        if row["term_type"] == "bad_done":
            bad_reasons[row["bad_reason"]] = bad_reasons.get(row["bad_reason"], 0) + 1
    print("mode summary:")
    for row in mode_rows:
        print(row)
    print("bad reasons:", bad_reasons)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
