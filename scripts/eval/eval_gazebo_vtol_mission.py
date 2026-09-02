#!/usr/bin/env python
"""Evaluate a VTOL mission actor on complete ground-to-ground episodes."""

import argparse
import pathlib
import sys

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.ppo.ppo_actor import PPOActor
from config import get_config
from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=11)
    return parser.parse_args()


def build_actor_args():
    args = get_config().parse_args([])
    args.use_safety_aux = True
    return args


def load_actor(checkpoint, env, device):
    actor = PPOActor(
        build_actor_args(), env.observation_space, env.action_space, device
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        state = state.get('policy', state.get('state_dict', state))
    actor.load_state_dict(state)
    actor.eval()
    return actor


def main():
    cli = parse_args()
    device = torch.device(
        cli.device if torch.cuda.is_available() else 'cpu'
    )

    def make_env():
        env = ControlEnv(
            num_envs=cli.num_envs,
            config='gazebo_vtol_mission',
            model='GAZEBO',
            random_seed=cli.seed,
            device=device,
        )
        env.task.curriculum_enabled = False
        return env

    env = GPUVecEnv([make_env])
    actor = load_actor(cli.checkpoint, env, device)
    obs = env.reset()
    actor_args = build_actor_args()
    rnn = np.zeros((
        cli.num_envs,
        actor_args.recurrent_hidden_layers,
        actor_args.recurrent_hidden_size,
    ), dtype=np.float32)
    masks = np.ones((cli.num_envs, 1), dtype=np.float32)

    completed = 0
    success_count = 0
    failure_count = 0
    timeout_count = 0
    success_times = []
    landing_errors = []
    touchdown_speeds = []
    while completed < cli.episodes:
        with torch.no_grad():
            action, _, rnn_tensor = actor(
                obs.reshape(cli.num_envs, -1),
                rnn,
                masks,
                deterministic=True,
            )
        rnn = rnn_tensor.detach().cpu().numpy()
        action = action.detach().cpu().numpy().reshape(
            cli.num_envs, 1, -1
        )
        obs, _, done, bad, timeout, info = env.step(action)
        done = done.reshape(-1).astype(bool)
        bad = bad.reshape(-1).astype(bool)
        timeout = timeout.reshape(-1).astype(bool)
        terminal = done | bad | timeout
        if not terminal.any():
            masks.fill(1.0)
            continue

        success = info['mission_success'].detach().cpu().numpy().astype(bool)
        elapsed = info['mission_elapsed_steps'].detach().cpu().numpy()
        error = info['mission_landing_error'].detach().cpu().numpy()
        touchdown_speed = info['mission_touchdown_speed'].detach().cpu().numpy()
        indices = np.flatnonzero(terminal)
        remaining = cli.episodes - completed
        indices = indices[:remaining]
        completed += len(indices)
        success_count += int(success[indices].sum())
        failure_count += int(bad[indices].sum())
        timeout_count += int(timeout[indices].sum())
        success_indices = indices[success[indices]]
        success_times.extend((elapsed[success_indices] * 0.02).tolist())
        landing_errors.extend(error[success_indices].tolist())
        touchdown_speeds.extend(touchdown_speed[success_indices].tolist())
        masks.fill(1.0)
        masks[terminal] = 0.0
        rnn[terminal] = 0.0

    success_rate = success_count / max(completed, 1)
    mean_time = float(np.mean(success_times)) if success_times else float('nan')
    mean_error = float(np.mean(landing_errors)) if landing_errors else float('nan')
    mean_touchdown_speed = (
        float(np.mean(touchdown_speeds)) if touchdown_speeds else float('nan')
    )
    print(f'episodes={completed}')
    print(f'safe_landing_rate={success_rate:.4f}')
    print(f'safe_landings={success_count}')
    print(f'safety_failures={failure_count}')
    print(f'timeouts={timeout_count}')
    print(f'mean_success_time_s={mean_time:.3f}')
    print(f'mean_landing_error_m={mean_error:.3f}')
    print(f'mean_touchdown_speed_mps={mean_touchdown_speed:.3f}')


if __name__ == '__main__':
    main()
