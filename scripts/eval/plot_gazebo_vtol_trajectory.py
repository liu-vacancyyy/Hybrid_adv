#!/usr/bin/env python
"""Plot one deterministic ground-to-ground VTOL mission trajectory."""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.ppo.ppo_actor import PPOActor
from config import get_config
from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv


PHASE_NAMES = (
    'takeoff',
    'rotor climb',
    'transition',
    'fixed wing',
    'back transition',
    'vertical landing',
)
PHASE_COLORS = (
    '#2563eb',
    '#0891b2',
    '#ca8a04',
    '#dc2626',
    '#9333ea',
    '#16a34a',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--csv-output', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--max-steps', type=int, default=None)
    return parser.parse_args()


def build_actor_args():
    args = get_config().parse_args([])
    args.use_safety_aux = False
    return args


def load_actor(checkpoint, env, device):
    actor = PPOActor(
        build_actor_args(), env.observation_space, env.action_space, device
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        state = state.get('policy', state.get('state_dict', state))
    state = {
        key: value for key, value in state.items()
        if not key.startswith('safety_out.')
    }
    actor.load_state_dict(state)
    actor.eval()
    return actor


def append_state(positions, phases, model, info):
    state = model.s[0, :3].detach().cpu().numpy().astype(np.float64)
    positions.append(state)
    if info is None:
        phases.append(0)
    else:
        phase = info['mission_phase'][0].detach().cpu().item()
        phases.append(int(phase))


def plot_trajectory(positions, phases, target, output, metadata):
    positions = np.asarray(positions)
    phases = np.asarray(phases, dtype=np.int64)
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.5, 8.0), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('North [m]')
    ax.set_ylabel('East [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title('standard_vtol deterministic mission trajectory')
    ax.view_init(elev=24.0, azim=-62.0)

    if len(positions) >= 2:
        points = positions[:, [0, 1, 2]].reshape(-1, 1, 3)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        segment_phases = phases[1:]
        colors = [
            PHASE_COLORS[min(max(int(phase), 0), len(PHASE_COLORS) - 1)]
            for phase in segment_phases
        ]
        collection = Line3DCollection(segments, colors=colors, linewidths=2.2)
        ax.add_collection3d(collection)

    start = positions[0]
    finish = positions[-1]
    ax.scatter(*start, color='#111827', marker='o', s=64, label='start')
    ax.scatter(*target, color='#16a34a', marker='*', s=150, label='landing target')
    ax.scatter(*finish, color='#f97316', marker='X', s=78, label='terminal state')

    # Keep the target visible even when the learned policy turns early.
    all_points = np.vstack((positions, np.asarray(target, dtype=np.float64)))
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    pad = np.maximum((high - low) * 0.08, np.array([5.0, 5.0, 2.0]))
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set_zlim(max(0.0, low[2] - pad[2]), high[2] + pad[2])

    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=name)
        for name, color in zip(PHASE_NAMES, PHASE_COLORS)
    ]
    handles.extend(ax.get_legend_handles_labels()[0])
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.98))
    ax.grid(True, alpha=0.3)
    fig.text(
        0.02,
        0.02,
        metadata,
        fontsize=9,
        color='#374151',
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def main():
    cli = parse_args()
    device = torch.device(
        cli.device if torch.cuda.is_available() else 'cpu'
    )
    actor_args = build_actor_args()

    def make_env():
        env = ControlEnv(
            num_envs=1,
            config='gazebo_vtol_mission',
            model='GAZEBO',
            random_seed=cli.seed,
            device=device,
        )
        env.task.curriculum_enabled = False
        # Preserve the terminal simulator state so the final trajectory point
        # is not replaced by the vector wrapper's reset observation.
        env.config.vec_auto_reset_on_done = False
        return env

    env = GPUVecEnv([make_env])
    actor = load_actor(cli.checkpoint, env, device)
    obs = env.reset()
    rnn = np.zeros((1, actor_args.recurrent_hidden_layers,
                    actor_args.recurrent_hidden_size), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    positions = []
    phases = []
    append_state(positions, phases, env.gpu_vec_env.model, None)

    max_steps = cli.max_steps
    if max_steps is None:
        max_steps = int(getattr(env.gpu_vec_env.config, 'max_steps', 3000))
    terminal_info = None
    terminal_kind = 'step limit'
    for _ in range(max_steps):
        with torch.no_grad():
            action, _, rnn_tensor = actor(
                obs.reshape(1, -1), rnn, masks, deterministic=True
            )
        rnn = rnn_tensor.detach().cpu().numpy()
        action = action.detach().cpu().numpy().reshape(1, 1, -1)
        obs, _, done, bad, timeout, info = env.step(action)
        append_state(positions, phases, env.gpu_vec_env.model, info)
        terminal = bool((done | bad | timeout).reshape(-1)[0])
        if terminal:
            terminal_info = info
            terminal_kind = (
                'safe landing' if bool(done.reshape(-1)[0])
                else 'safety failure' if bool(bad.reshape(-1)[0])
                else 'timeout'
            )
            break

    positions = np.asarray(positions)
    phases = np.asarray(phases, dtype=np.int64)
    target = np.array([
        float(getattr(env.gpu_vec_env.task, 'landing_n', 200.0)),
        float(getattr(env.gpu_vec_env.task, 'landing_e', 0.0)),
        float(getattr(env.gpu_vec_env.task, 'landing_altitude', 0.095)),
    ])
    elapsed_s = (len(positions) - 1) * float(env.gpu_vec_env.model.dt)
    landing_error = float(np.linalg.norm(positions[-1, :2] - target[:2]))
    phase_text = ', '.join(
        f'{PHASE_NAMES[i]}={int((phases == i).sum())}'
        for i in range(len(PHASE_NAMES))
        if np.any(phases == i)
    )
    reason_text = ''
    if terminal_info is not None:
        reason_keys = (
            'mission_off_route', 'mission_altitude_violation',
            'mission_premature_contact', 'extreme_aero_state',
            'extreme_angle', 'extreme_omega', 'high_speed', 'overload',
            'ground_crash_hard_touchdown', 'ground_crash_penetration',
            'ground_crash_tipover', 'ground_crash_force',
        )
        reasons = []
        for key in reason_keys:
            value = terminal_info.get(key)
            try:
                if bool(value[0].detach().cpu().item()):
                    reasons.append(key)
            except (KeyError, AttributeError, TypeError, ValueError, IndexError):
                continue
        if reasons:
            reason_text = ' | reasons=' + ','.join(reasons)
    metadata = (
        f'checkpoint={cli.checkpoint} | outcome={terminal_kind} | '
        f'time={elapsed_s:.2f} s | horizontal terminal error={landing_error:.2f} m | '
        f'samples={len(positions)} | {phase_text}{reason_text}'
    )
    plot_trajectory(positions, phases, target, cli.output, metadata)

    csv_output = cli.csv_output
    if csv_output is None:
        csv_output = str(pathlib.Path(cli.output).with_suffix('.csv'))
    np.savetxt(
        csv_output,
        np.column_stack((
            np.arange(len(positions)) * float(env.gpu_vec_env.model.dt),
            positions,
            phases,
        )),
        delimiter=',',
        header='time_s,north_m,east_m,altitude_m,phase',
        comments='',
    )
    print(metadata)
    print(f'plot={cli.output}')
    print(f'csv={csv_output}')


if __name__ == '__main__':
    main()
