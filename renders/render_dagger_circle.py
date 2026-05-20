"""Render a trained DAgger student on the circle tracking task.

Usage:
    python renders/render_dagger_circle.py --device cuda:0 --steps 2000 --no-show
    python renders/render_dagger_circle.py --ckpt algorithms/dagger/checkpoints_circle/<run>/dagger_latest.pt
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from algorithms.dagger.policy import PPOActorStudent, default_actor_args  # noqa: E402
from envs.control_env import ControlEnv                                    # noqa: E402
from envs.utils.utils import _t2n                                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='',
                   help='Path to dagger_latest.pt. Leave empty to use the newest circle checkpoint.')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--clean-obs', action='store_true',
                   help='disable CircleTask sensor/observation noise during render')
    p.add_argument('--verbose-env', action='store_true',
                   help='do not suppress verbose prints from env.step')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(ROOT, 'renders', 'result', 'circle_dagger'))
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def _find_latest_ckpt():
    roots = sorted(Path(ROOT).glob('algorithms/dagger/checkpoints_circle/*/dagger_latest.pt'),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not roots:
        raise FileNotFoundError(
            'No circle DAgger checkpoint found under algorithms/dagger/checkpoints_circle/*/dagger_latest.pt'
        )
    return str(roots[0])


def _scalar(x):
    return float(_t2n(x).reshape(-1)[0])


def _make_buf():
    return dict(
        t=[],
        npos=[], epos=[], altitude=[],
        target_npos=[], target_epos=[], target_altitude=[],
        target_vn=[], target_ve=[],
        roll=[], pitch=[], heading=[], target_heading=[],
        vx=[], vy=[], vz=[],
        reward=[],
        pos_err=[], xy_err=[],
        f_head=[], f_lift_mean=[], f_lift_spread=[],
    )


def _push(buf, env, reward=None):
    npos, epos, altitude = env.model.get_position()
    roll, pitch, heading = env.model.get_posture()
    vx, vy = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    u = env.model.get_control()

    dn = env.task.target_npos - npos
    de = env.task.target_epos - epos
    da = env.task.target_altitude - altitude
    xy_err = torch.sqrt(dn * dn + de * de)
    pos_err = torch.sqrt(dn * dn + de * de + da * da)

    buf['npos'].append(_scalar(npos))
    buf['epos'].append(_scalar(epos))
    buf['altitude'].append(_scalar(altitude))
    buf['target_npos'].append(_scalar(env.task.target_npos))
    buf['target_epos'].append(_scalar(env.task.target_epos))
    buf['target_altitude'].append(_scalar(env.task.target_altitude))
    buf['target_vn'].append(_scalar(env.task.target_vn))
    buf['target_ve'].append(_scalar(env.task.target_ve))
    buf['roll'].append(_scalar(roll))
    buf['pitch'].append(_scalar(pitch))
    buf['heading'].append(_scalar(heading))
    buf['target_heading'].append(_scalar(env.task.target_heading))
    buf['vx'].append(_scalar(vx))
    buf['vy'].append(_scalar(vy))
    buf['vz'].append(_scalar(vz))
    buf['reward'].append(0.0 if reward is None else _scalar(reward))
    buf['xy_err'].append(_scalar(xy_err))
    buf['pos_err'].append(_scalar(pos_err))

    un = _t2n(u).reshape(-1)
    lift = un[1:]
    buf['f_head'].append(float(un[0]))
    buf['f_lift_mean'].append(float(np.mean(lift)))
    buf['f_lift_spread'].append(float(np.max(lift) - np.min(lift)))


def _load_policy(env, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    actor_args = default_actor_args(**ckpt.get('args', {}))
    policy = PPOActorStudent(env.observation_space, env.action_space,
                             args=actor_args, device=device)
    state = ckpt.get('policy', ckpt.get('state_dict', ckpt))
    try:
        policy.load_state_dict(state)
    except RuntimeError as exc:
        if 'feature_norm' in str(exc) or 'size mismatch' in str(exc):
            raise RuntimeError(
                'Checkpoint observation dimension does not match the current circle task. '
                'Retrain circle DAgger after the observation update, or set '
                'circle_include_target_motion: false only when rendering old 24-dim checkpoints.'
            ) from exc
        raise
    policy.eval()
    policy.reset_rollout_state(env.n)
    print(f"[policy] loaded from {ckpt_path}")
    print(
        f"         recurrent={actor_args.use_recurrent_policy} "
        f"hidden='{actor_args.hidden_size}' act_hidden='{actor_args.act_hidden_size}'"
    )
    return policy


def _run(args, device):
    env = ControlEnv(num_envs=1, config='circle', model='HYBRID',
                     random_seed=args.seed, device=device)
    if args.clean_obs:
        env.config.enable_sensor_noise = False
        env.config.noise_scale = 0.0
        env.task.enable_sensor_noise = False
        env.task.noise_scale = 0.0
    obs = env.reset()

    ckpt_path = args.ckpt or _find_latest_ckpt()
    policy = _load_policy(env, ckpt_path, device)

    buf = _make_buf()
    buf['t'].append(0.0)
    _push(buf, env)

    bad_done_total = 0
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            action = policy.act(obs)
        if args.verbose_env:
            obs, reward, done, bad_done, exceed, _info = env.step(action)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, done, bad_done, exceed, _info = env.step(action)

        reset_mask = done | bad_done | exceed
        policy.set_done_mask(reset_mask)

        buf['t'].append(step * env.model.dt)
        _push(buf, env, reward)
        bad_done_total += int(_t2n(bad_done).sum())

        if step % 100 == 0:
            print(
                f"[dagger] step={step} t={buf['t'][-1]:.1f}s "
                f"xy_err={buf['xy_err'][-1]:.3f} pos_err={buf['pos_err'][-1]:.3f} "
                f"alt={buf['altitude'][-1]:.2f} reward={buf['reward'][-1]:.2f}"
            )

        if torch.any(reset_mask):
            print(f"[dagger] terminated at step={step}")
            break

    env.close()
    return buf, ckpt_path, bad_done_total


def _save_csv(buf, out_csv):
    keys = list(buf.keys())
    rows = zip(*(buf[k] for k in keys))
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')


def _plot(buf, out_png, title='Circle DAgger tracking'):
    t = np.asarray(buf['t'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    ax = axes[0, 0]
    ax.plot(buf['target_epos'], buf['target_npos'], 'k--', label='target circle')
    ax.plot(buf['epos'], buf['npos'], label='DAgger trajectory')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('East position (m)')
    ax.set_ylabel('North position (m)')
    ax.set_title('Horizontal trajectory')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, buf['xy_err'], label='xy error')
    ax.plot(t, buf['pos_err'], label='3-D error')
    ax.set_title('Tracking error (m)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(t, buf['altitude'], label='altitude')
    ax.plot(t, buf['target_altitude'], '--', label='target')
    ax.set_title('Altitude (m)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, np.degrees(buf['roll']), label='roll')
    ax.plot(t, np.degrees(buf['pitch']), label='pitch')
    ax.set_title('Roll / pitch (deg)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, np.degrees(np.unwrap(buf['heading'])), label='heading')
    ax.plot(t, np.degrees(np.unwrap(buf['target_heading'])), '--', label='target')
    ax.set_title('Heading (deg)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(t, buf['vx'], label='vx')
    ax.plot(t, buf['vy'], label='vy')
    ax.plot(t, buf['vz'], label='vz')
    ax.plot(t, buf['target_vn'], '--', label='target vn')
    ax.plot(t, buf['target_ve'], '--', label='target ve')
    ax.set_title('Velocity (m/s)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(t, buf['f_head'], label='F_head')
    ax.plot(t, buf['f_lift_mean'], label='F_1-4 mean')
    ax.plot(t, buf['f_lift_spread'], label='F_1-4 spread')
    ax.set_title('Final motor force (N)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(t, buf['reward'])
    ax.set_title('Per-step reward')
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.plot(t, np.cumsum(buf['reward']))
    ax.set_title('Cumulative reward')
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved: {out_png}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buf, ckpt_path, bad_done_total = _run(args, device)
    save_name = 'circle_dagger_tracking'
    _save_csv(buf, out_dir / f'{save_name}.csv')
    _plot(buf, out_dir / f'{save_name}.png',
          title='Circle task - DAgger student')

    total_reward = float(np.sum(buf['reward']))
    mean_xy_err = float(np.mean(buf['xy_err']))
    rmse_xy_err = float(np.sqrt(np.mean(np.square(buf['xy_err']))))
    print(
        f"[dagger] ckpt={ckpt_path}\n"
        f"[dagger] mean_xy_err={mean_xy_err:.3f} rmse_xy_err={rmse_xy_err:.3f} "
        f"final_xy_err={buf['xy_err'][-1]:.3f} total_reward={total_reward:.1f} "
        f"bad_done={bad_done_total}"
    )

    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
