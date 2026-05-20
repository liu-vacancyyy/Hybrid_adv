"""Render the 10 m circle tracking task with the existing position PID.

任务几何：
    - 圆半径 R = circle_radius，默认 10 m。
    - 圆心 = 初始位置 + circle_offset_left * 机体左侧方向，默认 10 m。
    - 因此初始飞行器到圆心的连线垂直于初始机头方向。
    - 目标点从飞行器初始位置开始，随后沿圆周运动。

控制方式：
    每个仿真 step 从 CircleTask 读取移动目标位置 / 高度 / 航向，
    然后调用已有的 HoverPIDController 位置环进行跟踪控制。
    可选开启 head motor 小前馈，用于提供沿机头方向的切向推力。

Usage:
    python renders/render_pid_circle.py --device cuda:0 --steps 2000 --no-show
    python renders/render_pid_circle.py --compare-head --no-show
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from algorithms.pid.circle_pid import CirclePIDController      # noqa: E402
from envs.control_env import ControlEnv                        # noqa: E402
from envs.utils.utils import _t2n                               # noqa: E402


def scalar(x):
    return float(_t2n(x).reshape(-1)[0])


def make_buf():
    return dict(
        t=[],
        npos=[], epos=[], altitude=[],
        target_npos=[], target_epos=[], target_altitude=[],
        roll=[], pitch=[], heading=[], target_heading=[],
        vx=[], vy=[], vz=[],
        reward=[],
        pos_err=[], xy_err=[],
        f_head=[], f_lift_mean=[], f_lift_spread=[],
    )


def push(buf, env, reward=None):
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

    buf['npos'].append(scalar(npos))
    buf['epos'].append(scalar(epos))
    buf['altitude'].append(scalar(altitude))
    buf['target_npos'].append(scalar(env.task.target_npos))
    buf['target_epos'].append(scalar(env.task.target_epos))
    buf['target_altitude'].append(scalar(env.task.target_altitude))
    buf['roll'].append(scalar(roll))
    buf['pitch'].append(scalar(pitch))
    buf['heading'].append(scalar(heading))
    buf['target_heading'].append(scalar(env.task.target_heading))
    buf['vx'].append(scalar(vx))
    buf['vy'].append(scalar(vy))
    buf['vz'].append(scalar(vz))
    buf['reward'].append(0.0 if reward is None else scalar(reward))
    buf['xy_err'].append(scalar(xy_err))
    buf['pos_err'].append(scalar(pos_err))

    un = _t2n(u).reshape(-1)
    lift = un[1:]
    buf['f_head'].append(float(un[0]))
    buf['f_lift_mean'].append(float(np.mean(lift)))
    buf['f_lift_spread'].append(float(np.max(lift) - np.min(lift)))


def run(args, device, with_head=False):
    env = ControlEnv(num_envs=1, config='circle', model='HYBRID',
                     random_seed=args.seed, device=device)
    env.reset()

    pid = CirclePIDController(
        n=env.n,
        device=device,
        dt=env.model.dt,
        max_thrust_per_motor=env.model.max_F,
        use_head_motor=with_head,
        head_max_force=args.head_max_force,
    )
    pid.reset()
    pid.set_circle_targets(env.task)

    buf = make_buf()
    buf['t'].append(0.0)
    push(buf, env)

    for step in range(1, args.steps + 1):
        pid.set_circle_targets(env.task)
        action = pid.compute_action(env.model)
        _obs, reward, done, bad_done, exceed, _info = env.step(action)

        reset_mask = done | bad_done | exceed
        if torch.any(reset_mask):
            pid.reset(mask=reset_mask)

        buf['t'].append(step * env.model.dt)
        push(buf, env, reward)

        if step % args.log_interval == 0:
            tag = 'circle-pid+head' if with_head else 'circle-pid'
            print(
                f"[{tag}] step={step} t={buf['t'][-1]:.1f}s "
                f"xy_err={buf['xy_err'][-1]:.3f} pos_err={buf['pos_err'][-1]:.3f} "
                f"alt={buf['altitude'][-1]:.2f} reward={buf['reward'][-1]:.2f}"
            )

        if torch.any(reset_mask):
            tag = 'circle-pid+head' if with_head else 'circle-pid'
            print(f"[{tag}] terminated at step={step}")
            break

    env.close()
    return buf


def save_csv(buf, out_csv):
    keys = list(buf.keys())
    rows = zip(*(buf[k] for k in keys))
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')


def plot(buf, out_png, title='Circle PID tracking'):
    t = np.asarray(buf['t'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    ax = axes[0, 0]
    ax.plot(buf['target_epos'], buf['target_npos'], 'k--', label='target circle')
    ax.plot(buf['epos'], buf['npos'], label='PID trajectory')
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
    ax.set_title('Velocity (m/s)')
    ax.grid(alpha=0.3)
    ax.legend()

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


def plot_compare(base, head, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(base['target_epos'], base['target_npos'], 'k--', label='target')
    ax.plot(base['epos'], base['npos'], label='PID')
    ax.plot(head['epos'], head['npos'], label='PID + head')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('East position (m)')
    ax.set_ylabel('North position (m)')
    ax.set_title('Horizontal trajectory')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(base['t'], base['xy_err'], label='PID')
    ax.plot(head['t'], head['xy_err'], label='PID + head')
    ax.set_title('XY tracking error (m)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(base['t'], base['f_head'], label='PID F_head')
    ax.plot(head['t'], head['f_head'], label='PID + head F_head')
    ax.set_title('Head motor force (N)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(base['t'], np.cumsum(base['reward']), label='PID')
    ax.plot(head['t'], np.cumsum(head['reward']), label='PID + head')
    ax.set_title('Cumulative reward')
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved: {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--log-interval', type=int, default=250)
    p.add_argument('--with-head', action='store_true',
                   help='Enable a small tangent-speed feedforward on the head motor.')
    p.add_argument('--compare-head', action='store_true',
                   help='Run both baseline PID and PID+head, then save comparison plots.')
    p.add_argument('--head-max-force', type=float, default=0.8,
                   help='Maximum head motor feedforward force in Newtons.')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(ROOT, 'renders', 'result', 'circle_pid'))
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_head:
        base = run(args, device, with_head=False)
        head = run(args, device, with_head=True)
        save_csv(base, out_dir / 'circle_pid_tracking_no_head.csv')
        save_csv(head, out_dir / 'circle_pid_tracking_with_head.csv')
        plot(base, out_dir / 'circle_pid_tracking_no_head.png',
             title='Circle PID tracking - no head motor')
        plot(head, out_dir / 'circle_pid_tracking_with_head.png',
             title='Circle PID tracking - with head motor feedforward')
        plot_compare(base, head, out_dir / 'circle_pid_compare_head.png')
        print(
            f"[circle-pid] no_head mean_xy={np.mean(base['xy_err']):.3f} "
            f"rmse_xy={np.sqrt(np.mean(np.square(base['xy_err']))):.3f} "
            f"final_xy={base['xy_err'][-1]:.3f}"
        )
        print(
            f"[circle-pid+head] mean_xy={np.mean(head['xy_err']):.3f} "
            f"rmse_xy={np.sqrt(np.mean(np.square(head['xy_err']))):.3f} "
            f"final_xy={head['xy_err'][-1]:.3f}"
        )
        if not args.no_show:
            plt.show()
        plt.close('all')
        return

    buf = run(args, device, with_head=args.with_head)
    suffix = 'with_head' if args.with_head else 'tracking'
    save_csv(buf, out_dir / f'circle_pid_{suffix}.csv')
    plot(buf, out_dir / f'circle_pid_{suffix}.png',
         title='Circle PID tracking - with head motor' if args.with_head else 'Circle PID tracking')

    tag = 'circle-pid+head' if args.with_head else 'circle-pid'
    print(
        f"[{tag}] mean_xy_err={np.mean(buf['xy_err']):.3f} "
        f"rmse_xy_err={np.sqrt(np.mean(np.square(buf['xy_err']))):.3f} "
        f"final_xy_err={buf['xy_err'][-1]:.3f}"
    )

    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
