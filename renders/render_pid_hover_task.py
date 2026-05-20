"""
Render the hover task on the Hybrid platform under PID control,
using the standard ControlEnv + HoverTask pipeline.

Usage:
    python renders/render_pid_hover_task.py
    python renders/render_pid_hover_task.py --model HYBRID_NEW --wind --clean-obs --no-render --no-show
"""
import argparse
import contextlib
import io
import os
import sys
import time
import numpy as np
import torch
import matplotlib
import os; os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from envs.control_env import ControlEnv                              # noqa: E402
from envs.utils.utils import _t2n                                    # noqa: E402
from algorithms.pid.hover_pid import HoverPIDController              # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='HYBRID',
                   choices=['HYBRID', 'HYBRID_NEW', 'HYBRID_WIND'])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--wind', action='store_true',
                   help='enable configured HYBRID_NEW Dryden turbulence')
    p.add_argument('--clean-obs', action='store_true',
                   help='disable HoverTask observation sensor noise')
    p.add_argument('--no-render', action='store_true',
                   help='skip Tacview/acmi logging during rollout')
    p.add_argument('--no-show', action='store_true',
                   help='save plots without opening a matplotlib window')
    p.add_argument('--verbose-env', action='store_true',
                   help='do not suppress verbose prints from env.step')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(ROOT, 'renders', 'result'))
    return p.parse_args()


def _configure_env(env, args):
    if args.clean_obs:
        env.config.enable_sensor_noise = False
        env.config.noise_scale = 0.0
        env.task.enable_sensor_noise = False
        env.task.noise_scale = 0.0

    if args.wind:
        if not hasattr(env.model, 'set_wind_gust_ned'):
            raise ValueError('--wind requires --model HYBRID_NEW or HYBRID_WIND')
        env.config.enable_wind = True
        env.config.enable_dryden_turbulence = True
        env.model.wind_enabled = True
        env.model.base_wind_ned.zero_()
        env.model.set_wind_ned(0.0, 0.0, 0.0)
        env.wind_disturbance = None
        env._init_wind_disturbance()


def _scalar(x):
    return float(_t2n(x).reshape(-1)[0])


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    total_steps = args.steps

    env = ControlEnv(num_envs=1, config='hover', model=args.model,
                     random_seed=args.seed, device=device)
    _configure_env(env, args)

    pid = HoverPIDController(n=env.n, device=device, dt=env.model.dt,
                             mass=1.779, gravity=9.807,
                             max_thrust_per_motor=env.model.max_F)
    pid.reset()

    obs = env.reset()
    pid.set_targets(target_altitude=env.task.target_altitude.clone(),
                    target_heading=env.task.target_heading.clone(),
                    target_npos=env.task.target_npos.clone(),
                    target_epos=env.task.target_epos.clone())

    result_dir = args.out_dir
    os.makedirs(result_dir, exist_ok=True)

    buf = dict(t=[], npos=[], epos=[], alt=[],
               roll=[], pitch=[], yaw=[],
               P=[], Q=[], R=[],
               vt=[], vx=[], vy=[], vz=[],
               aoa=[], aos=[],
               F_head=[], F_rf=[], F_lb=[], F_lf=[], F_rb=[],
               reward=[], target_alt=[], target_heading=[],
               target_npos=[], target_epos=[],
               wind_n=[], wind_e=[], wind_d=[], wind_mag=[],
               xy_err=[], alt_err=[], pos_err=[])

    def push():
        npos, epos, altitude = env.model.get_position()
        roll, pitch, yaw = env.model.get_posture()
        P, Q, R = env.model.get_angular_velocity()
        vx, vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        Fh, Frf, Flb, Flf, Frb = env.model.get_F()
        buf['npos'].append(_t2n(npos).mean())
        buf['epos'].append(_t2n(epos).mean())
        buf['alt'].append(_t2n(altitude).mean())
        buf['roll'].append(_t2n(roll).mean())
        buf['pitch'].append(_t2n(pitch).mean())
        buf['yaw'].append(_t2n(yaw).mean())
        buf['P'].append(_t2n(P).mean())
        buf['Q'].append(_t2n(Q).mean())
        buf['R'].append(_t2n(R).mean())
        buf['vt'].append(_t2n(env.model.get_vt()).mean())
        buf['vx'].append(_t2n(vx).mean())
        buf['vy'].append(_t2n(vy).mean())
        buf['vz'].append(_t2n(vz).mean())
        buf['F_head'].append(_t2n(Fh).mean())
        buf['F_rf'].append(_t2n(Frf).mean())
        buf['F_lb'].append(_t2n(Flb).mean())
        buf['F_lf'].append(_t2n(Flf).mean())
        buf['F_rb'].append(_t2n(Frb).mean())
        buf['aoa'].append(np.degrees(_t2n(env.model.get_AOA()).mean()))
        buf['aos'].append(np.degrees(_t2n(env.model.get_AOS()).mean()))
        buf['target_alt'].append(_t2n(env.task.target_altitude).mean())
        buf['target_heading'].append(_t2n(env.task.target_heading).mean())
        buf['target_npos'].append(_t2n(env.task.target_npos).mean())
        buf['target_epos'].append(_t2n(env.task.target_epos).mean())
        if hasattr(env.model, 'get_wind_ned'):
            wn, we, wd = env.model.get_wind_ned()
        else:
            wn = torch.zeros_like(npos)
            we = torch.zeros_like(npos)
            wd = torch.zeros_like(npos)
        wind_mag = torch.sqrt(wn * wn + we * we + wd * wd)
        xy_err = torch.sqrt((npos - env.task.target_npos) ** 2
                            + (epos - env.task.target_epos) ** 2)
        alt_err = altitude - env.task.target_altitude
        pos_err = torch.sqrt(xy_err * xy_err + alt_err * alt_err)
        buf['wind_n'].append(_scalar(wn))
        buf['wind_e'].append(_scalar(we))
        buf['wind_d'].append(_scalar(wd))
        buf['wind_mag'].append(_scalar(wind_mag))
        buf['xy_err'].append(_scalar(xy_err))
        buf['alt_err'].append(_scalar(alt_err))
        buf['pos_err'].append(_scalar(pos_err))

    push()
    buf['t'].append(0.0)
    buf['reward'].append(0.0)

    start = time.time()
    counts = 0
    if not args.no_render:
        env.render(count=counts, filename=os.path.join(ROOT, 'renders', 'tracks', 'HoverTaskPID-'))
    bad_done_total = 0
    for step in range(1, total_steps + 1):
        action = pid.compute_action(env.model)
        if args.verbose_env:
            obs, reward, done, bad_done, exceed, info = env.step(
                action, render=(not args.no_render), count=counts)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, done, bad_done, exceed, info = env.step(
                    action, render=(not args.no_render), count=counts)
        buf['t'].append(step * env.model.dt)
        buf['reward'].append(_t2n(reward).mean())
        push()
        bad_done_total += int(_t2n(bad_done).sum())
        if step % 100 == 0:
            print(f"[{step:4d}] alt={buf['alt'][-1]:7.3f}  "
                  f"roll={np.degrees(buf['roll'][-1]):+6.2f}  "
                  f"pitch={np.degrees(buf['pitch'][-1]):+6.2f}  "
                  f"yaw={np.degrees(buf['yaw'][-1]):+6.2f}  "
                  f"pos_err={buf['pos_err'][-1]:6.3f}  "
                  f"wind={buf['wind_mag'][-1]:5.3f}  "
                  f"r={buf['reward'][-1]:+5.2f}")
        if torch.any(done | bad_done | exceed):
            print(f"[terminated] step={step} done={bool(done.item())} "
                  f"bad_done={bool(bad_done.item())} exceed={bool(exceed.item())}")
            break

    print(f"sim={total_steps * env.model.dt:.1f}s  wall={time.time()-start:.2f}s  "
          f"bad_done={bad_done_total}  total_reward={float(np.sum(buf['reward'])):.1f}")
    print(f"metrics: rms_3d_err={np.sqrt(np.mean(np.square(buf['pos_err']))):.4f} m  "
          f"final_3d_err={buf['pos_err'][-1]:.4f} m  "
          f"max_3d_err={np.max(buf['pos_err']):.4f} m  "
          f"rms_wind={np.sqrt(np.mean(np.square(buf['wind_mag']))):.4f} m/s  "
          f"max_wind={np.max(buf['wind_mag']):.4f} m/s")

    for k, v in buf.items():
        np.save(os.path.join(result_dir, f'hover_task_pid_{k}.npy'), np.asarray(v))

    t = np.asarray(buf['t'])
    fig, axes = plt.subplots(4, 3, figsize=(14, 13))
    axes[0, 0].plot(t, buf['alt'], 'b'); axes[0, 0].plot(t, buf['target_alt'], 'r--')
    axes[0, 0].set_title('Altitude (m)'); axes[0, 0].grid()
    axes[0, 1].plot(t, buf['npos'], label='npos'); axes[0, 1].plot(t, buf['epos'], label='epos')
    axes[0, 1].plot(t, buf['target_npos'], 'k--'); axes[0, 1].plot(t, buf['target_epos'], 'k--')
    axes[0, 1].set_title('Horizontal pos (m)'); axes[0, 1].legend(); axes[0, 1].grid()
    axes[0, 2].plot(t, np.degrees(buf['roll']),  label='roll')
    axes[0, 2].plot(t, np.degrees(buf['pitch']), label='pitch')
    axes[0, 2].plot(t, np.degrees(buf['yaw']),   label='yaw')
    axes[0, 2].plot(t, np.degrees(buf['target_heading']), 'k--', label='tgt_yaw')
    axes[0, 2].set_title('Attitude (deg)'); axes[0, 2].legend(); axes[0, 2].grid()
    axes[1, 0].plot(t, buf['vx'], label='vx'); axes[1, 0].plot(t, buf['vy'], label='vy')
    axes[1, 0].plot(t, buf['vz'], label='vz')
    axes[1, 0].set_title('Velocity (m/s)'); axes[1, 0].legend(); axes[1, 0].grid()
    axes[1, 1].plot(t, np.degrees(buf['P']), label='P')
    axes[1, 1].plot(t, np.degrees(buf['Q']), label='Q')
    axes[1, 1].plot(t, np.degrees(buf['R']), label='R')
    axes[1, 1].set_title('Body rates (deg/s)'); axes[1, 1].legend(); axes[1, 1].grid()
    axes[1, 2].plot(t, buf['reward'])
    axes[1, 2].set_title('Per-step reward'); axes[1, 2].grid()
    axes[2, 0].plot(t, buf['F_head'], label='F_head')
    axes[2, 0].plot(t, buf['F_rf'],   label='F_rf')
    axes[2, 0].plot(t, buf['F_lb'],   label='F_lb')
    axes[2, 0].plot(t, buf['F_lf'],   label='F_lf')
    axes[2, 0].plot(t, buf['F_rb'],   label='F_rb')
    axes[2, 0].set_title('Motor thrust (N)'); axes[2, 0].legend(); axes[2, 0].grid()
    axes[2, 1].plot(t, np.cumsum(buf['reward']))
    axes[2, 1].set_title('Cumulative reward'); axes[2, 1].grid()
    axes[2, 2].plot(t, buf['vt'])
    axes[2, 2].plot(t, buf['wind_mag'], label='wind')
    axes[2, 2].set_title('|V| / wind (m/s)'); axes[2, 2].legend(); axes[2, 2].grid()
    axes[3, 0].plot(t, buf['aoa'], label='α (AOA)')
    axes[3, 0].axhline(0, color='k', linewidth=0.5)
    axes[3, 0].set_title('AOA α (deg)'); axes[3, 0].legend(); axes[3, 0].grid()
    axes[3, 1].plot(t, buf['aos'], label='β (AOS)', color='orange')
    axes[3, 1].axhline(0, color='k', linewidth=0.5)
    axes[3, 1].set_title('AOS β (deg)'); axes[3, 1].legend(); axes[3, 1].grid()
    axes[3, 2].set_visible(False)
    fig.suptitle(f"HoverTask + PID controller ({args.model}, wind={args.wind}, clean_obs={args.clean_obs})")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'hover_task_pid.png'), dpi=120)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"plot -> {result_dir}/hover_task_pid.png")


if __name__ == '__main__':
    main()
