"""
Render the hover task using a trained DAgger student policy.

Usage:
    python renders/render_dagger_hover.py \
        --ckpt algorithms/dagger/checkpoints/20260423_204808/dagger_latest.pt
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from envs.control_env import ControlEnv                                    # noqa: E402
from envs.utils.utils import _t2n                                          # noqa: E402
from algorithms.dagger.policy import PPOActorStudent, default_actor_args   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger',
                                        'checkpoints', '20260429_230311',
                                        'dagger_iter1789.pt'))
    p.add_argument('--steps',  type=int,   default=1500)
    p.add_argument('--device', type=str,   default='cuda:0')
    p.add_argument('--seed',   type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- env -------------------------------------------------------
    env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                     random_seed=args.seed, device=device)

    # ---- load student policy ---------------------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    actor_args_dict = ckpt.get('args', {})
    actor_args = default_actor_args(**actor_args_dict)
    policy = PPOActorStudent(env.observation_space, env.action_space,
                             args=actor_args, device=device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    policy.reset_rollout_state(env.n)
    print(f"[policy] loaded from {args.ckpt}")
    print(f"         recurrent={actor_args.use_recurrent_policy}  "
          f"hidden='{actor_args.hidden_size}'")

    # ---- rollout ---------------------------------------------------
    result_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(result_dir, exist_ok=True)

    buf = dict(t=[], npos=[], epos=[], alt=[],
               roll=[], pitch=[], yaw=[],
               P=[], Q=[], R=[],
               vt=[], vx=[], vy=[], vz=[],
               aoa=[], aos=[],
               F_head=[], F_rf=[], F_lb=[], F_lf=[], F_rb=[],
               reward=[], target_alt=[], target_heading=[],
               target_npos=[], target_epos=[])

    def push():
        npos, epos, altitude = env.model.get_position()
        roll, pitch, yaw     = env.model.get_posture()
        P, Q, R              = env.model.get_angular_velocity()
        vx, vy               = env.model.get_ground_speed()
        vz                   = env.model.get_climb_rate()
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

    obs = env.reset()
    push()
    buf['t'].append(0.0)
    buf['reward'].append(0.0)

    start = time.time()
    bad_done_total = 0
    counts = 0
    env.render(count=counts, filename=os.path.join(ROOT, 'renders', 'tracks', 'HoverTaskDAgger-'))

    for step in range(1, args.steps + 1):
        with torch.no_grad():
            action = policy.act(obs)

        obs, reward, done, bad_done, exceed, info = env.step(
            action, render=True, count=counts)

        reset_mask = done | bad_done | exceed
        policy.set_done_mask(reset_mask)

        buf['t'].append(step * env.model.dt)
        buf['reward'].append(_t2n(reward).mean())
        push()
        bad_done_total += int(_t2n(bad_done).sum())

        if step % 100 == 0:
            print(f"[{step:4d}] alt={buf['alt'][-1]:7.3f}  "
                  f"roll={np.degrees(buf['roll'][-1]):+6.2f}°  "
                  f"pitch={np.degrees(buf['pitch'][-1]):+6.2f}°  "
                  f"yaw={np.degrees(buf['yaw'][-1]):+6.2f}°  "
                  f"r={buf['reward'][-1]:+5.2f}")

    total_reward = float(np.sum(buf['reward']))
    print(f"sim={args.steps * env.model.dt:.1f}s  wall={time.time()-start:.2f}s  "
          f"bad_done={bad_done_total}  total_reward={total_reward:.1f}")

    # ---- plot -------------------------------------------------------
    t = np.asarray(buf['t'])
    fig, axes = plt.subplots(4, 3, figsize=(14, 13))

    axes[0, 0].plot(t, buf['alt'],         'b',  label='alt')
    axes[0, 0].plot(t, buf['target_alt'],  'r--',label='target')
    axes[0, 0].set_title('Altitude (m)'); axes[0, 0].legend(); axes[0, 0].grid()

    axes[0, 1].plot(t, buf['npos'], label='npos')
    axes[0, 1].plot(t, buf['epos'], label='epos')
    axes[0, 1].plot(t, buf['target_npos'], 'k--')
    axes[0, 1].plot(t, buf['target_epos'], 'k--')
    axes[0, 1].set_title('Horizontal pos (m)'); axes[0, 1].legend(); axes[0, 1].grid()

    axes[0, 2].plot(t, np.degrees(buf['roll']),  label='roll')
    axes[0, 2].plot(t, np.degrees(buf['pitch']), label='pitch')
    axes[0, 2].plot(t, np.degrees(buf['yaw']),   label='yaw')
    axes[0, 2].plot(t, np.degrees(buf['target_heading']), 'k--', label='tgt_yaw')
    axes[0, 2].set_title('Attitude (deg)'); axes[0, 2].legend(); axes[0, 2].grid()

    axes[1, 0].plot(t, buf['vx'], label='vx')
    axes[1, 0].plot(t, buf['vy'], label='vy')
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
    axes[2, 2].set_title('|V| (m/s)'); axes[2, 2].grid()

    axes[3, 0].plot(t, buf['aoa'], label='α (AOA)')
    axes[3, 0].axhline(0, color='k', linewidth=0.5)
    axes[3, 0].set_title('AOA α (deg)'); axes[3, 0].legend(); axes[3, 0].grid()

    axes[3, 1].plot(t, buf['aos'], label='β (AOS)', color='orange')
    axes[3, 1].axhline(0, color='k', linewidth=0.5)
    axes[3, 1].set_title('AOS β (deg)'); axes[3, 1].legend(); axes[3, 1].grid()

    axes[3, 2].set_visible(False)

    fig.suptitle(f'HoverTask + DAgger student  (total_reward={total_reward:.0f})')
    fig.tight_layout()
    out = os.path.join(result_dir, 'hover_task_dagger.png')
    fig.savefig(out, dpi=120)
    plt.show()
    plt.close(fig)
    print(f"plot -> {out}")


if __name__ == '__main__':
    main()
