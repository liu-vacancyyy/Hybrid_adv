"""
让飞行器先悬停稳定，然后给一个北向目标点让其加速到 ~1 m/s，
观察 AOA / AOS 在低速→过渡→巡速段的曲线是否平滑。

Usage:
    python renders/render_aoa_aos_transition.py
"""
import os, sys
os.environ.setdefault('DISPLAY', ':1')
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from envs.control_env import ControlEnv
from envs.utils.utils import _t2n
from algorithms.pid.hover_pid import HoverPIDController

HOVER_STEPS   = 200   # 先悬停稳定 (4 s)
TRANSIT_STEPS = 800   # 然后飞向目标 (16 s)
TOTAL_STEPS   = HOVER_STEPS + TRANSIT_STEPS
TARGET_NPOS   = 20.0  # 北向目标 20 m → 稳态速度约 1 m/s

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                     random_seed=0, device=device)
    pid = HoverPIDController(n=env.n, device=device, dt=env.model.dt,
                             mass=1.779, gravity=9.807,
                             max_thrust_per_motor=env.model.max_F)
    pid.reset()
    obs = env.reset()

    # Phase 1: hover in place
    pid.set_targets(
        target_altitude=env.task.target_altitude.clone(),
        target_heading=env.task.target_heading.clone(),
        target_npos=env.task.target_npos.clone(),
        target_epos=env.task.target_epos.clone(),
    )

    buf = dict(t=[], vt=[], vx=[], vy=[], vz=[],
               aoa=[], aos=[],
               roll=[], pitch=[], yaw=[],
               alt=[], npos=[])

    def push(step):
        npos, epos, alt = env.model.get_position()
        roll, pitch, yaw = env.model.get_posture()
        vx, vy = env.model.get_ground_speed()
        vz      = env.model.get_climb_rate()
        vt      = env.model.get_vt()
        aoa     = env.model.get_AOA()
        aos     = env.model.get_AOS()
        buf['t'].append(step * env.model.dt)
        buf['vt'].append(float(_t2n(vt).mean()))
        buf['vx'].append(float(_t2n(vx).mean()))
        buf['vy'].append(float(_t2n(vy).mean()))
        buf['vz'].append(float(_t2n(vz).mean()))
        buf['aoa'].append(float(np.degrees(_t2n(aoa).mean())))
        buf['aos'].append(float(np.degrees(_t2n(aos).mean())))
        buf['roll'].append(float(np.degrees(_t2n(roll).mean())))
        buf['pitch'].append(float(np.degrees(_t2n(pitch).mean())))
        buf['yaw'].append(float(np.degrees(_t2n(yaw).mean())))
        buf['alt'].append(float(_t2n(alt).mean()))
        buf['npos'].append(float(_t2n(npos).mean()))

    push(0)
    for step in range(1, TOTAL_STEPS + 1):
        # Phase 2: switch to moving target after HOVER_STEPS
        if step == HOVER_STEPS:
            pid.set_targets(
                target_altitude=env.task.target_altitude.clone(),
                target_heading=env.task.target_heading.clone(),
                target_npos=torch.full((env.n,), TARGET_NPOS, device=device),
                target_epos=env.task.target_epos.clone(),
            )
            print(f"[{step:4d}] Switching to npos target = {TARGET_NPOS} m")

        action = pid.compute_action(env.model)
        obs, rew, done, bad_done, exceed, _ = env.step(action)
        push(step)

        if step % 200 == 0:
            print(f"[{step:4d}] vt={buf['vt'][-1]:.2f} m/s  "
                  f"npos={buf['npos'][-1]:.2f} m  "
                  f"aoa={buf['aoa'][-1]:.2f}°  "
                  f"aos={buf['aos'][-1]:.2f}°  "
                  f"pitch={buf['pitch'][-1]:.2f}°")

    # ---- plot --------------------------------------------------------
    t = np.array(buf['t'])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    axes[0, 0].plot(t, buf['vt'], label='|V|')
    axes[0, 0].plot(t, buf['vx'], label='vx (north)')
    axes[0, 0].plot(t, buf['vy'], label='vy (east)')
    axes[0, 0].plot(t, buf['vz'], label='vz (climb)')
    axes[0, 0].axvline(HOVER_STEPS * env.model.dt, color='k', ls='--', lw=0.8, label='target switch')
    axes[0, 0].set_title('Speed (m/s)'); axes[0, 0].legend(); axes[0, 0].grid()

    axes[0, 1].plot(t, buf['npos'], label='npos')
    axes[0, 1].axhline(TARGET_NPOS, color='r', ls='--', label=f'target {TARGET_NPOS} m')
    axes[0, 1].set_title('North position (m)'); axes[0, 1].legend(); axes[0, 1].grid()

    axes[1, 0].plot(t, buf['aoa'], color='tab:blue')
    axes[1, 0].axvline(HOVER_STEPS * env.model.dt, color='k', ls='--', lw=0.8)
    axes[1, 0].set_title('AOA α (deg)'); axes[1, 0].grid()

    axes[1, 1].plot(t, buf['aos'], color='tab:orange')
    axes[1, 1].axvline(HOVER_STEPS * env.model.dt, color='k', ls='--', lw=0.8)
    axes[1, 1].set_title('AOS β (deg)'); axes[1, 1].grid()

    axes[2, 0].plot(t, buf['roll'],  label='roll')
    axes[2, 0].plot(t, buf['pitch'], label='pitch')
    axes[2, 0].axvline(HOVER_STEPS * env.model.dt, color='k', ls='--', lw=0.8)
    axes[2, 0].set_title('Roll / Pitch (deg)'); axes[2, 0].legend(); axes[2, 0].grid()

    axes[2, 1].plot(t, buf['alt'], label='altitude')
    axes[2, 1].axvline(HOVER_STEPS * env.model.dt, color='k', ls='--', lw=0.8)
    axes[2, 1].set_title('Altitude (m)'); axes[2, 1].legend(); axes[2, 1].grid()

    fig.suptitle('AOA / AOS smoothness: hover → 1 m/s transition')
    fig.tight_layout()
    result_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(result_dir, 'aoa_aos_transition.png')
    fig.savefig(out, dpi=120)
    plt.show()
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == '__main__':
    main()
