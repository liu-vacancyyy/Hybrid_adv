"""
Render the hover task on the Hybrid platform under PID control.

Usage:
    cd NeuralPlane_stable_V2
    python renders/render_pid_hover.py

Outputs:
    - renders/tracks/HoverPID-0.txt.acmi   (Tacview file)
    - renders/result/hover_pid_*.npy       (state / control trajectories)
    - renders/result/hover_pid_state.png
    - renders/result/hover_pid_motor.png

The controller, mixing and gains follow LearningToFly's C++ PID cascade.
"""
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'envs'))
sys.path.append(os.path.join(ROOT, 'envs', 'models'))

from envs.models.hybrid_model import HybridModel                  # noqa: E402
from envs.utils.utils import parse_config, enu_to_geodetic, _t2n  # noqa: E402
from algorithms.pid.hover_pid import HoverPIDController           # noqa: E402


def write_acmi_header(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.0\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")


def write_acmi_frame(path, t, model, n):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"#{t:.2f}\n")
        npos_t, epos_t, alt_t = model.get_position()
        roll_t, pitch_t, yaw_t = model.get_posture()
        for i in range(n):
            lat, lon, alt = enu_to_geodetic(_t2n(epos_t[i]), _t2n(npos_t[i]),
                                            _t2n(alt_t[i]), 0, 0, 0)
            roll_d = _t2n(roll_t[i]) * 180 / np.pi
            pitch_d = _t2n(pitch_t[i]) * 180 / np.pi
            yaw_d = _t2n(yaw_t[i]) * 180 / np.pi
            f.write(f"{100 + i},T={lon}|{lat}|{alt}|{roll_d}|{pitch_d}|{yaw_d},"
                    f"Name=Hybrid,Color=Blue\n")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n = 1                                           # single agent
    dt = 0.02
    total_steps = 1500                              # 30 seconds

    # ---------- model ------------------------------------------------
    config = parse_config('rc')                     # reuse rc.yaml defaults
    config.num_agents = n
    config.dt = dt
    model = HybridModel(config, n, device, random_seed=0)

    # Initial state: hover at 10 m, zero attitude, zero rates, zero body velocity
    model.s[:] = 0.0
    model.s[:, 2] = 10.0                            # altitude (m)
    # Initial control = exact hover thrust on 4 lift rotors, 0 on head
    hover_F = 1.779 * 9.807 / 4.0
    model.u[:] = 0.0
    model.u[:, 1:] = hover_F
    model.recent_s[:] = model.s
    model.recent_u[:] = model.u

    # ---------- controller ------------------------------------------
    pid = HoverPIDController(n=n, device=device, dt=dt,
                             mass=1.779, gravity=9.807,
                             max_thrust_per_motor=model.max_F)
    pid.reset()
    # Target: hold current altitude, heading = 0
    pid.set_targets(target_altitude=model.s[:, 2].clone(),
                    target_heading=torch.zeros(n, device=device))

    # ---------- output paths ----------------------------------------
    tracks_dir = os.path.join(ROOT, 'renders', 'tracks')
    result_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(tracks_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    acmi_path = os.path.join(tracks_dir, 'HoverPID-0.txt.acmi')
    write_acmi_header(acmi_path)

    # ---------- buffers ---------------------------------------------
    buf = dict(t=[], npos=[], epos=[], alt=[],
               roll=[], pitch=[], yaw=[],
               P=[], Q=[], R=[],
               vt=[], vx=[], vy=[], vz=[],
               alpha=[], beta=[],
               F_head=[], F_rf=[], F_lb=[], F_lf=[], F_rb=[],
               throttle=[], target_climb=[], target_yaw_rate=[],
               target_alt=[], target_heading=[])

    def push_state(t):
        npos_t, epos_t, alt_t = model.get_position()
        roll_t, pitch_t, yaw_t = model.get_posture()
        P_t, Q_t, R_t = model.get_angular_velocity()
        vx_t, vy_t = model.get_ground_speed()
        vz_t = model.get_climb_rate()
        F_head_t, F_rf_t, F_lb_t, F_lf_t, F_rb_t = model.get_F()

        buf['t'].append(t)
        buf['npos'].append(_t2n(npos_t).mean())
        buf['epos'].append(_t2n(epos_t).mean())
        buf['alt'].append(_t2n(alt_t).mean())
        buf['roll'].append(_t2n(roll_t).mean())
        buf['pitch'].append(_t2n(pitch_t).mean())
        buf['yaw'].append(_t2n(yaw_t).mean())
        buf['P'].append(_t2n(P_t).mean())
        buf['Q'].append(_t2n(Q_t).mean())
        buf['R'].append(_t2n(R_t).mean())
        buf['vt'].append(_t2n(model.get_vt()).mean())
        buf['vx'].append(_t2n(vx_t).mean())
        buf['vy'].append(_t2n(vy_t).mean())
        buf['vz'].append(_t2n(vz_t).mean())
        buf['alpha'].append(_t2n(model.get_AOA()).mean())
        buf['beta'].append(_t2n(model.get_AOS()).mean())
        buf['F_head'].append(_t2n(F_head_t).mean())
        buf['F_rf'].append(_t2n(F_rf_t).mean())
        buf['F_lb'].append(_t2n(F_lb_t).mean())
        buf['F_lf'].append(_t2n(F_lf_t).mean())
        buf['F_rb'].append(_t2n(F_rb_t).mean())
        buf['target_alt'].append(_t2n(pid.target_altitude).mean())
        buf['target_heading'].append(_t2n(pid.target_heading).mean())
        if hasattr(pid, 'debug') and pid.debug:
            buf['throttle'].append(_t2n(pid.debug['throttle']).mean())
            buf['target_climb'].append(_t2n(pid.debug['target_climb']).mean())
            buf['target_yaw_rate'].append(_t2n(pid.debug['target_yaw_rate']).mean())
        else:
            buf['throttle'].append(0.0)
            buf['target_climb'].append(0.0)
            buf['target_yaw_rate'].append(0.0)

    # ---------- run --------------------------------------------------
    start = time.time()
    push_state(0.0)
    write_acmi_frame(acmi_path, 0.0, model, n)

    for step in range(1, total_steps + 1):
        action = pid.compute_action(model)
        model.update(action)
        t_now = step * dt
        push_state(t_now)
        if step % 2 == 0:                           # 25 Hz tacview logging
            write_acmi_frame(acmi_path, t_now, model, n)
        if step % 100 == 0:
            a = _t2n(model.s[0])
            print(f"[{step:4d}] alt={a[2]:7.3f} m  "
                  f"roll={np.degrees(a[3]):+6.2f}  "
                  f"pitch={np.degrees(a[4]):+6.2f}  "
                  f"yaw={np.degrees(a[5]):+6.2f}  "
                  f"vt={a[6]:5.2f}  vz={buf['vz'][-1]:+5.2f}")

    end = time.time()
    print(f"sim time = {total_steps * dt:.1f} s,  wallclock = {end - start:.2f} s")

    # ---------- save npys --------------------------------------------
    for k, v in buf.items():
        np.save(os.path.join(result_dir, f'hover_pid_{k}.npy'), np.asarray(v))

    # ---------- plots ------------------------------------------------
    t = np.asarray(buf['t'])

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(t, buf['alt'], 'b', label='altitude')
    ax.plot(t, buf['target_alt'], 'r--', label='target')
    ax.set_title('Altitude (m)'); ax.legend(); ax.grid()

    ax = axes[0, 1]
    ax.plot(t, buf['npos'], label='npos')
    ax.plot(t, buf['epos'], label='epos')
    ax.set_title('Horizontal position (m)'); ax.legend(); ax.grid()

    ax = axes[0, 2]
    ax.plot(t, np.degrees(buf['roll']),  label='roll')
    ax.plot(t, np.degrees(buf['pitch']), label='pitch')
    ax.plot(t, np.degrees(buf['yaw']),   label='yaw')
    ax.plot(t, np.degrees(buf['target_heading']), 'k--', label='target_yaw')
    ax.set_title('Attitude (deg)'); ax.legend(); ax.grid()

    ax = axes[1, 0]
    ax.plot(t, buf['vx'], label='vx')
    ax.plot(t, buf['vy'], label='vy')
    ax.plot(t, buf['vz'], label='vz (climb)')
    ax.plot(t, buf['target_climb'], 'k--', label='target climb')
    ax.set_title('Velocity (m/s)'); ax.legend(); ax.grid()

    ax = axes[1, 1]
    ax.plot(t, np.degrees(buf['P']), label='P')
    ax.plot(t, np.degrees(buf['Q']), label='Q')
    ax.plot(t, np.degrees(buf['R']), label='R')
    ax.set_title('Body rates (deg/s)'); ax.legend(); ax.grid()

    ax = axes[1, 2]
    ax.plot(t, np.degrees(buf['alpha']), label='alpha')
    ax.plot(t, np.degrees(buf['beta']),  label='beta')
    ax.set_title('Aero angles (deg)'); ax.legend(); ax.grid()

    ax = axes[2, 0]
    ax.plot(t, buf['vt'], label='vt')
    ax.set_title('|V| (m/s)'); ax.grid()

    ax = axes[2, 1]
    ax.plot(t, buf['throttle'])
    ax.set_title('Throttle command [0,1]'); ax.grid()

    ax = axes[2, 2]
    ax.plot(t, np.degrees(buf['target_yaw_rate']))
    ax.set_title('Target yaw rate (deg/s)'); ax.grid()

    fig.suptitle('Hover PID - State')
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'hover_pid_state.png'), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, buf['F_head'], label='F_head (pusher)')
    ax.plot(t, buf['F_rf'],   label='F_rf')
    ax.plot(t, buf['F_lb'],   label='F_lb')
    ax.plot(t, buf['F_lf'],   label='F_lf')
    ax.plot(t, buf['F_rb'],   label='F_rb')
    ax.axhline(1.779 * 9.807 / 4.0, color='k', linestyle='--',
               label='hover per-motor (mg/4)')
    ax.set_title('Motor thrusts (N)'); ax.legend(); ax.grid()
    ax.set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'hover_pid_motor.png'), dpi=120)
    plt.close(fig)

    print(f"[done] plots saved to {result_dir}/hover_pid_state.png, hover_pid_motor.png")
    print(f"[done] tacview acmi saved to {acmi_path}")


if __name__ == '__main__':
    main()
