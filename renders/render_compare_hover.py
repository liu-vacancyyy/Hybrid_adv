"""
Compare DAgger student policy vs cascade PID controller on the hover task,
with both controllers operating from the SAME noisy sensor measurements as
configured in ``envs/configs/hover.yaml``.

The two controllers are evaluated in two independent rollouts that share the
same random seed (so spawn perturbation, body-mass randomisation and the
sensor-noise sequence are identical step-by-step).  Trajectories are then
overlaid in a single figure.

Usage
-----
    python renders/render_compare_hover.py \
        --ckpt algorithms/dagger/checkpoints/20260429_230311/dagger_iter1789.pt
"""
import os
import sys
import time
import math
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
from algorithms.pid.hover_pid import HoverPIDController                    # noqa: E402


# --------------------------------------------------------------------------- #
#  Noisy "view" of the model that emulates the sensor noise the policy sees   #
# --------------------------------------------------------------------------- #
class NoisyModelView:
    """Wrap a HybridModel and inject zero-mean Gaussian noise into the
    accessors used by ``HoverPIDController.compute_action``.

    The noise standard deviations are taken from ``env.task`` so the PID
    sees measurements drawn from exactly the same distribution as the
    DAgger student observation.

    Optionally apply a per-channel first-order IIR low-pass filter
    (``y[k] = y[k-1] + alpha * (x[k] - y[k-1])``) to mimic a small amount
    of pre-filtering before the PID sees the measurement.  Cut-off
    frequencies are deliberately conservative so the comparison stays
    apples-to-apples with the unfiltered student.
    """

    def __init__(self, model, task, dt=0.02, lpf=False,
                 fc_pos=1.0, fc_vel=3.0, fc_att=8.0, fc_omega=15.0):
        self.model = model
        self.pos_std   = float(task.sensor_pos_std)
        self.vel_std   = float(task.sensor_vel_std)
        self.att_std   = float(task.sensor_att_std)
        self.omega_std = float(task.sensor_omega_std)
        self.enabled   = bool(task.enable_sensor_noise)
        self.dt        = float(dt)
        self.lpf       = bool(lpf)
        # First-order IIR coefficients alpha = dt / (dt + 1/(2*pi*fc))
        def _alpha(fc):
            if fc <= 0.0:
                return 1.0
            rc = 1.0 / (2.0 * math.pi * float(fc))
            return self.dt / (self.dt + rc)
        self.a_pos   = _alpha(fc_pos)
        self.a_vel   = _alpha(fc_vel)
        self.a_att   = _alpha(fc_att)
        self.a_omega = _alpha(fc_omega)
        # State for IIR (lazy-init on first call)
        self._y = {}

    @staticmethod
    def _wrap_pi(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _noise(self, x, std):
        if not self.enabled or std == 0.0:
            return x
        return x + torch.randn_like(x) * std

    def _filter(self, key, x, alpha, is_angle=False):
        """First-order IIR; for angles the residual is wrapped to (-pi, pi]."""
        if not self.lpf or alpha >= 1.0:
            return x
        y_prev = self._y.get(key)
        if y_prev is None or y_prev.shape != x.shape:
            self._y[key] = x.clone()
            return x
        if is_angle:
            err = self._wrap_pi(x - y_prev)
            y = self._wrap_pi(y_prev + alpha * err)
        else:
            y = y_prev + alpha * (x - y_prev)
        self._y[key] = y
        return y

    # --- accessors used by the PID -----------------------------------------
    def get_position(self):
        n, e, a = self.model.get_position()
        n = self._noise(n, self.pos_std); e = self._noise(e, self.pos_std)
        a = self._noise(a, self.pos_std)
        n = self._filter('npos', n, self.a_pos)
        e = self._filter('epos', e, self.a_pos)
        a = self._filter('alt',  a, self.a_pos)
        return n, e, a

    def get_posture(self):
        r, p, y = self.model.get_posture()
        if self.enabled:
            r = self._wrap_pi(r + torch.randn_like(r) * self.att_std)
            p = self._wrap_pi(p + torch.randn_like(p) * self.att_std)
            y = self._wrap_pi(y + torch.randn_like(y) * self.att_std)
        r = self._filter('roll',  r, self.a_att, is_angle=True)
        p = self._filter('pitch', p, self.a_att, is_angle=True)
        y = self._filter('yaw',   y, self.a_att, is_angle=True)
        return r, p, y

    def get_ground_speed(self):
        vx, vy = self.model.get_ground_speed()
        vx = self._noise(vx, self.vel_std); vy = self._noise(vy, self.vel_std)
        vx = self._filter('vx', vx, self.a_vel)
        vy = self._filter('vy', vy, self.a_vel)
        return vx, vy

    def get_climb_rate(self):
        vz = self._noise(self.model.get_climb_rate(), self.vel_std)
        return self._filter('vz', vz, self.a_vel)

    def get_angular_velocity(self):
        P, Q, R = self.model.get_angular_velocity()
        P = self._noise(P, self.omega_std)
        Q = self._noise(Q, self.omega_std)
        R = self._noise(R, self.omega_std)
        P = self._filter('P', P, self.a_omega)
        Q = self._filter('Q', Q, self.a_omega)
        R = self._filter('R', R, self.a_omega)
        return P, Q, R


# --------------------------------------------------------------------------- #
#  Per-step state logger                                                      #
# --------------------------------------------------------------------------- #
def _new_buf():
    return dict(t=[], npos=[], epos=[], alt=[],
                roll=[], pitch=[], yaw=[],
                P=[], Q=[], R=[],
                vt=[], vx=[], vy=[], vz=[],
                F_head=[], F_rf=[], F_lb=[], F_lf=[], F_rb=[],
                reward=[], target_alt=[], target_heading=[],
                target_npos=[], target_epos=[])


def _push(buf, env):
    npos, epos, alt = env.model.get_position()
    roll, pitch, yaw = env.model.get_posture()
    P, Q, R          = env.model.get_angular_velocity()
    vx, vy           = env.model.get_ground_speed()
    vz               = env.model.get_climb_rate()
    Fh, Frf, Flb, Flf, Frb = env.model.get_F()
    buf['npos'].append(_t2n(npos).mean())
    buf['epos'].append(_t2n(epos).mean())
    buf['alt'].append(_t2n(alt).mean())
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
    buf['target_alt'].append(_t2n(env.task.target_altitude).mean())
    buf['target_heading'].append(_t2n(env.task.target_heading).mean())
    buf['target_npos'].append(_t2n(env.task.target_npos).mean())
    buf['target_epos'].append(_t2n(env.task.target_epos).mean())


# --------------------------------------------------------------------------- #
#  Rollout drivers                                                            #
# --------------------------------------------------------------------------- #
def rollout_dagger(args, device):
    env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                     random_seed=args.seed, device=device)

    ckpt = torch.load(args.ckpt, map_location=device)
    actor_args = default_actor_args(**ckpt.get('args', {}))
    policy = PPOActorStudent(env.observation_space, env.action_space,
                             args=actor_args, device=device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    policy.reset_rollout_state(env.n)
    print(f"[dagger] policy loaded from {args.ckpt}")

    buf = _new_buf()
    obs = env.reset()
    _push(buf, env)
    buf['t'].append(0.0)
    buf['reward'].append(0.0)

    t0 = time.time()
    bad = 0
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            action = policy.act(obs)
        obs, reward, done, bad_done, exceed, _ = env.step(
            action, render=True, count=0)
        policy.set_done_mask(done | bad_done | exceed)
        buf['t'].append(step * env.model.dt)
        buf['reward'].append(_t2n(reward).mean())
        _push(buf, env)
        bad += int(_t2n(bad_done).sum())
        if step % 200 == 0:
            print(f"[dagger][{step:4d}] alt={buf['alt'][-1]:7.3f}  "
                  f"r={buf['reward'][-1]:+5.2f}")
    print(f"[dagger] sim={args.steps*env.model.dt:.1f}s "
          f"wall={time.time()-t0:.2f}s  bad_done={bad}  "
          f"total_reward={float(np.sum(buf['reward'])):.1f}")
    return buf


def rollout_pid(args, device, use_noisy=True, lpf=False, label='pid'):
    # Re-create env with the SAME seed so initial state, body randomisation
    # and the upcoming noise stream match the dagger rollout step-by-step.
    env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                     random_seed=args.seed, device=device)

    # When use_noisy=False, the PID reads ground-truth from env.model directly
    # (upper-bound reference: "how good can this PID be without sensor noise").
    if use_noisy:
        noisy = NoisyModelView(env.model, env.task, dt=env.model.dt, lpf=lpf)
    else:
        noisy = env.model

    pid = HoverPIDController(n=env.n, device=device, dt=env.model.dt,
                             mass=1.779, gravity=9.807,
                             max_thrust_per_motor=env.model.max_F)
    pid.reset()

    buf = _new_buf()
    # First reset populates targets and applies spawn perturbations.
    _ = env.reset()
    pid.set_targets(target_altitude=env.task.target_altitude.clone(),
                    target_heading=env.task.target_heading.clone(),
                    target_npos=env.task.target_npos.clone(),
                    target_epos=env.task.target_epos.clone())
    _push(buf, env)
    buf['t'].append(0.0)
    buf['reward'].append(0.0)

    t0 = time.time()
    bad = 0
    for step in range(1, args.steps + 1):
        action = pid.compute_action(noisy)
        obs, reward, done, bad_done, exceed, _ = env.step(
            action, render=True, count=0)
        reset_mask = (done | bad_done | exceed)
        if torch.any(reset_mask):
            pid.reset(mask=reset_mask)
            pid.set_targets(target_altitude=env.task.target_altitude.clone(),
                            target_heading=env.task.target_heading.clone(),
                            target_npos=env.task.target_npos.clone(),
                            target_epos=env.task.target_epos.clone())
        buf['t'].append(step * env.model.dt)
        buf['reward'].append(_t2n(reward).mean())
        _push(buf, env)
        bad += int(_t2n(bad_done).sum())
        if step % 200 == 0:
            print(f"[{label:6s}][{step:4d}] alt={buf['alt'][-1]:7.3f}  "
                  f"r={buf['reward'][-1]:+5.2f}")
    print(f"[{label:6s}] sim={args.steps*env.model.dt:.1f}s "
          f"wall={time.time()-t0:.2f}s  bad_done={bad}  "
          f"total_reward={float(np.sum(buf['reward'])):.1f}")
    return buf


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def _arr(buf, key):
    return np.asarray(buf[key])


def plot_compare(d, p, out_path, buf_pid_clean=None, buf_pid_lpf=None):
    t_d = _arr(d, 't')
    t_p = _arr(p, 't')
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))

    cD, cP, cT, cC, cL = 'tab:blue', 'tab:orange', 'k', 'tab:green', 'tab:red'
    pc = buf_pid_clean
    pl = buf_pid_lpf
    t_c = _arr(pc, 't') if pc is not None else None
    t_l = _arr(pl, 't') if pl is not None else None

    def _add_extra(ax, key, deg=False):
        if pl is not None:
            y = _arr(pl, key)
            if deg:
                y = np.degrees(y)
            ax.plot(t_l, y, cL, label='PID+LPF')
        if pc is not None:
            y = _arr(pc, key)
            if deg:
                y = np.degrees(y)
            ax.plot(t_c, y, cC, label='PID-clean')

    ax = axes[0, 0]
    ax.plot(t_d, _arr(d, 'alt'), cD, label='DAgger')
    ax.plot(t_p, _arr(p, 'alt'), cP, label='PID')
    _add_extra(ax, 'alt')
    ax.plot(t_d, _arr(d, 'target_alt'), cT, ls='--', lw=1, label='target')
    ax.set_title('Altitude (m)'); ax.legend(); ax.grid()

    ax = axes[0, 1]
    ax.plot(t_d, _arr(d, 'npos'), cD, label='DAgger npos')
    ax.plot(t_p, _arr(p, 'npos'), cP, label='PID npos')
    ax.plot(t_d, _arr(d, 'epos'), cD, ls=':', label='DAgger epos')
    ax.plot(t_p, _arr(p, 'epos'), cP, ls=':', label='PID epos')
    ax.plot(t_d, _arr(d, 'target_npos'), cT, ls='--', lw=1)
    ax.plot(t_d, _arr(d, 'target_epos'), cT, ls='--', lw=1)
    ax.set_title('Horizontal position N/E (m)'); ax.legend(fontsize=8); ax.grid()

    ax = axes[0, 2]
    ax.plot(t_d, np.degrees(_arr(d, 'yaw')), cD, label='DAgger yaw')
    ax.plot(t_p, np.degrees(_arr(p, 'yaw')), cP, label='PID yaw')
    ax.plot(t_d, np.degrees(_arr(d, 'target_heading')), cT, ls='--', lw=1,
            label='target')
    ax.set_title('Yaw (deg)'); ax.legend(); ax.grid()

    ax = axes[1, 0]
    ax.plot(t_d, np.degrees(_arr(d, 'roll')), cD, label='DAgger')
    ax.plot(t_p, np.degrees(_arr(p, 'roll')), cP, label='PID')
    ax.set_title('Roll (deg)'); ax.legend(); ax.grid()

    ax = axes[1, 1]
    ax.plot(t_d, np.degrees(_arr(d, 'pitch')), cD, label='DAgger')
    ax.plot(t_p, np.degrees(_arr(p, 'pitch')), cP, label='PID')
    ax.set_title('Pitch (deg)'); ax.legend(); ax.grid()

    ax = axes[1, 2]
    ax.plot(t_d, _arr(d, 'vz'), cD, label='DAgger')
    ax.plot(t_p, _arr(p, 'vz'), cP, label='PID')
    ax.set_title('Climb rate vz (m/s)'); ax.legend(); ax.grid()

    ax = axes[2, 0]
    ax.plot(t_d, _arr(d, 'vx'), cD, label='DAgger')
    ax.plot(t_p, _arr(p, 'vx'), cP, label='PID')
    ax.set_title('Ground vx (m/s)'); ax.legend(); ax.grid()

    ax = axes[2, 1]
    ax.plot(t_d, _arr(d, 'vy'), cD, label='DAgger')
    ax.plot(t_p, _arr(p, 'vy'), cP, label='PID')
    ax.set_title('Ground vy (m/s)'); ax.legend(); ax.grid()

    ax = axes[2, 2]
    ax.plot(t_d, _arr(d, 'reward'), cD, label='DAgger')
    ax.plot(t_p, _arr(p, 'reward'), cP, label='PID')
    ax.set_title('Per-step reward'); ax.legend(); ax.grid()

    ax = axes[3, 0]
    ax.plot(t_d, np.cumsum(_arr(d, 'reward')), cD, label='DAgger')
    ax.plot(t_p, np.cumsum(_arr(p, 'reward')), cP, label='PID')
    ax.set_title('Cumulative reward'); ax.legend(); ax.grid()

    # 3-D position error norm against target
    err_d = np.sqrt((_arr(d, 'npos') - _arr(d, 'target_npos'))**2
                    + (_arr(d, 'epos') - _arr(d, 'target_epos'))**2
                    + (_arr(d, 'alt')  - _arr(d, 'target_alt'))**2)
    err_p = np.sqrt((_arr(p, 'npos') - _arr(p, 'target_npos'))**2
                    + (_arr(p, 'epos') - _arr(p, 'target_epos'))**2
                    + (_arr(p, 'alt')  - _arr(p, 'target_alt'))**2)
    ax = axes[3, 1]
    ax.plot(t_d, err_d, cD, label=f'DAgger (RMS={np.sqrt(np.mean(err_d**2)):.2f})')
    ax.plot(t_p, err_p, cP, label=f'PID    (RMS={np.sqrt(np.mean(err_p**2)):.2f})')
    if pl is not None:
        err_l = np.sqrt((_arr(pl, 'npos') - _arr(pl, 'target_npos'))**2
                        + (_arr(pl, 'epos') - _arr(pl, 'target_epos'))**2
                        + (_arr(pl, 'alt')  - _arr(pl, 'target_alt'))**2)
        ax.plot(t_l, err_l, cL,
                label=f'PID+LPF  (RMS={np.sqrt(np.mean(err_l**2)):.2f})')
    if pc is not None:
        err_c = np.sqrt((_arr(pc, 'npos') - _arr(pc, 'target_npos'))**2
                        + (_arr(pc, 'epos') - _arr(pc, 'target_epos'))**2
                        + (_arr(pc, 'alt')  - _arr(pc, 'target_alt'))**2)
        ax.plot(t_c, err_c, cC,
                label=f'PID-clean (RMS={np.sqrt(np.mean(err_c**2)):.2f})')
    ax.set_title('3-D position error (m)'); ax.legend(); ax.grid()

    # Total motor thrust
    sumF_d = (_arr(d, 'F_head') + _arr(d, 'F_rf') + _arr(d, 'F_lb')
              + _arr(d, 'F_lf') + _arr(d, 'F_rb'))
    sumF_p = (_arr(p, 'F_head') + _arr(p, 'F_rf') + _arr(p, 'F_lb')
              + _arr(p, 'F_lf') + _arr(p, 'F_rb'))
    ax = axes[3, 2]
    ax.plot(t_d, sumF_d, cD, label='DAgger')
    ax.plot(t_p, sumF_p, cP, label='PID')
    ax.axhline(1.779 * 9.807, color=cT, ls='--', lw=1, label='m·g')
    ax.set_title('Total motor thrust (N)'); ax.legend(); ax.grid()

    fig.suptitle('Hover task — DAgger vs PID under identical sensor noise',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] saved to {out_path}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger',
                                        'checkpoints', '20260429_230311',
                                        'dagger_iter1789.pt'))
    p.add_argument('--steps',  type=int, default=1500)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed',   type=int, default=12345,
                   help='Evaluation seed.  NOTE: training defaults to --seed 0; '
                        'use a different value here to avoid evaluating on the '
                        'exact RNG state seen during training.')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--pid-clean', action='store_true',
                   help='also run a third rollout: PID with ground-truth '
                        'feedback (upper bound for PID)')
    p.add_argument('--pid-lpf', action='store_true',
                   help='also run a rollout with PID + first-order LPF on the '
                        'noisy measurements')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 60); print('Rollout 1/2: DAgger student'); print('=' * 60)
    buf_dagger = rollout_dagger(args, device)

    print('=' * 60); print('Rollout 2: PID controller (noisy, no filter)'); print('=' * 60)
    buf_pid = rollout_pid(args, device, use_noisy=True, lpf=False, label='pid')

    buf_pid_lpf = None
    if args.pid_lpf:
        print('=' * 60); print('Rollout 3: PID controller (noisy + LPF)'); print('=' * 60)
        buf_pid_lpf = rollout_pid(args, device, use_noisy=True, lpf=True,
                                  label='pid+lp')

    buf_pid_clean = None
    if args.pid_clean:
        print('=' * 60); print('Rollout 4: PID controller (clean GT)'); print('=' * 60)
        buf_pid_clean = rollout_pid(args, device, use_noisy=False,
                                    label='pid_gt')

    result_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(result_dir, exist_ok=True)
    out_png = os.path.join(result_dir, 'hover_compare_dagger_vs_pid.png')
    plot_compare(buf_dagger, buf_pid, out_png,
                 buf_pid_clean=buf_pid_clean, buf_pid_lpf=buf_pid_lpf)

    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
