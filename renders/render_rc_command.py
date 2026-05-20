"""Standalone visualization of the human-like OU RC command generator.

This script re-implements the OU + piecewise-constant-mu command generator
that lives in ``envs/tasks/rc_task.py`` (with the same default parameters as
``envs/configs/rc.yaml``) so you can see what the simulated "pilot stick"
looks like *without* spinning up the full HybridModel / ControlEnv.

Usage:
    python renders/render_rc_command.py
    python renders/render_rc_command.py --duration 30 --num 4 --seed 0 \
        --out renders/result/rc_command.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def simulate(num_envs, T, dt, params, rng):
    """Vectorised numpy port of RCTask's OU command generator.

    Returns dict of arrays shaped (T, num_envs).
    """
    n = num_envs
    sqdt = np.sqrt(dt)

    # state
    target_vx = np.zeros(n)
    target_vz = np.zeros(n)
    target_yawr = np.zeros(n)
    target_heading = np.zeros(n)
    mu_vx = np.zeros(n)
    mu_vz = np.zeros(n)
    mu_yawr = np.zeros(n)
    dwell_left = np.zeros(n, dtype=np.int64)  # 0 -> resample on first step

    # logs
    log_vx = np.zeros((T, n))
    log_vz = np.zeros((T, n))
    log_yawr = np.zeros((T, n))
    log_heading = np.zeros((T, n))
    log_mu_vx = np.zeros((T, n))
    log_mu_vz = np.zeros((T, n))
    log_mu_yawr = np.zeros((T, n))

    for t in range(T):
        # 1) resample mu where dwell timer hit zero
        mask = dwell_left <= 0
        k = int(mask.sum())
        if k > 0:
            mu_vx[mask] = (rng.random(k) * 2 - 1) * params['mu_vx_range']
            mu_vz[mask] = (rng.random(k) * 2 - 1) * params['mu_vz_range']
            mu_yawr[mask] = (rng.random(k) * 2 - 1) * params['mu_yaw_rate_range']
            dwell_left[mask] = rng.integers(params['dwell_min'],
                                            params['dwell_max'] + 1, size=k)
        dwell_left = np.maximum(dwell_left - 1, 0)

        # 2) OU step
        eps_vx = rng.standard_normal(n)
        eps_vz = rng.standard_normal(n)
        eps_yr = rng.standard_normal(n)
        target_vx += params['theta_vel'] * (mu_vx - target_vx) * dt + params['sigma_vx'] * sqdt * eps_vx
        target_vz += params['theta_vel'] * (mu_vz - target_vz) * dt + params['sigma_vz'] * sqdt * eps_vz
        target_yawr += params['theta_yaw'] * (mu_yawr - target_yawr) * dt + params['sigma_yawr'] * sqdt * eps_yr

        # 3) saturate
        np.clip(target_vx, -params['max_vx'], params['max_vx'], out=target_vx)
        np.clip(target_vz, -params['max_vz'], params['max_vz'], out=target_vz)
        np.clip(target_yawr, -params['max_yaw_rate'], params['max_yaw_rate'], out=target_yawr)

        # 4) integrate yaw_rate -> heading
        target_heading = wrap_pi(target_heading + target_yawr * dt)

        log_vx[t] = target_vx
        log_vz[t] = target_vz
        log_yawr[t] = target_yawr
        log_heading[t] = target_heading
        log_mu_vx[t] = mu_vx
        log_mu_vz[t] = mu_vz
        log_mu_yawr[t] = mu_yawr

    return dict(vx=log_vx, vz=log_vz, yawr=log_yawr, heading=log_heading,
                mu_vx=log_mu_vx, mu_vz=log_mu_vz, mu_yawr=log_mu_yawr)


def load_yaml_params(yaml_path):
    """Read the rc_* keys from rc.yaml without requiring PyYAML."""
    defaults = dict(
        theta_vel=0.6, theta_yaw=0.8,
        sigma_vx=0.4, sigma_vz=0.3, sigma_yawr=0.4,
        max_vx=2.5, max_vz=2.0, max_yaw_rate=0.6,
        mu_vx_range=2.5, mu_vz_range=1.5, mu_yaw_rate_range=0.4,
        dwell_min=100, dwell_max=400,
    )
    key_map = {
        'rc_ou_theta_vel':      'theta_vel',
        'rc_ou_theta_yaw':      'theta_yaw',
        'rc_ou_sigma_vx':       'sigma_vx',
        'rc_ou_sigma_vz':       'sigma_vz',
        'rc_ou_sigma_yawr':     'sigma_yawr',
        'rc_max_vx':            'max_vx',
        'rc_max_vz':            'max_vz',
        'rc_max_yaw_rate':      'max_yaw_rate',
        'rc_mu_vx_range':       'mu_vx_range',
        'rc_mu_vz_range':       'mu_vz_range',
        'rc_mu_yaw_rate_range': 'mu_yaw_rate_range',
        'rc_dwell_min_steps':   'dwell_min',
        'rc_dwell_max_steps':   'dwell_max',
    }
    if not (yaml_path and os.path.isfile(yaml_path)):
        return defaults
    with open(yaml_path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip()
            v = v.strip()
            if k in key_map and v:
                try:
                    defaults[key_map[k]] = int(v) if 'dwell' in k else float(v)
                except ValueError:
                    pass
    return defaults


def plot(logs, dt, params, out_path):
    T, n = logs['vx'].shape
    t = np.arange(T) * dt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    for i in range(n):
        ax.plot(t, logs['vx'][:, i], lw=1.0, label=f'env {i}')
        ax.plot(t, logs['mu_vx'][:, i], lw=0.8, alpha=0.45, ls='--')
    ax.axhline(+params['max_vx'], color='k', lw=0.5, ls=':')
    ax.axhline(-params['max_vx'], color='k', lw=0.5, ls=':')
    ax.set_ylabel('target_vx [m/s]')
    ax.set_title('OU RC command (solid) vs piecewise-constant mu (dashed)')
    ax.grid(alpha=0.3)
    if n <= 6:
        ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    for i in range(n):
        ax.plot(t, logs['vz'][:, i], lw=1.0)
        ax.plot(t, logs['mu_vz'][:, i], lw=0.8, alpha=0.45, ls='--')
    ax.axhline(+params['max_vz'], color='k', lw=0.5, ls=':')
    ax.axhline(-params['max_vz'], color='k', lw=0.5, ls=':')
    ax.set_ylabel('target_vz [m/s]')
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i in range(n):
        ax.plot(t, logs['yawr'][:, i], lw=1.0)
        ax.plot(t, logs['mu_yawr'][:, i], lw=0.8, alpha=0.45, ls='--')
    ax.axhline(+params['max_yaw_rate'], color='k', lw=0.5, ls=':')
    ax.axhline(-params['max_yaw_rate'], color='k', lw=0.5, ls=':')
    ax.set_ylabel('target_yaw_rate [rad/s]')
    ax.grid(alpha=0.3)

    ax = axes[3]
    for i in range(n):
        ax.plot(t, np.rad2deg(logs['heading'][:, i]), lw=1.0)
    ax.axhline(+180, color='k', lw=0.5, ls=':')
    ax.axhline(-180, color='k', lw=0.5, ls=':')
    ax.set_ylabel('target_heading [deg]')
    ax.set_xlabel('time [s]')
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"theta_vel={params['theta_vel']}, theta_yaw={params['theta_yaw']}, "
        f"sigma=(vx={params['sigma_vx']}, vz={params['sigma_vz']}, "
        f"yawr={params['sigma_yawr']}), "
        f"dwell={params['dwell_min']}..{params['dwell_max']} steps",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f'saved figure -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=4, help='number of parallel pilots to draw')
    ap.add_argument('--duration', type=float, default=20.0, help='simulated seconds')
    ap.add_argument('--dt', type=float, default=0.02, help='timestep')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config', type=str,
                    default=os.path.join(os.path.dirname(__file__), '..', 'envs', 'configs', 'rc.yaml'),
                    help='rc.yaml to read parameters from')
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'result', 'rc_command.png'))
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    params = load_yaml_params(args.config)
    print('params:', params)

    rng = np.random.default_rng(args.seed)
    T = int(round(args.duration / args.dt))
    logs = simulate(args.num, T, args.dt, params, rng)

    # quick stats
    print(f'tgt_vx       range = [{logs["vx"].min():+.2f}, {logs["vx"].max():+.2f}]')
    print(f'tgt_vz       range = [{logs["vz"].min():+.2f}, {logs["vz"].max():+.2f}]')
    print(f'tgt_yaw_rate range = [{logs["yawr"].min():+.2f}, {logs["yawr"].max():+.2f}] rad/s')
    print(f'tgt_heading  range = [{np.rad2deg(logs["heading"].min()):+.1f}, '
          f'{np.rad2deg(logs["heading"].max()):+.1f}] deg')

    plot(logs, args.dt, params, args.out)
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()
