#!/usr/bin/env python
"""Verify HYBRID_NEW matches HYBRID when wind is zero."""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT)

from envs.control_env import ControlEnv  # noqa: E402


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def _reset_env(model_name, args, device):
    env = ControlEnv(num_envs=args.num_envs,
                     config=args.config,
                     model=model_name,
                     random_seed=args.seed,
                     device=device)
    env.reset()
    return env


def _compare_airdata(env_old, env_new):
    tas_diff = _max_abs(env_old.model.get_TAS(), env_new.model.get_TAS())
    sincos_old = torch.stack(env_old.model.get_aero_sincos(), dim=1)
    sincos_new = torch.stack(env_new.model.get_aero_sincos(), dim=1)
    sincos_diff = _max_abs(sincos_old, sincos_new)

    # AOA/AOS are only meaningful away from hover/zero airspeed.  Use a
    # controlled forward-flight state to compare the formulas in no wind.
    env_old.model.s[:, 6] = 3.0
    env_old.model.s[:, 7] = 0.2
    env_old.model.s[:, 8] = 0.15
    env_new.model.s.copy_(env_old.model.s)
    aoa_diff = _max_abs(env_old.model.get_AOA(), env_new.model.get_AOA())
    aos_diff = _max_abs(env_old.model.get_AOS(), env_new.model.get_AOS())
    return tas_diff, sincos_diff, aoa_diff, aos_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='circle')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--tol', type=float, default=1e-6)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    env_old = _reset_env('HYBRID', args, device)
    env_new = _reset_env('HYBRID_NEW', args, device)
    env_new.model.set_wind_ned(0.0, 0.0, 0.0)

    reset_state_diff = _max_abs(env_old.model.s, env_new.model.s)
    reset_control_diff = _max_abs(env_old.model.u, env_new.model.u)

    x_old = torch.hstack((env_old.model.s, env_old.model.u))
    x_new = torch.hstack((env_new.model.s, env_new.model.u))
    xdot_diff = _max_abs(env_old.model.dynamics.nlplant(x_old),
                         env_new.model.dynamics.nlplant(x_new))

    generator = torch.Generator(device=device).manual_seed(args.seed + 999)
    max_state_diff = 0.0
    max_xdot_diff = xdot_diff
    for _ in range(args.steps):
        action = torch.rand((env_old.n, env_old.num_actions),
                            generator=generator,
                            device=device) * 2.0 - 1.0
        env_old.model.update(action)
        env_new.model.update(action)
        max_state_diff = max(max_state_diff, _max_abs(env_old.model.s, env_new.model.s))
        x_old = torch.hstack((env_old.model.s, env_old.model.u))
        x_new = torch.hstack((env_new.model.s, env_new.model.u))
        max_xdot_diff = max(max_xdot_diff, _max_abs(
            env_old.model.dynamics.nlplant(x_old),
            env_new.model.dynamics.nlplant(x_new),
        ))

    tas_diff, sincos_diff, aoa_diff, aos_diff = _compare_airdata(env_old, env_new)
    print(f"reset_state_diff={reset_state_diff:.3e}")
    print(f"reset_control_diff={reset_control_diff:.3e}")
    print(f"max_xdot_diff={max_xdot_diff:.3e}")
    print(f"max_state_diff_after_{args.steps}_steps={max_state_diff:.3e}")
    print(f"tas_diff_zero_wind={tas_diff:.3e}")
    print(f"aero_sincos_diff_zero_wind={sincos_diff:.3e}")
    print(f"aoa_diff_forward_flight_zero_wind={aoa_diff:.3e}")
    print(f"aos_diff_forward_flight_zero_wind={aos_diff:.3e}")

    ok = (
        reset_state_diff <= args.tol
        and reset_control_diff <= args.tol
        and max_xdot_diff <= args.tol
        and max_state_diff <= args.tol
        and tas_diff <= args.tol
        and sincos_diff <= args.tol
    )
    if not ok:
        raise SystemExit(1)
    print("HYBRID_NEW zero-wind check passed.")


if __name__ == '__main__':
    main()
