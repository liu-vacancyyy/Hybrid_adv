#!/usr/bin/env python
"""Sanity checks for rc_human curriculum sampling and command ranges."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from envs.control_env import ControlEnv  # noqa: E402


def main():
    os.environ.setdefault("RC_HUMAN_MODE_ORDER", "0 1 2 5 3 4")
    os.environ.setdefault("RC_HUMAN_MAX_MODE_SLOTS", "6")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ControlEnv(num_envs=4096, config="rc_human", model="HYBRID_NEW",
                     random_seed=123, device=device)
    task = env.task
    task.curriculum_enable = True
    mode_order = [int(x.item()) for x in task.mode_order[:task.active_mode_slots]]

    for current_level in range(task.max_curriculum_level + 1):
        idx = torch.arange(env.n, device=device)
        task.curriculum_level[:] = current_level
        sampled = task._sample_command_level(idx)
        if int(sampled.max().item()) > current_level:
            raise AssertionError(
                f"sampled future level: current={current_level}, "
                f"max_sampled={int(sampled.max().item())}"
            )

    hardest = []
    for slot, mode in enumerate(mode_order):
        level = slot * task.levels_per_mode + task.levels_per_mode - 1
        level_tensor = torch.tensor([level], device=device)
        mode_tensor = torch.tensor([mode], device=device)
        amp = task._curriculum_amplitude(level_tensor, mode_tensor)
        vx_limit = task._curriculum_vx_forward_limit(level_tensor, mode_tensor)
        hardest.append((mode, level, float((amp * vx_limit).item())))

    expected_forward_modes = [m for m in mode_order if m != 0]
    for mode, level, target in hardest:
        if mode in expected_forward_modes and abs(target - 5.0) > 1e-5:
            raise AssertionError(
                f"mode {mode} hardest level {level} reaches {target:.3f}, not 5.0"
            )

    env.close()
    print("rc_human curriculum sanity check passed")
    print("mode_order:", mode_order)
    print("hardest_forward_targets:", hardest)


if __name__ == "__main__":
    main()
