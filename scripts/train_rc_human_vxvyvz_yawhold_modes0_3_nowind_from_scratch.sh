#!/usr/bin/env bash
# Train vx/vy/vz yaw-hold curriculum on mode0-3.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 3}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-4}"
# 2048 rollout envs * 1500 buffer ~= 3.072e6 samples/update.
# 3.2e9 env steps gives about 1041 PPO updates, so training runs past 1000 generations.
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-3.2e9}"

exec bash "${SCRIPT_DIR}/train_rc_human_vxvyvz_yawhold_nowind_from_scratch.sh" "$@"
