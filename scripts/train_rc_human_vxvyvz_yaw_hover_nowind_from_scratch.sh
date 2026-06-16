#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for the vx/vy/vz + yaw rc_human task, hover-only, no wind.
#
# Important mode mapping in envs/tasks/rc_human_task.py:
#   mode0 = hover/release, all raw sticks centered
#   mode1 = continuous random correction, not hover
#
# This script pins the curriculum to mode0 only:
#   - RC_HUMAN_MODE_ORDER=0
#   - RC_HUMAN_MAX_MODE_SLOTS=1
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_vxvyvz_yaw_hover_nowind_from_scratch.sh
#
# Useful overrides:
#   DEVICE=cuda:0 SEED=31 NUM_ENV_STEPS=8e8 \
#   RC_HUMAN_EXP_NAME=rc_human_vxvyvz_yaw_hover \
#   bash scripts/train_rc_human_vxvyvz_yaw_hover_nowind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-1}"
export RC_HUMAN_COMMAND_RATE_LIMIT_FRAC="${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC:-0.05}"
export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_yaw_hover_nowind_runtime}"

MODE_TAG=$(printf '%s' "${RC_HUMAN_MODE_ORDER}" | tr -c '0-9A-Za-z' '_')
export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_vxvyvz_yaw_hover_mode${MODE_TAG}_nowind_from_scratch}"

exec bash "${SCRIPT_DIR}/train_rc_human_rl_nowind_from_scratch.sh" "$@"
