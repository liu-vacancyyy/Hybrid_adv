#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for rc_human vx/vy/vz command tracking, no yaw command.
#
# The synthetic human command generator never samples yaw/yaw-rate stick input.
# Yaw is not part of the curriculum success gate; a weak, broad yaw-hold reward
# plus angular-rate reward only discourages uncontrolled spinning.
# Vertical commands are altitude-aware: near the low/high altitude guard bands,
# the command generator clamps vz commands that would push farther out.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_vxvyvz_nowind_from_scratch.sh
#
# Useful overrides:
#   DEVICE=cuda:0 SEED=31 NUM_ENV_STEPS=1.5e9 \
#   RC_HUMAN_ALT_LOW=0.5 RC_HUMAN_ALT_HIGH=10 \
#   bash scripts/train_rc_human_vxvyvz_nowind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 3 4 5}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export RC_HUMAN_COMMAND_RATE_LIMIT_FRAC="${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC:-0.05}"
export RC_HUMAN_MIX_CURRENT="${RC_HUMAN_MIX_CURRENT:-0.35}"
export RC_HUMAN_MIX_EASY_REPLAY="${RC_HUMAN_MIX_EASY_REPLAY:-0.25}"
export RC_HUMAN_MIX_MEDIUM_REPLAY="${RC_HUMAN_MIX_MEDIUM_REPLAY:-0.25}"
export RC_HUMAN_MIX_RANDOM_REPLAY="${RC_HUMAN_MIX_RANDOM_REPLAY:-0.15}"
export RC_HUMAN_VX_MIN="${RC_HUMAN_VX_MIN:--1.0}"
export RC_HUMAN_VX_MAX="${RC_HUMAN_VX_MAX:-1.0}"
export RC_HUMAN_VX_LIMIT="${RC_HUMAN_VX_LIMIT:-1.0}"
export RC_HUMAN_VY_LIMIT="${RC_HUMAN_VY_LIMIT:-1.0}"
export RC_HUMAN_VZ_LIMIT="${RC_HUMAN_VZ_LIMIT:-1.0}"

export RC_HUMAN_YAW_COMMAND_ENABLE="${RC_HUMAN_YAW_COMMAND_ENABLE:-0}"
export RC_HUMAN_YAW_HOLD_ENABLE="${RC_HUMAN_YAW_HOLD_ENABLE:-1}"
export RC_HUMAN_YAW_TRACKING_ENABLE="${RC_HUMAN_YAW_TRACKING_ENABLE:-0}"
export RC_HUMAN_YAW_REWARD_ENABLE="${RC_HUMAN_YAW_REWARD_ENABLE:-1}"
export RC_HUMAN_W_YAW="${RC_HUMAN_W_YAW:-0.3}"
export RC_HUMAN_SIG_YAW="${RC_HUMAN_SIG_YAW:-0.35}"
export RC_HUMAN_TRACKING_BAD_DONE_ENABLE="${RC_HUMAN_TRACKING_BAD_DONE_ENABLE:-0}"
export RC_HUMAN_SUCCESS_IGNORE_TRANSIENT="${RC_HUMAN_SUCCESS_IGNORE_TRANSIENT:-1}"
export RC_HUMAN_ALTITUDE_AWARE_VZ_ENABLE="${RC_HUMAN_ALTITUDE_AWARE_VZ_ENABLE:-1}"
export RC_HUMAN_ALT_LOW="${RC_HUMAN_ALT_LOW:-0.5}"
export RC_HUMAN_ALT_HIGH="${RC_HUMAN_ALT_HIGH:-10.0}"
export RC_HUMAN_ALT_GUARD_ZONE="${RC_HUMAN_ALT_GUARD_ZONE:-0.0}"
export RC_HUMAN_REL_ERROR_FIXED_SCALE="${RC_HUMAN_REL_ERROR_FIXED_SCALE:-1.0}"

export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_nowind_runtime}"

RATE_TAG=$(printf '%s' "${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC}" | tr -c '0-9A-Za-z' '_')
MODE_TAG=$(printf '%s' "${RC_HUMAN_MODE_ORDER}" | tr -c '0-9A-Za-z' '_')
export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_vxvyvz_no_yaw_nowind_rate${RATE_TAG}_from_scratch_modes${MODE_TAG}}"

exec bash "${SCRIPT_DIR}/train_rc_human_rl_nowind_from_scratch.sh" "$@"
