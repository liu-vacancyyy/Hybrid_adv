#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Mode0-only training for rc_human vx/vy/vz hover stabilization.
#
# This run is intentionally focused on the best possible robust mode0 policy:
#   - mode order pinned to mode0 only
#   - no wind, no Dryden turbulence, no wind loading
#   - target vx/vy/vz/yaw_rate are exactly zero
#   - initial roll/pitch/body uvw are zero; altitude/yaw/omega still randomize
#   - mass/inertia and small sensor noise randomization stay on
#   - yaw is not commanded; yaw reward holds the reset-time heading target
#   - velocity reward adds Simulink-style relative tracking/precision/damping terms
#   - old fixed tracking bad_done is off; dynamic vx/vy/vz divergence bad_done is on
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_vxvyvz_mode0_hover_nowind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-1}"
export RC_HUMAN_CURRICULUM_ENABLE="${RC_HUMAN_CURRICULUM_ENABLE:-0}"
export RC_HUMAN_VX_FORWARD_CURRICULUM_ENABLE="${RC_HUMAN_VX_FORWARD_CURRICULUM_ENABLE:-0}"
export RC_HUMAN_COMMAND_RATE_LIMIT_FRAC="${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC:-0.02}"
export RC_HUMAN_VX_MIN="${RC_HUMAN_VX_MIN:--1.0}"
export RC_HUMAN_VX_MAX="${RC_HUMAN_VX_MAX:-1.0}"
export RC_HUMAN_VX_LIMIT="${RC_HUMAN_VX_LIMIT:-1.0}"
export RC_HUMAN_VY_LIMIT="${RC_HUMAN_VY_LIMIT:-1.0}"
export RC_HUMAN_VZ_LIMIT="${RC_HUMAN_VZ_LIMIT:-1.0}"

export RC_HUMAN_YAW_COMMAND_ENABLE="${RC_HUMAN_YAW_COMMAND_ENABLE:-0}"
export RC_HUMAN_YAW_HOLD_ENABLE="${RC_HUMAN_YAW_HOLD_ENABLE:-1}"
export RC_HUMAN_YAW_TRACKING_ENABLE="${RC_HUMAN_YAW_TRACKING_ENABLE:-1}"
export RC_HUMAN_YAW_REWARD_ENABLE="${RC_HUMAN_YAW_REWARD_ENABLE:-1}"
export RC_HUMAN_YAW_REWARD_MODE="${RC_HUMAN_YAW_REWARD_MODE:-target}"

export RC_HUMAN_TRACKING_BAD_DONE_ENABLE="${RC_HUMAN_TRACKING_BAD_DONE_ENABLE:-0}"
export RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_ENABLE="${RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_ENABLE:-1}"
export RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_MARGIN="${RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_MARGIN:-0.5}"
export RC_HUMAN_SUCCESS_IGNORE_TRANSIENT="${RC_HUMAN_SUCCESS_IGNORE_TRANSIENT:-1}"
export RC_HUMAN_ALTITUDE_AWARE_VZ_ENABLE="${RC_HUMAN_ALTITUDE_AWARE_VZ_ENABLE:-1}"
export RC_HUMAN_ALT_LOW="${RC_HUMAN_ALT_LOW:-0.5}"
export RC_HUMAN_ALT_HIGH="${RC_HUMAN_ALT_HIGH:-10.0}"
export RC_HUMAN_ALT_GUARD_ZONE="${RC_HUMAN_ALT_GUARD_ZONE:-0.0}"

export RC_HUMAN_W_VEL="${RC_HUMAN_W_VEL:-8.0}"
export RC_HUMAN_W_YAW="${RC_HUMAN_W_YAW:-3.0}"
export RC_HUMAN_W_ATTITUDE="${RC_HUMAN_W_ATTITUDE:-2.5}"
export RC_HUMAN_W_OMEGA="${RC_HUMAN_W_OMEGA:-1.5}"
export RC_HUMAN_W_YAW_RATE="${RC_HUMAN_W_YAW_RATE:-0.6}"
export RC_HUMAN_W_SMOOTH="${RC_HUMAN_W_SMOOTH:-0.2}"
export RC_HUMAN_W_VEL_PRECISION="${RC_HUMAN_W_VEL_PRECISION:-0.0}"
export RC_HUMAN_W_YAW_PRECISION="${RC_HUMAN_W_YAW_PRECISION:-0.4}"
export RC_HUMAN_REL_ERROR_FIXED_SCALE="${RC_HUMAN_REL_ERROR_FIXED_SCALE:-1.0}"
export RC_HUMAN_REL_ERROR_FLOOR="${RC_HUMAN_REL_ERROR_FLOOR:-0.20}"
export RC_HUMAN_W_REL_TRACKING="${RC_HUMAN_W_REL_TRACKING:-2.0}"
export RC_HUMAN_W_REL_PRECISION="${RC_HUMAN_W_REL_PRECISION:-1.0}"
export RC_HUMAN_REL_PRECISION_GAIN="${RC_HUMAN_REL_PRECISION_GAIN:-20.0}"
export RC_HUMAN_W_OVERSHOOT="${RC_HUMAN_W_OVERSHOOT:-0.8}"
export RC_HUMAN_OVERSHOOT_DEADBAND="${RC_HUMAN_OVERSHOOT_DEADBAND:-0.05}"
export RC_HUMAN_W_ADAPTIVE_DAMPING="${RC_HUMAN_W_ADAPTIVE_DAMPING:-0.12}"
export RC_HUMAN_ADAPTIVE_DAMPING_GAIN="${RC_HUMAN_ADAPTIVE_DAMPING_GAIN:-12.0}"
export RC_HUMAN_SIG_VX="${RC_HUMAN_SIG_VX:-0.20}"
export RC_HUMAN_SIG_VY="${RC_HUMAN_SIG_VY:-0.20}"
export RC_HUMAN_SIG_VZ="${RC_HUMAN_SIG_VZ:-0.18}"
export RC_HUMAN_SIG_YAW="${RC_HUMAN_SIG_YAW:-0.12}"
export RC_HUMAN_SIG_YAW_DELTA="${RC_HUMAN_SIG_YAW_DELTA:-0.015}"
export RC_HUMAN_SIG_YAW_RATE="${RC_HUMAN_SIG_YAW_RATE:-0.35}"
export RC_HUMAN_SIG_ATTITUDE="${RC_HUMAN_SIG_ATTITUDE:-0.08}"
export RC_HUMAN_SIG_OMEGA="${RC_HUMAN_SIG_OMEGA:-0.60}"
export RC_HUMAN_SIG_SMOOTH="${RC_HUMAN_SIG_SMOOTH:-0.35}"

export RC_HUMAN_SUCCESS_VEL_ERROR="${RC_HUMAN_SUCCESS_VEL_ERROR:-0.08}"
export RC_HUMAN_SUCCESS_YAW_ERROR="${RC_HUMAN_SUCCESS_YAW_ERROR:-0.05}"
export RC_HUMAN_SUCCESS_ATTITUDE_ERROR="${RC_HUMAN_SUCCESS_ATTITUDE_ERROR:-0.04}"

export RC_HUMAN_ENABLE_SENSOR_NOISE="${RC_HUMAN_ENABLE_SENSOR_NOISE:-1}"
export RC_HUMAN_SENSOR_POS_STD="${RC_HUMAN_SENSOR_POS_STD:-0.02}"
export RC_HUMAN_SENSOR_VEL_STD="${RC_HUMAN_SENSOR_VEL_STD:-0.02}"
export RC_HUMAN_SENSOR_ATT_STD="${RC_HUMAN_SENSOR_ATT_STD:-0.005}"
export RC_HUMAN_SENSOR_OMEGA_STD="${RC_HUMAN_SENSOR_OMEGA_STD:-0.0005}"

export RC_HUMAN_INIT_ROLL_RANGE="${RC_HUMAN_INIT_ROLL_RANGE:-0.0}"
export RC_HUMAN_INIT_PITCH_RANGE="${RC_HUMAN_INIT_PITCH_RANGE:-0.0}"
export RC_HUMAN_INIT_YAW_RANGE="${RC_HUMAN_INIT_YAW_RANGE:-3.141592653590}"
export RC_HUMAN_INIT_VEL_RANGE="${RC_HUMAN_INIT_VEL_RANGE:-0.0}"
export RC_HUMAN_INIT_OMEGA_RANGE="${RC_HUMAN_INIT_OMEGA_RANGE:-0.02}"
export RC_HUMAN_DR_MASS="${RC_HUMAN_DR_MASS:-0.05}"
export RC_HUMAN_DR_INERTIA="${RC_HUMAN_DR_INERTIA:-0.20}"
export RC_HUMAN_MIN_ALTITUDE="${RC_HUMAN_MIN_ALTITUDE:-1.5}"
export RC_HUMAN_MAX_ALTITUDE="${RC_HUMAN_MAX_ALTITUDE:-8.5}"

export NUM_ENV_STEPS="${NUM_ENV_STEPS:-8e8}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export ENTROPY_COEF="${ENTROPY_COEF:-5e-4}"

export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_mode0_hover_nowind_runtime}"
export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_vxvyvz_mode0_hover_scale1_tuned_nowind_from_scratch}"

exec bash "${SCRIPT_DIR}/train_rc_human_rl_nowind_from_scratch.sh" "$@"
