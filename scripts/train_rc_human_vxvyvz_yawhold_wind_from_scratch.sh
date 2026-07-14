#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for rc_human vx/vy/vz tracking with yaw-hold. Wind
# disturbance is disabled by default; set RC_HUMAN_RANDOM_WIND_ENABLE=1 only
# when intentionally training with randomized Dryden wind.
#
# This preset keeps the current vx/vy/vz + yaw-hold target design unchanged:
# yaw command is disabled, target_vx/target_vy are current-heading local
# velocity commands, and target_heading is used for yaw-hold tracking.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_vxvyvz_yawhold_wind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_RANDOM_WIND_ENABLE="${RC_HUMAN_RANDOM_WIND_ENABLE:-0}"
export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 4 3}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export RC_HUMAN_YAW_COMMAND_ENABLE="${RC_HUMAN_YAW_COMMAND_ENABLE:-0}"
export RC_HUMAN_YAW_HOLD_ENABLE="${RC_HUMAN_YAW_HOLD_ENABLE:-1}"
export RC_HUMAN_YAW_TRACKING_ENABLE="${RC_HUMAN_YAW_TRACKING_ENABLE:-1}"

RATE_TAG=$(printf '%s' "${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC:-0.05}" | tr -c '0-9A-Za-z' '_')
MODE_TAG=$(printf '%s' "${RC_HUMAN_MODE_ORDER}" | tr -c '0-9A-Za-z' '_')

case "${RC_HUMAN_RANDOM_WIND_ENABLE}" in
    1|true|TRUE|yes|YES|on|ON)
        export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_yawhold_currheading_modes${MODE_TAG}_wind_runtime}"
        export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_vxvyvz_yawhold_currheading_rate${RATE_TAG}_from_scratch_modes${MODE_TAG}_random_wind}"
        ;;
    *)
        export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_yawhold_currheading_modes${MODE_TAG}_nowind_runtime}"
        export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_vxvyvz_yawhold_currheading_rate${RATE_TAG}_from_scratch_modes${MODE_TAG}}"
        ;;
esac

exec bash "${SCRIPT_DIR}/train_rc_human_vxvyvz_yawhold_nowind_from_scratch.sh" "$@"
