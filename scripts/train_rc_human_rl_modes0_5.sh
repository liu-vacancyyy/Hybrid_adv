#!/bin/sh
# Train rc_human with curriculum over modes 0, 1, 2, 5, 3, and 4.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS=6
export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_rl_gru_wind_modes012534}"
exec sh "${SCRIPT_DIR}/train_rc_human_rl.sh" "$@"
