#!/bin/sh
# Train rc_human with curriculum limited to the first 4 mode slots.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export RC_HUMAN_MAX_MODE_SLOTS=4
export RC_HUMAN_EXP_NAME="${RC_HUMAN_EXP_NAME:-rc_human_rl_gru_wind_first4}"
exec sh "${SCRIPT_DIR}/train_rc_human_rl.sh" "$@"
