#!/usr/bin/env bash
# Train vx/vy/vz yaw-hold curriculum on mode0-1 only.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-2}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-1.2e9}"

exec bash "${SCRIPT_DIR}/train_rc_human_vxvyvz_yawhold_nowind_from_scratch.sh" "$@"
