#!/usr/bin/env bash
# RPY + throttle reach no-forward training with damped / low-overshoot reward.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export SCENARIO_NAME="rpy_throttle_reach_no_forward_nowind_damped"
export RPY_THROTTLE_REACH_EXP_NAME="${RPY_THROTTLE_REACH_EXP_NAME:-rpy_throttle_reach_no_forward_nowind_damped_from_scratch_modes0123456789}"

exec bash "${SCRIPT_DIR}/train_rpy_throttle_reach_no_forward_nowind_from_scratch.sh" "$@"
