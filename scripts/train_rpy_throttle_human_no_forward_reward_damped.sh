#!/usr/bin/env bash
# RPY + throttle no-forward training with stronger damping and anti-overshoot weights.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RPY_THROTTLE_REWARD_VARIANT="damped"
exec bash "${SCRIPT_DIR}/train_rpy_throttle_human_no_forward_rl_nowind_from_scratch.sh" "$@"
