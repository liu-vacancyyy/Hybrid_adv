#!/bin/sh
# Train a PPO adversary against the current best rc_human policy on modes 0-1.
# Command attack is disabled; rc_human's randomized command generator is used.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1}"
export RC_HUMAN_MAX_MODE_SLOTS=2
export VICTIM_CKPT="${VICTIM_CKPT:-${REPO_ROOT}/scripts/runs/2026-05-20_23-15-17_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_first2/episode_880/actor_latest.ckpt}"
export RC_HUMAN_ADV_EXP_NAME="${RC_HUMAN_ADV_EXP_NAME:-rc_human_adv_modes0_1_random_command_ep880}"
exec sh "${SCRIPT_DIR}/train_rc_human_adversary.sh" --adv-use-random-command "$@"
