#!/bin/sh
# Render tracking curves for the selected rc_human checkpoint on curriculum
# levels 0-79. Defaults to episode_600 from the 2026-05-28 modes012534 run,
# which is the strongest saved checkpoint among those that reached level 70.

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/scripts/runs/2026-05-28_18-37-31_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_modes012534}"
EPISODE="${EPISODE:-600}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/episode_${EPISODE}/actor_latest.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/render_episode_${EPISODE}_levels0_79}"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"

echo "render ckpt: ${CKPT_PATH}"
echo "output dir:  ${OUTPUT_DIR}"
echo "mode order:  ${RC_HUMAN_MODE_ORDER}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/eval_rc_human_curriculum_levels.py" \
    --ckpt-path "${CKPT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --seed 600 \
    --episodes-per-level 1 \
    --min-level 0 \
    --max-level 79 \
    --max-steps 1500 \
    --mode-order "${RC_HUMAN_MODE_ORDER}" \
    --save-per-level-plots \
    "$@"
