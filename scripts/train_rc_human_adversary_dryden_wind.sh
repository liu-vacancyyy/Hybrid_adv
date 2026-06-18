#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Dryden wind-only adversary training for rc_human / HYBRID_NEW.
#
# The victim policy is fixed.  rc_human's random PX4 stick command generator is
# kept, observation attacks are disabled, and the adversary only drives Dryden
# N/E/D velocity plus body p/q/r gust targets.  The model receives Dryden-shaped
# wind outputs and converts them into body force / moment disturbances.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_adversary_dryden_wind.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
VICTIM_CKPT="${VICTIM_CKPT:-${REPO_ROOT}/scripts/runs/2026-06-03_15-49-07_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_modes012534_to_ep650_20260603_154906/episode_650/actor_latest.ckpt}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP="${RC_HUMAN_ADV_EXP_NAME:-rc_human_adv_dryden_wind_only_from_ep650_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/scripts/runs/${RUN_ID}_Control_rc_human_HYBRID_NEW_ppo_${EXP}}"

N_ROLLOUT_THREADS="${ADV_N_ROLLOUT_THREADS:-1024}"
BUFFER_SIZE="${ADV_BUFFER_SIZE:-1500}"
MAX_ITERATIONS="${ADV_MAX_ITERATIONS:-800}"
SAVE_INTERVAL="${ADV_SAVE_INTERVAL:-10}"
NUM_ENV_STEPS="${ADV_NUM_ENV_STEPS:-$((N_ROLLOUT_THREADS * BUFFER_SIZE * MAX_ITERATIONS))}"

if [ ! -f "${VICTIM_CKPT}" ]; then
    echo "Victim checkpoint not found: ${VICTIM_CKPT}" >&2
    exit 1
fi

echo "Dryden wind adversary training"
echo "  victim: ${VICTIM_CKPT}"
echo "  run_dir: ${RUN_DIR}"
echo "  rollout_threads: ${N_ROLLOUT_THREADS}"
echo "  buffer_size: ${BUFFER_SIZE}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  num_env_steps: ${NUM_ENV_STEPS}"

"${PYTHON_BIN}" scripts/adversarial/train_rc_human_adversary.py \
    --victim-ckpt "${VICTIM_CKPT}" \
    --scenario-name rc_human --model-name HYBRID_NEW --experiment-name "${EXP}" \
    --run-dir "${RUN_DIR}" \
    --seed 23 --device "${DEVICE}" --cuda \
    --n-rollout-threads "${N_ROLLOUT_THREADS}" \
    --buffer-size "${BUFFER_SIZE}" \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --log-interval 1 --save-interval "${SAVE_INTERVAL}" \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 \
    --ppo-epoch 8 --num-mini-batch 8 --clip-param 0.2 \
    --entropy-coef 2e-3 --max-grad-norm 1.0 \
    --hidden-size "128 128 128" --data-chunk-length 8 \
    --adv-use-random-command \
    --adv-command-frac 0.0 --adv-obs-frac 0.0 --adv-wind-frac 1.0 \
    --adv-command-alpha 1.0 --adv-obs-alpha 1.0 --adv-wind-alpha 1.0 \
    --adv-command-rate-limit-frac 0.0 \
    --adv-obs-rate-limit-frac 0.0 \
    --adv-wind-rate-limit-frac 0.20 \
    --adv-init-log-std -1.0 \
    --adv-lipschitz-coef 1e-6 \
    --adv-alive-penalty 0.01 \
    --adv-policy-reward-weight 0.30 \
    --adv-policy-reward-window 20 \
    --adv-w-vel-error 5.0 \
    --adv-w-axis-vel-error 3.0 \
    --adv-w-yaw-error 4.0 \
    --adv-axis-vel-margin 0.20 \
    --adv-yaw-margin-deg 4.0 \
    --adv-w-vel-bad-margin 10.0 \
    --adv-w-yaw-bad-margin 6.0 \
    --adv-w-attitude 8.0 \
    --adv-w-omega 1.5 \
    --adv-w-force-margin 0.4 \
    --adv-bad-done-bonus 80.0 \
    --adv-linf-penalty 0.0 \
    --adv-command-target-rms-min 0.0 \
    --adv-obs-target-rms-min 0.0 \
    --adv-wind-target-rms-min 0.35 \
    --adv-command-range-penalty 0.0 \
    --adv-obs-range-penalty 0.0 \
    --adv-wind-range-penalty 0.0 \
    --adv-saturation-penalty 0.0 \
    --adv-raw-excess-penalty 0.20 \
    --adv-obs-energy-window 50 \
    --adv-obs-energy-budget 50.0 \
    --adv-obs-energy-penalty 0.0 \
    --adv-attitude-safe-rad 0.16 \
    --adv-omega-safe-rad 0.8 \
    "$@"
