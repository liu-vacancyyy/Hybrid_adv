#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for rc_human from scratch after PX4 RC-link changes.
#
# This script intentionally starts a fresh policy:
#   - no --model-dir
#   - no --init-actor-ckpt
#
# Main differences from scripts/train_rc_human_rl.sh:
#   - uses the project conda Python by default;
#   - aligns buffer-size with rc_human.yaml max_steps=1500;
#   - exports the full PX4 VTOL-MC mode order explicitly.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_rl_from_scratch.sh
#
# Useful overrides:
#   DEVICE=cuda:0 SEED=31 NUM_ENV_STEPS=1.5e9 \
#   RC_HUMAN_EXP_NAME=rc_human_rl_px4_from_scratch \
#   bash scripts/train_rc_human_rl_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-7}"

ENV_NAME="Control"
SCENARIO_NAME="rc_human"
MODEL_NAME="HYBRID_NEW"
ALGO_NAME="ppo"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"

EXP="${RC_HUMAN_EXP_NAME:-rc_human_rl_px4_from_scratch_modes012534}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-1500}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-1.5e9}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

LR="${LR:-3e-4}"
PPO_EPOCH="${PPO_EPOCH:-10}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-8}"
ENTROPY_COEF="${ENTROPY_COEF:-2e-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2}"
DATA_CHUNK_LENGTH="${DATA_CHUNK_LENGTH:-8}"

for arg in "$@"; do
    case "${arg}" in
        --use-recurrent-policy
            echo "Do not pass --use-recurrent-policy: config.py defines it as store_false and it would disable GRU." >&2
            exit 2
            ;;
        --model-dir|--model-dir=*|--init-actor-ckpt|--init-actor-ckpt=*)
            echo "This script is for from-scratch training; do not pass ${arg}." >&2
            exit 2
            ;;
    esac
done

if [ -x "${PYTHON_BIN}" ]; then
    :
elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v "${PYTHON_BIN}")
else
    echo "Python executable not found: ${PYTHON_BIN}. Activate your environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

echo "rc_human from-scratch training"
echo "  experiment: ${EXP}"
echo "  env/model: ${ENV_NAME}/${SCENARIO_NAME}/${MODEL_NAME}"
echo "  seed/device: ${SEED}/${DEVICE}"
echo "  mode_order: ${RC_HUMAN_MODE_ORDER}"
echo "  max_mode_slots: ${RC_HUMAN_MAX_MODE_SLOTS}"
echo "  rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, num_env_steps=${NUM_ENV_STEPS}"
echo "  lr=${LR}, ppo_epoch=${PPO_EPOCH}, num_mini_batch=${NUM_MINI_BATCH}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train/train_F16sim.py" \
    --env-name "${ENV_NAME}" \
    --algorithm-name "${ALGO_NAME}" \
    --scenario-name "${SCENARIO_NAME}" \
    --model-name "${MODEL_NAME}" \
    --experiment-name "${EXP}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --cuda \
    --n-training-threads 1 \
    --n-rollout-threads "${N_ROLLOUT_THREADS}" \
    --buffer-size "${BUFFER_SIZE}" \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --lr "${LR}" \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ppo-epoch "${PPO_EPOCH}" \
    --num-mini-batch "${NUM_MINI_BATCH}" \
    --clip-param 0.2 \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --hidden-size "128 128" \
    --act-hidden-size "128 128" \
    --activation-id 1 \
    --gain 0.01 \
    --recurrent-hidden-size 128 \
    --recurrent-hidden-layers 1 \
    --data-chunk-length "${DATA_CHUNK_LENGTH}" \
    "$@"
