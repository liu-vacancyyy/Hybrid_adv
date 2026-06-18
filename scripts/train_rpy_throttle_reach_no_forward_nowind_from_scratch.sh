#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for the RPY + throttle one-shot reach task, no wind, with
# the forward/head motor physically disabled.
#
# Task target:
#   one episode: initial attitude -> commanded roll/pitch/yaw/throttle target
#
# Curriculum:
#   10 modes x 10 levels.  Modes 0-4 start from hover; modes 5-9 start from
#   the online pose pool and reuse the difficulty of mode(x-5).
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rpy_throttle_reach_no_forward_nowind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-7}"

ENV_NAME="Control"
SCENARIO_NAME="${SCENARIO_NAME:-rpy_throttle_reach_no_forward_nowind}"
MODEL_NAME="HYBRID_NEW_NO_FORWARD"
ALGO_NAME="ppo"

export RPY_THROTTLE_REACH_MAX_MODE_SLOTS="${RPY_THROTTLE_REACH_MAX_MODE_SLOTS:-10}"

SCENARIO_TAG=$(printf '%s' "${SCENARIO_NAME}" | tr -c '0-9A-Za-z' '_')
EXP="${RPY_THROTTLE_REACH_EXP_NAME:-${SCENARIO_TAG}_from_scratch_modes0123456789}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-2048}"
BUFFER_SIZE="${BUFFER_SIZE:-1000}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-2.5e9}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

LR="${LR:-1e-4}"
PPO_EPOCH="${PPO_EPOCH:-5}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-16}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-0.25}"
USE_CLIPPED_VALUE_LOSS="${USE_CLIPPED_VALUE_LOSS:-1}"
DATA_CHUNK_LENGTH="${DATA_CHUNK_LENGTH:-8}"
TARGET_KL="${TARGET_KL:-0.02}"
MAX_LOG_RATIO="${MAX_LOG_RATIO:-10.0}"
USE_SAFETY_AUX="${USE_SAFETY_AUX:-1}"
SAFETY_AUX_HORIZON="${SAFETY_AUX_HORIZON:-25}"
SAFETY_AUX_LOSS_COEF="${SAFETY_AUX_LOSS_COEF:-0.10}"
SAFETY_AUX_POS_WEIGHT="${SAFETY_AUX_POS_WEIGHT:-5.0}"

for arg in "$@"; do
    case "${arg}" in
        --use-recurrent-policy)
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

CLIPPED_VALUE_ARGS=()
case "${USE_CLIPPED_VALUE_LOSS}" in
    1|true|TRUE|yes|YES|on|ON)
        CLIPPED_VALUE_ARGS=(--use-clipped-value-loss)
        ;;
    0|false|FALSE|no|NO|off|OFF)
        CLIPPED_VALUE_ARGS=()
        ;;
    *)
        echo "USE_CLIPPED_VALUE_LOSS must be 0/1 or true/false, got: ${USE_CLIPPED_VALUE_LOSS}" >&2
        exit 2
        ;;
esac

SAFETY_AUX_ARGS=()
case "${USE_SAFETY_AUX}" in
    1|true|TRUE|yes|YES|on|ON)
        SAFETY_AUX_ARGS=(
            --use-safety-aux
            --safety-aux-horizon "${SAFETY_AUX_HORIZON}"
            --safety-aux-loss-coef "${SAFETY_AUX_LOSS_COEF}"
            --safety-aux-pos-weight "${SAFETY_AUX_POS_WEIGHT}"
        )
        ;;
    0|false|FALSE|no|NO|off|OFF)
        SAFETY_AUX_ARGS=()
        ;;
    *)
        echo "USE_SAFETY_AUX must be 0/1 or true/false, got: ${USE_SAFETY_AUX}" >&2
        exit 2
        ;;
esac

TRAIN_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train/train_F16sim.py"
    --env-name "${ENV_NAME}"
    --algorithm-name "${ALGO_NAME}"
    --scenario-name "${SCENARIO_NAME}"
    --model-name "${MODEL_NAME}"
    --experiment-name "${EXP}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --cuda
    --n-training-threads 1
    --n-rollout-threads "${N_ROLLOUT_THREADS}"
    --buffer-size "${BUFFER_SIZE}"
    --num-env-steps "${NUM_ENV_STEPS}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --lr "${LR}"
    --gamma 0.99
    --gae-lambda 0.95
    --ppo-epoch "${PPO_EPOCH}"
    --num-mini-batch "${NUM_MINI_BATCH}"
    --clip-param 0.2
    "${CLIPPED_VALUE_ARGS[@]}"
    --value-loss-coef "${VALUE_LOSS_COEF}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --target-kl "${TARGET_KL}"
    --max-log-ratio "${MAX_LOG_RATIO}"
    "${SAFETY_AUX_ARGS[@]}"
    --entropy-coef "${ENTROPY_COEF}"
    --hidden-size "128 128"
    --act-hidden-size "128 128"
    --activation-id 1
    --gain 0.01
    --recurrent-hidden-size 128
    --recurrent-hidden-layers 1
    --data-chunk-length "${DATA_CHUNK_LENGTH}"
    "$@"
)

echo "rpy_throttle reach no-forward no-wind from-scratch training"
echo "  experiment: ${EXP}"
echo "  env/model: ${ENV_NAME}/${SCENARIO_NAME}/${MODEL_NAME}"
echo "  motor0: forced to 0N by HYBRID_NEW_NO_FORWARD model"
echo "  seed/device: ${SEED}/${DEVICE}"
echo "  max_mode_slots: ${RPY_THROTTLE_REACH_MAX_MODE_SLOTS}"
echo "  rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, num_env_steps=${NUM_ENV_STEPS}"
echo "  lr=${LR}, value_loss_coef=${VALUE_LOSS_COEF}, clipped_value_loss=${USE_CLIPPED_VALUE_LOSS}"
echo "  ppo_epoch=${PPO_EPOCH}, num_mini_batch=${NUM_MINI_BATCH}, entropy_coef=${ENTROPY_COEF}"
echo "  max_grad_norm=${MAX_GRAD_NORM}, target_kl=${TARGET_KL}, max_log_ratio=${MAX_LOG_RATIO}"
echo "  safety_aux=${USE_SAFETY_AUX}, horizon=${SAFETY_AUX_HORIZON}, loss_coef=${SAFETY_AUX_LOSS_COEF}, pos_weight=${SAFETY_AUX_POS_WEIGHT}"

case "${DRY_RUN:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        printf 'dry-run command:'
        printf ' %q' "${TRAIN_CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

exec "${TRAIN_CMD[@]}"
