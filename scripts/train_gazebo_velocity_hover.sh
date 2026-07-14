#!/usr/bin/env bash
# PPO-GRU training for GazeboModel velocity hover:
# upper-level vx/vy/vz commands are fixed at zero and the policy tracks them.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="Control"
SCENARIO_NAME="${SCENARIO_NAME:-gazebo_velocity_hover}"
MODEL_NAME="${MODEL_NAME:-GAZEBO}"
ALGO_NAME="${ALGO_NAME:-ppo}"
EXP_NAME="${EXP_NAME:-gazebo_velocity_hover_vxvyvz_zero}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda:0}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-512}"
BUFFER_SIZE="${BUFFER_SIZE:-1500}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-8e8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

LR="${LR:-3e-4}"
PPO_EPOCH="${PPO_EPOCH:-12}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-8}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2}"
DATA_CHUNK_LENGTH="${DATA_CHUNK_LENGTH:-8}"

CUDA_ARGS=()
case "${USE_CUDA:-auto}" in
    1|true|TRUE|yes|YES|on|ON)
        CUDA_ARGS=(--cuda)
        ;;
    0|false|FALSE|no|NO|off|OFF)
        CUDA_ARGS=()
        ;;
    *)
        case "${DEVICE}" in
            cpu|CPU)
                CUDA_ARGS=()
                ;;
            *)
                CUDA_ARGS=(--cuda)
                ;;
        esac
        ;;
esac

TRAIN_CMD=(
    "${PYTHON_BIN}" scripts/train/train_F16sim.py
    --env-name "${ENV_NAME}"
    --algorithm-name "${ALGO_NAME}"
    --scenario-name "${SCENARIO_NAME}"
    --model-name "${MODEL_NAME}"
    --experiment-name "${EXP_NAME}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --n-training-threads 1
    --n-rollout-threads "${N_ROLLOUT_THREADS}"
    "${CUDA_ARGS[@]}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --num-mini-batch "${NUM_MINI_BATCH}"
    --buffer-size "${BUFFER_SIZE}"
    --num-env-steps "${NUM_ENV_STEPS}"
    --lr "${LR}"
    --gamma 0.99
    --gae-lambda 0.95
    --ppo-epoch "${PPO_EPOCH}"
    --clip-param 0.2
    --max-grad-norm "${MAX_GRAD_NORM}"
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

echo "gazebo velocity-hover PPO training"
echo "  env/model: ${ENV_NAME}/${SCENARIO_NAME}/${MODEL_NAME}"
echo "  command: vx=0, vy=0, vz=0"
echo "  seed/device: ${SEED}/${DEVICE}"
echo "  rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, num_env_steps=${NUM_ENV_STEPS}"
echo "  lr=${LR}, ppo_epoch=${PPO_EPOCH}, mini_batch=${NUM_MINI_BATCH}, entropy=${ENTROPY_COEF}"

case "${DRY_RUN:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        printf 'dry-run command:'
        printf ' %q' "${TRAIN_CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

exec "${TRAIN_CMD[@]}"
