#!/usr/bin/env bash
# PPO-GRU training for HoverTask + GazeboModel with PX4 Gazebo-inspired domain
# randomization values from envs/configs/gazebo_hover.yaml.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="Control"
SCENARIO_NAME="${SCENARIO_NAME:-gazebo_hover}"
MODEL_NAME="${MODEL_NAME:-GAZEBO}"
ALGO_NAME="${ALGO_NAME:-ppo}"
EXP_NAME="${EXP_NAME:-gazebo_hover_px4_dr}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda:0}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-512}"
BUFFER_SIZE="${BUFFER_SIZE:-1500}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-8e8}"

echo "env=${ENV_NAME} scenario=${SCENARIO_NAME} model=${MODEL_NAME} algo=${ALGO_NAME} exp=${EXP_NAME} seed=${SEED}"
"${PYTHON_BIN}" scripts/train/train_F16sim.py \
    --env-name "${ENV_NAME}" \
    --algorithm-name "${ALGO_NAME}" \
    --scenario-name "${SCENARIO_NAME}" \
    --model-name "${MODEL_NAME}" \
    --experiment-name "${EXP_NAME}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n-training-threads 1 \
    --n-rollout-threads "${N_ROLLOUT_THREADS}" \
    --cuda \
    --log-interval 1 \
    --save-interval 10 \
    --num-mini-batch 8 \
    --buffer-size "${BUFFER_SIZE}" \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ppo-epoch 12 \
    --clip-param 0.2 \
    --max-grad-norm 2 \
    --entropy-coef 1e-3 \
    --hidden-size "128 128" \
    --act-hidden-size "128 128" \
    --activation-id 1 \
    --gain 0.01 \
    --recurrent-hidden-size 128 \
    --recurrent-hidden-layers 1 \
    --data-chunk-length 8 \
    "$@"
