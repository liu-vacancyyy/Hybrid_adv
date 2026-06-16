#!/usr/bin/env bash
# Train rc_human PPO to episode_650, then train the adversary to episode_800
# against that frozen episode_650 actor.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
RUNS_ROOT="${REPO_ROOT}/scripts/runs"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/a/anaconda3/envs/Neuralplane/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

NORMAL_TARGET_EPISODE="${NORMAL_TARGET_EPISODE:-650}"
ADV_TARGET_EPISODE="${ADV_TARGET_EPISODE:-800}"

NORMAL_N_ROLLOUT_THREADS="${NORMAL_N_ROLLOUT_THREADS:-1024}"
NORMAL_BUFFER_SIZE="${NORMAL_BUFFER_SIZE:-2000}"
NORMAL_SAVE_INTERVAL="${NORMAL_SAVE_INTERVAL:-10}"
NORMAL_SEED="${NORMAL_SEED:-7}"

ADV_N_ROLLOUT_THREADS="${ADV_N_ROLLOUT_THREADS:-1536}"
ADV_BUFFER_SIZE="${ADV_BUFFER_SIZE:-1500}"
ADV_SAVE_INTERVAL="${ADV_SAVE_INTERVAL:-10}"

# The runners use zero-based update names.  To create episode_N, run N + 1
# PPO updates so the loop reaches update index N.
NORMAL_ITERATIONS=$((NORMAL_TARGET_EPISODE + 1))
ADV_ITERATIONS=$((ADV_TARGET_EPISODE + 1))
NORMAL_NUM_ENV_STEPS=$((NORMAL_N_ROLLOUT_THREADS * NORMAL_BUFFER_SIZE * NORMAL_ITERATIONS))
ADV_NUM_ENV_STEPS=$((ADV_N_ROLLOUT_THREADS * ADV_BUFFER_SIZE * ADV_ITERATIONS))

NORMAL_EXP="${NORMAL_EXP:-rc_human_rl_gru_wind_modes012534_to_ep${NORMAL_TARGET_EPISODE}_${RUN_ID}}"
ADV_EXP="${ADV_EXP:-rc_human_adv_from_ep${NORMAL_TARGET_EPISODE}_to_ep${ADV_TARGET_EPISODE}_${RUN_ID}}"
ADV_RUN_DIR="${ADV_RUN_DIR:-${RUNS_ROOT}/${RUN_ID}_Control_rc_human_HYBRID_NEW_ppo_${ADV_EXP}}"

export PYTHON_BIN
export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export RC_HUMAN_EXP_NAME="${NORMAL_EXP}"

find_normal_run_dir() {
    find "${RUNS_ROOT}" -maxdepth 1 -type d \
        -name "*_Control_rc_human_HYBRID_NEW_ppo_${NORMAL_EXP}" \
        -print | sort | tail -n 1
}

echo "==> Normal PPO training"
echo "    target episode: ${NORMAL_TARGET_EPISODE}"
echo "    iterations: ${NORMAL_ITERATIONS}"
echo "    num_env_steps: ${NORMAL_NUM_ENV_STEPS}"
echo "    experiment: ${NORMAL_EXP}"

sh "${SCRIPT_DIR}/train_rc_human_rl.sh" \
    --device "${DEVICE}" \
    --seed "${NORMAL_SEED}" \
    --n-rollout-threads "${NORMAL_N_ROLLOUT_THREADS}" \
    --buffer-size "${NORMAL_BUFFER_SIZE}" \
    --num-env-steps "${NORMAL_NUM_ENV_STEPS}" \
    --save-interval "${NORMAL_SAVE_INTERVAL}"

NORMAL_RUN_DIR=$(find_normal_run_dir)
if [ -z "${NORMAL_RUN_DIR}" ]; then
    echo "Could not locate normal training run directory for experiment ${NORMAL_EXP}" >&2
    exit 1
fi

VICTIM_CKPT="${NORMAL_RUN_DIR}/episode_${NORMAL_TARGET_EPISODE}/actor_latest.ckpt"
if [ ! -f "${VICTIM_CKPT}" ]; then
    echo "Expected victim checkpoint was not created: ${VICTIM_CKPT}" >&2
    exit 1
fi

echo "==> Adversarial PPO training"
echo "    victim checkpoint: ${VICTIM_CKPT}"
echo "    target episode: ${ADV_TARGET_EPISODE}"
echo "    iterations: ${ADV_ITERATIONS}"
echo "    num_env_steps: ${ADV_NUM_ENV_STEPS}"
echo "    run dir: ${ADV_RUN_DIR}"

export DEVICE
export VICTIM_CKPT
export RC_HUMAN_ADV_EXP_NAME="${ADV_EXP}"
export ADV_N_ROLLOUT_THREADS
export ADV_BUFFER_SIZE
export ADV_MAX_ITERATIONS="${ADV_ITERATIONS}"
export ADV_NUM_ENV_STEPS

sh "${SCRIPT_DIR}/train_rc_human_adversary.sh" \
    --run-dir "${ADV_RUN_DIR}" \
    --save-interval "${ADV_SAVE_INTERVAL}"

ADV_ACTOR="${ADV_RUN_DIR}/episode_${ADV_TARGET_EPISODE}/adv_actor_latest.ckpt"
if [ ! -f "${ADV_ACTOR}" ]; then
    echo "Expected adversary checkpoint was not created: ${ADV_ACTOR}" >&2
    exit 1
fi

echo "Done."
echo "Normal run: ${NORMAL_RUN_DIR}"
echo "Victim checkpoint: ${VICTIM_CKPT}"
echo "Adversary run: ${ADV_RUN_DIR}"
