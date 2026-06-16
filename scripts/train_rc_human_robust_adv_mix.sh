#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Robust PPO fine-tuning for rc_human.
#
# Environment mixture:
#   - 10% of vectorized envs use a frozen adversary to generate command,
#     observation, and Dryden-wind perturbations.
#   - 90% of vectorized envs sample rc_human curriculum levels uniformly over
#     all 120 levels (0..119), instead of adaptive curriculum progression.
#
# Defaults continue training the episode_650 victim against the episode_999
# adversary selected from the 2026-06-10 adversarial run.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/a/anaconda3/envs/Neuralplane/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-23}"
EXP="${RC_HUMAN_ROBUST_EXP_NAME:-rc_human_rl_robust_adv10_uniform120_from_ep650_adv999}"

INIT_ACTOR_CKPT="${INIT_ACTOR_CKPT:-${REPO_ROOT}/scripts/runs/2026-06-03_15-49-07_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_modes012534_to_ep650_20260603_154906/episode_650/actor_latest.ckpt}"
ADV_CKPT="${ADV_CKPT:-${REPO_ROOT}/scripts/runs/2026-06-10_20-48-13_Control_rc_human_HYBRID_NEW_ppo_rc_human_adv_vtol_mc_px4stick_cmd_obs_dryden_wind_from_ep650/episode_999/adv_actor_latest.ckpt}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-1200}"
BUFFER_SIZE="${BUFFER_SIZE:-1500}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-1.8e9}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

ADV_MIX_FRAC="${ADV_MIX_FRAC:-0.10}"
UNIFORM_CURRICULUM_LEVELS="${UNIFORM_CURRICULUM_LEVELS:-120}"

PPO_EPOCH="${PPO_EPOCH:-10}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-24}"
LR="${LR:-2e-4}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ ! -f "${INIT_ACTOR_CKPT}" ]; then
    echo "Initial actor checkpoint not found: ${INIT_ACTOR_CKPT}" >&2
    exit 1
fi
if [ ! -f "${ADV_CKPT}" ]; then
    echo "Adversary checkpoint not found: ${ADV_CKPT}" >&2
    exit 1
fi

echo "robust rc_human training"
echo "  experiment: ${EXP}"
echo "  init_actor: ${INIT_ACTOR_CKPT}"
echo "  adv_ckpt: ${ADV_CKPT}"
echo "  mix: adv=${ADV_MIX_FRAC}, uniform_levels=${UNIFORM_CURRICULUM_LEVELS}"
echo "  rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, num_env_steps=${NUM_ENV_STEPS}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train/train_rc_human_robust_adv_mix.py" \
    --env-name Control --scenario-name rc_human --model-name HYBRID_NEW \
    --algorithm-name ppo --experiment-name "${EXP}" \
    --seed "${SEED}" --device "${DEVICE}" --cuda \
    --n-training-threads 1 \
    --n-rollout-threads "${N_ROLLOUT_THREADS}" \
    --buffer-size "${BUFFER_SIZE}" \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --init-actor-ckpt "${INIT_ACTOR_CKPT}" \
    --adv-ckpt "${ADV_CKPT}" \
    --adv-mix-frac "${ADV_MIX_FRAC}" \
    --uniform-curriculum-levels "${UNIFORM_CURRICULUM_LEVELS}" \
    --lr "${LR}" \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ppo-epoch "${PPO_EPOCH}" \
    --num-mini-batch "${NUM_MINI_BATCH}" \
    --clip-param 0.2 \
    --max-grad-norm 2 \
    --entropy-coef "${ENTROPY_COEF}" \
    --hidden-size "128 128" \
    --act-hidden-size "128 128" \
    --activation-id 1 \
    --gain 0.01 \
    --recurrent-hidden-size 128 \
    --recurrent-hidden-layers 1 \
    --data-chunk-length 8 \
    --adv-hidden-size "128 128 128" \
    --adv-command-frac 1.0 \
    --adv-obs-frac 1.0 \
    --adv-wind-frac 1.0 \
    --adv-command-alpha 1.0 \
    --adv-obs-alpha 1.0 \
    --adv-wind-alpha 1.0 \
    --adv-command-rate-limit-frac 0.0 \
    --adv-obs-rate-limit-frac 0.1 \
    --adv-wind-rate-limit-frac 0.1 \
    --adv-obs-default-scale 0.02 \
    --adv-obs-max-scale 0.10 \
    "$@"
