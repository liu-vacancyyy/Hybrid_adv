#!/bin/sh
# ---------------------------------------------------------------------------
# PPO adversary training for rc_human with command, observation, and Dryden
# wind attacks enabled.
#
# The victim policy is fixed.  Sensor noise and Dryden random noise are disabled.
# Wind attacks still pass through a deterministic Dryden shaping filter.
# By default the adversary generates:
#   - command space: raw PX4 vx/vy/vz/yaw sticks
#   - observation space: normalized policy observation perturbations
#   - wind space: Dryden N/E/D velocity + body p/q/r gust targets
# Add --adv-use-random-command to keep rc_human's original random command
# generator and attack only observation/wind.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_adversary.sh
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
VICTIM_CKPT="${VICTIM_CKPT:-/home/a/demo/Hybrid_adv/scripts/runs/2026-06-03_15-49-07_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_modes012534_to_ep650_20260603_154906/episode_650/actor_latest.ckpt}"
EXP="${RC_HUMAN_ADV_EXP_NAME:-rc_human_adv_cmd_obs_dryden_wind_from_ep650}"
N_ROLLOUT_THREADS="${ADV_N_ROLLOUT_THREADS:-1024}"
BUFFER_SIZE="${ADV_BUFFER_SIZE:-3000}"
MAX_ITERATIONS="${ADV_MAX_ITERATIONS:-1000}"
NUM_ENV_STEPS="${ADV_NUM_ENV_STEPS:-$((N_ROLLOUT_THREADS * BUFFER_SIZE * MAX_ITERATIONS))}"
PPO_EPOCH="${ADV_PPO_EPOCH:-8}"
NUM_MINI_BATCH="${ADV_NUM_MINI_BATCH:-32}"

echo "adversary training: rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, max_iterations=${MAX_ITERATIONS}, num_env_steps=${NUM_ENV_STEPS}, ppo_epoch=${PPO_EPOCH}, num_mini_batch=${NUM_MINI_BATCH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/adversarial/train_rc_human_adversary.py" \
    --victim-ckpt "${VICTIM_CKPT}" \
    --scenario-name rc_human --model-name HYBRID_NEW --experiment-name "${EXP}" \
    --seed 17 --device "${DEVICE}" --cuda \
    --n-rollout-threads "${N_ROLLOUT_THREADS}" --buffer-size "${BUFFER_SIZE}" --num-env-steps "${NUM_ENV_STEPS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --log-interval 1 --save-interval 10 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 \
    --ppo-epoch "${PPO_EPOCH}" --num-mini-batch "${NUM_MINI_BATCH}" --clip-param 0.2 \
    --entropy-coef 2e-3 --max-grad-norm 1.0 \
    --hidden-size "128 128 128" --data-chunk-length 8 \
    --adv-command-frac 1.0 --adv-obs-frac 1.0 --adv-wind-frac 1.0 \
    --adv-command-alpha 1.0 --adv-obs-alpha 1.0 --adv-wind-alpha 1.0 \
    --adv-command-rate-limit-frac 0.0 --adv-obs-rate-limit-frac 0.1 --adv-wind-rate-limit-frac 0.1 \
    --adv-init-log-std -1.2 \
    --adv-lipschitz-coef 1e-6 \
    --adv-alive-penalty 0.01 --adv-policy-reward-weight 0.0 \
    --adv-policy-reward-window 10 \
    --adv-w-vel-error 4.0 --adv-w-axis-vel-error 2.0 --adv-w-yaw-error 2.0 \
    --adv-axis-vel-margin 0.25 --adv-yaw-margin-deg 6.0 \
    --adv-w-vel-bad-margin 8.0 --adv-w-yaw-bad-margin 4.0 \
    --adv-w-attitude 5.0 --adv-w-omega 0.8 --adv-w-force-margin 0.2 \
    --adv-bad-done-bonus 50.0 \
    --adv-linf-penalty 0.0 \
    --adv-command-target-rms-min 0.25 \
    --adv-obs-target-rms-min 0.45 \
    --adv-wind-target-rms-min 0.35 \
    --adv-command-range-penalty 0.0 --adv-obs-range-penalty 0.0 --adv-wind-range-penalty 0.0 \
    --adv-saturation-penalty 0.0 --adv-raw-excess-penalty 0.20 \
    --adv-obs-energy-window 50 --adv-obs-energy-budget 50.0 --adv-obs-energy-penalty 0.02 \
    "$@"
