#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO adversary training for the current rc_human vx/vy/vz + yawhold victim.
#
# Attack surfaces enabled:
#   - command: raw vx/vy/vz/yaw sticks before PX4 VTOL-MC stick mapping
#   - observation: bounded normalized victim-observation perturbations
#
# Wind attack is intentionally disabled.  The default scenario is the no-wind
# runtime config trained with mode order 0 1 2 5 4 3.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_adversary_cmd_obs_nowind_ep630.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SCENARIO_NAME="${SCENARIO_NAME:-rc_human_vxvyvz_yawhold_currheading_modes0_1_2_5_4_3_nowind_runtime}"
MODEL_NAME="${MODEL_NAME:-HYBRID_NEW}"
VICTIM_RUN="${VICTIM_RUN:-${REPO_ROOT}/scripts/runs/2026-06-29_13-48-43_Control_rc_human_vxvyvz_yawhold_currheading_modes0_1_2_5_4_3_nowind_runtime_HYBRID_NEW_ppo_rc_human_vxvyvz_yawhold_currheading_rate0_05_from_scratch_modes0_1_2_5_4_3}"
VICTIM_CKPT="${VICTIM_CKPT:-${VICTIM_RUN}/episode_630/actor_latest.ckpt}"
EXP="${RC_HUMAN_ADV_EXP_NAME:-rc_human_adv_cmd_obs_nowind_from_ep630}"
RUN_DIR="${RUN_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

# Conservative defaults.  Increase ADV_N_ROLLOUT_THREADS after confirming
# memory/GPU headroom on your machine.
N_ROLLOUT_THREADS="${ADV_N_ROLLOUT_THREADS:-512}"
BUFFER_SIZE="${ADV_BUFFER_SIZE:-1500}"
MAX_ITERATIONS="${ADV_MAX_ITERATIONS:-1000}"
NUM_ENV_STEPS="${ADV_NUM_ENV_STEPS:-$((N_ROLLOUT_THREADS * BUFFER_SIZE * MAX_ITERATIONS))}"
PPO_EPOCH="${ADV_PPO_EPOCH:-8}"
NUM_MINI_BATCH="${ADV_NUM_MINI_BATCH:-16}"
SAVE_INTERVAL="${ADV_SAVE_INTERVAL:-10}"
SEED="${SEED:-17}"
COMMAND_RATE_LIMIT_FRAC="${ADV_COMMAND_RATE_LIMIT_FRAC:-0.05}"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 4 3}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export RC_HUMAN_RANDOM_WIND_ENABLE="${RC_HUMAN_RANDOM_WIND_ENABLE:-0}"

if [ -x "${PYTHON_BIN}" ]; then
    :
elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v "${PYTHON_BIN}")
else
    echo "Python executable not found: ${PYTHON_BIN}. Activate your environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/envs/configs/${SCENARIO_NAME}.yaml" ]; then
    echo "Scenario config not found: ${REPO_ROOT}/envs/configs/${SCENARIO_NAME}.yaml" >&2
    exit 1
fi

if [ ! -f "${VICTIM_CKPT}" ]; then
    echo "Victim checkpoint not found: ${VICTIM_CKPT}" >&2
    exit 1
fi

RUN_DIR_ARGS=()
if [ -n "${RUN_DIR}" ]; then
    RUN_DIR_ARGS=(--run-dir "${RUN_DIR}")
fi

CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/adversarial/train_rc_human_adversary.py"
    --victim-ckpt "${VICTIM_CKPT}"
    --scenario-name "${SCENARIO_NAME}" --model-name "${MODEL_NAME}" --experiment-name "${EXP}"
    --seed "${SEED}" --device "${DEVICE}" --cuda
    --n-rollout-threads "${N_ROLLOUT_THREADS}" --buffer-size "${BUFFER_SIZE}" --num-env-steps "${NUM_ENV_STEPS}"
    --max-iterations "${MAX_ITERATIONS}"
    --log-interval 1 --save-interval "${SAVE_INTERVAL}"
    "${RUN_DIR_ARGS[@]}"
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95
    --ppo-epoch "${PPO_EPOCH}" --num-mini-batch "${NUM_MINI_BATCH}" --clip-param 0.2
    --entropy-coef 2e-3 --max-grad-norm 1.0
    --hidden-size "128 128 128" --data-chunk-length 8
    --adv-command-frac 1.0 --adv-obs-frac 1.0 --adv-wind-frac 0.0
    --adv-command-alpha 1.0 --adv-obs-alpha 1.0 --adv-wind-alpha 1.0
    --adv-command-rate-limit-frac "${COMMAND_RATE_LIMIT_FRAC}" --adv-obs-rate-limit-frac 0.1 --adv-wind-rate-limit-frac 0.0
    --adv-init-log-std -1.2
    --adv-lipschitz-coef 1e-6
    --adv-alive-penalty 0.01 --adv-policy-reward-weight 0.0
    --adv-policy-reward-window 10
    --adv-w-vel-error 4.0 --adv-w-axis-vel-error 2.0 --adv-w-yaw-error 2.0
    --adv-axis-vel-margin 0.25 --adv-yaw-margin-deg 6.0
    --adv-w-vel-bad-margin 8.0 --adv-w-yaw-bad-margin 4.0
    --adv-w-attitude 5.0 --adv-w-omega 0.8 --adv-w-force-margin 0.2
    --adv-bad-done-bonus 50.0
    --adv-linf-penalty 0.0
    --adv-command-target-rms-min 0.25
    --adv-obs-target-rms-min 0.45
    --adv-wind-target-rms-min 0.0 --adv-wind-target-rms-max 0.0
    --adv-command-range-penalty 0.0 --adv-obs-range-penalty 0.0 --adv-wind-range-penalty 0.0
    --adv-saturation-penalty 0.0 --adv-raw-excess-penalty 0.20
    --adv-obs-energy-window 50 --adv-obs-energy-budget 50.0 --adv-obs-energy-penalty 0.02
    "$@"
)

echo "adversary training: ${EXP}"
echo "  scenario: ${SCENARIO_NAME}"
echo "  victim_ckpt: ${VICTIM_CKPT}"
echo "  attack: command + observation, wind disabled"
echo "  rollout_threads: ${N_ROLLOUT_THREADS}"
echo "  buffer_size: ${BUFFER_SIZE}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo "  num_env_steps: ${NUM_ENV_STEPS}"
echo "  ppo_epoch: ${PPO_EPOCH}"
echo "  num_mini_batch: ${NUM_MINI_BATCH}"
echo "  command_rate_limit_frac: ${COMMAND_RATE_LIMIT_FRAC}"
if [ -n "${RUN_DIR}" ]; then
    echo "  run_dir: ${RUN_DIR}"
fi

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
