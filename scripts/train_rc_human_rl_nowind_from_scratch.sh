#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PPO-GRU training for rc_human from scratch with all wind effects disabled.
#
# This is a clean normal-training baseline:
#   - no adversary
#   - no --model-dir
#   - no --init-actor-ckpt
#   - no steady wind, no Dryden turbulence, no angular gust, no wind loading
#
# The script does not modify envs/configs/rc_human.yaml. It derives a runtime
# scenario config at envs/configs/rc_human_nowind_runtime.yaml, then trains with
# --scenario-name rc_human_nowind_runtime.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_rl_nowind_from_scratch.sh
#
# Useful overrides:
#   DEVICE=cuda:0 SEED=31 NUM_ENV_STEPS=2.0e9 \
#   RC_HUMAN_COMMAND_RATE_LIMIT_FRAC=0.05 \
#   RC_HUMAN_EXP_NAME=rc_human_rl_px4_nowind_rate0_05 \
#   bash scripts/train_rc_human_rl_nowind_from_scratch.sh
#
# Print the generated command without starting training:
#   DRY_RUN=1 bash scripts/train_rc_human_rl_nowind_from_scratch.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-7}"

ENV_NAME="Control"
BASE_SCENARIO_NAME="${BASE_SCENARIO_NAME:-rc_human}"
NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_nowind_runtime}"
SCENARIO_NAME="${NO_WIND_CONFIG_NAME}"
MODEL_NAME="HYBRID_NEW"
ALGO_NAME="ppo"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 5 3 4}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export RC_HUMAN_COMMAND_RATE_LIMIT_FRAC="${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC:-0.05}"

RATE_TAG=$(printf '%s' "${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC}" | tr -c '0-9A-Za-z' '_')
EXP="${RC_HUMAN_EXP_NAME:-rc_human_rl_px4_nowind_rate${RATE_TAG}_from_scratch_modes012534}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-2048}"
BUFFER_SIZE="${BUFFER_SIZE:-1500}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-2.5e9}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

LR="${LR:-2e-4}"
PPO_EPOCH="${PPO_EPOCH:-10}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-8}"
ENTROPY_COEF="${ENTROPY_COEF:-2e-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2}"
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-0.5}"
USE_CLIPPED_VALUE_LOSS="${USE_CLIPPED_VALUE_LOSS:-1}"
DATA_CHUNK_LENGTH="${DATA_CHUNK_LENGTH:-8}"

BASE_CONFIG_PATH="${REPO_ROOT}/envs/configs/${BASE_SCENARIO_NAME}.yaml"
NO_WIND_CONFIG_PATH="${REPO_ROOT}/envs/configs/${NO_WIND_CONFIG_NAME}.yaml"

for arg in "$@"; do
    case "${arg}" in
        --use-recurrent-policy)
            echo "Do not pass --use-recurrent-policy: config.py defines it as store_false and it would disable GRU." >&2
            exit 2
            ;;
        --scenario-name|--scenario-name=*)
            echo "Do not pass ${arg}: this script must use the generated no-wind scenario ${SCENARIO_NAME}." >&2
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

if [ ! -f "${BASE_CONFIG_PATH}" ]; then
    echo "Base config not found: ${BASE_CONFIG_PATH}" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${BASE_CONFIG_PATH}" "${NO_WIND_CONFIG_PATH}" <<'PY'
from pathlib import Path
import os
import sys
import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

with base_path.open("r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

false_keys = [
    "enable_wind",
    "enable_dryden_turbulence",
    "dryden_randomize",
    "dryden_domain_randomization",
    "dryden_sigma_curriculum_enable",
    "dryden_mean_wind_curriculum_enable",
    "enable_dryden_angular_turbulence",
    "wind_drag_enable",
    "wind_pqr_torque_enable",
]
for key in false_keys:
    data[key] = False

zero_keys = [
    "wind_north",
    "wind_east",
    "wind_down",
    "gust_north",
    "gust_east",
    "gust_down",
    "dryden_sigma_ref",
    "dryden_sigma_ref_min",
    "dryden_sigma_ref_max",
    "dryden_sigma_scale_min",
    "dryden_sigma_scale_max",
    "dryden_mean_wind_scale_min",
    "dryden_mean_wind_scale_max",
    "dryden_mean_wind_north_min",
    "dryden_mean_wind_north_max",
    "dryden_mean_wind_east_min",
    "dryden_mean_wind_east_max",
    "dryden_mean_wind_down_min",
    "dryden_mean_wind_down_max",
    "dryden_pqr_sigma_ref",
    "dryden_pqr_sigma_scale_min",
    "dryden_pqr_sigma_scale_max",
    "dryden_pqr_attack_p_max",
    "dryden_pqr_attack_q_max",
    "dryden_pqr_attack_r_max",
    "wind_body_cda_x",
    "wind_body_cda_y",
    "wind_body_cda_z",
    "wind_cp_x",
    "wind_cp_y",
    "wind_cp_z",
    "wind_pqr_accel_gain_p",
    "wind_pqr_accel_gain_q",
    "wind_pqr_accel_gain_r",
    "wind_force_clip",
    "wind_moment_clip",
]
for key in zero_keys:
    data[key] = 0.0

data["task_name"] = "rc_human"
mode_order = os.environ.get("RC_HUMAN_MODE_ORDER")
if mode_order:
    data["rc_human_mode_order"] = mode_order
max_mode_slots = os.environ.get("RC_HUMAN_MAX_MODE_SLOTS")
if max_mode_slots:
    data["rc_human_max_mode_slots"] = int(max_mode_slots)
rate_limit_frac = os.environ.get("RC_HUMAN_COMMAND_RATE_LIMIT_FRAC")
if rate_limit_frac:
    data["rc_human_command_rate_limit_frac"] = float(rate_limit_frac)

header = (
    "# Generated by scripts/train_rc_human_rl_nowind_from_scratch.sh.\n"
    "# Do not edit this file directly; edit envs/configs/rc_human.yaml or the script.\n"
)
out_path.write_text(
    header + yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
    encoding="utf-8",
)
PY

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

echo "rc_human no-wind from-scratch training"
echo "  experiment: ${EXP}"
echo "  env/model: ${ENV_NAME}/${SCENARIO_NAME}/${MODEL_NAME}"
echo "  base_config: ${BASE_CONFIG_PATH}"
echo "  no_wind_config: ${NO_WIND_CONFIG_PATH}"
echo "  wind: disabled; Dryden: disabled; wind loading: disabled"
echo "  seed/device: ${SEED}/${DEVICE}"
echo "  mode_order: ${RC_HUMAN_MODE_ORDER}"
echo "  max_mode_slots: ${RC_HUMAN_MAX_MODE_SLOTS}"
echo "  command_rate_limit_frac: ${RC_HUMAN_COMMAND_RATE_LIMIT_FRAC}"
echo "  rollout_threads=${N_ROLLOUT_THREADS}, buffer_size=${BUFFER_SIZE}, num_env_steps=${NUM_ENV_STEPS}"
echo "  lr=${LR}, value_loss_coef=${VALUE_LOSS_COEF}, clipped_value_loss=${USE_CLIPPED_VALUE_LOSS}"
echo "  ppo_epoch=${PPO_EPOCH}, num_mini_batch=${NUM_MINI_BATCH}, entropy_coef=${ENTROPY_COEF}"

case "${DRY_RUN:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        printf 'dry-run command:'
        printf ' %q' "${TRAIN_CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

exec "${TRAIN_CMD[@]}"
