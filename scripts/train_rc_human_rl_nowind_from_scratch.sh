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
yaw_reward_mode = os.environ.get("RC_HUMAN_YAW_REWARD_MODE")
if yaw_reward_mode:
    data["rc_human_yaw_reward_mode"] = yaw_reward_mode
for env_key, data_key in [
    ("RC_HUMAN_TRACKING_BAD_DONE_ENABLE", "rc_human_tracking_bad_done_enable"),
    ("RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_ENABLE", "rc_human_vxyvz_dynamic_bad_done_enable"),
    ("RC_HUMAN_SUCCESS_IGNORE_TRANSIENT", "rc_human_success_ignore_transient"),
    ("RC_HUMAN_CURRICULUM_ENABLE", "rc_human_curriculum_enable"),
    ("RC_HUMAN_VX_FORWARD_CURRICULUM_ENABLE", "rc_human_vx_forward_curriculum_enable"),
    ("RC_HUMAN_YAW_COMMAND_ENABLE", "rc_human_yaw_command_enable"),
    ("RC_HUMAN_YAW_HOLD_ENABLE", "rc_human_yaw_hold_enable"),
    ("RC_HUMAN_YAW_TRACKING_ENABLE", "rc_human_yaw_tracking_enable"),
    ("RC_HUMAN_YAW_REWARD_ENABLE", "rc_human_yaw_reward_enable"),
    ("RC_HUMAN_ALTITUDE_AWARE_VZ_ENABLE", "rc_human_altitude_aware_vz_enable"),
    ("RC_HUMAN_ENABLE_SENSOR_NOISE", "enable_sensor_noise"),
]:
    raw = os.environ.get(env_key)
    if raw is not None:
        data[data_key] = raw.strip().lower() in {"1", "true", "yes", "on"}
for env_key, data_key in [
    ("RC_HUMAN_ALT_GUARD_ZONE", "rc_human_alt_guard_zone"),
    ("RC_HUMAN_ALT_LOW", "rc_human_alt_low"),
    ("RC_HUMAN_ALT_HIGH", "rc_human_alt_high"),
    ("RC_HUMAN_MIX_CURRENT", "rc_human_mix_current"),
    ("RC_HUMAN_MIX_EASY_REPLAY", "rc_human_mix_easy_replay"),
    ("RC_HUMAN_MIX_MEDIUM_REPLAY", "rc_human_mix_medium_replay"),
    ("RC_HUMAN_MIX_RANDOM_REPLAY", "rc_human_mix_random_replay"),
    ("RC_HUMAN_W_YAW", "rc_human_w_yaw"),
    ("RC_HUMAN_W_OMEGA", "rc_human_w_omega"),
    ("RC_HUMAN_W_YAW_RATE", "rc_human_w_yaw_rate"),
    ("RC_HUMAN_W_VEL", "rc_human_w_vel"),
    ("RC_HUMAN_W_ATTITUDE", "rc_human_w_attitude"),
    ("RC_HUMAN_W_SMOOTH", "rc_human_w_smooth"),
    ("RC_HUMAN_W_VEL_PRECISION", "rc_human_w_vel_precision"),
    ("RC_HUMAN_W_YAW_PRECISION", "rc_human_w_yaw_precision"),
    ("RC_HUMAN_REL_ERROR_FIXED_SCALE", "rc_human_rel_error_fixed_scale"),
    ("RC_HUMAN_REL_ERROR_FLOOR", "rc_human_rel_error_floor"),
    ("RC_HUMAN_W_REL_TRACKING", "rc_human_w_rel_tracking"),
    ("RC_HUMAN_W_REL_PRECISION", "rc_human_w_rel_precision"),
    ("RC_HUMAN_REL_PRECISION_GAIN", "rc_human_rel_precision_gain"),
    ("RC_HUMAN_W_OVERSHOOT", "rc_human_w_overshoot"),
    ("RC_HUMAN_OVERSHOOT_DEADBAND", "rc_human_overshoot_deadband"),
    ("RC_HUMAN_W_ADAPTIVE_DAMPING", "rc_human_w_adaptive_damping"),
    ("RC_HUMAN_ADAPTIVE_DAMPING_GAIN", "rc_human_adaptive_damping_gain"),
    ("RC_HUMAN_SIG_YAW", "rc_human_sig_yaw"),
    ("RC_HUMAN_SIG_YAW_DELTA", "rc_human_sig_yaw_delta"),
    ("RC_HUMAN_SIG_YAW_RATE", "rc_human_sig_yaw_rate"),
    ("RC_HUMAN_SIG_VX", "rc_human_sig_vx"),
    ("RC_HUMAN_SIG_VY", "rc_human_sig_vy"),
    ("RC_HUMAN_SIG_VZ", "rc_human_sig_vz"),
    ("RC_HUMAN_SIG_ATTITUDE", "rc_human_sig_attitude"),
    ("RC_HUMAN_SIG_OMEGA", "rc_human_sig_omega"),
    ("RC_HUMAN_SIG_SMOOTH", "rc_human_sig_smooth"),
    ("RC_HUMAN_SUCCESS_VEL_ERROR", "rc_human_success_vel_error"),
    ("RC_HUMAN_SUCCESS_YAW_ERROR", "rc_human_success_yaw_error"),
    ("RC_HUMAN_SUCCESS_ATTITUDE_ERROR", "rc_human_success_attitude_error"),
    ("RC_HUMAN_MODE5_RELEASE_SPEED_ERROR", "rc_human_mode5_release_speed_error"),
    ("RC_HUMAN_MODE5_RELEASE_TARGET_FRAC", "rc_human_mode5_release_target_frac"),
    ("RC_HUMAN_MODE5_HOLD_MIN_STEPS", "rc_human_mode5_hold_min_steps"),
    ("RC_HUMAN_MODE5_HOLD_MAX_STEPS", "rc_human_mode5_hold_max_steps"),
    ("RC_HUMAN_MODE5_RELEASE_RECOVERY_STEPS", "rc_human_mode5_release_recovery_steps"),
    ("RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_MARGIN", "rc_human_vxyvz_dynamic_bad_done_margin"),
    ("RC_HUMAN_VX_MIN", "rc_human_vx_min"),
    ("RC_HUMAN_VX_MAX", "rc_human_vx_max"),
    ("RC_HUMAN_VX_LIMIT", "rc_human_vx_limit"),
    ("RC_HUMAN_VY_LIMIT", "rc_human_vy_limit"),
    ("RC_HUMAN_VZ_LIMIT", "rc_human_vz_limit"),
    ("RC_HUMAN_MAX_VELOCITY", "max_velocity"),
    ("RC_HUMAN_ALTITUDE_LIMIT", "altitude_limit"),
    ("RC_HUMAN_MAX_PITCH", "max_pitch"),
    ("RC_HUMAN_MAX_ROLL", "max_roll"),
    ("RC_HUMAN_MAX_OMEGA", "max_omega"),
    ("RC_HUMAN_MIN_ALPHA", "min_alpha"),
    ("RC_HUMAN_MAX_ALPHA", "max_alpha"),
    ("RC_HUMAN_MIN_BETA", "min_beta"),
    ("RC_HUMAN_MAX_BETA", "max_beta"),
    ("RC_HUMAN_INIT_ROLL_RANGE", "init_roll_range"),
    ("RC_HUMAN_INIT_PITCH_RANGE", "init_pitch_range"),
    ("RC_HUMAN_INIT_YAW_RANGE", "init_yaw_range"),
    ("RC_HUMAN_INIT_VEL_RANGE", "init_vel_range"),
    ("RC_HUMAN_INIT_OMEGA_RANGE", "init_omega_range"),
    ("RC_HUMAN_DR_MASS", "dr_mass"),
    ("RC_HUMAN_DR_INERTIA", "dr_inertia"),
    ("RC_HUMAN_MIN_ALTITUDE", "min_altitude"),
    ("RC_HUMAN_MAX_ALTITUDE", "max_altitude"),
    ("RC_HUMAN_SENSOR_POS_STD", "sensor_pos_std"),
    ("RC_HUMAN_SENSOR_VEL_STD", "sensor_vel_std"),
    ("RC_HUMAN_SENSOR_ATT_STD", "sensor_att_std"),
    ("RC_HUMAN_SENSOR_OMEGA_STD", "sensor_omega_std"),
]:
    raw = os.environ.get(env_key)
    if raw is not None:
        data[data_key] = float(raw)

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
echo "  yaw_reward_mode=${RC_HUMAN_YAW_REWARD_MODE:-target}, yaw_cmd=${RC_HUMAN_YAW_COMMAND_ENABLE:-1}, yaw_hold=${RC_HUMAN_YAW_HOLD_ENABLE:-1}, yaw_track=${RC_HUMAN_YAW_TRACKING_ENABLE:-1}"
echo "  yaw_w=${RC_HUMAN_W_YAW:-}, yaw_rate_w=${RC_HUMAN_W_YAW_RATE:-0}, yaw_sig=${RC_HUMAN_SIG_YAW:-}, yaw_rate_sig=${RC_HUMAN_SIG_YAW_RATE:-}"
echo "  success(v=${RC_HUMAN_SUCCESS_VEL_ERROR:-}, yaw=${RC_HUMAN_SUCCESS_YAW_ERROR:-}, att=${RC_HUMAN_SUCCESS_ATTITUDE_ERROR:-}), bad_done(track=${RC_HUMAN_TRACKING_BAD_DONE_ENABLE:-}, dyn=${RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_ENABLE:-}, margin=${RC_HUMAN_VXYVZ_DYNAMIC_BAD_DONE_MARGIN:-})"
echo "  curriculum(enable=${RC_HUMAN_CURRICULUM_ENABLE:-}, mode_order=${RC_HUMAN_MODE_ORDER}, levels=${RC_HUMAN_LEVELS_PER_MODE:-20}, mix=${RC_HUMAN_MIX_CURRENT:-}/${RC_HUMAN_MIX_EASY_REPLAY:-}/${RC_HUMAN_MIX_MEDIUM_REPLAY:-}/${RC_HUMAN_MIX_RANDOM_REPLAY:-})"

case "${DRY_RUN:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        printf 'dry-run command:'
        printf ' %q' "${TRAIN_CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

exec "${TRAIN_CMD[@]}"
