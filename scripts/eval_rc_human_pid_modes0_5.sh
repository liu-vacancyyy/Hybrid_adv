#!/usr/bin/env bash
# Evaluate the LearningToFly-style PID baseline on rc_human vx/vy/vz yaw-hold
# curriculum levels 0..119 (mode0..5), without starting PPO training.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-230}"

export RC_HUMAN_MODE_ORDER="${RC_HUMAN_MODE_ORDER:-0 1 2 3 4 5}"
export RC_HUMAN_MAX_MODE_SLOTS="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
export NO_WIND_CONFIG_NAME="${NO_WIND_CONFIG_NAME:-rc_human_vxvyvz_yawhold_currheading_modes0_1_2_3_4_5_nowind_runtime}"

# Generate the runtime no-wind YAML used by this task. DRY_RUN exits before PPO.
DRY_RUN=1 PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/train_rc_human_vxvyvz_yawhold_modes0_5_nowind_from_scratch.sh" >/tmp/rc_human_pid_config_dryrun.log

CONFIG_NAME="${NO_WIND_CONFIG_NAME}"
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/scripts/runs/pid_rc_human_modes0_5_${RUN_ID}}"
EPISODES_PER_LEVEL="${EPISODES_PER_LEVEL:-3}"
MAX_STEPS="${MAX_STEPS:-1500}"

CUDA_ARGS=(--no-cuda)
case "${USE_CUDA:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        CUDA_ARGS=()
        ;;
esac

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval/eval_rc_human_pid_curriculum_levels_parallel.py" \
    --output-dir "${OUTPUT_DIR}" \
    --config-name "${CONFIG_NAME}" \
    --model-name HYBRID_NEW \
    --mode-order "${RC_HUMAN_MODE_ORDER}" \
    --device "${DEVICE}" \
    "${CUDA_ARGS[@]}" \
    --seed "${SEED}" \
    --disable-tracking-bad-done \
    --episodes-per-level "${EPISODES_PER_LEVEL}" \
    --min-level 0 \
    --max-level 119 \
    --max-steps "${MAX_STEPS}" \
    --xy-kp "${PID_XY_KP:-0.75}" \
    --xy-ki "${PID_XY_KI:-0.06}" \
    --xy-kd "${PID_XY_KD:-0.010}" \
    --xy-imax "${PID_XY_IMAX:-0.25}" \
    --z-kp "${PID_Z_KP:-1.6}" \
    --z-ki "${PID_Z_KI:-0.10}" \
    --z-kd "${PID_Z_KD:-0.006}" \
    --z-imax "${PID_Z_IMAX:-0.25}" \
    --max-tilt-deg "${PID_MAX_TILT_DEG:-9}" \
    --max-horiz-accel "${PID_MAX_HORIZ_ACCEL:-1.3}" \
    --max-z-throttle-corr "${PID_MAX_Z_THROTTLE_CORR:-0.06}" \
    --yaw-p "${PID_YAW_P:-1.2}" \
    --max-yaw-rate "${PID_MAX_YAW_RATE:-0.45}" \
    --side-damp-p "${PID_SIDE_DAMP_P:-0.25}" \
    --head-max-scaled "${PID_HEAD_MAX_SCALED:-0.30}" \
    --alt-floor "${PID_ALT_FLOOR:-0.5}" \
    --alt-floor-buffer "${PID_ALT_FLOOR_BUFFER:-0.8}" \
    --alt-floor-vz-bias "${PID_ALT_FLOOR_VZ_BIAS:-0.6}" \
    "$@"

echo "PID evaluation saved to: ${OUTPUT_DIR}"
