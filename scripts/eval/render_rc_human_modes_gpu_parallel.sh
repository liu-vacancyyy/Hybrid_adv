#!/usr/bin/env bash
# Vectorized rc_human render/eval for selected modes.
#
# Defaults evaluate the latest rate-limited run's episode_620 on the hard end
# levels of modes 0/1/2: level 19, 39, 59.  CUDA is required by default because
# this script is intended for fast render sweeps; set ALLOW_CPU_FALLBACK=1 only
# when running in an environment without a visible GPU.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/a/anaconda3/envs/Neuralplane/bin/python}"
RUN_DIR="${RUN_DIR:-scripts/runs/2026-06-15_19-30-28_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_px4_rate0_05_from_scratch_modes012534}"
EPISODE="${EPISODE:-620}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/episode_${EPISODE}/actor_latest.ckpt}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-230}"
EPISODES_PER_MODE="${EPISODES_PER_MODE:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
MODEL_NAME="${MODEL_NAME:-HYBRID_NEW}"
MODE_ORDER="${MODE_ORDER:-0 1 2 5 3 4}"
MODE_LEVELS="${MODE_LEVELS:-0:19 1:39 2:59}"
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-0}"

CUDA_AVAILABLE="$("${PYTHON_BIN}" - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)"

DEVICE_ARGS=(--device "${DEVICE}")
DEVICE_TAG="gpu"
if [[ "${CUDA_AVAILABLE}" != "1" ]]; then
    if [[ "${ALLOW_CPU_FALLBACK}" == "1" ]]; then
        echo "CUDA is not available; running vectorized CPU fallback." >&2
        DEVICE_ARGS=(--no-cuda)
        DEVICE_TAG="cpu_fallback"
    else
        echo "CUDA is not available. Set ALLOW_CPU_FALLBACK=1 to run vectorized CPU fallback." >&2
        exit 2
    fi
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi

echo "rc_human parallel render"
echo "  run_dir: ${RUN_DIR}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  device: ${DEVICE_TAG} ${DEVICE_ARGS[*]}"
echo "  mode_levels: ${MODE_LEVELS}"
echo "  episodes_per_mode: ${EPISODES_PER_MODE}"

POST_ARGS=()
for pair in ${MODE_LEVELS}; do
    mode="${pair%%:*}"
    level="${pair##*:}"
    out_dir="${RUN_DIR}/render_episode_${EPISODE}_mode${mode}_level${level}_${EPISODES_PER_MODE}_gpu_parallel"
    echo "[eval] mode=${mode} level=${level} -> ${out_dir}"
    "${PYTHON_BIN}" scripts/eval/eval_rc_human_curriculum_levels_parallel.py \
        --ckpt-path "${CKPT_PATH}" \
        --output-dir "${out_dir}" \
        --episodes-per-level "${EPISODES_PER_MODE}" \
        --min-level "${level}" \
        --max-level "${level}" \
        --max-steps "${MAX_STEPS}" \
        --model-name "${MODEL_NAME}" \
        --mode-order "${MODE_ORDER}" \
        "${DEVICE_ARGS[@]}"
    POST_ARGS+=(--mode-level "${pair}")
done

combined_dir="${RUN_DIR}/render_episode_${EPISODE}_modes_gpu_parallel_${EPISODES_PER_MODE}eps"
echo "[plot] ${combined_dir}"
"${PYTHON_BIN}" scripts/eval/plot_rc_human_mode_eval_curves.py \
    --run-dir "${RUN_DIR}" \
    --episode "${EPISODE}" \
    --episodes-per-mode "${EPISODES_PER_MODE}" \
    --output-dir "${combined_dir}" \
    "${POST_ARGS[@]}"

echo "[saved] ${combined_dir}"
