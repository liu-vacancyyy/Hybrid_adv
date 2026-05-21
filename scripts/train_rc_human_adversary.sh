#!/bin/sh
# ---------------------------------------------------------------------------
# PPO adversary training for rc_human.
#
# The victim policy is fixed.  Sensor noise and Dryden random wind are disabled.
# By default the adversary generates:
#   - command space: vx/vy/vz/yaw-rate commands
#   - observation space: normalized policy observation perturbations
#   - wind space: N/E/D gust commands within rc_human wind ranges
# Add --adv-use-random-command to keep rc_human's original random command
# generator and attack only observation/wind.
#
# Run:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/train_rc_human_adversary.sh
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
VICTIM_CKPT="${VICTIM_CKPT:-/home/a/demo/Hybrid_adv/scripts/runs/2026-05-20_23-15-17_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_first2/episode_880/actor_latest.ckpt}"
EXP="${RC_HUMAN_ADV_EXP_NAME:-rc_human_adv_cmd_obs_wind_ep880}"

"${PYTHON_BIN}" scripts/adversarial/train_rc_human_adversary.py \
    --victim-ckpt "${VICTIM_CKPT}" \
    --scenario-name rc_human --model-name HYBRID_NEW --experiment-name "${EXP}" \
    --seed 17 --device "${DEVICE}" --cuda \
    --n-rollout-threads 1024 --buffer-size 256 --num-env-steps 2.0e7 \
    --max-iterations 1000 \
    --log-interval 1 --save-interval 10 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 \
    --ppo-epoch 8 --num-mini-batch 8 --clip-param 0.2 \
    --entropy-coef 2e-3 --max-grad-norm 1.0 \
    --hidden-size "128 128" --act-hidden-size "128 128" \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --adv-command-frac 0.18 --adv-obs-frac 0.6 --adv-wind-frac 0.5 \
    --adv-command-alpha 0.20 --adv-obs-alpha 0.25 --adv-wind-alpha 0.15 \
    --adv-init-log-std -1.2 \
    --adv-alive-penalty 0.01 --adv-policy-reward-weight 0.05 \
    --adv-w-vel-error 2.0 --adv-w-yaw-error 1.0 \
    --adv-w-attitude 5.0 --adv-w-omega 0.8 --adv-w-force-margin 0.2 \
    --adv-bad-done-bonus 50.0 \
    --adv-linf-penalty 0.20 --adv-smooth-penalty 0.10 --adv-raw-excess-penalty 0.20 \
    --adv-obs-energy-window 50 --adv-obs-energy-budget 50.0 --adv-obs-energy-penalty 0.05 \
    "$@"
