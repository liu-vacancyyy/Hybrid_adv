#!/bin/sh
# ---------------------------------------------------------------------------
# PPO-GRU training for the human-like RC command task.
#
# Task:
#   envs/tasks/rc_human_task.py
#   envs/reward_functions/rc_human_reward.py
#   envs/configs/rc_human.yaml
#
# Key settings:
#   - 30 s episode: max_steps/buffer-size = 1500 at dt=0.02
#   - 1024 rollout envs
#   - HYBRID_NEW model so env Dryden wind is applied
#   - GRU is enabled by default in config.py. Do not pass
#     --use-recurrent-policy because that flag disables it.
#
# Run:
#   cd /home/a/demo/NeuralPlane_stable_V2
#   bash scripts/train_rc_human_rl.sh
# ---------------------------------------------------------------------------
env="Control"
scenario="rc_human"
model="HYBRID_NEW"
algo="ppo"
max_mode_slots="${RC_HUMAN_MAX_MODE_SLOTS:-6}"
exp="${RC_HUMAN_EXP_NAME:-rc_human_rl_gru_wind_modes${max_mode_slots}}"
seed=7
device="cuda:0"
PYTHON_BIN="${PYTHON_BIN:-python}"

for arg in "$@"; do
    if [ "$arg" = "--use-recurrent-policy" ]; then
        echo "Do not pass --use-recurrent-policy: config.py defines it as store_false and it would disable GRU." >&2
        exit 2
    fi
done

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}, max_mode_slots is ${max_mode_slots}"
"${PYTHON_BIN}" scripts/train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 1024 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 8 --buffer-size 1500 --num-env-steps 1.5e9 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --ppo-epoch 12 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 2e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --activation-id 1 --gain 0.01 \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    "$@"
