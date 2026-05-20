#!/bin/sh
# ---------------------------------------------------------------------------
# Pure PPO training on the CircleTask + HybridModel (QuadPlane).
#
# This mirrors scripts/train_hover.sh:
#   - same PPO runner: scripts/train/train_F16sim.py
#   - same actor/critic network family as DAgger's PPOActorStudent
#   - GRU is enabled by default; do not pass --use-recurrent-policy because
#     config.py defines that flag as store_false.
#
# Circle-specific setup lives in:
#   envs/tasks/circle_task.py
#   envs/reward_functions/circle_reward.py
#   envs/configs/circle.yaml
#
# Run:
#   cd NeuralPlane_stable_V2
#   bash scripts/train_circle_rl.sh
# ---------------------------------------------------------------------------
env="Control"
scenario="circle"
model='HYBRID'
algo="ppo"
exp="circle_rl_gru"
seed=5
device="cuda:0"

# Optional warm start from a DAgger actor checkpoint:
# init_actor_ckpt="algorithms/dagger/checkpoints_circle/YYYYMMDD_HHMMSS/dagger_latest.pt"
init_actor_ckpt=""
PYTHON_BIN="${PYTHON_BIN:-python}"

for arg in "$@"; do
    if [ "$arg" = "--use-recurrent-policy" ]; then
        echo "Do not pass --use-recurrent-policy: config.py defines it as store_false and it would disable GRU." >&2
        exit 2
    fi
done

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
"${PYTHON_BIN}" scripts/train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 1024 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 8 --buffer-size 2000 --num-env-steps 1.5e9 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --ppo-epoch 12 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 2e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --activation-id 1 --gain 0.01 \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    ${init_actor_ckpt:+--init-actor-ckpt ${init_actor_ckpt}} \
    "$@"
