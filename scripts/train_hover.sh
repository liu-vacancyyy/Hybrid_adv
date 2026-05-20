#!/bin/sh
# ---------------------------------------------------------------------------
# Pure PPO training on the HoverTask + HybridModel (QuadPlane).
#
# Network architecture is intentionally identical to the DAgger student
# (algorithms.dagger.policy.PPOActorStudent), so any checkpoint produced
# here can be loaded by render_dagger_hover.py / render_compare_hover.py
# and directly compared against the DAgger-trained one.
#
# GRU is enabled (config.py: --use-recurrent-policy is on by default; this
# script keeps that default and forces the matching recurrent hyper-params).
#
# Run:
#   cd NeuralPlane_stable_V2/scripts
#   bash train_hover.sh
# ---------------------------------------------------------------------------
env="Control"
scenario="hover"
model='HYBRID'
algo="ppo"
exp="v1"
seed=5
device="cuda:0"
# Optional warm start from a DAgger actor checkpoint.
# If running this script from NeuralPlane_stable_V2/scripts, use a path like:
# init_actor_ckpt="../algorithms/dagger/checkpoints/20260429_230311/dagger_iter1789.pt"
init_actor_ckpt=""

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 892 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1.5e9 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 16 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --activation-id 1 --gain 0.01 \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    ${init_actor_ckpt:+--init-actor-ckpt ${init_actor_ckpt}} \
    --freeze-head-motor
