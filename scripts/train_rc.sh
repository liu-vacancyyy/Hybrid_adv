#!/bin/sh
env="Control"
scenario="rc"
model='HYBRID'
algo="ppo"
exp="v1"
seed=5
device="cuda:0"

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 1024 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1.5e9 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 16 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --activation-id 1 --gain 0.01 \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8
