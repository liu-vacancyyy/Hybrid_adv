#!/bin/sh
env="Control"
scenario="rc"
model='HYBRID'
algo="ppo"
exp="v1"
seed=10
device="cuda:0"

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python eval/eval_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-eval-threads 1 --n-rollout-threads 1 --cuda \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1