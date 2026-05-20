#!/bin/sh
env="Control"
scenario="rc"
model='HYBRID'
seed_train=5
seed_eval=13
device="cuda:0"
episodes=50

echo "env=${env}, scenario=${scenario}, model=${model}, train_seed=${seed_train}, eval_seed=${seed_eval}, episodes=${episodes}"

python eval/eval_rc_policy_metrics.py \
  --env-name ${env} --scenario-name ${scenario} --model-name ${model} \
  --train-seed ${seed_train} --eval-seed ${seed_eval} --episodes ${episodes} \
  --device ${device} --cuda
