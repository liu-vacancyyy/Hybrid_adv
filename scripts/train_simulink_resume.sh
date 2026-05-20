#!/bin/sh
# Resume training from the last checkpoint of a previous run.
# Change MODEL_DIR to point to the episode directory you want to resume from.

env="Control"
scenario="simulink"
model='Simulink'
algo="ppo"
exp="v1"
seed=0
device="cuda:0"

# ====== Resume from this checkpoint ======
MODEL_DIR="/home/a/demo/NeuralPlane_stable_V2/scripts/runs/2026-04-17_10-16-11_Control_simulink_Simulink_ppo_v1/episode_25"

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Resuming from: ${MODEL_DIR}"
python train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 2048 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1.5e9 \
    --lr 1e-4 --gamma 0.99 --ppo-epoch 16 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "64 64" --act-hidden-size "64 64" --recurrent-hidden-size 64 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --model-dir "${MODEL_DIR}"
