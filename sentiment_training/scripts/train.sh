#!/bin/bash

for seed in 1 2 3
do
    model_name="checkpoints/${1}/sentiment_seed=${seed}"
    mkdir $model_name
    python main.py --model_type $1 --lr 5e-6 --seed $seed --epochs 5 \
        --batchsize 32 --save $model_name --disable_tqdm
done
