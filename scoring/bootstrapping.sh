#! /bin/bash

python scoring/bootstrapping.py \
    --dataset "TheFinAI/MED_SYN1_CLEVELAND_train" \
    --n_samples 5 \
    --sample_size 100 \
    --seed 42 \
    --token $HF_TOKEN \
    --private
    