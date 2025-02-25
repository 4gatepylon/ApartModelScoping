#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G

python ../../lm-evaluation-harness/lm_eval --model hf \
    --model_args pretrained=/om/user/dbaek/ApartModelScoping/FineTuning/results/checkpoint-458/,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results/
