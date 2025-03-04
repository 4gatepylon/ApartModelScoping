#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G

export HF_HOME="/om/user/dbaek/.cache"

python ../../lm-evaluation-harness/lm_eval --model hf \
    --model_args pretrained=./results/llama-3.2-1b-ft/,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results/

python ../../lm-evaluation-harness/lm_eval --model hf \
    --model_args pretrained=./results/llama-3.2-3b-ft/,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results/

python ../../lm-evaluation-harness/lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results/

python ../../lm-evaluation-harness/lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results/