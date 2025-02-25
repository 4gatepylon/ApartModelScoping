#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G

python fine-tuning.py