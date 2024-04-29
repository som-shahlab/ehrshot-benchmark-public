#!/bin/bash
#SBATCH --job-name=7_make_figures
#SBATCH --output=logs/7_make_figures_%A.out
#SBATCH --error=logs/7_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

python3 ../7_make_figures.py \
    --model_heads "[('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf'), ('clmbr', 'lr_lbfgs'), ('clmbr_custom_batch', 'lr_lbfgs')]" 

# --model_heads "[('clmbr', 'lr_femr'), ('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf'), ('gpt2-base', 'lr_lbfgs')]" \
    # --model_heads "[('gpt2-base', 'lr_lbfgs'), ('gpt2-base', 'gbm'), ('gpt2-base', 'rf')]" \
    # --model_heads "[('gpt2-base-v8_chunk:last_embed:last', 'lr_lbfgs'), ('gpt2-base-v8_chunk:last_embed:last', 'gbm'), ('gpt2-base-v8_chunk:last_embed:last', 'rf')]" \
