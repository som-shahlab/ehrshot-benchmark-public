#!/bin/bash
#SBATCH --job-name=6_eval__helper
#SBATCH --output=logs/6_eval__helper_%A.out
#SBATCH --error=logs/6_eval__helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu,nigam-v100
#SBATCH --mem=360G
#SBATCH --cpus-per-task=30
#SBATCH --exclude=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

python3 ../6_eval.py \
    --path_to_output_dir $1 \
    --labeler $2 \
    --shot_strat $3 \
    --num_threads $4