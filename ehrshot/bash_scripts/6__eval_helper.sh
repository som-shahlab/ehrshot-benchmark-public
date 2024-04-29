#!/bin/bash
#SBATCH --job-name=6_eval__helper
#SBATCH --output=logs/6_eval__helper_%A.out
#SBATCH --error=logs/6_eval__helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu
#SBATCH --mem=350G
#SBATCH --cpus-per-task=20
#SBATCH --exclude=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

conda activate /home/ehrshot/ehrshot_env

python3 ../6_eval.py \
    --path_to_output_dir $1 \
    --labeler $2 \
    --num_threads $3
