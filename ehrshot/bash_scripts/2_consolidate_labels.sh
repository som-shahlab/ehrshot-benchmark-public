#!/bin/bash
#SBATCH --job-name=2_consolidate_labels
#SBATCH --output=logs/2_consolidate_labels_%A.out
#SBATCH --error=logs/2_consolidate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5

# Time to run: 10 secs

python3 ../2_consolidate_labels.py