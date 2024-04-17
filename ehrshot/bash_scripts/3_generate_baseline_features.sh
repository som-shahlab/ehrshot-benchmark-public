#!/bin/bash
#SBATCH --job-name=3_generate_baseline_features
#SBATCH --output=logs/3_generate_baseline_features_%A.out
#SBATCH --error=logs/3_generate_baseline_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 10 mins

python3 ../3_generate_baseline_features.py \
     --is_ontology_expansion \
     --num_threads 10