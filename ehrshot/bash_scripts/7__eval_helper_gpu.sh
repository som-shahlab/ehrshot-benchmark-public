#!/bin/bash
#SBATCH --job-name=7__eval_helper_gpu
#SBATCH --output=logs/7__eval_helper_gpu_%A.out
#SBATCH --error=logs/7__eval_helper_gpu_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,nigam-v100,nigam-a100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

# For running multiple labeling functions in parallel per node:
CUDA_VISIBLE_DEVICES=0 && python3 ../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --num_threads 5 \
    --labeling_function $8 \
    --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen,finetune_frozen-logregfirst &

CUDA_VISIBLE_DEVICES=1 && python3 ../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --num_threads 5 \
    --labeling_function $9 \
    --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen,finetune_frozen-logregfirst &

CUDA_VISIBLE_DEVICES=2 && python3 ../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --num_threads 5 \
    --labeling_function ${10} \
    --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen,finetune_frozen-logregfirst &

wait

# For running one job per node:
#
# python3 ../7_eval_finetune.py \
#     --path_to_database $1 \
#     --path_to_labels_dir $2 \
#     --path_to_features_dir $3 \
#     --path_to_split_csv $4 \
#     --path_to_output_dir $5 \
#     --shot_strat $6 \
#     --num_threads 3 \
#     --labeling_function $8 \
#     --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen,finetune_frozen-logregfirst


# For debugging:
#
# python3 ../7_eval_finetune.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/features' \
#     --path_to_split_csv '../../EHRSHOT_ASSETS/splits/person_id_map.csv' \
#     --path_to_output_dir '../../EHRSHOT_ASSETS/results' \
#     --labeling_function guo_los \
#     --shot_strat all \
#     --num_threads 1 \
#     --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen,finetune_frozen-logregfirst


# For debugging:
#
# python3 ../7_eval_finetune.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/features' \
#     --path_to_split_csv '../../EHRSHOT_ASSETS/splits/person_id_map.csv' \
#     --path_to_output_dir '../../EHRSHOT_ASSETS/results' \
#     --labeling_function guo_los \
#     --shot_strat all \
#     --num_threads 1 \
#     --heads finetune_frozen-logregfirst

