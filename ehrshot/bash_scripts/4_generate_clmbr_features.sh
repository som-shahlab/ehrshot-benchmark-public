#!/bin/bash
#SBATCH --job-name=5_generate_clmbr_features
#SBATCH --output=logs/5_generate_clmbr_features_%A.out
#SBATCH --error=logs/5_generate_clmbr_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=20

/share/sw/open/anaconda/3.10.2/bin/conda activate /home/ehrshot/ehrshot_env
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 0,999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 1000,1999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 2000,2999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 3000,3999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 4000,4999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 5000,5999
python3 ../4_generate_clmbr_features.py --num_threads 10 --patient_range 6000,6732
# Time to run: XXXX mins

# python3 ../5_generate_clmbr_features.py \
#     --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    # --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
#     --path_to_models_dir ../../EHRSHOT_ASSETS/models \
#     --model motor \
#     --is_force_refresh