#!/bin/bash
#SBATCH --job-name=6_generate_shots
#SBATCH --output=logs/6_generate_shots_%A.out
#SBATCH --error=logs/6_generate_shots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 2 mins

labeling_functions=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    # "chexpert" # TODO
)
shot_strats=("all")

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
    python3 ../5_generate_shots.py \
        --labeler ${labeling_function} \
        --shot_strat ${shot_strat} \
        --n_replicates 5
    done
done

echo "Done!" >&2