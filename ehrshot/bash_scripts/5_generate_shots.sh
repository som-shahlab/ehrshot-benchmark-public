#!/bin/bash
#SBATCH --job-name=5_generate_shots
#SBATCH --output=logs/5_generate_shots_%A.out
#SBATCH --error=logs/5_generate_shots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 5 mins per labeler

labelers=(
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

for labeler in "${labelers[@]}"; do
    python3 ../5_generate_shots.py \
        --labeler ${labeler} \
        --n_replicates 5
done

echo "Done!"