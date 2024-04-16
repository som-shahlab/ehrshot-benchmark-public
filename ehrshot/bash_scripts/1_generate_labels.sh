#!/bin/bash
#SBATCH --job-name=1_generate_labels
#SBATCH --output=logs/1_generate_labels_%A.out
#SBATCH --error=logs/1_generate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=22

# Time to run: 6 mins

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
    "chexpert"
)

for labeler in "${labelers[@]}"
do
    python3 ../1_generate_labels.py \
        --labeler ${labeler}
done