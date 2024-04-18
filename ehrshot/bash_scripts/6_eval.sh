#!/bin/bash

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

path_to_output_dir='../../assets/outputs/'

labelers=(
    # "chexpert" # CheXpert first b/c slowest
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    # Labs take long time -- need more GB
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_anemia"
    "lab_hypoglycemia" # will OOM at 200G on `gpu` partition
)
num_threads=20

for labeler in "${labelers[@]}"; do
    sbatch 6__eval_helper.sh $path_to_output_dir ${labeler} $num_threads
done
