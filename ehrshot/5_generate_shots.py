"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/labeler/{shot_strat}_shots_data.json` containing:

Output:
    few_shots_dict = {
        sub_task_1: : {
            k_1: {
                replicate_1: {
                    "patient_ids_train_k": List[int] = patient_ids['train'][train_idxs_k].tolist(),
                    "patient_ids_val_k": List[int] = patient_ids['val'][val_idxs_k].tolist(),
                    "label_times_train_k": List[str] = [ label_time.isoformat() for label_time in label_times['train'][train_idxs_k] ], 
                    "label_times_val_k": List[str] = [ label_time.isoformat() for label_time in label_times['val'][val_idxs_k] ], 
                    'label_values_train_k': List[int] = y_train[train_idxs_k].tolist(),
                    'label_values_val_k': List[int] = y_val[val_idxs_k].tolist(),
                    "train_idxs": List[int] = train_idxs_k.tolist(),
                    "val_idxs": List[int] = val_idxs_k.tolist(),
                },
                ... 
            },
            ...
        },
        ...
    }
"""
import argparse
import collections
import datetime
import json
import os
from typing import Dict, List, Union
import meds
import numpy as np
from loguru import logger
from utils import (
    LABELING_FUNCTIONS, 
    CHEXPERT_LABELS, 
    SHOT_STRATS,
    convert_csv_labels_to_meds,
    get_labels_and_features,
    get_rel_path, 
    process_chexpert_labels, 
    convert_multiclass_to_binary_labels,
    split_labels
)
import datasets

def get_k_samples(y: List[int], k: int, max_k: int, is_preserve_prevalence: bool = False, seed=0) -> List[int]:
    """_summary_

    Args:
        y (List[int]): Ground truth labels
        k (int): number of samples per class
        max_k (int): largest size of k that we'll feed into model. This is needed to ensure that 
            we always sample a subset of the larger k when sampling data points for smaller k's, 
            to control for randomness and isolate out the effect of increasing k.
        is_preserve_prevalence (bool, optional): If TRUE, then preserve prevalence of each class in k-shot sample. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        List[int]: List of idxs in `y` that are included in this k-shot sample
    """    
    valid_idxs: List[int] = []
    if k == -1:
        # Return all samples
        valid_idxs = list(range(len(y)))
    else:
        # Do sampling
        classes = np.unique(y)
        for c in classes:
            # Get idxs corresponding to this class
            class_idxs = np.where(y == c)[0]
            # Get k random labels
            # Note that instead of directly sampling `k` random samples, we instead
            # sample `max_k` examples all at once, and then take the first `k` subset of them.
            # This ensures that `k = N` always contains a superset of the examples sampled
            # for `k = \hat{N}`, where `\hat{N} < N`.
            np.random.seed(seed)
            idxs = np.random.choice(class_idxs, size=min(len(class_idxs), max_k), replace=False)
            if max_k > len(class_idxs):
                # Fill rest of `k` with oversampling if not enough examples to fill `k` without replacement
                idxs = np.hstack([idxs, np.random.choice(class_idxs, size=max_k - len(class_idxs), replace=True)])
                logger.warning(f"Oversampling class {c} with replacement from {len(class_idxs)} -> {max_k} examples.")
            # If we want to preserve the prevalence of each class, then we need to adjust our sample size
            if is_preserve_prevalence:
                prev_k = max(1, int(len(classes) * k * len(class_idxs) / len(y)))
                idxs = idxs[:prev_k]
            else:
                idxs = idxs[:k]
            valid_idxs.extend([ int(x) for x in idxs ]) # need to cast to normal `int` for JSON serializability
    return valid_idxs

def generate_shots(k: int, 
                   max_k: int, 
                   y_train: List[int], 
                   y_val: List[int], 
                   patient_ids: Dict[str, np.ndarray], 
                   label_times: Dict[str, np.ndarray], 
                   seed: int = 0) -> Dict[str, List[Union[int, str]]]:
    train_idxs_k: List[int] = get_k_samples(y_train, k=k, max_k=max_k, seed=seed)
    val_idxs_k: List[int] = get_k_samples(y_val, k=k, max_k=max_k, seed=seed)

    shot_dict: Dict[str, List[Union[int, str]]] = {
        "patient_ids_train_k": patient_ids['train'][train_idxs_k].tolist(),
        "patient_ids_val_k": patient_ids['val'][val_idxs_k].tolist(),
        "label_times_train_k": [ label_time.astype(datetime.datetime).isoformat() for label_time in label_times['train'][train_idxs_k] ], 
        "label_times_val_k": [ label_time.astype(datetime.datetime).isoformat() for label_time in label_times['val'][val_idxs_k] ], 
        'label_values_train_k': y_train[train_idxs_k].tolist(),
        'label_values_val_k': y_val[val_idxs_k].tolist(),
        "train_idxs": train_idxs_k,
        "val_idxs": val_idxs_k,
    }
    
    return shot_dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate few-shot data for eval")
    parser.add_argument("--path_to_dataset", default=get_rel_path(__file__, "../assets/ehrshot-meds-stanford/"), type=str, help="Path to MEDS formatted version of EHRSHOT")
    parser.add_argument("--path_to_labels_dir", default=get_rel_path(__file__, "../assets/labels/"), type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_splits_csv", default=get_rel_path(__file__, "../assets/splits.csv"), type=str, help="Path to patient train/val/test split CSV")
    parser.add_argument("--labeler", required=True, type=str, help="Labeling function for which we will create k-shot samples.", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--shot_strat", type=str, choices=SHOT_STRATS.keys(), help="What type of X-shot evaluation we are interested in.", required=True )
    parser.add_argument("--n_replicates", type=int, help="Number of replicates to run for each `k`. Useful for creating std bars in plots", default=3, )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_dataset: str = os.path.join(args.path_to_dataset, 'data/*.parquet')
    labeler: str = args.labeler
    path_to_labels_dir: str = args.path_to_labels_dir
    path_to_splits_csv: str = args.path_to_splits_csv
    shot_strat: str = args.shot_strat
    n_replicates: int = args.n_replicates
    path_to_labels_csv: str = os.path.join(path_to_labels_dir, f"{labeler}_labels.csv")
    path_to_output_file: str = os.path.join(path_to_labels_dir, labeler, f"{shot_strat}_shots_data.json")
    
    assert os.path.exists(args.path_to_dataset), f"Path to dataset does not exist: {args.path_to_dataset}"
    assert os.path.exists(path_to_labels_csv), f"Path to labels CSV does not exist: {path_to_labels_csv}"
    assert os.path.exists(path_to_splits_csv), f"Path to splits CSV does not exist: {path_to_splits_csv}"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)

    # Few v. long shot
    if shot_strat in SHOT_STRATS:
        SHOTS: List[int] = SHOT_STRATS[shot_strat]
    else:
        raise ValueError(f"Invalid `shot_strat`: {shot_strat}")

    # Load labels for this task
    labels: List[meds.Label] = convert_csv_labels_to_meds(path_to_labels_csv)
    
    if labeler == "chexpert":
        # CheXpert is multilabel, convert to binary for EHRSHOT
        labels = process_chexpert_labels(labels)
    elif labeler.startswith('lab_'):
        # Lab values is multiclass, convert to binary for EHRSHOT
        labels = convert_multiclass_to_binary_labels(labels, threshold=1)

    # Train/val/test splits
    labels_split, label_values, label_times, patient_ids = split_labels(labels, path_to_splits_csv)
    logger.info(f"Train prevalence: {np.sum(label_values['train'] != 0) / label_values['train'].size}")
    logger.info(f"Val prevalence: {np.sum(label_values['val'] != 0) / label_values['val'].size}")
    logger.info(f"Test prevalence: {np.sum(label_values['test'] != 0) / label_values['test'].size}")

    if labeler == 'chexpert':
        # Multilabel -- create one task per class
        sub_tasks: List[str] = CHEXPERT_LABELS
    else:
        # Binary classification
        sub_tasks: List[str] = [labeler]
    
    # Create shots
    few_shots_dict: Dict[str, Dict] = collections.defaultdict(dict)
    for idx, sub_task in enumerate(sub_tasks):
        few_shots_dict[sub_task]: Dict[int, Dict[int, Dict]] = collections.defaultdict(dict)
        # Get ground truth labels
        if labeler == 'chexpert':
            y_train, y_val = label_values['train'][:, idx], label_values['val'][:, idx]
        else:
            y_train, y_val = label_values['train'], label_values['val']
        # Create a sample for each k, for each replicate
        for k in SHOTS:
            for replicate in range(n_replicates):
                if k == -1 and replicate > 0:
                    # Only need one copy of `all` dataset (for speed)
                    continue
                logger.critical(f"Label: {sub_task} | k: {k} | Replicate: {replicate}")
                shot_dict: Dict[str, List[Union[int, str]]] = generate_shots(k, 
                                                                            max_k=max(SHOTS), 
                                                                            y_train=y_train, 
                                                                            y_val=y_val, 
                                                                            patient_ids=patient_ids, 
                                                                            label_times=label_times, 
                                                                            seed=replicate)
                few_shots_dict[sub_task][k][replicate] = shot_dict
    
    # Save patients selected for each shot
    logger.info(f"Saving few shot data to: {path_to_output_file}")
    with open(path_to_output_file, 'w') as f:
        json.dump(few_shots_dict, f)
    logger.success(f"Done with {labeler}!")
    