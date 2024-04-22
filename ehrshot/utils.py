import ast
import pickle
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import meds
import pandas as pd
import numpy as np
import femr.featurizers
import femr
from loguru import logger
from tqdm import tqdm

# SPLITS
SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: int = 70
SPLIT_VAL_CUTOFF: int = 85

# Types of base models to test
MODEL_2_NAME: Dict[str, str] = {
    'count' : 'Count-based',
    'clmbr' : 'CLMBR',
    # 'gpt2-base' : 'GPT2-base',
    # 'gpt2-medium' : 'GP2-medium',
    # 'gpt2-large' : 'GP2-large',
    # 'bert-base' : 'BERT-base',
    # 'gpt2-base-v8_chunk:last_embed:last' : 'GPT2-base',
    # 'bert-base-v8_chunk:last_embed:last' : 'BERT-base',
}
BASE_MODELS: List[str] = list(MODEL_2_NAME.keys())

# Map each base model to a set of heads to test
BASE_MODEL_2_HEADS: Dict[str, List[str]] = {
    'count' : ['gbm', 'lr_lbfgs', 'rf', ], 
    'clmbr' : ['lr_lbfgs', 'gbm', 'rf', ],
    'gpt2-base-v8_chunk:last_embed:last' : ['gbm', 'lr_lbfgs', 'rf', ], 
    'bert-base-v8_chunk:last_embed:last' : ['gbm', 'lr_lbfgs', 'rf', ], 
}
HEAD_2_NAME: Dict[str, str] = {
    'gbm' : 'GBM',
    'lr_lbfgs' : 'LR',
    'lr_femr' : 'LR',
    'lr_newton-cg' : 'LR',
    'protonet' : 'ProtoNet',
    'rf' : 'Random Forest',
}

# Labeling functions
LABELERS: List[str] = [
    # Guo et al. 2023
    "guo_los",
    "guo_readmission",
    "guo_icu",
    # New diagnosis
    'new_pancan',
    'new_celiac',
    'new_lupus',
    'new_acutemi',
    'new_hypertension',
    'new_hyperlipidemia',
    # Instant lab values
    "lab_thrombocytopenia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_anemia",
    # Custom tasks
    "chexpert"
]

LABELER_2_PAPER_NAME = {
    "guo_los": "Long LOS",
    "guo_readmission": "30-day Readmission",
    "guo_icu": "ICU Admission",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    "new_hypertension": "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    "new_pancan": "Pancreatic Cancer",
    "new_celiac": "Celiac",
    "new_lupus": "Lupus",
    "new_acutemi": "Acute MI",
    "chexpert": "Chest X-ray Findings"
}

TASK_GROUP_2_PAPER_NAME = {
    "operational_outcomes": "Operational Outcomes",
    "lab_values": "Anticipating Lab Test Results",
    "new_diagnoses": "Assignment of New Diagnoses",
    "chexpert": "Anticipating Chest X-ray Findings",
}

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TASK_GROUP_2_LABELER = {
    "operational_outcomes": [
        "guo_los",
        "guo_readmission",
        "guo_icu"
    ],
    "lab_values": [
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ],
    "new_diagnoses": [
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi"
    ],
    "chexpert": [
        "chexpert"
    ]
}

# Hyperparameter search
XGB_PARAMS = {
    'max_depth': [3, 6, -1],
    'learning_rate': [0.02, 0.1, 0.5],
    'num_leaves' : [10, 25, 100],
}
LR_PARAMS = {
    "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
    "penalty": ['l2'],
    'max_iter' : [5000],
}
RF_PARAMS = {
    'n_estimators': [10, 20, 50, 100, 300],
    'max_depth' : [3, 5, 10, 20, 50],
}

# Few shot settings
SHOTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1]

# Plotting
SCORE_MODEL_HEAD_2_COLOR = {
    'auroc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'gbm' : 'aqua',
            'lr_lbfgs' : 'tab:blue',
            'rf' : 'darkblue',
        },
    },
    'auprc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'gbm' : 'aqua',
            'lr_lbfgs' : 'tab:blue',
            'rf' : 'darkblue',
        },
    },
}

def map_patient_id_to_split(path_to_split_csv: str) -> Dict[str, np.ndarray]:
    df_split = pd.read_csv(path_to_split_csv)
    return {
        'train' : np.array(df_split[df_split['split'] == 'train']['omop_person_id'].values),
        'val' : np.array(df_split[df_split['split'] == 'val']['omop_person_id'].values),
        'test' : np.array(df_split[df_split['split'] == 'test']['omop_person_id'].values),
    }

def split_labels(labels: List[meds.Label], path_to_split_csv: str) -> Tuple:
    """Split labels into train/val/test sets."""
    patient_id_2_split: Dict[str, np.ndarray] = map_patient_id_to_split(path_to_split_csv)
    labels_split: Dict[str, meds.Label] = {
        'train' : [ label for label in labels if label['patient_id'] in patient_id_2_split['train'] ],
        'val' : [ label for label in labels if label['patient_id'] in patient_id_2_split['val'] ],
        'test' : [ label for label in labels if label['patient_id'] in patient_id_2_split['test'] ],
    }
    label_values: Dict[str, np.ndarray] = {
        'train': np.array([ 1 if x['boolean_value'] else 0 for x in labels_split['train'] ] ),
        'val': np.array([ 1 if x['boolean_value'] else 0 for x in labels_split['val'] ]),
        'test': np.array([ 1 if x['boolean_value'] else 0 for x in labels_split['test'] ]),
    }
    patient_ids: Dict[str, np.ndarray] = {
        'train': np.array([ x['patient_id'] for x in labels_split['train'] ]),
        'val': np.array([ x['patient_id'] for x in labels_split['val'] ]),
        'test': np.array([ x['patient_id'] for x in labels_split['test'] ]),
    }
    label_times: Dict[str, np.ndarray] = {
        'train': np.array([ x['prediction_time'].to_numpy() if not isinstance(x['prediction_time'], np.datetime64) else x['prediction_time'] for x in labels_split['train'] ]),
        'val': np.array([ x['prediction_time'].to_numpy() if not isinstance(x['prediction_time'], np.datetime64) else x['prediction_time'] for x in labels_split['val'] ]),
        'test': np.array([ x['prediction_time'].to_numpy() if not isinstance(x['prediction_time'], np.datetime64) else x['prediction_time'] for x in labels_split['test'] ]),
    }
    return labels_split, label_values, label_times, patient_ids


def join_labels_clmbr(features: Dict[str, np.ndarray], labels: List[meds.Label]) -> Dict[str, np.ndarray]:
    """CLMBR returns feature['times'] <= label['time'], so we take max(feature_time | feature_time <= label time)"""
    # Sort arrays by (1) patient ID and (2) label time
    ## Labels
    labels = sorted(labels, key=lambda x: (x["patient_id"], x["prediction_time"]))
    
    ## Feats
    sort_order: np.ndarray = np.lexsort((features['feature_times'], features['patient_ids'])) # Note: Last col is primary sort key
    patient_ids = features['patient_ids'][sort_order]
    times = features['feature_times'][sort_order]
    feats = features['features'][sort_order]

    current_feature_idx: int = 0
    feature_indices: List[int] = []
    for label in tqdm(labels):
        while True:
            assert patient_ids[current_feature_idx] <= label['patient_id'], f"No features found for patient_id={label['patient_id']} | time={label['prediction_time']}"
            if patient_ids[current_feature_idx] < label['patient_id']:
                # Continue until we reach this patient
                current_feature_idx += 1
                continue

            if (
                current_feature_idx + 1 >= len(patient_ids) # reached last feature
                or patient_ids[current_feature_idx + 1] != label['patient_id'] # reached last feature for patient
                or times[current_feature_idx + 1] > label['prediction_time'] # past label time
            ): 
                # The next feature is invalid for this label (b/c either: we hit the last feature possible, we past the label's time with the next feature, or we've hit the last feature for this patient), so save this feature
                assert label['patient_id'] == patient_ids[current_feature_idx], f"No features found for patient_id={label['patient_id']} | time={label['prediction_time']}"
                assert times[current_feature_idx] <= label['prediction_time'], f"No features found for patient_id={label['patient_id']} | time={label['prediction_time']}"
                feature_indices.append(current_feature_idx)
                break

            current_feature_idx += 1
    
    return {
        "patient_ids": patient_ids[feature_indices],
        "times": times[feature_indices],
        "features": feats[feature_indices, :],
    }

def join_labels_for_baseline(features: Dict[str, np.ndarray], labels: List[meds.Label]) -> Dict[str, np.ndarray]:
    """Baseline returns feature['times'] == label['time'], so we expect exact matches"""
    # Sort arrays by (1) patient ID and (2) label time
    ## Labels
    labels = sorted(labels, key=lambda x: (x["patient_id"], x["prediction_time"]))
    
    ## Feats
    sort_order: np.ndarray = np.lexsort((features['feature_times'], features['patient_ids'])) # Note: Last col is primary sort key
    patient_ids = features['patient_ids'][sort_order]
    times = features['feature_times'][sort_order]
    feats = features['features'][sort_order]
    
    current_feature_idx: int = 0
    feature_indices: List[int] = []
    for label in tqdm(labels):
        while True:
            if (
                patient_ids[current_feature_idx] == label['patient_id']
                and times[current_feature_idx] == label['prediction_time']
            ):
                feature_indices.append(current_feature_idx)
                break
            else:
                if (
                    patient_ids[current_feature_idx] > label['patient_id']
                ):
                    # We have no more features for this patient, but no features were 
                    # found that match this label. Thus, raise Exception
                    raise ValueError(f"No features found for patient_id={label['patient_id']} | time={label['prediction_time']}")
            current_feature_idx += 1
            
    return {
        "patient_ids": patient_ids[feature_indices],
        "times": times[feature_indices],
        "features": feats[feature_indices, :],
    }

def get_labels_and_features(labels: List[meds.Label], value_type: str = 'boolean_value', path_to_features_dir: Optional[str] = None) -> Union[List[meds.Label], Tuple]:
    """Given a path to a directory containing labels and features as well as a LabeledPatients object, returns
        the labels and features for each patient. Note that this function is more complex b/c we need to align
        the labels with their corresponding features based on their prediction times."""
    # Set label time to np.datetime64
    for idx in range(len(labels)):
        labels[idx]['prediction_time'] = labels[idx]['prediction_time'].to_numpy()

    # Sort arrays by (1) patient ID and (2) label time
    labels = sorted(labels, key=lambda x: (x['patient_id'], x['prediction_time']))
    label_patient_ids = np.array([ x['patient_id'] for x in labels ])
    label_times = np.array([ x['prediction_time'] for x in labels ])
    if value_type == 'boolean_value':
        label_values = np.array([ 1 if x[value_type] else 0 for x in labels ])
    else:
        label_values = np.array([ x[value_type] for x in labels ])
        
    # Just return labels, ignore features
    if path_to_features_dir is None:
        return labels

    # Go through every featurization we've created (e.g. count, clmbr)
    # and align the label times with the featurization times
    featurizations: Dict[str, np.ndarray] = {}
    for model in BASE_MODELS:
        if model == 'count':
            # TODO - remove
            path_to_feats_file: str = os.path.join(path_to_features_dir, 'count_features__ont_exp_False__pruned.pkl')
        else:
            path_to_feats_file: str = os.path.join(path_to_features_dir, f'{model}_features.pkl')
        assert os.path.exists(path_to_feats_file), f'Path to file containing `{model}` features does not exist at this path: {path_to_feats_file}. Maybe you forgot to run `generate_features.py` first?'

        with open(path_to_feats_file, 'rb') as f:
            # Load data and do type checking
            features: Dict[str, np.ndarray] = pickle.load(f)

            # Align label times with feature times
            if model == 'count':
                joined_features: Dict[str, np.ndarray] = join_labels_for_baseline(features, labels)
                assert np.all(joined_features['patient_ids'] == label_patient_ids)
                assert np.all(joined_features['times'] == label_times)
            elif model == 'clmbr':
                joined_features: Dict[str, np.ndarray] = join_labels_clmbr(features, labels)
                assert np.all(joined_features['patient_ids'] == label_patient_ids)
            else:
                raise ValueError(f"Unrecognized model: {model}")

            # Save featurizations for this model
            featurizations[model] = joined_features['features']
    
    return label_patient_ids, label_values, label_times, featurizations

def process_chexpert_meds_labels(labels: List[meds.Label]) -> List[meds.Label]:
    for label in labels:
        label_str = bin(label['categorical_value'])[2:]
        rem_bin = 14 - len(label_str)
        label_str = "0"*rem_bin + label_str
        label_list = [*label_str]
        label_list = [int(label) for label in label_list]
        label['boolean_value'] = label_list # ! Hacky, but MEDS doesn't support list values
    return labels

def convert_multiclass_to_binary_meds_labels(labels: List[meds.Label], threshold: int = 1) -> List[meds.Label]:
    for label in labels:
        label['boolean_value'] = label['integer_value'] >= threshold
        del label['integer_value']
    return labels

def process_chexpert_ndarray_labels(labels: np.ndarray) -> np.ndarray:
    # TODO
    raise ValueError("NOT IMPLEMENTED")
    # for label in labels:
    #     label_str = bin(label['categorical_value'])[2:]
    #     rem_bin = 14 - len(label_str)
    #     label_str = "0"*rem_bin + label_str
    #     label_list = [*label_str]
    #     label_list = [int(label) for label in label_list]
    #     label['boolean_value'] = label_list # ! Hacky, but MEDS doesn't support list values
    # return labels

def convert_multiclass_to_binary_ndarray_labels(labels: np.ndarray, threshold: int = 1) -> np.ndarray:
    return (labels >= threshold).astype(int)

def check_file_existence_and_handle_force_refresh(path_to_file_or_dir: str, is_force_refresh: bool):
    """Checks if file/folder exists. If it does, deletes it if `is_force_refresh` is True."""
    if is_force_refresh:
        if os.path.exists(path_to_file_or_dir):
            if os.path.isdir(path_to_file_or_dir):
                logger.critical(f"Deleting existing directory at `{path_to_file_or_dir}`")
                os.system(f"rm -r {path_to_file_or_dir}")
            else:
                logger.critical(f"Deleting existing file at `{path_to_file_or_dir}`")
                os.system(f"rm {path_to_file_or_dir}")
    else:
        if os.path.exists(path_to_file_or_dir):
            if os.path.isdir(path_to_file_or_dir):
                raise ValueError(f"Error -- Directory already exists at `{path_to_file_or_dir}`. Please delete it and try again.")
            else:
                raise ValueError(f"Error -- File already exists at `{path_to_file_or_dir}`. Please delete it and try again.")
    if os.path.isdir(path_to_file_or_dir):
        os.makedirs(path_to_file_or_dir, exist_ok=True)

def type_tuple_list(s):
    """For parsing List[Tuple] from command line using `argparse`"""
    try:
        # Convert the string representation of list of tuples into actual list of tuples
        val = ast.literal_eval(s)
        if not isinstance(val, list):
            raise ValueError("Argument should be a list of tuples")
        for item in val:
            if not isinstance(item, tuple) or not all(isinstance(i, str) for i in item):
                raise ValueError("Argument items should be tuples of strings")
        return val
    except ValueError:
        raise ValueError("Argument should be a list of tuples of strings")

def filter_df(df: pd.DataFrame, 
            score: Optional[str] = None, 
            labeler: Optional[str] = None, 
            task_group: Optional[str] = None,
            sub_tasks: Optional[List[str]] = None,
            model_heads: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Filters results df based on various criteria."""
    df = df.copy()
    if score:
        df = df[df['score'] == score]
    if labeler:
        df = df[df['labeler'] == labeler]
    if task_group:
        labelers: List[str] = TASK_GROUP_2_LABELER[task_group]
        df = df[df['labeler'].isin(labelers)]
    if sub_tasks:
        df = df[df['sub_task'].isin(sub_tasks)]
    if model_heads:
        mask = [ False ] * df.shape[0]
        for model_head in model_heads:
            mask = mask | ((df['model'] == model_head[0]) & (df['head'] == model_head[1]))
        df = df[mask]
    return df

def write_table_to_latex(df: pd.DataFrame, path_to_file: str, is_ignore_index: bool = False):
    with open(path_to_file, 'a') as f:
        latex = df.to_latex(index=not is_ignore_index, escape=True)
        f.write("=======================================\n")
        f.write("=======================================\n")
        f.write("\n\nFigure:\n\n")
        f.write("\n")
        f.write(re.sub(r'\& +', '& ', latex))
        f.write("\n")
        f.write("=======================================\n")
        f.write("=======================================\n")

def get_rel_path(file: str, rel_path: str) -> str:
    """Transforms a relative path from a specific file in the package `eclair/src/eclair/ into an absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file)), rel_path)
    )

def convert_csv_labels_to_meds(path_to_labels_csv: str, is_force_refresh: bool = False) -> List[meds.Label]:
    dir_name: str = os.path.dirname(path_to_labels_csv)
    base_name: str = os.path.basename(path_to_labels_csv).replace(".csv", "")
    
    # Load cached labels if they exist
    path_to_cache: str = os.path.join(dir_name, f'{base_name}_meds.pkl')
    if not is_force_refresh and os.path.exists(path_to_cache):
        logger.info(f"Loading cached labels from `{path_to_cache}`")
        with open(path_to_cache, 'rb') as f:
            return pickle.load(f)
    else:
        logger.info(f"No cached labels found at `{path_to_cache}`. Generating labels from CSV file.")
    
    df_labels = pd.read_csv(path_to_labels_csv)
    df_labels['prediction_time'] = pd.to_datetime(df_labels['prediction_time'])
    # Conver to list of MEDS labels
    labels: List[meds.Label] = [
        meds.Label(
            patient_id=row['patient_id'],
            prediction_time=row['prediction_time'],
            boolean_value=row['boolean_value'],
            integer_value=row['integer_value'],
            categorical_value=row['categorical_value'],
            float_value=row['float_value'],
        )
        for _, row in tqdm(df_labels.iterrows(), desc='Converting labels to MEDS format', total=df_labels.shape[0])
    ]
    pickle.dump(labels, open(path_to_cache, 'wb'))

    return labels
