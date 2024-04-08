import ast
import pickle
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import meds
import pandas as pd
import numpy as np
import datasets
import femr
from loguru import logger
from tqdm import tqdm

# SPLITS
SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: int = 70
SPLIT_VAL_CUTOFF: int = 85

# Types of base models to test
MODEL_2_NAME: Dict[str, str] = {
    'count' : 'Count-based (v8)',
    'clmbr' : 'CLMBR (v8)',
    # 'gpt2-base' : 'GPT2-base (v9)',
    # 'gpt2-medium' : 'GP2-medium (v9)',
    # 'gpt2-large' : 'GP2-large (v9)',
    # 'bert-base' : 'BERT-base (v9)',
    'gpt2-base-v8_chunk:last_embed:last' : 'GPT2-base (v8)',
    'bert-base-v8_chunk:last_embed:last' : 'BERT-base (v8)',
}
BASE_MODELS: List[str] = list(MODEL_2_NAME.keys())

# Map each base model to a set of heads to test
BASE_MODEL_2_HEADS: Dict[str, List[str]] = {
    'count' : ['gbm', 'lr_lbfgs', 'rf', ], 
    'clmbr' : ['lr_lbfgs', 'lr_femr', 'rf', ],
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
LABELING_FUNCTIONS: List[str] = [
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

LABELING_FUNCTION_2_PAPER_NAME = {
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

TASK_GROUP_2_LABELING_FUNCTION = {
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
    "penalty": ['l2']
}
RF_PARAMS = {
    'n_estimators': [10, 20, 50, 100, 300],
    'max_depth' : [3, 5, 10, 20, 50],
}

# Few shot settings
SHOT_STRATS = {
    'few' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128],
    'long' : [-1],
    'all' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1],
    'debug' : [10],
}

# Plotting
SCORE_MODEL_HEAD_2_COLOR = {
    'auroc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'lr_femr' : 'tab:blue',
        },
        'gpt2-base-v8_chunk:last_embed:last' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'bert-base-v8_chunk:last_embed:last' : {
            'lr_lbfgs' : 'tab:olive',
        },
    },
    'auprc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'lr_femr' : 'tab:blue',
        },
        'gpt2-base-v8_chunk:last_embed:last' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'bert-base-v8_chunk:last_embed:last' : {
            'lr_lbfgs' : 'tab:olive',
        },
    },
}

def split_labels(labels: List[meds.Label], path_to_split_csv: str) -> Tuple:
    """Split labels into train/val/test sets."""
    df_split = pd.read_csv(path_to_split_csv)
    labels_split: Dict[str, meds.Label] = {
        'train' : [ label for label in labels if label['patient_id'] in df_split[df_split['split'] == 'train']['omop_person_id'].values ],
        'val' : [ label for label in labels if label['patient_id'] in df_split[df_split['split'] == 'val']['omop_person_id'].values ],
        'test' : [ label for label in labels if label['patient_id'] in df_split[df_split['split'] == 'test']['omop_person_id'].values ],
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
        'train': np.array([ x['prediction_time'] for x in labels_split['train'] ]),
        'val': np.array([ x['prediction_time'] for x in labels_split['val'] ]),
        'test': np.array([ x['prediction_time'] for x in labels_split['test'] ]),
    }
    return labels_split, label_values, label_times, patient_ids


def get_labels_and_features(labels: List[meds.Label], path_to_features_dir: Optional[str]) -> List[meds.Label]:
    """Given a path to a directory containing labels and features as well as a LabeledPatients object, returns
        the labels and features for each patient. Note that this function is more complex b/c we need to align
        the labels with their corresponding features based on their prediction times."""
    # Sort arrays by (1) patient ID and (2) label time
    labels = sorted(labels, key=lambda x: (x.patient_id, x.prediction_time))

    # Just return labels, ignore features
    if path_to_features_dir is None:
        return labels

    # TODO - check rest of this
    # Go through every featurization we've created (e.g. count, clmbr)
    # and align the label times with the featurization times
    featurizations: Dict[str, np.ndarray] = {}
    for model in BASE_MODELS:
        path_to_feats_file: str = os.path.join(path_to_features_dir, f'{model}_features.pkl')
        assert os.path.exists(path_to_feats_file), f'Path to file containing `{model}` features does not exist at this path: {path_to_feats_file}. Maybe you forgot to run `generate_features.py` first?'
        
        with open(path_to_feats_file, 'rb') as f:
            # Load data and do type checking
            feats: Tuple[Any, np.ndarray, np.ndarray, np.ndarray] = pickle.load(f)
            feature_matrix, feature_patient_ids, feature_times = (
                feats[0],
                feats[1],
                feats[3], # NOTE: skip label_values in [2]
            )
            feature_patient_ids = feature_patient_ids.astype(label_patient_ids.dtype)
            feature_times = feature_times.astype(label_times.dtype)
            assert feature_patient_ids.dtype == label_patient_ids.dtype, f'Error -- mismatched types between feature_patient_ids={feature_patient_ids.dtype} and label_patient_ids={label_patient_ids.dtype}'
            assert feature_times.dtype == label_times.dtype, f'Error -- mismatched types between feature_times={feature_times.dtype} and label_times={label_times.dtype}'

            # Sort arrays by (1) patient ID and (2) label time
            sort_order: np.ndarray = np.lexsort((feature_times, feature_patient_ids))
            feature_patient_ids, feature_times = feature_patient_ids[sort_order], feature_times[sort_order]

            # Align label times with feature times
            join_indices = femr.extension.dataloader.compute_feature_label_alignment(label_patient_ids, 
                                                                                     label_times.astype(np.int64), 
                                                                                     feature_patient_ids, 
                                                                                     feature_times.astype(np.int64))
            feature_matrix = feature_matrix[sort_order[join_indices], :]

            # Validate that our alignment was successful
            assert np.all(feature_patient_ids[join_indices] == label_patient_ids)
            assert np.all(feature_times[join_indices] == label_times)

            featurizations[model] = feature_matrix
    
    return label_patient_ids, label_values, label_times, featurizations

def process_chexpert_labels(labels: List[meds.Label]) -> List[meds.Label]:
    for label in labels:
        label_str = bin(label['categorical_value'])[2:]
        rem_bin = 14 - len(label_str)
        label_str = "0"*rem_bin + label_str
        label_list = [*label_str]
        label_list = [int(label) for label in label_list]
        label['boolean_value'] = label_list # ! Hacky, but MEDS doesn't support list values
    return labels

def convert_multiclass_to_binary_labels(labels: List[meds.Label], threshold: int = 1) -> List[meds.Label]:
    for label in labels:
        label['boolean_value'] = label['integer_value'] >= threshold
        del label['integer_value']
    return labels

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
            labeling_function: Optional[str] = None, 
            task_group: Optional[str] = None,
            sub_tasks: Optional[List[str]] = None,
            model_heads: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Filters results df based on various criteria."""
    df = df.copy()
    if score:
        df = df[df['score'] == score]
    if labeling_function:
        df = df[df['labeling_function'] == labeling_function]
    if task_group:
        labeling_functions: List[str] = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        df = df[df['labeling_function'].isin(labeling_functions)]
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
        )
        for _, row in tqdm(df_labels.iterrows(), desc='Converting labels to MEDS format', total=df_labels.shape[0])
    ]
    pickle.dump(labels, open(path_to_cache, 'wb'))

    return labels