"""
Usage:
python3 2_generate_labels.py
"""
import argparse
import datetime
import json
import os
import meds
import datasets
import pickle
import pyarrow
import pyarrow.csv
import argparse
import femr
from femr.labelers.ehrshot import (
    PancreaticCancerCodeLabeler,
    CeliacDiseaseCodeLabeler,
    LupusCodeLabeler,
    AcuteMyocardialInfarctionCodeLabeler,
    EssentialHypertensionCodeLabeler,
    HyperlipidemiaCodeLabeler,
    Guo_LongLOSLabeler,
    Guo_30DayReadmissionLabeler,
    Guo_ICUAdmissionLabeler,
    ThrombocytopeniaInstantLabValueLabeler,
    HyperkalemiaInstantLabValueLabeler,
    HypoglycemiaInstantLabValueLabeler,
    HyponatremiaInstantLabValueLabeler,
    AnemiaInstantLabValueLabeler,
    ChexpertLabeler,
)

from utils import get_rel_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labels for a specific task")
    parser.add_argument("--path_to_dataset", default=get_rel_path(__file__, "../assets/ehrshot-meds-stanford/"), type=str, help="Path to MEDS formatted version of EHRSHOT")
    parser.add_argument("--path_to_ontology", default=get_rel_path(__file__, "../assets/ontology.pkl"), type=str, help="Path to OMOP ontology .pkl file for EHRSHOT")
    parser.add_argument("--path_to_labels_dir", default=get_rel_path(__file__, "../assets/labels/"), type=str, help="Path to directory containing saved labels")
    parser.add_argument("--labeler", required=True, type=str, help="Which labeler to run")
    # TODO (@Rahul)
    # parser.add_argument("--path_to_chexpert_csv", type=str, help="Path to CheXpert CSV file. Specific to CheXpert labeler", default=None,)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_dataset: str = os.path.join(args.path_to_dataset, 'data/*.parquet')
    path_to_ontology: str = args.path_to_ontology
    path_to_labels_dir: str = args.path_to_labels_dir
    labeler: str = args.labeler
    
    assert os.path.exists(args.path_to_dataset), f"Path to dataset does not exist: {args.path_to_dataset}"
    assert os.path.exists(path_to_ontology), f"Path to ontology does not exist: {path_to_ontology}"
    assert os.path.exists(path_to_labels_dir), f"Path to labels directory does not exist: {path_to_labels_dir}"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)

    # Load ontology
    ontology = pickle.load(open(path_to_ontology, "rb"))

    # Choose labeling function
    labeler_cls = {
        'new_pancan': PancreaticCancerCodeLabeler,
        'new_celiac': CeliacDiseaseCodeLabeler,
        'new_lupus': LupusCodeLabeler,
        'new_acutemi': AcuteMyocardialInfarctionCodeLabeler,
        'new_hypertension': EssentialHypertensionCodeLabeler,
        'new_hyperlipidemia': HyperlipidemiaCodeLabeler,
        'guo_los': Guo_LongLOSLabeler,
        'guo_readmission': Guo_30DayReadmissionLabeler,
        'guo_icu': Guo_ICUAdmissionLabeler,
        'lab_thrombocytopenia': ThrombocytopeniaInstantLabValueLabeler,
        'lab_hyperkalemia': HyperkalemiaInstantLabValueLabeler,
        'lab_hypoglycemia': HypoglycemiaInstantLabValueLabeler,
        'lab_hyponatremia': HyponatremiaInstantLabValueLabeler,
        'lab_anemia': AnemiaInstantLabValueLabeler,
        'chexpert' : ChexpertLabeler,
    }
    if labeler not in labeler_cls:
        raise ValueError(f"Labeler {labeler} not found. Must be one of {list(labeler_cls.keys())}")
    labeler_cls = labeler_cls[args.labeler]

    # Apply labeler
    labeler = labeler_cls(ontology)
    labeled_patients = labeler.apply(dataset, batch_size=1, num_proc=1)
    table = pyarrow.Table.from_pylist(labeled_patients, schema=meds.label)
    pyarrow.csv.write_csv(table, os.path.join(path_to_labels_dir, f"{args.labeler}_labels.csv"))



