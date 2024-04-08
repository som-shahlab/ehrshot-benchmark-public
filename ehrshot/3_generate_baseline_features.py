"""
Timing:
    23 mins to preprocess featurizers on a Macbook Pro
    5 hrs 32 mins to featurize on a Macbook Pro

Usage:
    python 3_generate_baseline_features.py
"""
import argparse
import pickle
import os
from typing import Any, Dict, List
from loguru import logger
from femr.featurizers import AgeFeaturizer, CountFeaturizer, FeaturizerList
from femr.index import PatientIndex
from femr.ontology import Ontology
import meds
from utils import get_rel_path, convert_csv_labels_to_meds
import datasets

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate count-based featurizations for GBM models (for all tasks at once)")
    parser.add_argument("--path_to_dataset", default=get_rel_path(__file__, "../assets/ehrshot-meds-stanford/"), type=str, help="Path to MEDS formatted version of EHRSHOT")
    parser.add_argument("--path_to_labels_csv", default=get_rel_path(__file__, "../assets/labels/merged_labels.csv"), type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_ontology", default=get_rel_path(__file__, "../assets/ontology.pkl"), type=str, help="Path to OMOP ontology .pkl file for EHRSHOT")
    parser.add_argument("--path_to_features_dir", default=get_rel_path(__file__, "../assests/features/"), type=str, help="Path to directory where features will be saved")
    parser.add_argument("--num_threads", default=5, type=int, help="Number of threads to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_labels_csv: str = args.path_to_labels_csv
    path_to_dataset: str = os.path.join(args.path_to_dataset, 'data/*.parquet')
    path_to_features_dir: str = args.path_to_features_dir
    path_to_ontology: str = args.path_to_ontology
    num_threads: int = args.num_threads
    os.makedirs(path_to_features_dir, exist_ok=True)
    
    assert os.path.exists(args.path_to_dataset), f"Path to dataset does not exist: {args.path_to_dataset}"
    assert os.path.exists(path_to_features_dir), f"Path to features directory does not exist: {path_to_features_dir}"
    assert os.path.exists(path_to_labels_csv), f"Path to labels CSV does not exist: {path_to_labels_csv}"
    assert os.path.exists(path_to_ontology), f"Path to ontology does not exist: {path_to_ontology}"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)
    patient_index: PatientIndex = PatientIndex(dataset)
    
    # Load ontology
    ontology: Ontology = pickle.load(open(path_to_ontology, "rb"))

    # Load consolidated labels across all patients for all tasks
    labels: List[meds.Label] = convert_csv_labels_to_meds(path_to_labels_csv)
    
    # Combine two featurizations of each patient: one for the patient's age, and one for the count of every code
    # they've had in their record up to the prediction timepoint for each label
    age = AgeFeaturizer()
    count = CountFeaturizer(ontology, is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers -- this includes processes such as normalizing age
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers( dataset, patient_index, labels, num_proc=num_threads )
    logger.info("Finish | Preprocess featurizers")

    # Run actual featurization for each patient
    logger.info("Start | Featurize patients")
    results: Dict[str, Any] = featurizer_age_count.featurize(dataset, patient_index, labels, num_proc=num_threads)
    logger.info("Finish | Featurize patients")

    # Save results
    path_to_output_file = os.path.join(path_to_features_dir, "count_features.pkl")
    logger.info(f"Saving results to `{path_to_output_file}`")
    with open(path_to_output_file, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    patient_ids, feature_times, features = results['patient_ids'], results['feature_times'], results['features']
    logger.info("FeaturizedPatient stats:\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"features={repr(features)}\n"
                f"feature_times={repr(feature_times)}\n")
    logger.success("Done!")
    