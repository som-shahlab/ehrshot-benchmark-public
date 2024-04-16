import argparse
import os
import pickle
from typing import List
import femr.models.transformer
from typing import Any, Dict, List, Optional
import meds
import torch
from loguru import logger
import femr.models.tokenizer
import femr.models.processor
import datasets
from utils import get_rel_path, convert_csv_labels_to_meds

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLMBR-T-Base patient representations (for all tasks at once)")
    parser.add_argument("--path_to_dataset", default=get_rel_path(__file__, "../assets/ehrshot-meds-stanford/"), type=str, help="Path to MEDS formatted version of EHRSHOT")
    parser.add_argument("--path_to_labels_csv", default=get_rel_path(__file__, "../assets/labels/merged_labels.csv"), type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", default=get_rel_path(__file__, "../assets/features/"), type=str, help="Path to directory where features will be saved")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="GPU device to use (if available)")
    parser.add_argument("--num_threads", default=5, type=int, help="Number of threads to use")
    parser.add_argument("--patient_range", type=str, default=None, help="Comma-separated patient range to featurize (inclusive) -- e.g. '0,40' gets patients in dataset at indices 0 through 40, inclusive")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_dataset: str = os.path.join(args.path_to_dataset, 'data/*.parquet')
    path_to_labels_csv: str = args.path_to_labels_csv
    path_to_features_dir: str = args.path_to_features_dir
    num_threads: int = args.num_threads
    device: str = args.device
    patient_range: Optional[str] = args.patient_range
    os.makedirs(path_to_features_dir, exist_ok=True)

    assert os.path.exists(args.path_to_dataset), f"Path to dataset does not exist: {args.path_to_dataset}"
    assert os.path.exists(path_to_labels_csv), f"Path to labels CSV does not exist: {path_to_labels_csv}"
    assert os.path.exists(path_to_features_dir), f"Path to features directory does not exist: {path_to_features_dir}"

    model_name: str = "StanfordShahLab/clmbr-t-base"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)
    
    if patient_range is not None:
        dataset = dataset.select(range(int(patient_range.split(",")[0]), int(patient_range.split(",")[1])))

    print("==> Len dataset:", len(dataset))

    # Load consolidated labels across all patients for all tasks
    labels: List[meds.Label] = convert_csv_labels_to_meds(path_to_labels_csv)
    
    labels = [ x for x in labels if x['patient_id'] in dataset['patient_id'] ]
    
    # Load model
    print("Loading model", model_nme)
    model = femr.models.transformer.FEMRModel.from_pretrained(model_name)
    
    # Generate features
    tokens_per_batch = 64 * 1024
    print("Generating batches of size", tokens_per_batch)
    breakpoint()
    results: Dict[str, Any] = femr.models.transformer.compute_features(dataset, model_name, labels, ontology=None, num_proc=num_threads, tokens_per_batch=tokens_per_batch, device=device)

    # Save results
    path_to_output_file = os.path.join(path_to_features_dir, f"clmbr_features_{patient_range}.pkl")
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