import os
from typing import List
from utils import get_rel_path
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    path_to_labels_dir: str = get_rel_path(__file__, "../assets/labels/")
    labelers: List[str] = os.listdir(path_to_labels_dir)
    print(f"Found {len(labelers)} labelers to merge: {labelers}")

    # Merge all predictions times for all labels across all tasks into a single file,
    # so that we can later generate features for all of them at once.
    dfs: List[pd.DataFrame] = []
    for lf in tqdm(labelers, desc="Merging label times"):
        dfs.append(pd.read_csv(os.path.join(path_to_labels_dir, lf)))
    df = pd.concat(dfs)
    df['prediction_time'] = pd.to_datetime(df['prediction_time'])
    
    # Drop labels that occur at the same time for the same patient (b/c featurization will be redundant)
    df = df.drop_duplicates(['patient_id', 'prediction_time'])
    df['boolean_value'] = False

    # Resort all labels to be in chronological order
    df = df.sort_values(['patient_id', 'prediction_time'])
    
    # Save the merged labels to a CSV file
    df.to_csv(get_rel_path(__file__, "../assets/labels/merged_labels.csv"), index=False)
    
    