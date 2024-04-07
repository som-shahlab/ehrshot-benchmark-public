# 👂 💉 EHRSHOT

A benchmark/dataset for few-shot evaluation of foundation models for electronic health records (EHRs). You can **[read the paper here](https://arxiv.org/abs/2307.02028)**. 

Please note that the dataset + model are still being reviewed, and a download link will be provided once they are approved for public release.

----

Whereas most prior EHR benchmarks are limited to the ICU setting, **EHRSHOT** contains the **full longitudinal health records of 6,732 patients from Stanford Medicine** and a diverse set of **15 classification tasks** tailored towards few-shot evaluation of pre-trained models. 

# 📖 Table of Contents
1. [Pre-trained Foundation Model](#models)
2. [Dataset + Tasks](#dataset)
3. [Comparison to Prior Work](#prior_work)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Citation](#citation)

<a name="models"/>

# 🔮 Foundation Model for EHRs

We publish the weights of **CLMBR-T-Base**, a **141 million parameter** clinical foundation model pre-trained on the deidentified structured EHR data of **2.57M patients** from Stanford Medicine.

### Setup

```python
from transformers import AutoModel, AutoTokenizer
import femr.models.transformer
import femr.models.tokenizer
import femr.models.dataloader

# Random weight model (non-gated access, immediately available)
tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained("StanfordShahLab/clmbr-t-base-random")
batch_processor = femr.models.dataloader.FEMRBatchProcessor("StanfordShahLab/clmbr-t-base-random")
model = femr.models.transformer.FEMRModel.from_pretrained("StanfordShahLab/clmbr-t-base-random")

# Pretrained model (gated access, requires research usage agreement)
tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained("StanfordShahLab/clmbr-t-base")
batch_processor = femr.models.dataloader.FEMRBatchProcessor("StanfordShahLab/clmbr-t-base")
model = femr.models.transformer.FEMRModel.from_pretrained("StanfordShahLab/clmbr-t-base")
```

Our pretrained CLMBR-T-Base model is [available at HuggingFace at this link.](https://huggingface.co/StanfordShahLab/clmbr-t-base-random). Access to this model is gated, so you will need to request access and fill out a research use agreement before you can get the model weights.

For testing purposes, a non-gated model with random weights [is available on HuggingFace here](https://huggingface.co/StanfordShahLab/clmbr-t-base-random) and can be immediately downloaded by anyone.

### Usage

An example for how to run inference on a single patient using CLMBR-T-Base is shown below:

```python
import femr.models.transformer
import torch
import femr.models.tokenizer
import femr.models.dataloader
import datetime

model_name = "StanfordShahLab/clmbr-t-base-random"

# Load tokenizer / batch loader
tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_name)
batch_processor = femr.models.dataloader.FEMRBatchProcessor(tokenizer)

# Load model
model = femr.models.transformer.FEMRModel.from_pretrained(model_name)

# Create an example patient to run inference on
example_patient = {
    'patient_id': 30,
    'events': [{
        'time': datetime.datetime(2011, 5, 8),
        'measurements': [
            {'code': 'SNOMED/1'},
        ],
    },
    {
        'time': datetime.datetime(2012, 6, 9),
        'measurements': [
            {'code': 'SNOMED/30'},
            {'code': 'SNOMED/103'}
        ],
    }]
}
batch = batch_processor.convert_patient(example_patient, tensor_type="pt")

# Run model
with torch.no_grad():
    patient_ids, times, reprs = model(batch)
    print(patient_ids)
    print(times)
    print(reprs)
```

### Prior Work

We are [one of the first](https://arxiv.org/abs/2303.12961) to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data  (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR. 


### Model Architecture
We use [Clinical Language-Model-Based Representations (CLMBR)](https://www.sciencedirect.com/science/article/pii/S1532046420302653) as our model architeceture. CLMBR is an autoregressive model designed to predict the next medical code in a patient's timeline given previous codes. CLMBR employs causally masked local attention, ensuring forward-only flow of information which is vital for prediction tasks and is in contrast to BERT-based models which are bidirectional in nature. We utilize a transformer as our base model with 141 million trainable parameters and a next code prediction objective, providing minute-level EHR resolution rather than the day-level aggregation of the original model formulation. 


<a name="dataset"/>

# 🗃️ Dataset + Tasks

**Please Note:** Dataset release is currently being reviewed and the download link will be updated once it is publicly available.

The EHRSHOT (version 1) dataset contains:
* **6,732 patients**
* **43.2 million clinical events**
* **1,004,148 visits**
* **15 prediction tasks**

Each patient consists of an ordered timeline of clinical events taken from the structured data of their EHR (e.g. diagnoses, procedures, prescriptions, etc.). 

Each task is a predictive classification task, and includes a canonical train/val/test split. The tasks are defined as follows:

|         Task         | Type              | Prediction Time                       | Time Horizon           |
|:--------------------:|-------------------|---------------------------------------|------------------------|
| Long Length of Stay  | Binary            | 11:59pm on day of admission           | Admission duration     |
| 30-day Readmission   | Binary            | 11:59pm on day of discharge           | 30-days post discharge |
| ICU Transfer         | Binary            | 11:59pm on day of admission           | Admission duration     |
| Thrombocytopenia     | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hyperkalemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hypoglycemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hyponatremia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Anemia               | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hypertension         | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Hyperlipidemia       | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Pancreatic Cancer    | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Celiac               | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Lupus                | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Acute MI             | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Chest X-Ray Findings | 14-way Multilabel | 24hrs before report is recorded       | Next report            |


1. Download EHRSHOT to `[PATH_TO_SOURCE_OMOP]`

2. Convert EHRSHOT => [MEDS data format](https://github.com/Medical-Event-Data-Standard/meds) using the following:

```bash
# Convert OMOP => MEDS data format (takes ~5 min using 100 shards and 5 processes)
meds_etl_omop [PATH_TO_SOURCE_OMOP] [PATH_TO_OUTPUT_MEDS]_raw

# Apply some EHRSHOT-specific fixes (takes ~45 mins)
femr_stanford_omop_fixer [PATH_TO_OUTPUT_MEDS]_raw [PATH_TO_OUTPUT_MEDS]
```

3. Use HuggingFace's [Datasets](https://huggingface.co/docs/datasets/en/index) library to load our dataset in Python

```python
import datasets
dataset = datasets.Dataset.from_parquet(PATH_TO_OUTPUT_MEDS + 'data/*')

# Print dataset stats
print(dataset)
>>> Dataset({
>>>   features: ['patient_id', 'events'],
>>>   num_rows: 6732
>>> })

# Print number of events in first patient in dataset
print(len(dataset[0]['events']))
>>> 2287
```


```

<a name="prior_work"/>

# 📊 Comparison to Prior Work

Most prior benchmarks are (1) limited to the ICU setting and (2) not tailored towards few-shot evaluation of pre-trained models.

In contrast, **EHRSHOT** contains (1) the full breadth of longitudinal data that a health system would expect to have on the patients it treats and (2) a broad range of tasks designed to evaluate models' task adaptation and few-shot capabilities:

<table>
  <tr> <th rowspan="3">Benchmark</th> <th colspan="1">Source</th> <th colspan="3">EHR Properties</th> <th colspan="2">Evaluation</th> <th colspan="3">Reproducibility</th> </tr>
  <tr> <td rowspan="2">Dataset</td> <td rowspan="2">ICU/ED Visits</td> <td rowspan="2">Non-ICU/ED Visits</td> <td rowspan="2"># of Patients</td> <td rowspan="2"># of Tasks</td> <td rowspan="2">Few Shot</td> <td rowspan="2">Dataset via DUA</td> <td rowspan="2">Preprocessing Code</td> <td rowspan="2">Model Weights</td> </tr>
  <tr></tr>
  <tr></tr>
  <tr> <td><b>EHRSHOT</b></td> <td><b>Stanford Medicine</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>7k</b></td> <td><b>15</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>✓</b></td> </tr>
  <tr> <td><a href="https://github.com/MLforHealth/MIMIC_Extract">MIMIC-Extract</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>34k</td> <td>5</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/USC-Melady/Benchmarking_DL_MIMICIII">Purushotham 2018</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>35k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/YerevaNN/mimic3-benchmarks">Harutyunyan 2019</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>33k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/healthylaife/MIMIC-IV-Data-Pipeline">Gupta 2022</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>257k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/aishwarya-rm/cop-e-cat">COP-E-CAT</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>257k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/nliulab/mimic4ed-benchmark">Xie 2022</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>216k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/mostafaalishahi/eICU_Benchmark">eICU</a></td> <td>eICU</td> <td>✓</td> <td>--</td> <td>73k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/mmcdermott/comprehensive_MTL_EHR">EHR PT</a></td> <td>MIMIC-III / eICU</td> <td>✓</td> <td>--</td> <td>86k</td> <td>11</td> <td>✓</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/MLD3/FIDDLE">FIDDLE</a></td> <td>MIMIC-III / eICU</td> <td>✓</td> <td>--</td> <td>157k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/ratschlab/HIRID-ICU-Benchmark">HiRID-ICU</a></td> <td>HiRID</td> <td>✓</td> <td>--</td> <td>33k</td> <td>6</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://www.sciencedirect.com/science/article/pii/S1532046419302564?via%3Dihub">Solares 2020</a></td> <td>CPRD</td> <td>✓</td> <td>✓</td> <td>4M</td> <td>2</td> <td>--</td> <td>--</td> <td>--</td> <td>--</td> </tr>
</table>


<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running the EHRSHOT benchmark.

**1)**: Install **EHRSHOT**

```bash
conda create -n EHRSHOT_ENV python=3.10 -y
conda activate EHRSHOT_ENV

git clone https://github.com/som-shahlab/ehrshot-benchmark.git
cd ehrshot-benchmark
pip install -r requirements.txt
```

**2)**: Install **FEMR**

For our data preprocessing pipeline we use **[FEMR  (Framework for Electronic Medical Records)](https://github.com/som-shahlab/femr)**, a Python package for building deep learning models with EHR data. 

You must also have CUDA/cuDNN installed (we recommend CUDA 11.8 and cuDNN 8.7.0)

Note that this currently only works on Linux machines.

```bash
pip install --upgrade "jax[cuda11_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install bazel=6 -y
pip install git+https://github.com/som-shahlab/femr.git@ehrshot_branch
```

## Download Private Assets

You will need to separately download several assets that we cannot redistribute publicly on Github.

This includes the dataset itself, the weights of the pre-trained foundation model we benchmark, and the Athena OHDSI Ontology. 

### A) Dataset & Foundation Model for EHRs

**Please Note:** Dataset + model release is currently being reviewed and the download link will be updated once it is publicly available.

Once this is downloaded, unzip it to get a folder called `EHRSHOT_ASSETS/`. Please move this folder to the root of this repo.

### B) Athena OHDSI Ontology

Our pipeline requires the user to provide an ontology in order to map medical codes to their parents/children. We use the default Athena OHDSI Ontology for this. 

Unfortunately, we cannot redistribute the Athena OHDSI Ontology ourselves, so you must separately download it by following these steps:

1. Go to the [Athena website at this link](https://athena.ohdsi.org/vocabulary/list). You may need to create an account.
2. Click the green "Download" button at the top right of the website
3. Click the purple "Download Vocabularies" button below the green "Download" button
4. Name the bundle "athena_download" and select 5.x version
5. Scroll to the bottom of the list, and click the blue "Download" button
6. It will take some time for the download to be ready. Please [refresh the webpage here](https://athena.ohdsi.org/vocabulary/download-history) to check whether your download is ready. Once the download is ready, click "Download"
7. After the download is complete, unzip the file and move all the files into the `EHRSHOT_ASSETS/athena_download/` folder in your repo.

After downloading the Athena OHDSI Ontology, you will have to separately download the CPT subset of the ontology. You can follow the instructions in the `readme.txt` in your Athena download, or follow the steps below:

1. Create a [UMLS account here](https://uts.nlm.nih.gov/uts/signup-login)
2. Get your [UMLS API key here](https://uts.nlm.nih.gov/uts/edit-profile)
3. From the `EHRSHOT_ASSETS/athena_download/` folder, run this command: `bash cpt.sh <YOUR UMLS API KEY>`

Your ontology will then be ready to go!

## Folder Structure

Your final folder structure should look like this:

- `ehrshot-benchmark/`
  - `EHRSHOT_ASSETS/`
    - `data/`
      - *We provide this asset, which contains deidentified EHR data as CSVs.*
    - `benchmark/`
      - *We provide this asset, which contains labels and few-shot samples for all our tasks.*
    - `models`
      - *We provide this asset, which contains our pretrained foundation model for EHRs.*
    - `athena_download/`
      - *You will need to download and put the Athena OHDSI Ontology inside this folder. Please follow the instructions above to download it.*
  - `ehrshot/`
    - *We provide the scripts to run the benchmark here*

<a name="usage"/>

# 👩‍💻 Usage

To execute the entire benchmark end-to-end, please run:

```bash
python3 run_all.py
```

----

You can also run each of the steps individually by directly calling their corresponding Python/Bash files in the `ehrshot/` folder. Note that depending on your system, you may need to change the Bash scripts.

Here is a breakdown of what each step in the pipeline does:

**1)**: Convert the **EHRSHOT** CSV files into a format that the [FEMR library](https://github.com/som-shahlab/femr) can process.

```bash
python3 1_create_femr_database.py \
    --path_to_input ../EHRSHOT_ASSETS/data \
    --path_to_target ../EHRSHOT_ASSETS/femr \
    --athena_download ../EHRSHOT_ASSETS/athena_download \
    --num_threads 10
```

Alternatively, you can also run
```bash
sbatch 1_create_femr_database_slurm.sh
```

Please make sure you change the Bash script according to your system. You may not be able to run it as a slurm job.

**2)**: Apply the labeling functions defined in [FEMR](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) to our dataset to generate labels for our benchmark tasks.

Note that as part of our dataset release, we also include these labels in a CSV. Thus, you should skip to the label generation part of the script by setting the `--is_skip_label` flag.

```bash
python3 2_generate_labels_and_features.py \
    --path_to_database ../EHRSHOT_ASSETS/femr/extract \
    --path_to_output_dir ../EHRSHOT_ASSETS/benchmark \
    --path_to_chexpert_csv ../EHRSHOT_ASSETS/benchmark/chexpert/chexpert_labeled_radiology_notes.csv \
    --labeling_function guo_los \
    --is_skip_label \
    --num_threads 10
```

In case you want to regenerate your labels, you can run the above command without the `--is_skip_label` flag.

The above command runs it only for `guo_los` (Long Length of Stay) labeling function. You will need to individually run this script for each of the 15 tasks. Alternatively, you can run the Bash script shown below to iterate through every task automatically.

```bash
sbatch 2_generate_labels_and_features_slurm.sh
```

**3)**: Generate a CLMBR representation for each patient for each label. Below is an example of how to run it for one task (`guo_los`). 

Note that this job **requires a GPU.**

```bash
python3 3_generate_clmbr_representations.py \
    --path_to_clmbr_data ../EHRSHOT_ASSETS/models/clmbr_model \
    --path_to_database ../EHRSHOT_ASSETS/femr/extract \
    --path_to_labeled_featurized_data ../EHRSHOT_ASSETS/benchmark \
    --path_to_save ../EHRSHOT_ASSETS/clmbr_reps \
    --labeling_function guo_los
```

To run it for all tasks automatically, run the following Bash script:

```bash
sbatch 3_generate_clmbr_representations_slurm.sh
```

**4)**: Generate our `k`-shots for few-shot evaluation.

Note that we provide the exact `k`-shots used in our paper with our data release. Please do not run this script if you want to use the `k`-shots we used in our paper.

```bash
python3 4_generate_shot.py \
    --path_to_data ../EHRSHOT_ASSETS \
    --labeling_function guo_los \
    --num_replicates 1 \
    --path_to_save ../EHRSHOT_ASSETS/benchmark \
    --shot_strat few
```

To run it for all tasks automatically, run the following Bash script:

```bash
sbatch 4_generate_shot_slurm.sh
```

**5)**: Train our baseline models and generate performance metrics.

```bash
python3 5_eval.py \
    --path_to_data ../EHRSHOT_ASSETS \
    --labeling_function guo_los \
    --num_replicates 5 \
    --model_head logistic \
    --is_tune_hyperparams \
    --path_to_save ../EHRSHOT_ASSETS/output \
    --shot_strat few
```

To run it for all tasks automatically, run the following Bash script:

```bash
sbatch 5_eval_slurm.sh
```

**6)**: Generate the plots we included in our paper.

```bash
python3 6_make_figures.py \
    --path_to_eval ../EHRSHOT_ASSETS/output \
    --path_to_save ../EHRSHOT_ASSETS/figures
```

or 

```bash
sbatch 6_make_figures_slurm.sh
```

# Citation

If you find this project helpful, please cite [our paper](https://arxiv.org/abs/2307.02028):

```
@article{wornow2023ehrshot,
      title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models}, 
      author={Michael Wornow and Rahul Thapa and Ethan Steinberg and Jason Fries and Nigam Shah},
      year={2023},
      eprint={2307.02028},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# License

The source code of this repo is released under the Apache License 2.0. The dataset and model license are listed on their corresponding Stanford AIMI Center webpage.
