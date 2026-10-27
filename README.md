# AI-Based Seizure Pattern Detection from EEG Signals

A reproducible research prototype that classifies EEG segments as **seizure** or **non-seizure** using signal preprocessing, time/frequency features, and scikit-learn models.

> **Safety note:** This is an educational/research prototype, not a medical device and not suitable for diagnosis, treatment, or real-time clinical decisions. The built-in demo data is synthetic and must not be presented as clinical evidence.

## Features
- Band-pass filtering and optional notch filtering
- Time-domain features (mean, standard deviation, RMS, line length, Hjorth parameters)
- Frequency-domain features (Welch band powers, spectral entropy, dominant frequency)
- SVM and Random Forest classifiers
- Subject-independent evaluation when subject IDs are provided
- Sensitivity, specificity, precision, recall, F1, confusion matrix, ROC-AUC (where defined)
- False alarms per hour and missed-seizure rate reporting
- Synthetic demo data generator, plots, tests, and command-line workflow

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m eeg_seizure.demo --output data/demo_eeg.csv
python -m eeg_seizure.train --input data/demo_eeg.csv --output-dir outputs --model random_forest
```

Run commands from the project root. If using a source checkout without installation, set `PYTHONPATH=src` (Windows PowerShell: `$env:PYTHONPATH='src'`). Alternatively install the package with `pip install -e .`.

## Dataset format

CSV input contains **one EEG segment per row**:
- `label`: `0` = non-seizure, `1` = seizure
- `subject_id`: optional patient/subject identifier; recommended for subject-independent evaluation
- `sampling_rate`: optional numeric sampling rate per segment; if absent, `--sampling-rate` is used
- all remaining numeric columns are consecutive EEG samples, e.g. `sample_0000`, `sample_0001`, ...

Example:

```csv
subject_id,label,sampling_rate,sample_0000,sample_0001,sample_0002
S01,0,256,0.02,-0.01,0.03
S02,1,256,0.50,-0.32,0.11
```

Use consistent segment lengths and units across rows. Avoid mixing segments from the same patient across train/test folds; the trainer uses group-based splitting when there are enough distinct subjects. For a true independent test set, provide separate train/test files or hold out entire subjects before model development.

## Real EEG dataset

A commonly used public research dataset is the **CHB-MIT Scalp EEG Database** hosted by PhysioNet: https://physionet.org/content/chbmit/1.0.0/ . Read and follow the dataset's access, citation, and license/terms information. It is distributed as EDF recordings with seizure annotations, so it needs a separate EDF-to-segment conversion step before using this CSV pipeline. Ensure labels and segment boundaries are derived from the official annotations and document your conversion process. Do not use synthetic demo data for claims about clinical performance.

## Outputs

The training command creates:
- `metrics.json` — metrics and split details
- `classification_report.txt` — class-wise precision/recall/F1
- `confusion_matrix.png` — false positives and false negatives
- `example_segments.png` — example raw segments by class
- `model.joblib` — fitted pipeline
- `feature_importance.csv` — Random Forest feature importance, when available

## Evaluation cautions
- Overall accuracy can be misleading because seizures are typically rare. Inspect sensitivity, specificity, precision, recall, F1, false alarms, and missed detections.
- The false-alarms-per-hour estimate is only meaningful when the evaluated segments represent continuous monitoring with known segment duration and realistic prevalence.
- The included synthetic data is intentionally separable enough to demonstrate the software pipeline; it is **not representative** of real EEG complexity.
- Never split overlapping windows randomly across train and test sets. Split by subject/session first to avoid data leakage.

## Project structure

```text
src/eeg_seizure/   preprocessing, feature extraction, training, demo generator
 tests/            unit tests
 data/             generated demo CSV (created by command)
 outputs/          model and reports (created by command)
```

## Reproducibility
The CLI fixes random seeds by default. Record the dataset version, subject IDs, preprocessing settings, and model settings for every real-data experiment.
