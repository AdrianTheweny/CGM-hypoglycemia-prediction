# CGM Hypoglycemia Prediction

Machine learning pipeline for predicting hypoglycemia events using continuous
glucose monitoring (CGM) data from a 30-patient subset.

## Project Structure
├── data/
│   ├── raw/                  # raw CGM data
│   └── processed/            # cleaned, labeled, model-ready data
├── notebooks/
│   ├── BE175_project_data_processing.ipynb   # data cleaning + feature engineering
│   └── BE175_CGM_Data_Labeling_Clean.ipynb   # labeling pipeline
├── src/
│   ├── train_baseline.py     # baseline models (LR, RF, GB)
│   ├── train_experimental.py # full pipeline w/ CV, RUSBoost, imbalance handling
│   ├── train_rusboost_only.py
│   ├── safora_evaluation.py  # model evaluation + plots
│   └── training_tests.py     # unit tests for training pipeline
├── tests/
│   └── test_safora_evaluation.py
├── results/
│   ├── figures/
│   ├── metrics/              # model performance CSVs
│   ├── models/               # trained model .pkl files
│   └── safora_evaluation/    # evaluation results + ROC/PR curves
└── README.md
```
## Setup

```bash
pip install -r requirements.txt
```

(or list out: pandas, numpy, scikit-learn, imbalanced-learn, joblib)

## Usage

Run the baseline models:
```bash
python src/train_baseline.py
```

Run the experimental pipeline (cross-validation + RUSBoost):
```bash
python src/train_experimental.py
```

Run unit tests:
```bash
python src/training_tests.py
```

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- RUSBoost (handles class imbalance via random undersampling)

## Data

Input data: `data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv`
(not tracked in repo due to size — see `.gitignore`)

Features include rolling glucose readings (t-55 to t-0 minutes), 60-minute
rolling mean, and 60-minute glucose slope.


## Team Contributions

- **Adrian** – Repo/folder structure, shared data format, baseline modeling
  (Logistic Regression / Random Forest / Gradient Boosting), final pipeline
  integration across all components.
- **Sarah** – Data preprocessing: loading REPLACE-BG CGM data, cleaning
  timestamps, handling missing values/gaps, sorting by patient/time, reducing
  to a 30-50 patient subset.
- **Jonathon** – Feature engineering and labeling: 1-hour CGM windows,
  hypoglycemia labels (next 40 min), features including rolling glucose
  values, slope, and rolling mean.
- **Safora** – Model evaluation: ROC AUC, PR AUC, confusion matrix,
  sensitivity/recall, and simplified event-based detection.
- **Dani** – Figures and writeup: ROC/PR curve plots, example CGM traces with
  predicted alarms, report drafting (background, methods, results), and
  comparison to the original paper's approach.