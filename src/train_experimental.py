import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.ensemble import RUSBoostClassifier

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DATA_PATH    = "data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv"
MODEL_DIR    = "results/models"
METRICS_PATH = "results/metrics/model_results.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "glucose_t_minus_55",
    "glucose_t_minus_50",
    "glucose_t_minus_45",
    "glucose_t_minus_40",
    "glucose_t_minus_35",
    "glucose_t_minus_30",
    "glucose_t_minus_25",
    "glucose_t_minus_20",
    "glucose_t_minus_15",
    "glucose_t_minus_10",
    "glucose_t_minus_5",
    "glucose_t_0",
    "glucose_rolling_mean_60min",
    "glucose_slope_60min"
]

X = df[feature_cols]
y = df["label"]

print("\nLABEL COUNTS:")
print(y.value_counts())
print(f"\nClass imbalance ratio: {round(y.value_counts()[0] / y.value_counts()[1], 1)}:1")

# ──────────────────────────────────────────────
# TRAIN/TEST SPLIT
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────

# compute scale factor for GB (handles imbalance since it doesn't support class_weight)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
        # note: GradientBoostingClassifier doesn't support class_weight
        # imbalance handled via sample_weight at fit time below
    ),

    # RUSBoost: the actual ensemble method from the paper
    # randomly undersamples majority class at each boosting round
    "RUSBoost": RUSBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

# ──────────────────────────────────────────────
# CROSS-VALIDATION + TRAINING LOOP
# ──────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # cross-validation on training set
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    print(f"CV ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # fit on full training set
    # GB gets sample weights to handle imbalance
    if name == "Gradient Boosting":
        sample_weights = np.where(y_train == 1, scale_pos, 1.0)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # metrics
    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall   = recall_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)

    print(f"\nTest ROC AUC : {roc_auc:.3f}")
    print(f"Test PR AUC  : {pr_auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # save trained model for Safora's evaluation script
    model_filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_filename))
    print(f"Model saved: results/models/{model_filename}")

    results.append({
        "Model"       : name,
        "CV_ROC_AUC"  : round(cv_scores.mean(), 3),
        "CV_Std"      : round(cv_scores.std(), 3),
        "Test_ROC_AUC": round(roc_auc, 3),
        "Test_PR_AUC" : round(pr_auc, 3),
        "Precision"   : round(precision, 3),
        "Recall"      : round(recall, 3),
        "F1"          : round(f1, 3)
    })

# ──────────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────────

results_df = pd.DataFrame(results)

# sort by test ROC AUC descending
results_df = results_df.sort_values("Test_ROC_AUC", ascending=False).reset_index(drop=True)

results_df.to_csv(METRICS_PATH, index=False)

print(f"\n{'='*50}")
print("FINAL RESULTS SUMMARY")
print('='*50)
print(results_df.to_string(index=False))
print(f"\nResults saved to: {METRICS_PATH}")
print("Models saved to:  results/models/")