import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


DATA_PATH = "data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv"
MODEL_DIR = "results/models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

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

# load dataset
df = pd.read_csv(DATA_PATH)

print(df.columns)

# features and labels
X = df[feature_cols]
y = df["label"]

# label balance
print("\nLABEL COUNTS:\n")
print(y.value_counts())

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# models
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        random_state=42
    )
}


def clean_filename(name):
    """Convert model names into filenames."""
    return name.lower().replace(" ", "_")


# store results
results = []

# training loop
for name, model in models.items():

    print(f"\n--- {name} ---")

    # train
    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # metrics
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # print metrics
    print("ROC AUC:", auc)
    print(classification_report(y_test, y_pred))

    # save trained model
    model_filename = clean_filename(name) + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_filename))
    print(f"Model saved: {MODEL_DIR}/{model_filename}")

    # save metrics
    results.append({
        "Model": name,
        "ROC_AUC": round(auc, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3)
    })

# convert to dataframe
results_df = pd.DataFrame(results)

# save results
results_df.to_csv(
    "results/metrics/baseline_model_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/metrics/baseline_model_results.csv")