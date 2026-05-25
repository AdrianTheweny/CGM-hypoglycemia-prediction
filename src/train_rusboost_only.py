import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.ensemble import RUSBoostClassifier


DATA_PATH = "data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv"
MODEL_DIR = "results/models"

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
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


def main():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df["label"]

    print("\nLabel counts:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining RUSBoost only...")

    model = RUSBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nMaking predictions...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nRUSBoost Results")
    print("=" * 50)
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"PR AUC: {average_precision_score(y_test, y_prob):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall/Sensitivity: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(MODEL_DIR, "rusboost.pkl")
    joblib.dump(model, model_path)

    print(f"\nSaved RUSBoost model to: {model_path}")


if __name__ == "__main__":
    main()