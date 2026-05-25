import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)


DATA_PATH = "data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv"
MODEL_DIR = "results/models"
OUTPUT_DIR = "results/safora_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def load_test_data():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


def evaluate_model(model_name, model_path, X_test, y_test):
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall/Sensitivity: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    safe_name = model_name.lower().replace(" ", "_").replace(".", "")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.savefig(f"{OUTPUT_DIR}/{safe_name}_roc_curve.png")
    plt.close()

    # Precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.savefig(f"{OUTPUT_DIR}/{safe_name}_pr_curve.png")
    plt.close()

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{OUTPUT_DIR}/{safe_name}_confusion_matrix.png")
    plt.close()

    return {
        "Model": model_name,
        "ROC_AUC": round(roc_auc, 3),
        "PR_AUC": round(pr_auc, 3),
        "Precision": round(precision, 3),
        "Recall_Sensitivity": round(recall, 3),
        "F1": round(f1, 3)
    }


def main():
    X_test, y_test = load_test_data()

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "RUSBoost": "rusboost.pkl"
    }

    results = []

    for model_name, filename in model_files.items():
        model_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(model_path):
            result = evaluate_model(model_name, model_path, X_test, y_test)
            results.append(result)
        else:
            print(f"Skipping {model_name}: model file not found at {model_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/safora_model_evaluation_results.csv", index=False)

    print("\nSaved Safora evaluation results to:")
    print(f"{OUTPUT_DIR}/safora_model_evaluation_results.csv")
    print("\nSaved graphs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()