import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


DATA_PATH = "data/processed/8May_labeled_model_ready_cgm_30patient_subset.csv"
RESULTS_DIR = "results/safora_evaluation"
METRICS_DIR = "results/metrics"
MODELS_DIR = "results/models"

MODELS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "Gradient Boosting": os.path.join(MODELS_DIR, "gradient_boosting.pkl"),
    "RUSBoost": os.path.join(MODELS_DIR, "rusboost.pkl"),
}


def clean_filename(name):
    """
    Changes the model name into a cleaner filename.

    This is mostly just so the saved plot names do not have spaces in them.
    """
    return name.lower().replace(" ", "_")


def load_data():
    """
    Loads the processed CGM data and separates the labels from the features.

    The model should only get feature columns as input, so this removes the
    target column and also removes columns like patient_id and timestamp that
    should not be used directly for prediction.
    """
    data = pd.read_csv(DATA_PATH)

    # find the target column, since different files may name it slightly differently
    if "hypoglycemia_event" in data.columns:
        target_col = "hypoglycemia_event"
    elif "label" in data.columns:
        target_col = "label"
    elif "target" in data.columns:
        target_col = "target"
    else:
        raise ValueError("Could not find target column.")

    y = data[target_col]

    # start by dropping the label column so it is not accidentally used as a feature
    drop_cols = [target_col]

    # these columns are useful for organizing data, but not as direct model inputs
    if "patient_id" in data.columns:
        drop_cols.append("patient_id")

    if "timestamp_numeric" in data.columns:
        drop_cols.append("timestamp_numeric")

    if "timestamp" in data.columns:
        drop_cols.append("timestamp")

    X = data.drop(columns=drop_cols)

    return X, y


def evaluate_model(model_name, model_path, X, y):
    """
    Loads one trained model and evaluates how well it predicts hypoglycemia.

    This calculates the main metrics we need for the project, including ROC AUC,
    PR AUC, precision, recall/sensitivity, F1 score, confusion matrix, and the
    ROC/precision-recall plots.
    """
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: model file not found at {model_path}")
        return None

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # load the saved model file
    model = joblib.load(model_path)

    # get the class predictions
    y_pred = model.predict(X)

    # get prediction scores/probabilities for ROC and PR curves
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        raise ValueError(f"{model_name} does not provide probability scores.")

    # calculate the main evaluation metrics
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall/Sensitivity: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    cm = confusion_matrix(y, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    file_name = clean_filename(model_name)

    # save confusion matrix figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"{file_name}_confusion_matrix.png"),
        dpi=300,
    )
    plt.close()

    # save ROC curve with the AUC value on the graph
    fpr, tpr, _ = roc_curve(y, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"{file_name}_roc_curve.png"),
        dpi=300,
    )
    plt.close()

    # save precision-recall curve with the PR AUC value on the graph
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"{file_name}_pr_curve.png"),
        dpi=300,
    )
    plt.close()

    return {
        "Model": model_name,
        "ROC_AUC": round(roc_auc, 3),
        "PR_AUC": round(pr_auc, 3),
        "Precision": round(precision, 3),
        "Recall_Sensitivity": round(recall, 3),
        "F1": round(f1, 3),
    }


def main():
    """
    Runs the full evaluation workflow for all the models.

    This makes the results folders, loads the data, evaluates each model that
    exists, and then saves the final metrics and plots.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X, y = load_data()

    results = []

    # go through each model and evaluate it if the file exists
    for model_name, model_path in MODELS.items():
        result = evaluate_model(model_name, model_path, X, y)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)

    safora_results_path = os.path.join(
        RESULTS_DIR,
        "safora_model_evaluation_results.csv",
    )

    model_results_path = os.path.join(
        METRICS_DIR,
        "model_results.csv",
    )

    # save the metrics in CSV form so they are easier to use in the report
    results_df.to_csv(safora_results_path, index=False)
    results_df.to_csv(model_results_path, index=False)

    print("\nSaved Safora evaluation results to:")
    print(safora_results_path)

    print("\nSaved model results to:")
    print(model_results_path)

    print("\nSaved graphs to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()