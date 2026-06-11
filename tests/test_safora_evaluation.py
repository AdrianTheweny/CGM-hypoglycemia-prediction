import os
import sys

# ------------------------------------------------------------
# Testing Overview
#
# This file has both unit tests and integration tests.
#
# Unit tests:
# - Check small helper functions like clean_filename().
# - Check that DATA_PATH, MODELS, and EXPECTED_FEATURES are set up correctly.
#
# Integration tests:
# - Check that the real processed CSV exists and loads.
# - Check that load_data() returns valid feature/label data.
# - Check that preprocessing removes columns we should not use directly.
# - Check that expected CGM feature columns are present.
#
# These tests check that the code, data loading, preprocessing,
# and project setup work as expected. Model performance is checked
# separately in safora_evaluation.py using ROC AUC, PR AUC,
# precision, recall, F1 score, confusion matrices, and ROC/PR curves.
#
# To run these tests successfully, the processed dataset must be in:
# data/processed/
#
# If trained model files are missing, run:
#
# python3 src/train_baseline.py
# python3 src/train_rusboost_only.py
#
# before running:
#
# python3 src/safora_evaluation.py
# ------------------------------------------------------------

# lets this test file import safora_evaluation.py from src
sys.path.append(os.path.abspath("src"))

from safora_evaluation import load_data, clean_filename, MODELS, DATA_PATH


EXPECTED_FEATURES = [
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
    "glucose_slope_60min",
]


def test_clean_filename():
    """
    Checks that model names get converted into filenames correctly.
    """
    assert clean_filename("Random Forest") == "random_forest"
    assert clean_filename("Gradient Boosting") == "gradient_boosting"
    assert clean_filename("Logistic Regression") == "logistic_regression"


def test_clean_filename_lowercase():
    """
    Checks that clean_filename makes names lowercase.
    """
    assert clean_filename("RUSBoost") == "rusboost"


def test_clean_filename_replaces_spaces():
    """
    Checks that spaces are changed into underscores.
    """
    assert clean_filename("Logistic Regression") == "logistic_regression"


def test_data_path_is_string():
    """
    Checks that DATA_PATH is stored as text.
    """
    assert isinstance(DATA_PATH, str)


def test_data_path_points_to_csv():
    """
    Checks that DATA_PATH points to a csv file.
    """
    assert DATA_PATH.endswith(".csv")


def test_data_path_points_to_processed_folder():
    """
    Checks that the dataset path points to the processed data folder.
    """
    assert DATA_PATH.startswith("data/processed")


def test_data_file_exists():
    """
    Checks that the processed CSV is actually where the code expects it.
    """
    assert os.path.exists(DATA_PATH)


def test_expected_features_list_not_empty():
    """
    Checks that the expected feature list actually has columns in it.
    """
    assert len(EXPECTED_FEATURES) > 0


def test_expected_features_are_strings():
    """
    Checks that all expected feature names are stored as text.
    """
    for col in EXPECTED_FEATURES:
        assert isinstance(col, str)


def test_expected_features_have_no_duplicates():
    """
    Checks that the expected feature list does not repeat columns.
    """
    assert len(EXPECTED_FEATURES) == len(set(EXPECTED_FEATURES))


def test_load_data_returns_matching_rows():
    """
    Checks that load_data() returns features and labels with matching rows.
    """
    X, y = load_data()

    # make sure both X and y contain data
    assert len(X) > 0
    assert len(y) > 0

    # every row should have a matching label
    assert len(X) == len(y)


def test_dataset_is_large_enough():
    """
    Makes sure the real dataset loaded and not some empty or wrong file.
    """
    X, y = load_data()

    assert X.shape[0] > 1000
    assert y.shape[0] > 1000


def test_target_is_binary():
    """
    Checks that the target labels are only 0 and 1.
    """
    X, y = load_data()

    # hypoglycemia prediction is a binary classification problem
    assert set(y.unique()).issubset({0, 1})


def test_target_has_both_classes():
    """
    Checks that both non-hypoglycemia and hypoglycemia labels are present.
    """
    X, y = load_data()

    assert 0 in set(y.unique())
    assert 1 in set(y.unique())


def test_target_values_are_integers():
    """
    Checks that target labels are stored as integer-like values.
    """
    X, y = load_data()

    assert all(float(value).is_integer() for value in y.unique())


def test_no_missing_values_in_target():
    """
    Checks that the target labels do not contain missing values.
    """
    X, y = load_data()

    # every example should have a label
    assert y.isnull().sum() == 0


def test_expected_feature_columns_exist():
    """
    Checks that all expected CGM features are still in the feature matrix.
    """
    X, y = load_data()

    for col in EXPECTED_FEATURES:
        assert col in X.columns


def test_loaded_features_match_expected_features():
    """
    Checks that the loaded feature columns match the expected feature list.
    """
    X, y = load_data()

    assert set(X.columns) == set(EXPECTED_FEATURES)


def test_nonfeature_columns_removed():
    """
    Checks that columns like patient_id, timestamp, and label are not features.
    """
    X, y = load_data()

    # these columns should not be model inputs
    assert "patient_id" not in X.columns
    assert "timestamp_numeric" not in X.columns
    assert "timestamp" not in X.columns
    assert "label" not in X.columns
    assert "hypoglycemia_event" not in X.columns


def test_feature_matrix_has_correct_number_of_columns():
    """
    Checks that load_data() returns the expected number of feature columns.
    """
    X, y = load_data()

    assert X.shape[1] == len(EXPECTED_FEATURES)


def test_features_are_numeric():
    """
    Checks that all model input columns are numeric.
    """
    X, y = load_data()

    assert all(dtype.kind in "biufc" for dtype in X.dtypes)


def test_no_missing_values_in_features():
    """
    Checks that the feature matrix does not contain missing values.
    """
    X, y = load_data()

    assert X.isnull().sum().sum() == 0


def test_target_length_matches_feature_rows():
    """
    Checks that the number of labels matches the number of feature rows.
    """
    X, y = load_data()

    assert y.shape[0] == X.shape[0]


def test_models_dictionary_has_expected_models():
    """
    Checks that all models used in the project are listed.
    """
    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "RUSBoost",
    }

    assert expected_models.issubset(MODELS.keys())


def test_models_dictionary_has_four_models():
    """
    Checks that the project is set up to evaluate four models.
    """
    assert len(MODELS) == 4


def test_models_dictionary_is_not_empty():
    """
    Checks that the MODELS dictionary actually has model entries.
    """
    assert len(MODELS) > 0


def test_model_names_are_strings():
    """
    Checks that all model names are stored as text.
    """
    for model_name in MODELS.keys():
        assert isinstance(model_name, str)


def test_model_names_are_not_empty():
    """
    Checks that no model name is blank.
    """
    for model_name in MODELS.keys():
        assert len(model_name) > 0


def test_models_dictionary_values_are_strings():
    """
    Checks that each model path is stored as text.
    """
    for model_path in MODELS.values():
        assert isinstance(model_path, str)


def test_model_paths_are_pickle_files():
    """
    Checks that all saved model paths point to pkl files.
    """
    for model_path in MODELS.values():
        assert model_path.endswith(".pkl")


def test_model_paths_are_in_results_models_folder():
    """
    Checks that the model paths are pointing to the results/models folder.
    """
    for model_path in MODELS.values():
        assert model_path.startswith("results/models")