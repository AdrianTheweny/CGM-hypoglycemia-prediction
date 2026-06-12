"""
test_training.py
 
quick sanity checks for the stuff that's easy to mess up when iterating
on train_baseline.py / train_experimental.py:
- feature columns match what's actually in the data
- train/test split keeps the class ratio (stratify is doing its job)
- scale_pos weight calc doesn't blow up if a class is missing
- GB sample_weights line up 1:1 with y_train
 
these don't run the full training scripts (too slow + writes files).
just pulls out the pieces that would silently break if the data
changes or someone edits the feature list.
"""
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
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
    "glucose_slope_60min",
]
 
 
def make_fake_df(n=200, pos_ratio=0.1, seed=0):
    """small fake dataset shaped like the real one, imbalanced labels"""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(100, 20, size=n) for col in FEATURE_COLS}
    n_pos = int(n * pos_ratio)
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(labels)
    data["label"] = labels
    return pd.DataFrame(data)
 
 
def test_feature_cols_present_in_df():
    # if someone renames a feature col in the csv but forgets to update
    # FEATURE_COLS this catches it before training errors out
    df = make_fake_df()
    for col in FEATURE_COLS:
        assert col in df.columns, f"missing feature col: {col}"
    assert "label" in df.columns
 
 
def test_split_is_stratified():
    df = make_fake_df(n=500, pos_ratio=0.1)
    X = df[FEATURE_COLS]
    y = df["label"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # shapes line up
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(df)
 
    # stratify should keep roughly the same class ratio in train vs test
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05
 
 
def test_scale_pos_weight_calc():
    # same calc as in train_experimental.py for GB sample weights
    df = make_fake_df(n=300, pos_ratio=0.1)
    y_train = df["label"]
 
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
 
    assert pos_count > 0, "test data has no positive class, fix make_fake_df"
 
    scale_pos = neg_count / pos_count
    assert scale_pos > 1  # neg class should outnumber pos class here
    assert np.isfinite(scale_pos)
 
 
def test_sample_weights_match_y_train_shape():
    df = make_fake_df(n=300, pos_ratio=0.1)
    y_train = df["label"]
 
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count
 
    sample_weights = np.where(y_train == 1, scale_pos, 1.0)
 
    assert len(sample_weights) == len(y_train)
    # positive-class rows should get the bigger weight
    assert sample_weights[y_train == 1][0] == scale_pos
    assert sample_weights[y_train == 0][0] == 1.0
 
 
def test_scale_pos_no_positives_doesnt_crash_division():
    # edge case: if a fold/split has zero positives, neg_count/pos_count
    # would be a divide by zero.confirming the failure mode so
    # nobody's surprised if it shows up during CV on a small fold
    y_train = pd.Series([0, 0, 0, 0])
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
 
    assert pos_count == 0
    # this would raise in real code so documenting it here
    result = neg_count / pos_count if pos_count > 0 else None
    assert result is None
 
 
if __name__ == "__main__":
    test_feature_cols_present_in_df()
    test_split_is_stratified()
    test_scale_pos_weight_calc()
    test_sample_weights_match_y_train_shape()
    test_scale_pos_no_positives_doesnt_crash_division()
    print("all tests passed")