# utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_and_target(df: pd.DataFrame, target_col: str = "SalePrice"):
    """Separate features and target from a DataFrame."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def ensure_columns_match(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Ensure test_df has same columns as train_df.
    Extra columns in test_df are dropped, missing columns are filled with 0.
    """
    missing_cols = [c for c in train_df.columns if c not in test_df.columns]
    for col in missing_cols:
        test_df[col] = 0

    extra_cols = [c for c in test_df.columns if c not in train_df.columns]
    if extra_cols:
        test_df = test_df.drop(columns=extra_cols)

    # Reorder
    test_df = test_df[train_df.columns]
    return test_df
