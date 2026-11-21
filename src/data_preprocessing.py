"""Loading, cleaning and preprocessing for the mushroom dataset."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET = "class"

# Features a person can observe in the field without smelling or spore-printing
FIELD_FEATURES = ["cap-shape", "cap-surface", "cap-color",
                  "bruises", "population", "habitat"]


def load_data(path="mushrooms.csv"):
    return pd.read_csv(path)


def clean_data(df):
    """Mark '?' in stalk-root as missing and drop the zero-variance veil-type column."""
    df_clean = df.copy()
    df_clean["stalk-root"] = df_clean["stalk-root"].replace("?", np.nan)
    df_clean = df_clean.drop(columns=["veil-type"])
    return df_clean


def split_data(df_clean, test_size=0.20):
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            random_state=RANDOM_STATE, stratify=y)


def build_preprocessor(features):
    """Missing values become their own 'missing' category, then one-hot encode.

    No scaling is needed: one-hot columns are already 0/1.
    """
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("cat", categorical_pipeline, list(features))])


def get_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
