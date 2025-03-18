import pandas as pd
import numpy as np
from utils.train_test_split import train_test_split

# Define student identity columns for merging
MERGE_COLUMNS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]


def load_processed_data():
    # Load the datasets from CSV files
    df_mat = pd.read_csv("./data/student_performance/student-mat.csv", sep=";")
    df_por = pd.read_csv("./data/student_performance/student-por.csv", sep=";")

    # Merge datasets based on student identity columns
    df = pd.merge(
        df_mat, df_por, on=MERGE_COLUMNS, suffixes=("_mat", "_por"), how="outer"
    )

    target_columns = ["G1_mat", "G2_mat", "G3_mat", "G1_por", "G2_por", "G3_por"]
    feature_columns = [col for col in df.columns if col not in target_columns]

    X = df[feature_columns].copy()

    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = X[column].astype("category").cat.codes

    y = df[target_columns]  # Targets

    # Convert categorical features to numeric codes
    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = pd.Categorical(X[column]).codes  # Convert categories to numeric

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric values to NaN

    # Handle missing values by filling with the mean
    X = X.fillna(X.mean())

    # Standardize features (zero mean, unit variance)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_scaled = (X - means) / stds

    # You can use G1, G2, and G3 separately, or combine them
    y = y.mean(axis=1)  # Average across all grades (optional)

    # Return the processed dataset
    return train_test_split(X_scaled, y)
