import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from utils.train_test_split import train_test_split


def load_processed_data():
    # Fetch dataset
    student_performance = fetch_ucirepo(id=320)  # student performance dataset ID

    # Data (as pandas dataframes)
    X = student_performance.data.features
    y = student_performance.data.targets

    # 1. Check for missing values (if any)
    if X.isnull().any().any():
        X = X.fillna(
            X.mean()
        )  # Replace missing values with the mean of the respective column

    # 2. Ensure all columns are numeric (convert categorical columns to numeric)
    for column in X.select_dtypes(include=["object"]).columns:
        X.loc[:, column] = pd.Categorical(
            X[column]
        ).codes  # Convert categorical to numeric codes

    # 3. Ensure that the data is of numeric type (to avoid dtype issues)
    X = X.apply(
        pd.to_numeric, errors="coerce"
    )  # Force conversion to numeric, non-numeric values will be NaN

    # 4. Fill any remaining NaN values (if any) after conversion
    X = X.fillna(X.mean())  # Fill NaNs with column mean

    # 5. Standardize the features (using mean and standard deviation)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_scaled = (X - means) / stds

    # Return the preprocessed data
    return train_test_split(X_scaled, y)
