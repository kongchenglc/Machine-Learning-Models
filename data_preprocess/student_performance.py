import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo


def load_processed_data():
    # Fetch dataset
    student_performance = fetch_ucirepo(id=320)  # student performance dataset ID

    # Data (as pandas dataframes)
    X = student_performance.data.features
    y = student_performance.data.targets

    print(y)

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

    # 6. Split data into training and testing sets
    np.random.seed(42)  # Ensure reproducibility
    indices = np.random.permutation(len(X_scaled))  # Shuffle indices
    train_size = int(len(X_scaled) * 0.8)  # 80% for training
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X_scaled.iloc[train_indices]
    X_test = X_scaled.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Return the preprocessed data
    return X_train, X_test, y_train, y_test
