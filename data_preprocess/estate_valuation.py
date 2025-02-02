import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo


def load_processed_data():
    # Fetch dataset
    real_estate_valuation = fetch_ucirepo(id=477)

    # Data (as pandas dataframes)
    X = real_estate_valuation.data.features
    y = real_estate_valuation.data.targets

    # Standardize the features
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_scaled = (X - means) / stds

    # Split data into training and testing sets
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
