import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from utils.train_test_split import train_test_split


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

    # Return the preprocessed data
    return train_test_split(X_scaled, y)
