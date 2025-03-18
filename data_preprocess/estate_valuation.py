import pandas as pd
import numpy as np
from utils.train_test_split import train_test_split  # Assuming you have this utility


def load_processed_data(
    file_path="./data/real_estate_valuation/real_estate_valuation.xlsx",
):
    # Load the dataset from an Excel file
    df = pd.read_excel(file_path)

    # Assuming the target column is named 'Y house price of unit area' and all others are features
    X = df.drop(
        columns=["Y house price of unit area"]
    )  # Replace 'Y house price of unit area' with your target column name if needed
    y = df["Y house price of unit area"]  # Target variable, adjust as per the dataset's actual column name

    # Standardize the features (zero mean, unit variance)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_scaled = (X - means) / stds

    # Return the preprocessed data using train_test_split
    return train_test_split(X_scaled, y)
