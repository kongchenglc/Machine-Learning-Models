# data_preprocess/mushroom.py

import pandas as pd
import numpy as np


def load_processed_data():
    """
    Load mushroom dataset
    Returns: DataFrame - Dataset containing features and labels
    """
    # Define column names
    columns = [
        "class",
        "cap_shape",
        "cap_surface",
        "cap_color",
        "bruises",
        "odor",
        "gill_attachment",
        "gill_spacing",
        "gill_size",
        "gill_color",
        "stalk_shape",
        "stalk_root",
        "stalk_surface_above_ring",
        "stalk_surface_below_ring",
        "stalk_color_above_ring",
        "stalk_color_below_ring",
        "veil_type",
        "veil_color",
        "ring_number",
        "ring_type",
        "spore_print_color",
        "population",
        "habitat",
    ]

    # Load data
    data = pd.read_csv(
        "./data/mushroom/agaricus-lepiota.data", header=None, names=columns
    )
    return preprocess_mushroom_data(data, "class")


def preprocess_mushroom_data(data, target_column):
    """
    Preprocess mushroom dataset
    Returns: (X_train, X_test, y_train, y_test) - Training and testing sets of features and labels
    """
    # Replace question marks with NaN
    data.replace("?", pd.NA, inplace=True)

    # Drop rows containing NaN
    data.dropna(how='any', inplace=True)

    # Process labels
    y = (data[target_column] == "p").astype(int)  # Convert 'poisonous' to 1 and 'edible' to 0
    X = data.drop(columns=[target_column])  # Drop label column

    # Convert categorical features to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Split into training and testing sets
    np.random.seed()
    # Create a randomly permuted index
    indices = np.random.permutation(len(X))
    # 80% for training set, 20% for testing set
    train_size = int(len(X) * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test