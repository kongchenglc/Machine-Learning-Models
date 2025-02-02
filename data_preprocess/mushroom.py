# data_preprocess/mushroom.py

import pandas as pd
import numpy as np
from utils.train_test_split import train_test_split


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
    data.dropna(how="any", inplace=True)

    # Process labels
    y = (data[target_column] == "p").astype(
        int
    )  # Convert 'poisonous' to 1 and 'edible' to 0
    X = data.drop(columns=[target_column])  # Drop label column

    # Convert categorical features to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y)
