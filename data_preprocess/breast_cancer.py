# data_preprocess/breast_cancer.py

import pandas as pd
import numpy as np
from utils.train_test_split import train_test_split

def load_processed_data():
    """
    Load breast cancer dataset
    Returns: DataFrame - Dataset containing features and labels
    """
    # Define column names
    columns = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]

    # Load data
    data = pd.read_csv("./data/breastcancer/wdbc.data", header=None, names=columns)
    return preprocess_breast_cancer_data(data, "diagnosis")


def preprocess_breast_cancer_data(data, target_column):
    """
    Preprocess breast cancer dataset
    Returns: (X_train, X_test, y_train, y_test) - Training and testing sets of features and labels
    """
    # Process labels
    y = (data[target_column] == "M").astype(int)  # Convert 'M' to 1 and 'B' to 0
    X = data.drop(columns=[target_column, "id"])  # Drop label and ID columns

    # Standardize features
    X = (X - X.mean()) / X.std()

    return train_test_split(X, y)