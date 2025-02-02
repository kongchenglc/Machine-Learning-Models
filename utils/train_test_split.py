import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split the dataset into training and testing sets.

    :param X: Feature data (DataFrame or NumPy array)
    :param y: Target labels (DataFrame or NumPy array)
    :param test_size: Proportion of the test set, default is 0.2 (20%)
    :param random_seed: Random seed, default is None
    :return: X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)  # Set random seed for reproducibility

    # Generate random indices
    indices = np.random.permutation(len(X))
    
    # Calculate training set size
    train_size = int(len(X) * (1 - test_size))
    
    # Split into training and testing sets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Adapt for Pandas DataFrame and NumPy array
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
