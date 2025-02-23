import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """Fit the linear regression model"""
        # Add constant term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of 1s to X
        # Calculate coefficients using the least squares method
        self.coefficients = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        """Make predictions using the fitted model"""
        return np.dot(X, self.coefficients) + self.intercept


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = LinearRegression()  # Use custom linear regression class
        self.data = processed_data  # Directly pass in processed data

    def load_dataset(self):
        """Return processed data"""
        return self.data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate the model"""
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        self.model.fit(X_train, y_train)  # Fit the model
        mse = np.mean(
            (self.model.predict(X_test) - y_test) ** 2
        )  # Calculate mean squared error
        return {"mse": mse, "rmse": np.sqrt(mse)}
