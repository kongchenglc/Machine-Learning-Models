import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def sigmoid(self, z):
        """Sigmoid function with numerical stability handling"""
        z = np.clip(z, -500, 500)  # Prevent exponential overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Dimension unification
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
            
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize parameters
        self.coefficients = np.zeros(X_b.shape[1])
        
        # Ensure y is 1D array
        y = np.asarray(y).ravel()
        
        # Optimized gradient calculation
        for iteration in range(1000):
            linear_model = X_b @ self.coefficients
            y_pred = self.sigmoid(linear_model)
            
            # Unified dimension calculation
            error = y_pred - y
            gradient = (X_b.T @ error) / len(y)
            
            self.coefficients -= 0.1 * gradient

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return (self.sigmoid(X_b @ self.coefficients) >= 0.5).astype(int).ravel()

    def score(self, X, y):
        """Calculate accuracy"""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = LogisticRegression()  # Use custom logistic regression
        self.data = processed_data         # Directly use processed data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate the model"""
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        self.model.fit(X_train, y_train)    # Train model
        accuracy = self.model.score(X_test, y_test)  # Get accuracy
        return {"accuracy": accuracy}
