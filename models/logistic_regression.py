import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.coefficients = np.zeros(X_b.shape[1])
        learning_rate = 0.01
        n_iterations = 1000

        for _ in range(n_iterations):
            linear_model = np.dot(X_b, self.coefficients)
            y_pred = self.sigmoid(linear_model)
            gradient = np.dot(X_b.T, (y_pred - y)) / y.size
            self.coefficients -= learning_rate * gradient

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        linear_model = np.dot(X_b, self.coefficients)
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int) # Convert probabilities to class labels

    def score(self, X, y):
        """Calculate accuracy"""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)  # Calculate accuracy
        return accuracy


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = LogisticRegression()  # Use custom logistic regression class
        self.data = processed_data  # Directly pass in processed data

    def load_dataset(self):
        """Return processed data"""
        return self.data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate the model"""
        self.model.fit(X_train, y_train)  # Fit the model
        accuracy = self.model.score(X_test, y_test)  # Calculate accuracy
        return {"accuracy": accuracy}
