import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3, task='classification'):
        self.k = k
        self.task = task  # Add task type parameter

    def fit(self, X, y):
        """Fit the model using training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the labels for the given test data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Helper function to predict the label for a single sample."""
        # Compute distances from x to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort distances and get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Modified prediction logic for regression support
        if self.task == 'regression':
            return np.mean(k_nearest_labels)  # Return mean for regression
        return Counter(k_nearest_labels).most_common(1)[0][0]  # Classification remains same

    def _euclidean_distance(self, x1, x2):
        """Compute the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))


class DatasetTrainer:
    def __init__(self, processed_data):
        self.data = processed_data
        # Convert y to Series if it's a DataFrame
        y = self.data[2].squeeze()  # Add squeeze() to handle DataFrame
        if self._is_regression(y):
            self.model = KNN(task='regression')
        else:
            self.model = KNN(task='classification')

    def _is_regression(self, y):
        # Determine regression task: float dtype or >25% unique values
        return y.dtype == float or len(np.unique(y)) > 0.25 * len(y)

    def load_dataset(self):
        return self.data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Convert all features to numerical values
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        # Return appropriate metrics
        if self.model.task == 'classification':
            accuracy = np.mean(predictions == y_test)
            return {"accuracy": accuracy}
        else:
            mse = np.mean((predictions - y_test) ** 2)
            return {"mse": mse, "rmse": np.sqrt(mse)}


# Example usage:
if __name__ == "__main__":
    # Example data for classification (2D points, binary classification)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 5], [7, 8], [8, 7]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # Initialize the KNN model with k=3
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Test data
    X_test = np.array([[5, 5], [1, 1]])

    # Make predictions
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)

    # New regression demo
    X_reg = np.array([[1], [2], [3], [4], [5], [6]])
    y_reg = np.array([2.1, 3.9, 6.2, 8.1, 9.8, 12.1])
    
    knn_reg = KNN(k=2, task='regression')
    knn_reg.fit(X_reg, y_reg)
    print("Regression prediction:", knn_reg.predict([[3.5]]))  # Should output ≈5.05
