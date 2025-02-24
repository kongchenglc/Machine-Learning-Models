import numpy as np
import pandas as pd


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # Store unique class labels
        self.class_prior = {}  # Prior probability P(C_k)
        self.mean = {}  # Mean of each feature per class
        self.variance = {}  # Variance of each feature per class

    def fit(self, X, y):
        """Convert data to numpy arrays and ensure numerical types"""
        self.X = np.asarray(X, dtype=np.float64)  # Ensure numerical input
        self.y = np.asarray(y).ravel()  # Flatten to 1D array
        
        # Convert to pandas Series with proper typing
        self.classes = pd.Series(np.unique(self.y))
        self.class_prior = pd.Series(
            [np.mean(self.y == c) for c in self.classes],
            index=self.classes
        )
        
        # Store means and variances as numpy arrays
        self.means = {
            cls: np.asarray([self.X[self.y == cls, i].mean() for i in range(self.X.shape[1])], 
                           dtype=np.float64)
            for cls in self.classes
        }
        self.variances = {
            cls: np.asarray([self.X[self.y == cls, i].var() for i in range(self.X.shape[1])],
                           dtype=np.float64)
            for cls in self.classes
        }

    def gaussian_pdf(self, x, mean, variance):
        """Compute Gaussian probability density function for a feature."""
        epsilon = 1e-9  # Avoid division by zero
        coefficient = 1.0 / np.sqrt(2 * np.pi * (variance + epsilon))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (variance + epsilon)))
        return coefficient * exponent

    def compute_class_probabilities(self, X):
        """Compute posterior probabilities for each sample."""
        probabilities = []
        for sample in X:
            class_probs = {}
            for cls in self.classes:
                prior_log = np.log(self.class_prior[cls])
                # Use logsumexp for numerical stability
                log_likelihoods = [
                    self._log_gaussian_pdf(
                        x_i, 
                        self.means[cls][i], 
                        self.variances[cls][i]
                    )
                    for i, x_i in enumerate(sample)
                ]
                class_probs[cls] = prior_log + np.sum(log_likelihoods)
            probabilities.append(class_probs)
        return probabilities

    def _log_gaussian_pdf(self, x, mean, variance):
        """Handle numerical type consistency"""
        x = np.asarray(x, dtype=np.float64)  # Ensure numerical type
        eps = 1e-6
        var = variance + eps
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    def predict(self, X):
        """Predict the class with the highest probability for each sample."""
        probabilities = self.compute_class_probabilities(X)
        return np.array(
            [max(prob, key=prob.get) for prob in probabilities]
        )  # Choose the class with the highest probability

    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = GaussianNaiveBayes()
        self.data = processed_data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Convert one-hot encoded labels to class indices
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        return {"accuracy": accuracy}
