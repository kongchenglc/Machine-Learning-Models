import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, task="classification"):
        """
        Decision Tree that supports both classification and regression.

        :param max_depth: Maximum depth of the tree.
        :param task: "classification" or "regression".
        """
        self.max_depth = max_depth
        self.task = task
        self.tree = None

    def fit(self, X, y):
        """Train the decision tree model."""
        self.task = (
            "classification"
            if len(np.unique(y)) < 10 and y.dtype in [int, np.int32, np.int64]
            else "regression"
        )
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """Predict class labels or regression values for given samples."""
        return np.array([self._traverse_tree(sample, self.tree) for sample in X])

    def _best_split(self, X, y):
        """Find the best feature and threshold for splitting."""
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        best_score = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                score = self._impurity_score(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score, best_feature, best_threshold = score, feature, threshold

        return best_feature, best_threshold

    def _impurity_score(self, left_labels, right_labels):
        """Compute impurity score (Gini for classification, MSE for regression)."""

        def gini(labels):
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / counts.sum()
            return 1 - np.sum(probs**2)

        def mse(values):
            if len(values) == 0:
                return 0
            mean_value = np.mean(values)
            return np.mean((values - mean_value) ** 2)

        num_left, num_right = len(left_labels), len(right_labels)
        total = num_left + num_right

        if self.task == "classification":
            return (num_left / total) * gini(left_labels) + (num_right / total) * gini(
                right_labels
            )
        else:
            return (num_left / total) * mse(left_labels) + (num_right / total) * mse(
                right_labels
            )

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Stopping conditions
        if len(unique_labels) == 1 or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return np.mean(y) if self.task == "regression" else np.bincount(y).argmax()

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y) if self.task == "regression" else np.bincount(y).argmax()

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _traverse_tree(self, sample, node):
        """Recursively traverse the decision tree to get the prediction."""
        if isinstance(node, dict):
            if sample[node["feature"]] <= node["threshold"]:
                return self._traverse_tree(sample, node["left"])
            else:
                return self._traverse_tree(sample, node["right"])
        return (
            node  # Leaf node returns class (classification) or mean value (regression)
        )


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = DecisionTree()  # Use default parameters initially
        self.data = processed_data

    def load_dataset(self):
        """Return processed data"""
        return self.data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate the decision tree model"""
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        # Calculate appropriate metrics based on task type
        if self.model.task == "classification":
            accuracy = np.mean(predictions == y_test)
            return {"accuracy": accuracy}
        else:
            mse = np.mean((predictions - y_test) ** 2)
            return {"mse": mse, "rmse": np.sqrt(mse)}


# Testing the Unified Decision Tree
if __name__ == "__main__":
    # Classification Example
    X_train_cls = np.array([[2, 3], [3, 3], [4, 2], [6, 6], [7, 7], [8, 6]])
    y_train_cls = np.array([0, 0, 0, 1, 1, 1])  # Binary classes

    # Regression Example
    X_train_reg = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y_train_reg = np.array(
        [1.2, 2.3, 3.1, 4.8, 5.0, 6.7, 7.2, 8.9]
    )  # Continuous values

    # Classification Test
    X_test_cls = np.array([[5, 5], [3, 2]])
    model_cls = DecisionTree(max_depth=3, task="classification")
    model_cls.fit(X_train_cls, y_train_cls)
    predictions_cls = model_cls.predict(X_test_cls)
    print("Classification Predictions:", predictions_cls)

    # Regression Test
    X_test_reg = np.array([[2.5], [6.5]])
    model_reg = DecisionTree(max_depth=3, task="regression")
    model_reg.fit(X_train_reg, y_train_reg)
    predictions_reg = model_reg.predict(X_test_reg)
    print("Regression Predictions:", predictions_reg)
