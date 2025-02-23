import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def sigmoid(self, z):
        """Sigmoid函数添加数值稳定性处理"""
        z = np.clip(z, -500, 500)  # 防止指数爆炸
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # 统一维度处理
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
            
        # 添加偏置项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 参数初始化
        self.coefficients = np.zeros(X_b.shape[1])
        
        # 确保y为一维数组
        y = np.asarray(y).ravel()
        
        # 优化梯度计算
        for iteration in range(1000):
            linear_model = X_b @ self.coefficients
            y_pred = self.sigmoid(linear_model)
            
            # 统一维度计算
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
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        self.model.fit(X_train, y_train)  # Fit the model
        accuracy = self.model.score(X_test, y_test)  # Calculate accuracy
        return {"accuracy": accuracy}
