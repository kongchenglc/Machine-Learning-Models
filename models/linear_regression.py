import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """拟合线性回归模型"""
        # 添加常数项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X前添加一列1
        # 计算最小二乘法的系数
        self.coefficients = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        """使用拟合的模型进行预测"""
        return np.dot(X, self.coefficients) + self.intercept

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        # 假设阈值为0.5，将预测值转换为分类结果
        y_pred_class = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_pred_class == y)  # 计算准确率
        return accuracy


class DatasetTrainer:
    def __init__(self, processed_data):
        self.model = LinearRegression()  # 使用自定义线性回归类
        self.data = processed_data  # 直接传入处理好的数据

    def load_dataset(self):
        """返回处理好的数据"""
        return self.data

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """训练和评估模型"""
        self.model.fit(X_train, y_train)  # 拟合模型
        mse = np.mean((self.model.predict(X_test) - y_test) ** 2)  # 计算均方误差
        accuracy = self.model.score(X_test, y_test)  # 计算准确率
        return mse, accuracy
