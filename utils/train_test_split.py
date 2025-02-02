import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    将数据集拆分为训练集和测试集。

    :param X: 特征数据 (DataFrame 或 NumPy 数组)
    :param y: 目标标签 (DataFrame 或 NumPy 数组)
    :param test_size: 测试集比例，默认 0.2 (20%)
    :param random_seed: 随机种子，默认为 None
    :return: X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)  # 设置随机种子以保证可复现性

    # 生成随机索引
    indices = np.random.permutation(len(X))
    
    # 计算训练集大小
    train_size = int(len(X) * (1 - test_size))
    
    # 划分训练集和测试集
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 适配 Pandas DataFrame 和 NumPy 数组
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
