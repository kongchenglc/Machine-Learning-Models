# data_preprocess/breast_cancer.py

import pandas as pd
import numpy as np

def load_processed_data():
    """
    加载乳腺癌数据集
    返回: DataFrame - 包含特征和标签的数据集
    """
    # 定义列名
    columns = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]

    # 加载数据
    data = pd.read_csv("./data/breastcancer/wdbc.data", header=None, names=columns)
    return preprocess_breast_cancer_data(data, "diagnosis")


def preprocess_breast_cancer_data(data, target_column):
    """
    预处理乳腺癌数据集
    返回: (X_train, X_test, y_train, y_test) - 特征和标签的训练集和测试集
    """
    # 处理标签
    y = (data[target_column] == "M").astype(int)  # 将'M'转换为1，'B'转换为0
    X = data.drop(columns=[target_column, "id"])  # 删除标签和ID列

    # 标准化特征
    X = (X - X.mean()) / X.std()

    # 划分训练集和测试集
    # 设置随机种子
    np.random.seed(42)
    # 创建一个随机排列的索引
    indices = np.random.permutation(len(X))
    # 80% 作为训练集，20% 作为测试集
    train_size = int(len(X) * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
