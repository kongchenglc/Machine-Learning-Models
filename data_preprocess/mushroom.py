# data_preprocess/mushroom.py

import pandas as pd
import numpy as np


def load_processed_data():
    """
    加载蘑菇数据集
    返回: DataFrame - 包含特征和标签的数据集
    """
    # 定义列名
    columns = [
        "class",
        "cap_shape",
        "cap_surface",
        "cap_color",
        "bruises",
        "odor",
        "gill_attachment",
        "gill_spacing",
        "gill_size",
        "gill_color",
        "stalk_shape",
        "stalk_root",
        "stalk_surface_above_ring",
        "stalk_surface_below_ring",
        "stalk_color_above_ring",
        "stalk_color_below_ring",
        "veil_type",
        "veil_color",
        "ring_number",
        "ring_type",
        "spore_print_color",
        "population",
        "habitat",
    ]

    # 加载数据
    data = pd.read_csv(
        "./data/mushroom/agaricus-lepiota.data", header=None, names=columns
    )
    return preprocess_mushroom_data(data, "class")


def preprocess_mushroom_data(data, target_column):
    """
    预处理蘑菇数据集
    返回: (X_train, X_test, y_train, y_test) - 特征和标签的训练集和测试集
    """
    # 将问号替换为 NaN
    data.replace("?", pd.NA, inplace=True)

    # 删除包含 NaN 的行
    data.dropna(how='any', inplace=True)

    # 处理标签
    y = (data[target_column] == "p").astype(
        int
    )  # 将'poisonous'转换为1，'edible'转换为0
    X = data.drop(columns=[target_column])  # 删除标签列

    # 将分类特征转换为虚拟变量
    X = pd.get_dummies(X, drop_first=True)

    # 划分训练集和测试集
    np.random.seed()
    # 创建一个随机排列的索引
    indices = np.random.permutation(len(X))
    # 80% 作为训练集，20% 作为测试集
    train_size = int(len(X) * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
