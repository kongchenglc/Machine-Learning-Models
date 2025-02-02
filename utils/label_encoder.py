import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.classes_dict_ = None
        self.inverse_classes_dict_ = None

    def fit(self, y):
        """拟合标签数据"""
        # 获取标签的唯一值，并按字典顺序排序
        self.classes_ = sorted(set(y))  
        # 为每个标签分配一个唯一的整数
        self.classes_dict_ = {label: idx for idx, label in enumerate(self.classes_)}
        # 反向字典，便于之后将数字标签转回字符串
        self.inverse_classes_dict_ = {idx: label for label, idx in self.classes_dict_.items()}

    def transform(self, y):
        """将标签数据转换为数字"""
        return np.array([self.classes_dict_[label] for label in y])

    def fit_transform(self, y):
        """拟合并转换标签"""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """将数字标签转换回原始标签"""
        return np.array([self.inverse_classes_dict_[label] for label in y])

