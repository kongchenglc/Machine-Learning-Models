import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.classes_dict_ = None
        self.inverse_classes_dict_ = None

    def fit(self, y):
        """Fit label data"""
        # Get unique values of labels and sort them in dictionary order
        self.classes_ = sorted(set(y))  
        # Assign a unique integer to each label
        self.classes_dict_ = {label: idx for idx, label in enumerate(self.classes_)}
        # Inverse dictionary for converting numeric labels back to strings
        self.inverse_classes_dict_ = {idx: label for label, idx in self.classes_dict_.items()}

    def transform(self, y):
        """Transform label data to numbers"""
        return np.array([self.classes_dict_[label] for label in y])

    def fit_transform(self, y):
        """Fit and transform labels"""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """Convert numeric labels back to original labels"""
        return np.array([self.inverse_classes_dict_[label] for label in y])

