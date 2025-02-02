import pandas as pd
import numpy as np
from utils.label_encoder import LabelEncoder
from utils.train_test_split import train_test_split


def add_time_features(df):
    """为每个序列添加统计特征"""
    df["mean"] = df["sequence"].apply(lambda x: x.mean())
    df["std"] = df["sequence"].apply(lambda x: x.std())
    df["max"] = df["sequence"].apply(lambda x: x.max(axis=0).mean())
    df["min"] = df["sequence"].apply(lambda x: x.min(axis=0).mean())
    return df


def parse_robot_file(file_path):
    """
    解析包含标签块的机器人执行数据文件
    :param file_path: 数据文件路径
    :return: 包含标签和时序数据的DataFrame
    """
    data = []
    current_label = None
    current_sequence = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # 检测标签行（非数字开头）
            if not line.split("\t")[0].lstrip("-").isdigit():
                # 保存前一个标签的数据
                if current_label is not None and len(current_sequence) == 15:
                    data.append(
                        {"label": current_label, "sequence": np.array(current_sequence)}
                    )
                current_label = line.strip()
                current_sequence = []
            elif line != "":
                # 解析数据行
                values = [float(x) for x in line.split("\t")]
                current_sequence.append(values)

        # 添加最后一个标签的数据
        if current_label is not None and len(current_sequence) == 15:
            data.append(
                {"label": current_label, "sequence": np.array(current_sequence)}
            )
    return pd.DataFrame(data)


def load_processed_data(output_format="flatten"):
    """
    完整数据处理流程
    :param output_format: 'flatten' 或 'time_series'
    :return: 特征数据和标签
    """
    # 读取并解析所有文件
    all_data = []
    for i in range(1, 6):
        df = parse_robot_file(f"./data/robotfailure/lp{i}.data")
        df["file_source"] = f"lp{i}"
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # 添加统计特征
    full_df = add_time_features(full_df)

    # 标签编码
    le = LabelEncoder()
    labels = le.fit_transform(full_df["label"])

    # 根据需求返回不同格式
    if output_format == "flatten":
        X = np.array([seq.flatten() for seq in full_df["sequence"]])

        return train_test_split(X, labels)
    elif output_format == "time_series":
        return np.array(full_df["sequence"]), labels
    else:
        raise ValueError("Supported formats: 'flatten' or 'time_series'")
