import pandas as pd
import numpy as np
from utils.label_encoder import LabelEncoder
from utils.train_test_split import train_test_split


def add_time_features(df):
    """Add statistical features for each sequence"""
    df["mean"] = df["sequence"].apply(lambda x: x.mean())
    df["std"] = df["sequence"].apply(lambda x: x.std())
    df["max"] = df["sequence"].apply(lambda x: x.max(axis=0).mean())
    df["min"] = df["sequence"].apply(lambda x: x.min(axis=0).mean())
    return df


def parse_robot_file(file_path):
    """
    Parse the robot execution data file containing label blocks
    :param file_path: Path to the data file
    :return: DataFrame containing labels and time series data
    """
    data = []
    current_label = None
    current_sequence = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Detect label line (not starting with a number)
            if not line.split("\t")[0].lstrip("-").isdigit():
                # Save the previous label's data
                if current_label is not None and len(current_sequence) == 15:
                    data.append(
                        {"label": current_label, "sequence": np.array(current_sequence)}
                    )
                current_label = line.strip()
                current_sequence = []
            elif line != "":
                # Parse data line
                values = [float(x) for x in line.split("\t")]
                current_sequence.append(values)

        # Add the last label's data
        if current_label is not None and len(current_sequence) == 15:
            data.append(
                {"label": current_label, "sequence": np.array(current_sequence)}
            )
    return pd.DataFrame(data)


def load_processed_data():
    """
    Complete data processing workflow
    :return: Feature data and labels
    """
    # Read and parse all files
    all_data = []
    for i in range(1, 6):
        df = parse_robot_file(f"./data/robotfailure/lp{i}.data")
        df["file_source"] = f"lp{i}"
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # Add statistical features
    full_df = add_time_features(full_df)

    # Label encoding
    le = LabelEncoder()
    labels = le.fit_transform(full_df["label"])

    X = np.array([seq.flatten() for seq in full_df["sequence"]])
    return train_test_split(X, labels)
