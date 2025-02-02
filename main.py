# main.py

import pandas as pd
from models.linear_regression import DatasetTrainer as linear_regression_trainer
from models.logistic_regression import DatasetTrainer as logistic_regression_trainer

from data_preprocess.mushroom import load_processed_data as mushroom_load_processed_data
from data_preprocess.breast_cancer import load_processed_data as breast_cancer_load_processed_data
from data_preprocess.estate_valuation import load_processed_data as estate_valuation_load_processed_data
from data_preprocess.student_performance import load_processed_data as student_performance_load_processed_data


# from data_preprocess.robot_failure import load_processed_data as robot_failure_load_processed_data


def main():
    # Configure model and dataset
    processed_data = student_performance_load_processed_data()  # Load processed data
    trainer = linear_regression_trainer(processed_data)  # Pass the processed data

    # Load data
    data = trainer.load_dataset()
    if data is None:
        print("\nDataset is empty")
        return

    # Train and evaluate
    X_train, X_test, y_train, y_test = data  # Assume data is returned as a tuple
    result = trainer.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Print results
    print("Training results for the dataset:")
    print(result)


if __name__ == "__main__":
    main()
