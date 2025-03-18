import pandas as pd
import numpy as np
from models.linear_regression import DatasetTrainer as linear_regression
from models.logistic_regression import DatasetTrainer as logistic_regression
from models.decision_tree import DatasetTrainer as decision_tree
from models.knn import DatasetTrainer as knn
from models.bayesian_classifier import DatasetTrainer as bayesian_classifier

from data_preprocess.mushroom import load_processed_data as mushroom_load_processed_data
from data_preprocess.breast_cancer import load_processed_data as breast_cancer_load_processed_data
from data_preprocess.estate_valuation import load_processed_data as estate_valuation_load_processed_data
from data_preprocess.student_performance import load_processed_data as student_performance_load_processed_data
from data_preprocess.robot_failure import load_processed_data as robot_failure_load_processed_data

MODEL_MAP = {
    "linear_regression": linear_regression,
    "logistic_regression": logistic_regression,
    "decision_tree": decision_tree,
    "knn": knn,
    "bayesian_classifier": bayesian_classifier
}

DATASET_MAP = {
    "mushroom": mushroom_load_processed_data,
    "breast_cancer": breast_cancer_load_processed_data,
    "estate_valuation": estate_valuation_load_processed_data,
    "student_performance": student_performance_load_processed_data,
    "robot_failure": robot_failure_load_processed_data
}

def main():
    for model_name in MODEL_MAP.keys():
        for dataset_name in DATASET_MAP.keys():
            print(f"\n{'-'*40}")
            print(f"Running {model_name} on {dataset_name}")
            print(f"{'-'*40}")
            
            try:
                dataset_loader = DATASET_MAP[dataset_name]
                processed_data = dataset_loader()
                
                model_class = MODEL_MAP[model_name]
                trainer = model_class(processed_data)

                X_train, X_test, y_train, y_test = processed_data
                result = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
                
                print(f"Results for {model_name} on {dataset_name}:")
                for metric, value in result.items():
                    print(f"{metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"Error occurred: {str(e)}")
            print(f"{'-'*40}\n")

if __name__ == "__main__":
    main()
