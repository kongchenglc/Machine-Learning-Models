# main.py

import pandas as pd
from models.linear_regression import DatasetTrainer as linear_regression_trainer

from data_preprocess.mushroom import load_processed_data as mushroom_load_processed_data
from data_preprocess.breast_cancer import load_processed_data as breast_cancer_load_processed_data

def main():
    # 配置模型和数据集
    processed_data = mushroom_load_processed_data()  # 加载处理后的数据
    trainer = linear_regression_trainer(processed_data)  # 传入处理好的数据
    results = []

    # 加载数据
    data = trainer.load_dataset()
    if data is None:
        print("\n数据集为空")
        return

    # 训练和评估
    X_train, X_test, y_train, y_test = data  # 假设数据是以元组形式返回的
    mse, accuracy = trainer.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 保存结果
    results.append({"mse": mse, "accuracy": accuracy})

    # 打印所有结果的摘要
    print("\n所有数据集的训练结果摘要:")
    results_df = pd.DataFrame(results)
    print(results)
    print(results_df)


if __name__ == "__main__":
    main()