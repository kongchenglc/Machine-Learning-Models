import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os

class ModelVisualizer:
    def __init__(self):
        # Set the style using seaborn's default style
        sns.set_theme(style="whitegrid")
        # Set the figure style parameters
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                            dataset_name: str, metric: str):
        """Plot comparison of different models for a specific metric."""
        # Filter out models that don't have the specified metric
        valid_models = {model: results[model] for model in results 
                       if metric in results[model]}
        
        if not valid_models:
            print(f"Warning: No models have the metric '{metric}' for dataset '{dataset_name}'. Skipping plot.")
            return
            
        models = list(valid_models.keys())
        values = [valid_models[model][metric] for model in models]
        
        plt.figure()
        sns.barplot(x=models, y=values)
        plt.title(f'{metric} Comparison for {dataset_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_scores: List[float], 
                          test_scores: List[float], 
                          model_name: str, 
                          dataset_name: str):
        """Plot learning curve showing training and test scores."""
        plt.figure()
        plt.plot(train_scores, label='Training Score', marker='o')
        plt.plot(test_scores, label='Test Score', marker='o')
        plt.title(f'Learning Curve - {model_name} on {dataset_name}')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            model_name: str, 
                            dataset_name: str):
        """Plot confusion matrix for classification results."""
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_regression_results(self, y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              model_name: str, 
                              dataset_name: str):
        """Plot actual vs predicted values for regression tasks."""
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5, s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
        plt.title(f'Actual vs Predicted - {model_name} on {dataset_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_model_across_datasets(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                                 model_name: str, metric: str):
        """Plot a model's performance across different datasets."""
        if model_name not in results:
            print(f"Warning: Model '{model_name}' not found in results.")
            return
            
        datasets = list(results[model_name].keys())
        values = []
        
        for dataset in datasets:
            if metric in results[model_name][dataset]:
                values.append(results[model_name][dataset][metric])
            else:
                values.append(None)
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            print(f"Warning: No valid values found for metric '{metric}' across datasets.")
            return
            
        datasets = [datasets[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        plt.figure()
        sns.barplot(x=datasets, y=values)
        plt.title(f'{model_name} - {metric} Across Datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_dataset_across_models(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                                 dataset_name: str, metric: str):
        """Plot different models' performance on a specific dataset."""
        models = list(results.keys())
        values = []
        
        for model in models:
            if dataset_name in results[model] and metric in results[model][dataset_name]:
                values.append(results[model][dataset_name][metric])
            else:
                values.append(None)
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            print(f"Warning: No valid values found for metric '{metric}' across models.")
            return
            
        models = [models[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        plt.figure()
        sns.barplot(x=models, y=values)
        plt.title(f'{dataset_name} - {metric} Across Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_comprehensive_comparison(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                                   metric: str):
        """Plot a comprehensive comparison matrix of all models across all datasets."""
        models = list(results.keys())
        datasets = list(next(iter(results.values())).keys())
        
        # Create a matrix of values
        matrix = np.zeros((len(models), len(datasets)))
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                if dataset in results[model] and metric in results[model][dataset]:
                    matrix[i, j] = results[model][dataset][metric]
                else:
                    matrix[i, j] = np.nan
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=datasets, yticklabels=models)
        plt.title(f'Comprehensive {metric} Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 