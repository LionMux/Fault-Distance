"""
Metrics calculation for regression and classification tasks.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import torch


class MetricsCalculator:
    """
    Calculate various metrics for model evaluation.
    """
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        Calculate regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            dict: MAE, MSE, RMSE, R2
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def classification_metrics(y_true, y_pred):
        """
        Calculate classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            dict: Accuracy, Precision, Recall, F1, Confusion Matrix
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def print_regression_metrics(y_true, y_pred, dataset_name="Test"):
        """
        Print regression metrics in formatted way.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            dataset_name: Name of dataset (e.g., 'Train', 'Test')
        """
        metrics = MetricsCalculator.regression_metrics(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"METRICS - {dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"MAE  (Mean Absolute Error):       {metrics['mae']:.6f} km")
        print(f"RMSE (Root Mean Squared Error):   {metrics['rmse']:.6f} km")
        print(f"MSE  (Mean Squared Error):        {metrics['mse']:.6f}")
        print(f"R²   (R-squared):                 {metrics['r2']:.6f}")
        print(f"MAPE (Mean Absolute % Error):    {metrics['mape']:.2f}%")
        print(f"{'='*60}\n")
        
        return metrics
    
    @staticmethod
    def error_statistics(y_true, y_pred):
        """
        Calculate error statistics.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
        
        Returns:
            dict: Error statistics
        """
        errors = y_true - y_pred
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'min_error': np.min(np.abs(errors)),
            'median_error': np.median(np.abs(errors)),
            '95_percentile_error': np.percentile(np.abs(errors), 95),
            '99_percentile_error': np.percentile(np.abs(errors), 99)
        }
