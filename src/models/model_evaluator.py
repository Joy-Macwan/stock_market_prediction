"""
Model Evaluator
===============

Comprehensive model evaluation and comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

from .base_model import BaseModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    
    Provides metrics calculation, visualization, and model comparison.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.results: Dict[str, Dict] = {}
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name for storing results
            
        Returns:
            Dictionary of metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        
        # Additional metrics
        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction) * 100
        
        # Tracking error (standard deviation of prediction errors)
        metrics['tracking_error'] = np.std(y_true - y_pred)
        
        # Max drawdown of predictions
        cumulative_true = np.cumsum(y_true)
        cumulative_pred = np.cumsum(y_pred)
        max_true = np.maximum.accumulate(cumulative_true)
        max_pred = np.maximum.accumulate(cumulative_pred)
        metrics['max_drawdown_true'] = np.min(cumulative_true - max_true)
        metrics['max_drawdown_pred'] = np.min(cumulative_pred - max_pred)
        
        self.results[model_name] = {'type': 'regression', 'metrics': metrics}
        
        return metrics
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            model_name: Name for storing results
            
        Returns:
            Dictionary of metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        self.results[model_name] = {'type': 'classification', 'metrics': metrics}
        
        return metrics
    
    def compare_models(
        self,
        models: List[BaseModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task: str = "regression"
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: List of trained models
            X_test: Test features
            y_test: Test targets
            task: 'regression' or 'classification'
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data: List[Dict[str, Any]] = []
        
        for model in models:
            predictions = model.predict(X_test)
            
            if task == "regression":
                metrics: Dict[str, Any] = self.evaluate_regression(y_test, predictions, model.name)
            else:
                metrics = self.evaluate_classification(y_test, predictions, model.name)
            
            metrics['model'] = model.name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('model')
        
        return df
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Time series plot
        ax1 = axes[0, 0]
        ax1.plot(y_true, label='Actual', alpha=0.8)
        ax1.plot(y_pred, label='Predicted', alpha=0.8)
        ax1.set_title(f'{title} - Time Series')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(y_true, y_pred, alpha=0.5)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
        ax2.set_title('Actual vs Predicted Scatter')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Residuals plot
        ax3 = axes[1, 0]
        residuals = y_true - y_pred
        ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_title('Residuals Distribution')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Residuals over time
        ax4 = axes[1, 1]
        ax4.plot(residuals, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('Residuals Over Time')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Residual')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importance: Dictionary of feature importances
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        # Sort and get top features
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        features = list(sorted_importance.keys())
        values = list(sorted_importance.values())
        
        cmap = plt.colormaps.get_cmap('RdYlGn')
        colors = cmap(np.linspace(0.2, 0.8, len(features)))
        
        ax.barh(features, values, color=colors)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot model comparison.
        
        Args:
            comparison_df: DataFrame from compare_models
            metrics: List of metrics to plot
            save_path: Path to save figure
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'directional_accuracy']
        
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            values = comparison_df[metric]
            cmap = plt.colormaps.get_cmap('viridis')
            colors = cmap(np.linspace(0.2, 0.8, len(values)))
            
            bars = ax.bar(values.index, values.values, color=colors)
            ax.set_title(metric.upper())
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate evaluation report.
        
        Returns:
            Formatted report string
        """
        report = ["=" * 60]
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        for model_name, result in self.results.items():
            report.append(f"\n{'─' * 40}")
            report.append(f"Model: {model_name}")
            report.append(f"Type: {result['type']}")
            report.append(f"{'─' * 40}")
            
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                elif metric != 'confusion_matrix':
                    report.append(f"  {metric}: {value}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
