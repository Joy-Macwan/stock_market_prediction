"""
Base Model Class
================

Abstract base class for all prediction models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import joblib
import logging

from ..config.settings import get_settings

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from keras.models import Model as KerasModel

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for stock prediction models.
    
    All models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = "base_model"):
        """
        Initialize base model.
        
        Args:
            name: Model name for identification
        """
        self.name = name
        self.settings = get_settings()
        self.model: Any = None
        self.is_trained = False
        self.history: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def build(self, input_shape: Tuple, **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional model parameters
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            r2_score, mean_absolute_percentage_error
        )
        
        predictions = self.predict(X_test)
        
        # Flatten arrays if needed
        y_test = np.array(y_test).flatten()
        predictions = np.array(predictions).flatten()
        
        self.metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'mape': mean_absolute_percentage_error(y_test, predictions) * 100,
            'r2': r2_score(y_test, predictions)
        }
        
        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            self.metrics['directional_accuracy'] = np.mean(
                actual_direction == pred_direction
            ) * 100
        
        return self.metrics
    
    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Save model to disk.
        
        Args:
            filepath: Custom filepath (optional)
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            save_path = self.settings.MODELS_DIR / f"{self.name}.joblib"
        else:
            save_path = Path(filepath)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'name': self.name,
            'model': self.model,
            'is_trained': self.is_trained,
            'history': self.history,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path
    
    def load(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        if filepath is None:
            load_path = self.settings.MODELS_DIR / f"{self.name}.joblib"
        else:
            load_path = Path(filepath)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.name = model_data['name']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.history = model_data['history']
        self.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary of feature importances or None
        """
        return None
    
    def summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'history_epochs': len(self.history.get('loss', []))
        }
