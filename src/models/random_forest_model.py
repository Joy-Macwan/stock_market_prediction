"""
Random Forest Model
===================

Random Forest ensemble model for stock prediction.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .base_model import BaseModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for stock prediction.
    
    Supports both regression (price prediction) and classification (direction prediction).
    """
    
    def __init__(
        self,
        name: str = "random_forest_model",
        task: str = "regression",
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name
            task: 'regression' or 'classification'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            feature_names: Names of input features
        """
        super().__init__(name)
        
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_names = feature_names
        self.feature_importances_ = None
    
    def build(self, input_shape: Optional[Tuple] = None, **kwargs) -> None:
        """
        Build Random Forest model.
        
        Args:
            input_shape: Not used for RF (kept for interface consistency)
            **kwargs: Additional model parameters
        """
        model_params = {
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'max_depth': kwargs.get('max_depth', self.max_depth),
            'min_samples_split': kwargs.get('min_samples_split', self.min_samples_split),
            'min_samples_leaf': kwargs.get('min_samples_leaf', self.min_samples_leaf),
            'random_state': self.settings.RANDOM_STATE,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if self.task == "regression":
            self.model = RandomForestRegressor(**model_params)
        else:
            self.model = RandomForestClassifier(**model_params)
        
        logger.info(f"Random Forest model built for {self.task}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        optimize_hyperparams: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (used for evaluation only)
            y_val: Validation targets
            optimize_hyperparams: Whether to perform hyperparameter optimization
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if self.model is None:
            self.build()
        
        if optimize_hyperparams:
            self.model = self._optimize_hyperparameters(X_train, y_train)
        
        # Flatten y if needed
        y_train = np.array(y_train).flatten()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        
        self.history = {
            'train_score': self.model.score(X_train, y_train)
        }
        
        if X_val is not None and y_val is not None:
            y_val = np.array(y_val).flatten()
            self.history['val_score'] = self.model.score(X_val, y_val)
        
        self.is_trained = True
        logger.info(f"Random Forest training completed. Train score: {self.history['train_score']:.4f}")
        
        return self.history
    
    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        Optimize hyperparameters using RandomizedSearchCV.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Best estimator
        """
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        if self.task == "regression":
            base_model = RandomForestRegressor(
                random_state=self.settings.RANDOM_STATE,
                n_jobs=-1
            )
        else:
            base_model = RandomForestClassifier(
                random_state=self.settings.RANDOM_STATE,
                n_jobs=-1
            )
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=20,
            cv=5,
            scoring='neg_mean_squared_error' if self.task == "regression" else 'accuracy',
            random_state=self.settings.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        y = np.array(y).flatten()
        search.fit(X, y)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importances
        """
        if self.feature_importances_ is None:
            return {}
        
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, self.feature_importances_))
        else:
            importance_dict = {f"feature_{i}": imp for i, imp in enumerate(self.feature_importances_)}
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_importance
