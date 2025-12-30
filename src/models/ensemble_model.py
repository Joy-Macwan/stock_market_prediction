"""
Ensemble Model
==============

Combines multiple models for improved prediction accuracy.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from .base_model import BaseModel
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple prediction models.
    
    Supports weighted averaging, voting, and stacking strategies.
    """
    
    def __init__(
        self,
        name: str = "ensemble_model",
        strategy: str = "weighted_average",
        models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            name: Model name
            strategy: Ensemble strategy ('weighted_average', 'voting', 'stacking')
            models: List of base models
            weights: Model weights for weighted averaging
        """
        super().__init__(name)
        
        self.strategy = strategy
        self.models = models or []
        self.weights = weights
        self.meta_model: Any = None
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
            weight: Model weight
        """
        self.models.append(model)
        
        if self.weights is None:
            self.weights = []
        self.weights.append(weight)
        
        logger.info(f"Added {model.name} to ensemble with weight {weight}")
    
    def build(self, input_shape: Optional[Tuple] = None, **kwargs) -> None:
        """
        Build ensemble model.
        
        Args:
            input_shape: Input shape for models
            **kwargs: Additional parameters
        """
        # Normalize weights
        if self.weights:
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        else:
            # Equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # Build individual models if needed
        for model in self.models:
            if model.model is None and input_shape is not None:
                model.build(input_shape, **kwargs)
        
        # Build meta-model for stacking
        if self.strategy == "stacking":
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
        
        logger.info(f"Ensemble built with {len(self.models)} models using {self.strategy} strategy")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Training history for all models
        """
        all_histories = {}
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            
            history = model.train(X_train, y_train, X_val, y_val, **kwargs)
            all_histories[model.name] = history
        
        # Train meta-model for stacking
        if self.strategy == "stacking" and self.meta_model is not None:
            # Get predictions from base models
            base_predictions = self._get_base_predictions(X_train)
            y_train_flat = np.array(y_train).flatten()
            self.meta_model.fit(base_predictions, y_train_flat)
        
        self.history = all_histories
        self.is_trained = True
        
        logger.info("Ensemble training completed")
        return self.history
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions from all models
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(np.array(pred).flatten())
        
        return np.column_stack(predictions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        base_predictions = self._get_base_predictions(X)
        
        if self.strategy == "weighted_average":
            # Weighted average of predictions
            ensemble_pred = np.average(base_predictions, axis=1, weights=self.weights)
        
        elif self.strategy == "voting":
            # Majority voting (for classification)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=1,
                arr=base_predictions
            )
        
        elif self.strategy == "stacking":
            # Use meta-model predictions
            ensemble_pred = self.meta_model.predict(base_predictions)
        
        else:
            # Simple average
            ensemble_pred = np.mean(base_predictions, axis=1)
        
        return ensemble_pred
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble and individual models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Metrics for ensemble and each model
        """
        results = {}
        
        # Evaluate ensemble
        results['ensemble'] = super().evaluate(X_test, y_test)
        
        # Evaluate individual models
        for model in self.models:
            results[model.name] = model.evaluate(X_test, y_test)
        
        self.metrics = results
        return results
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        contributions = {}
        
        for model in self.models:
            contributions[model.name] = model.predict(X)
        
        return contributions
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> List[float]:
        """
        Optimize ensemble weights using Optuna.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Optimized weights
        """
        import optuna
        from sklearn.metrics import mean_squared_error
        
        base_predictions = self._get_base_predictions(X_val)
        y_val_flat = np.array(y_val).flatten()
        
        def objective(trial):
            # Suggest weights
            weights = []
            for i in range(len(self.models)):
                w = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(w)
            
            # Normalize weights
            total = sum(weights)
            if total == 0:
                return float('inf')
            weights = [w / total for w in weights]
            
            # Calculate weighted prediction
            pred = np.average(base_predictions, axis=1, weights=weights)
            
            # Return MSE
            return mean_squared_error(y_val_flat, pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract optimized weights
        optimized_weights = []
        for i in range(len(self.models)):
            optimized_weights.append(study.best_params[f'weight_{i}'])
        
        # Normalize
        total = sum(optimized_weights)
        optimized_weights = [w / total for w in optimized_weights]
        
        self.weights = optimized_weights
        
        logger.info(f"Optimized weights: {dict(zip([m.name for m in self.models], optimized_weights))}")
        
        return optimized_weights
