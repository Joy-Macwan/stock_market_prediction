"""
XGBoost Model
=============

Gradient Boosting model for stock prediction using XGBoost.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna

from .base_model import BaseModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for stock prediction.
    
    High-performance gradient boosting with advanced regularization.
    """
    
    def __init__(
        self,
        name: str = "xgboost_model",
        task: str = "regression",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            task: 'regression' or 'classification'
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            feature_names: Names of input features
        """
        super().__init__(name)
        
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_names = feature_names
        self.feature_importances_ = None
    
    def build(self, input_shape: Optional[Tuple] = None, **kwargs) -> None:
        """
        Build XGBoost model.
        
        Args:
            input_shape: Not used for XGBoost
            **kwargs: Additional model parameters
        """
        model_params = {
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'max_depth': kwargs.get('max_depth', self.max_depth),
            'learning_rate': kwargs.get('learning_rate', self.learning_rate),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 1.0),
            'random_state': self.settings.RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.task == "regression":
            model_params['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**model_params)
        else:
            model_params['objective'] = 'binary:logistic'
            self.model = xgb.XGBClassifier(**model_params)
        
        logger.info(f"XGBoost model built for {self.task}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        optimize_hyperparams: bool = False,
        use_optuna: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize_hyperparams: Whether to optimize hyperparameters
            use_optuna: Use Optuna for optimization
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if self.model is None:
            self.build()
        
        y_train = np.array(y_train).flatten()
        
        if optimize_hyperparams:
            if use_optuna:
                self.model = self._optimize_with_optuna(X_train, y_train, X_val, y_val)
            else:
                self.model = self._optimize_hyperparameters(X_train, y_train)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            y_val = np.array(y_val).flatten()
            eval_set.append((X_val, y_val))
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Training metrics
        self.history = {
            'train_score': self.model.score(X_train, y_train),
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        }
        
        if X_val is not None and y_val is not None:
            self.history['val_score'] = self.model.score(X_val, y_val)
        
        self.is_trained = True
        logger.info(f"XGBoost training completed. Train score: {self.history['train_score']:.4f}")
        
        return self.history
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Optimize using RandomizedSearchCV."""
        param_distributions = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        
        if self.task == "regression":
            base_model = xgb.XGBRegressor(
                random_state=self.settings.RANDOM_STATE,
                n_jobs=-1
            )
            scoring = 'neg_mean_squared_error'
        else:
            base_model = xgb.XGBClassifier(
                random_state=self.settings.RANDOM_STATE,
                n_jobs=-1
            )
            scoring = 'accuracy'
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=30,
            cv=5,
            scoring=scoring,
            random_state=self.settings.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X, y)
        
        logger.info(f"Best parameters: {search.best_params_}")
        return search.best_estimator_
    
    def _optimize_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_trials: int = 50
    ):
        """Optimize using Optuna."""
        
        def objective(trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.settings.RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            if self.task == "regression":
                model = xgb.XGBRegressor(**params)
            else:
                model = xgb.XGBClassifier(**params)
            
            model.fit(X_train, y_train)
            
            if X_val is not None and y_val is not None:
                return float(model.score(X_val, y_val))
            else:
                return float(model.score(X_train, y_train))
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value:.4f}")
        
        # Create model with best parameters
        best_params = study.best_params
        best_params['random_state'] = self.settings.RANDOM_STATE
        best_params['n_jobs'] = -1
        
        if self.task == "regression":
            return xgb.XGBRegressor(**best_params)
        else:
            return xgb.XGBClassifier(**best_params)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importances_ is None:
            return {}
        
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, self.feature_importances_))
        else:
            importance_dict = {f"feature_{i}": imp for i, imp in enumerate(self.feature_importances_)}
        
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_importance
