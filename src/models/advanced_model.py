"""
Advanced Deep Learning Model for Stock Prediction
==================================================

Implements state-of-the-art techniques for high prediction accuracy:
1. Transformer-based Attention Mechanism
2. Multi-Head Self-Attention
3. Temporal Convolutional Networks (TCN)
4. Residual Connections
5. Feature Pyramid Networks
6. Ensemble of Multiple Architectures
7. Monte Carlo Dropout for Uncertainty
8. Advanced Feature Engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import (
    Input, Dense, LSTM, GRU, Bidirectional, Dropout, 
    BatchNormalization, LayerNormalization, Conv1D, 
    MaxPooling1D, GlobalAveragePooling1D, Flatten,
    Concatenate, Add, Multiply, Attention, MultiHeadAttention,
    Embedding, Reshape, Lambda, TimeDistributed
)
from keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, LearningRateScheduler
)
from keras.regularizers import l1_l2
from keras.optimizers import Adam, AdamW
import keras.backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base_model import BaseModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class PositionalEncoding(keras.layers.Layer):
    """Positional encoding for transformer-style models."""
    
    def __init__(self, max_len: int = 100, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
        super().build(input_shape)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


class TemporalBlock(keras.layers.Layer):
    """Temporal Convolutional Block with residual connection."""
    
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, 
                 dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        
    def build(self, input_shape):
        self.conv1 = Conv1D(
            self.filters, self.kernel_size, 
            dilation_rate=self.dilation_rate,
            padding='causal', activation='relu',
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        )
        self.conv2 = Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal', activation='relu',
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        )
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        # Residual connection
        if input_shape[-1] != self.filters:
            self.residual = Conv1D(self.filters, 1, padding='same')
        else:
            self.residual = Lambda(lambda x: x)
            
        super().build(input_shape)
        
    def call(self, x, training=None):
        res = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.dropout1(out, training=training)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out, training=training)
        
        return keras.activations.relu(out + res)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout': self.dropout
        })
        return config


class AdvancedStockPredictor(BaseModel):
    """
    Advanced Deep Learning Model with multiple architectures:
    
    1. Transformer Encoder - For capturing long-range dependencies
    2. TCN (Temporal Convolutional Network) - For local patterns
    3. BiLSTM with Attention - For sequential modeling
    4. Feature Pyramid Network - Multi-scale feature extraction
    5. Ensemble Head - Combines all architectures
    
    Features:
    - Monte Carlo Dropout for uncertainty estimation
    - Multi-task learning (price + direction)
    - Residual connections throughout
    - Advanced regularization (L1/L2, dropout, layer norm)
    """
    
    def __init__(
        self,
        name: str = "advanced_stock_predictor",
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        lstm_units: int = 128,
        tcn_filters: int = 64,
        dropout: float = 0.3,
        use_monte_carlo: bool = True
    ):
        super().__init__(name)
        self.settings = get_settings()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers
        self.lstm_units = lstm_units
        self.tcn_filters = tcn_filters
        self.dropout = dropout
        self.use_monte_carlo = use_monte_carlo
        
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler()
        
    def _build_transformer_branch(self, inputs):
        """Build Transformer encoder branch."""
        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(max_len=inputs.shape[1], d_model=self.d_model)(x)
        
        # Transformer encoder layers
        for _ in range(self.n_transformer_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout
            )(x, x)
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # Feed-forward network
            ff = Dense(self.d_model * 4, activation='gelu')(x)
            ff = Dropout(self.dropout)(ff)
            ff = Dense(self.d_model)(ff)
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        return x
    
    def _build_tcn_branch(self, inputs):
        """Build Temporal Convolutional Network branch."""
        x = inputs
        
        # TCN blocks with increasing dilation
        for i, dilation in enumerate([1, 2, 4, 8]):
            x = TemporalBlock(
                filters=self.tcn_filters,
                kernel_size=3,
                dilation_rate=dilation,
                dropout=self.dropout
            )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        return x
    
    def _build_lstm_attention_branch(self, inputs):
        """Build BiLSTM with Attention branch."""
        # Bidirectional LSTM
        x = Bidirectional(LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        # Second LSTM layer
        x = Bidirectional(LSTM(
            self.lstm_units // 2,
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))(x)
        x = BatchNormalization()(x)
        
        # Self-attention
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = keras.activations.softmax(attention)
        attention = keras.layers.RepeatVector(self.lstm_units)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = Multiply()([x, attention])
        x = Lambda(lambda z: K.sum(z, axis=1))(x)
        
        return x
    
    def _build_feature_pyramid(self, inputs):
        """Build Feature Pyramid Network for multi-scale features."""
        # Different kernel sizes for multi-scale
        scales = []
        
        for kernel_size in [3, 5, 7]:
            conv = Conv1D(32, kernel_size, padding='same', activation='relu')(inputs)
            conv = MaxPooling1D(2)(conv)
            conv = Conv1D(64, kernel_size, padding='same', activation='relu')(conv)
            conv = GlobalAveragePooling1D()(conv)
            scales.append(conv)
        
        # Concatenate multi-scale features
        x = Concatenate()(scales)
        return x
    
    def build(self, input_shape: Tuple, **kwargs) -> None:
        """
        Build the advanced ensemble model.
        
        Args:
            input_shape: Shape of input (timesteps, features)
        """
        inputs = Input(shape=input_shape, name='input')
        
        # Build all branches
        transformer_out = self._build_transformer_branch(inputs)
        tcn_out = self._build_tcn_branch(inputs)
        lstm_out = self._build_lstm_attention_branch(inputs)
        pyramid_out = self._build_feature_pyramid(inputs)
        
        # Concatenate all branches
        combined = Concatenate()([
            transformer_out, tcn_out, lstm_out, pyramid_out
        ])
        
        # Ensemble head with residual connections
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        res1 = Dense(128)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = BatchNormalization()(x)
        x = Add()([x, res1])
        x = Dropout(self.dropout)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout / 2)(x)
        
        # Multi-task outputs
        price_output = Dense(1, activation='linear', name='price')(x)
        direction_output = Dense(1, activation='sigmoid', name='direction')(x)
        
        self.model = Model(
            inputs=inputs,
            outputs=[price_output, direction_output],
            name=self.name
        )
        
        # Compile with custom loss weights
        self.model.compile(
            optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss={
                'price': 'huber',  # Robust to outliers
                'direction': 'binary_crossentropy'
            },
            loss_weights={'price': 1.0, 'direction': 0.3},
            metrics={
                'price': ['mae', 'mape'],
                'direction': ['accuracy']
            }
        )
        
        logger.info(f"Advanced model built with {self.model.count_params():,} parameters")
    
    def _cosine_decay_with_warmup(self, epoch, lr, warmup_epochs=5, total_epochs=100):
        """Cosine learning rate decay with warmup."""
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the advanced model with curriculum learning.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Prepare direction labels (1 if price goes up, 0 if down)
        y_train_direction = (np.diff(y_train.flatten(), prepend=y_train[0]) > 0).astype(float)
        y_train_dict = {'price': y_train, 'direction': y_train_direction}
        
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_direction = (np.diff(y_val.flatten(), prepend=y_val[0]) > 0).astype(float)
            validation_data = (X_val, {'price': y_val, 'direction': y_val_direction})
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_price_loss' if validation_data else 'price_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_price_loss' if validation_data else 'price_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(
                lambda epoch, lr: self._cosine_decay_with_warmup(epoch, lr, 5, epochs)
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train_dict,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        self.is_trained = True
        
        return self.history
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Make predictions with optional uncertainty estimation.
        
        Uses Monte Carlo Dropout for uncertainty quantification.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if return_uncertainty and self.use_monte_carlo:
            # Monte Carlo Dropout: multiple forward passes
            n_samples = 30
            predictions = []
            
            for _ in range(n_samples):
                # Enable dropout during inference
                pred = self.model(X, training=True)
                predictions.append(pred[0].numpy())  # Price prediction
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            return mean_pred, uncertainty
        else:
            pred = self.model.predict(X, verbose=0)
            return pred[0]  # Return price prediction
    
    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals.
        """
        mean_pred, uncertainty = self.predict(X, return_uncertainty=True)
        
        return {
            'prediction': mean_pred,
            'uncertainty': uncertainty,
            'lower_95': mean_pred - 1.96 * uncertainty,
            'upper_95': mean_pred + 1.96 * uncertainty,
            'direction_prob': self.model.predict(X, verbose=0)[1]
        }


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions.
    
    Combines:
    1. AdvancedStockPredictor (Transformer + TCN + LSTM)
    2. Gradient Boosting (XGBoost)
    3. Random Forest
    4. Simple LSTM
    
    Uses stacking with meta-learner for final prediction.
    """
    
    def __init__(self, n_models: int = 4):
        self.n_models = n_models
        self.models = []
        self.meta_learner = None
        self.is_trained = False
        
    def build(self, input_shape: Tuple):
        """Build ensemble of models."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        import xgboost as xgb
        
        # Advanced DL model
        self.dl_model = AdvancedStockPredictor()
        self.dl_model.build(input_shape)
        
        # ML models (will use flattened features)
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42
        )
        
        # Meta-learner
        self.meta_learner = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        logger.info("Ensemble model built")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict:
        """Train all models in ensemble."""
        # Flatten X for ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train DL model
        logger.info("Training Deep Learning model...")
        self.dl_model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        # Train ML models
        logger.info("Training Random Forest...")
        self.rf_model.fit(X_train_flat, y_train.flatten())
        
        logger.info("Training Gradient Boosting...")
        self.gb_model.fit(X_train_flat, y_train.flatten())
        
        logger.info("Training XGBoost...")
        self.xgb_model.fit(X_train_flat, y_train.flatten())
        
        # Get predictions for meta-learner training
        dl_pred = self.dl_model.predict(X_train)
        rf_pred = self.rf_model.predict(X_train_flat).reshape(-1, 1)
        gb_pred = self.gb_model.predict(X_train_flat).reshape(-1, 1)
        xgb_pred = self.xgb_model.predict(X_train_flat).reshape(-1, 1)
        
        # Stack predictions
        meta_features = np.hstack([dl_pred, rf_pred, gb_pred, xgb_pred])
        
        # Train meta-learner
        logger.info("Training Meta-learner...")
        self.meta_learner.fit(meta_features, y_train.flatten())
        
        self.is_trained = True
        logger.info("Ensemble training completed")
        
        return {'status': 'completed'}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        X_flat = X.reshape(X.shape[0], -1)
        
        # Get individual predictions
        dl_pred = self.dl_model.predict(X)
        rf_pred = self.rf_model.predict(X_flat).reshape(-1, 1)
        gb_pred = self.gb_model.predict(X_flat).reshape(-1, 1)
        xgb_pred = self.xgb_model.predict(X_flat).reshape(-1, 1)
        
        # Stack and predict with meta-learner
        meta_features = np.hstack([dl_pred, rf_pred, gb_pred, xgb_pred])
        
        return self.meta_learner.predict(meta_features)
    
    def predict_with_individual(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        X_flat = X.reshape(X.shape[0], -1)
        
        return {
            'dl_model': self.dl_model.predict(X),
            'random_forest': self.rf_model.predict(X_flat),
            'gradient_boosting': self.gb_model.predict(X_flat),
            'xgboost': self.xgb_model.predict(X_flat),
            'ensemble': self.predict(X)
        }


def create_sequences_advanced(
    data: pd.DataFrame,
    lookback: int = 60,
    forecast_horizon: int = 5,
    target_col: str = 'close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences with advanced feature engineering.
    
    Features:
    - Price returns at multiple scales
    - Volatility features
    - Momentum indicators
    - Volume features
    - Technical indicators
    """
    df = data.copy()
    
    # Price returns at multiple horizons
    for h in [1, 5, 10, 20]:
        df[f'return_{h}d'] = df[target_col].pct_change(h)
    
    # Volatility features
    df['volatility_5d'] = df[target_col].pct_change().rolling(5).std()
    df['volatility_20d'] = df[target_col].pct_change().rolling(20).std()
    
    # Volume features
    if 'volume' in df.columns:
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Drop NaN
    df = df.dropna()
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'datetime']]
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(df) - forecast_horizon + 1):
        X.append(df[feature_cols].iloc[i-lookback:i].values)
        y.append(df[target_col].iloc[i + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Scale
    scaler_X = RobustScaler()
    scaler_y = MinMaxScaler()
    
    X_shape = X.shape
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X_shape)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_y
