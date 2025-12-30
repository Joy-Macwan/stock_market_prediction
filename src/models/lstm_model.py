"""
LSTM Deep Learning Model
========================

Long Short-Term Memory neural network for stock price prediction.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras

from .base_model import BaseModel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM-based deep learning model for stock prediction.
    
    Supports bidirectional LSTM, multiple layers, and dropout regularization.
    """
    
    def __init__(
        self,
        name: str = "lstm_model",
        units: Optional[int] = None,
        layers: Optional[int] = None,
        dropout: Optional[float] = None,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model.
        
        Args:
            name: Model name
            units: Number of LSTM units per layer
            layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__(name)
        self.settings = get_settings()
        
        self.units = units or self.settings.LSTM_UNITS
        self.layers = layers or self.settings.LSTM_LAYERS
        self.dropout = dropout or self.settings.DROPOUT_RATE
        self.bidirectional = bidirectional
    
    def build(self, input_shape: Tuple, output_size: int = 1, **kwargs) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            output_size: Number of output predictions
            **kwargs: Additional parameters
        """
        self.model = Sequential(name=self.name)
        
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            
            lstm_layer = LSTM(
                units=self.units,
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            )
            
            if self.bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            self.model.add(lstm_layer)
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
        
        # Dense layers
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(self.dropout / 2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"LSTM model built: {self.model.summary()}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        epochs = epochs or self.settings.EPOCHS
        batch_size = batch_size or self.settings.BATCH_SIZE
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.settings.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add model checkpoint
        checkpoint_path = self.settings.MODELS_DIR / f"{self.name}_checkpoint.keras"
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose="auto"
        )
        
        self.history = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'val_loss': history.history.get('val_loss', []),
            'val_mae': history.history.get('val_mae', [])
        }
        
        self.is_trained = True
        logger.info(f"LSTM training completed. Final loss: {self.history['loss'][-1]:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose="0")
    
    def predict_future(
        self,
        last_sequence: np.ndarray,
        n_steps: int,
        scaler=None
    ) -> np.ndarray:
        """
        Predict future values recursively.
        
        Args:
            last_sequence: Last known sequence
            n_steps: Number of future steps to predict
            scaler: Scaler for inverse transformation
            
        Returns:
            Array of future predictions
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next value
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose="0")
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            new_row = np.zeros((1, current_sequence.shape[1]))
            new_row[0, 0] = pred[0, 0]  # Set predicted close price
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        predictions = np.array(predictions)
        
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def save(self, filepath: Optional[Union[str, 'Path']] = None):
        """Save LSTM model."""
        from pathlib import Path
        if filepath is None:
            filepath = str(self.settings.MODELS_DIR / f"{self.name}.keras")
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'is_trained': self.is_trained,
            'history': self.history,
            'metrics': self.metrics
        }
        
        import joblib
        metadata_path = str(filepath).replace('.keras', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"LSTM model saved to {filepath}")
        return filepath
    
    def load(self, filepath: Optional[Union[str, 'Path']] = None):
        """Load LSTM model."""
        from pathlib import Path
        if filepath is None:
            filepath = str(self.settings.MODELS_DIR / f"{self.name}.keras")
        
        self.model = keras.models.load_model(filepath)
        
        # Load metadata
        import joblib
        metadata_path = str(filepath).replace('.keras', '_metadata.joblib')
        metadata = joblib.load(metadata_path)
        
        self.name = metadata['name']
        self.units = metadata['units']
        self.layers = metadata['layers']
        self.dropout = metadata['dropout']
        self.bidirectional = metadata['bidirectional']
        self.is_trained = metadata['is_trained']
        self.history = metadata['history']
        self.metrics = metadata['metrics']
        
        logger.info(f"LSTM model loaded from {filepath}")
