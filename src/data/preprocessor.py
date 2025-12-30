"""
Data Preprocessing Module
=========================

Clean, transform, and prepare stock data for ML/DL models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Type alias for scalers
ScalerType = Union[MinMaxScaler, StandardScaler, RobustScaler]


class DataPreprocessor:
    """
    Data preprocessing pipeline for stock market data.
    
    Handles missing values, scaling, feature engineering, and data splitting.
    """
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        """
        self.settings = get_settings()
        self.scaler_type = scaler_type
        self.scalers: Dict[str, ScalerType] = {}
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize the scaler based on type."""
        if self.scaler_type == "minmax":
            self.scaler: ScalerType = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw stock data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Ensure OHLC consistency (High >= Low, etc.)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return calculations to the data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        
        if 'close' in df.columns:
            # Daily returns
            df['daily_return'] = df['close'].pct_change()
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Cumulative returns
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            
            # Rolling returns
            for window in [5, 10, 20]:
                df[f'return_{window}d'] = df['close'].pct_change(window)
        
        return df
    
    def add_volatility(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Add volatility measures.
        
        Args:
            df: DataFrame with return data
            windows: Rolling window sizes
            
        Returns:
            DataFrame with volatility columns
        """
        df = df.copy()
        
        if 'daily_return' not in df.columns:
            df = self.add_returns(df)
        
        for window in windows:
            # Rolling standard deviation of returns
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Average True Range (ATR) based volatility
            if all(col in df.columns for col in ['high', 'low', 'close']):
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift(1))
                low_close = np.abs(df['low'] - df['close'].shift(1))
                
                tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
                tr = tr_df.max(axis=1)
                df[f'atr_{window}d'] = tr.rolling(window=window).mean()
        
        return df
    
    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array
            lookback: Number of time steps to look back
            forecast_horizon: Number of steps to predict ahead
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(data[i:i + forecast_horizon, 0])  # Predict close price
        
        return np.array(X), np.array(y)
    
    def scale_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale the data using the configured scaler.
        
        Args:
            df: DataFrame to scale
            columns: Columns to scale (None for all numeric)
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (scaled DataFrame, scaler info)
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        scaler_info: Dict[str, Dict[str, Any]] = {}
        
        for col in columns:
            if col in df.columns:
                values = np.array(df[col].values).reshape(-1, 1)
                
                if col not in self.scalers:
                    self.scalers[col] = self._create_scaler()
                
                scaler = self.scalers[col]
                if fit:
                    scaled = scaler.fit_transform(values)
                else:
                    scaled = scaler.transform(values)
                
                df[col] = scaled.flatten()
                
                # Get scaler info if MinMaxScaler
                if isinstance(scaler, MinMaxScaler):
                    scaler_info[col] = {
                        'min': float(scaler.data_min_[0]),
                        'max': float(scaler.data_max_[0])
                    }
                else:
                    scaler_info[col] = {'min': None, 'max': None}
        
        return df, scaler_info
    
    def _create_scaler(self) -> ScalerType:
        """Create a new scaler instance."""
        if self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            return StandardScaler()
        else:  # robust
            return RobustScaler()
    
    def inverse_scale(self, data: np.ndarray, column: str) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data
            column: Column name for the scaler
            
        Returns:
            Original scale data
        """
        if column not in self.scalers:
            raise ValueError(f"No scaler found for column: {column}")
        
        data = np.array(data).reshape(-1, 1)
        scaler = self.scalers[column]
        return scaler.inverse_transform(data).flatten()
    
    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for traditional ML models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: Feature column names
            test_size: Test set proportion
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        df = df.copy().dropna()
        
        if feature_cols is None:
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=shuffle,
            random_state=self.settings.RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test
    
    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None,
        lookback: int = 60,
        test_size: float = 0.2,
        scale: bool = True
    ) -> Dict:
        """
        Prepare data for LSTM models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: Feature column names
            lookback: Sequence length
            test_size: Test set proportion
            scale: Whether to scale data
            
        Returns:
            Dictionary with train/test data and scalers
        """
        df = df.copy().dropna()
        
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Ensure target is first column
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        feature_cols = [target_col] + feature_cols
        
        data = df[feature_cols].values
        
        if scale:
            # Scale the data
            data_scaled = np.zeros_like(data, dtype=np.float64)
            for i, col in enumerate(feature_cols):
                values = np.array(data[:, i]).reshape(-1, 1)
                if col not in self.scalers:
                    self.scalers[col] = self._create_scaler()
                scaler = self.scalers[col]
                data_scaled[:, i] = scaler.fit_transform(values).flatten()
            data = data_scaled
        
        # Create sequences
        X, y = self.create_sequences(data, lookback)
        
        # Split data (time series split, no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'lookback': lookback,
            'scalers': self.scalers
        }
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_type: str = 'direction',
        horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: Input DataFrame
            target_type: 'direction' (up/down), 'return', or 'price'
            horizon: Prediction horizon in days
            threshold: Threshold for direction classification
            
        Returns:
            DataFrame with target column
        """
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        if target_type == 'direction':
            # Binary classification: 1 if price goes up, 0 if down
            future_return = df['close'].shift(-horizon) / df['close'] - 1
            df['target'] = (future_return > threshold).astype(int)
        
        elif target_type == 'return':
            # Regression: predict the return
            df['target'] = df['close'].shift(-horizon) / df['close'] - 1
        
        elif target_type == 'price':
            # Regression: predict the actual price
            df['target'] = df['close'].shift(-horizon)
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return df
