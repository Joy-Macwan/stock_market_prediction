"""
Technical Indicators Module
===========================

Calculate various technical indicators for stock analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from ..config.settings import get_settings


class TechnicalIndicators:
    """
    Technical indicators calculator for stock data.
    
    Provides various trend, momentum, volatility, and volume indicators.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_price_features(df)
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
        
        # Exponential Moving Averages
        for period in [12, 26, 50, 200]:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(
            close=df['close'],
            window_slow=self.settings.MACD_SLOW,
            window_fast=self.settings.MACD_FAST,
            window_sign=self.settings.MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Price relative to moving averages
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']
        df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        df = df.copy()
        
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        williams = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        )
        df['williams_r'] = williams.williams_r()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   df['close'].shift(period)) * 100
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators
        """
        df = df.copy()
        
        # Bollinger Bands
        bb = BollingerBands(
            close=df['close'],
            window=self.settings.BB_PERIOD,
            window_dev=self.settings.BB_STD
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        # Average True Range
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['atr'] = atr.average_true_range()
        
        # Historical Volatility
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(
                window=period
            ).std() * np.sqrt(252)
        
        # Keltner Channel
        ema_20 = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['keltner_upper'] = ema_20 + (2 * df['atr'])
        df['keltner_lower'] = ema_20 - (2 * df['atr'])
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        df = df.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        # VWAP (Volume Weighted Average Price)
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume Ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Price-Volume Trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / 
                    df['close'].shift(1) * df['volume']).cumsum()
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        df = df.copy()
        
        # Candlestick patterns
        df['body'] = df['close'] - df['open']
        df['body_percent'] = df['body'] / df['open'] * 100
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Price range
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_percent'] = df['daily_range'] / df['close'] * 100
        
        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = df['gap'] / df['close'].shift(1) * 100
        
        # High-Low range percentile
        df['hl_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Days since high/low
        df['days_since_high'] = df['high'].rolling(window=252, min_periods=1).apply(
            lambda x: len(x) - 1 - x.argmax(), raw=True
        )
        df['days_since_low'] = df['low'].rolling(window=252, min_periods=1).apply(
            lambda x: len(x) - 1 - x.argmin(), raw=True
        )
        
        return df
    
    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with signal columns
        """
        df = df.copy()
        
        # Ensure we have indicators
        if 'rsi_14' not in df.columns:
            df = self.add_all_indicators(df)
        
        # RSI signals
        df['rsi_signal'] = 0
        df.loc[df['rsi_14'] < 30, 'rsi_signal'] = 1  # Oversold - Buy
        df.loc[df['rsi_14'] > 70, 'rsi_signal'] = -1  # Overbought - Sell
        
        # MACD signals
        df['macd_signal_line'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'macd_signal_line'] = 1  # Bullish
        df.loc[df['macd'] < df['macd_signal'], 'macd_signal_line'] = -1  # Bearish
        
        # Moving Average signals
        df['ma_signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'ma_signal'] = 1  # Bullish
        df.loc[df['sma_20'] < df['sma_50'], 'ma_signal'] = -1  # Bearish
        
        # Bollinger Band signals
        df['bb_signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1  # Oversold
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1  # Overbought
        
        # Stochastic signals
        df['stoch_signal'] = 0
        df.loc[(df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']), 'stoch_signal'] = 1
        df.loc[(df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']), 'stoch_signal'] = -1
        
        # Combined signal
        signal_cols = ['rsi_signal', 'macd_signal_line', 'ma_signal', 'bb_signal', 'stoch_signal']
        df['combined_signal'] = df[signal_cols].mean(axis=1)
        
        # Final recommendation
        df['recommendation'] = 'HOLD'
        df.loc[df['combined_signal'] > 0.3, 'recommendation'] = 'BUY'
        df.loc[df['combined_signal'] < -0.3, 'recommendation'] = 'SELL'
        
        return df
