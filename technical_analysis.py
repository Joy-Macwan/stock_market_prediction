"""
Technical Analysis Module for Stock Market Prediction
"""

import pandas as pd
import numpy as np
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from config import TECHNICAL_SETTINGS


def add_all_technical_indicators(df):
    """Add all technical indicators to the dataframe"""
    df = df.copy()
    
    # Ensure we have required columns
    if 'close' not in df.columns:
        return df
    
    # Moving Averages
    df = add_moving_averages(df)
    
    # Momentum Indicators
    df = add_momentum_indicators(df)
    
    # Volatility Indicators
    df = add_volatility_indicators(df)
    
    # Volume Indicators
    if 'volume' in df.columns:
        df = add_volume_indicators(df)
    
    # Trend Indicators
    df = add_trend_indicators(df)
    
    return df


def add_moving_averages(df):
    """Add Simple and Exponential Moving Averages"""
    df = df.copy()
    
    # Simple Moving Averages
    for period in TECHNICAL_SETTINGS['SMA_PERIODS']:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in TECHNICAL_SETTINGS['EMA_PERIODS']:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    return df


def add_momentum_indicators(df):
    """Add RSI, MACD, Stochastic Oscillator"""
    df = df.copy()
    
    # RSI
    rsi_period = TECHNICAL_SETTINGS['RSI_PERIOD']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    fast = TECHNICAL_SETTINGS['MACD_FAST']
    slow = TECHNICAL_SETTINGS['MACD_SLOW']
    signal = TECHNICAL_SETTINGS['MACD_SIGNAL']
    
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic Oscillator
    stoch_period = TECHNICAL_SETTINGS['STOCH_PERIOD']
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Rate of Change (ROC)
    df['ROC'] = df['close'].pct_change(periods=10) * 100
    
    # Momentum
    df['Momentum'] = df['close'].diff(10)
    
    return df


def add_volatility_indicators(df):
    """Add Bollinger Bands, ATR"""
    df = df.copy()
    
    # Bollinger Bands
    bb_period = TECHNICAL_SETTINGS['BOLLINGER_PERIOD']
    bb_std = TECHNICAL_SETTINGS['BOLLINGER_STD']
    
    df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_val = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_val * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_val * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Average True Range (ATR)
    atr_period = TECHNICAL_SETTINGS['ATR_PERIOD']
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=atr_period).mean()
    
    # Historical Volatility
    df['Volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    return df


def add_volume_indicators(df):
    """Add Volume-based indicators"""
    df = df.copy()
    
    if 'volume' not in df.columns:
        return df
    
    # Volume Moving Average
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Volume Price Trend
    df['VPT'] = ((df['close'].diff() / df['close'].shift()) * df['volume']).fillna(0).cumsum()
    
    # Accumulation/Distribution Line
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)
    df['ADL'] = (clv * df['volume']).cumsum()
    
    return df


def add_trend_indicators(df):
    """Add trend indicators"""
    df = df.copy()
    
    # Average Directional Index (ADX)
    period = 14
    
    # Calculate +DM and -DM
    df['High_Diff'] = df['high'].diff()
    df['Low_Diff'] = -df['low'].diff()
    
    df['+DM'] = np.where((df['High_Diff'] > df['Low_Diff']) & (df['High_Diff'] > 0), df['High_Diff'], 0)
    df['-DM'] = np.where((df['Low_Diff'] > df['High_Diff']) & (df['Low_Diff'] > 0), df['Low_Diff'], 0)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values
    df['ATR_ADX'] = df['TR'].rolling(window=period).mean()
    df['+DI'] = 100 * (df['+DM'].rolling(window=period).mean() / df['ATR_ADX'])
    df['-DI'] = 100 * (df['-DM'].rolling(window=period).mean() / df['ATR_ADX'])
    
    # ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # Clean up temporary columns
    df.drop(['High_Diff', 'Low_Diff', '+DM', '-DM', 'TR', 'ATR_ADX', 'DX'], axis=1, inplace=True, errors='ignore')
    
    # Parabolic SAR (simplified)
    df['SAR'] = df['close'].shift(1) * 0.98  # Simplified approximation
    
    return df


def generate_trading_signals(df):
    """Generate buy/sell signals based on technical indicators"""
    df = df.copy()
    
    signals = pd.DataFrame(index=df.index)
    signals['Signal'] = 0
    signals['Signal_Strength'] = 0
    
    # RSI Signals
    if 'RSI' in df.columns:
        signals.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold - Buy
        signals.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought - Sell
        signals['RSI_Signal'] = signals.get('RSI_Signal', 0).fillna(0)
    
    # MACD Signals
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        signals['MACD_Signal_Line'] = 0
        signals.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal_Line'] = 1
        signals.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal_Line'] = -1
    
    # Moving Average Crossover Signals
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        signals['MA_Signal'] = 0
        signals.loc[df['SMA_20'] > df['SMA_50'], 'MA_Signal'] = 1
        signals.loc[df['SMA_20'] < df['SMA_50'], 'MA_Signal'] = -1
    
    # Bollinger Bands Signals
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        signals['BB_Signal'] = 0
        signals.loc[df['close'] < df['BB_Lower'], 'BB_Signal'] = 1  # Buy
        signals.loc[df['close'] > df['BB_Upper'], 'BB_Signal'] = -1  # Sell
    
    # Stochastic Signals
    if 'Stoch_K' in df.columns:
        signals['Stoch_Signal'] = 0
        signals.loc[df['Stoch_K'] < 20, 'Stoch_Signal'] = 1  # Oversold
        signals.loc[df['Stoch_K'] > 80, 'Stoch_Signal'] = -1  # Overbought
    
    # Combine signals
    signal_columns = [col for col in signals.columns if col.endswith('_Signal') or col.endswith('_Signal_Line')]
    if signal_columns:
        signals['Combined_Signal'] = signals[signal_columns].sum(axis=1)
        signals['Signal'] = np.sign(signals['Combined_Signal'])
        signals['Signal_Strength'] = signals['Combined_Signal'].abs()
    
    return signals


def get_support_resistance(df, window=20):
    """Calculate support and resistance levels"""
    df = df.copy()
    
    # Recent highs and lows
    recent_high = df['high'].rolling(window=window).max().iloc[-1]
    recent_low = df['low'].rolling(window=window).min().iloc[-1]
    
    # Pivot Points
    pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    
    r1 = 2 * pivot - df['low'].iloc[-1]
    r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
    r3 = r1 + (df['high'].iloc[-1] - df['low'].iloc[-1])
    
    s1 = 2 * pivot - df['high'].iloc[-1]
    s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
    s3 = s1 - (df['high'].iloc[-1] - df['low'].iloc[-1])
    
    return {
        'pivot': round(pivot, 2),
        'resistance_1': round(r1, 2),
        'resistance_2': round(r2, 2),
        'resistance_3': round(r3, 2),
        'support_1': round(s1, 2),
        'support_2': round(s2, 2),
        'support_3': round(s3, 2),
        'recent_high': round(recent_high, 2),
        'recent_low': round(recent_low, 2)
    }


def get_trend_analysis(df):
    """Analyze current trend"""
    if len(df) < 50:
        return {'trend': 'Insufficient Data', 'strength': 0}
    
    close = df['close']
    
    # Short-term trend (20 days)
    short_trend = 'UP' if close.iloc[-1] > close.iloc[-20] else 'DOWN'
    
    # Medium-term trend (50 days)
    medium_trend = 'UP' if close.iloc[-1] > close.iloc[-50] else 'DOWN'
    
    # Moving average alignment
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        ma_bullish = df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]
        ma_bearish = df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]
    else:
        ma_bullish = False
        ma_bearish = False
    
    # ADX trend strength
    if 'ADX' in df.columns:
        adx_value = df['ADX'].iloc[-1]
        if adx_value > 25:
            trend_strength = 'Strong'
        elif adx_value > 20:
            trend_strength = 'Moderate'
        else:
            trend_strength = 'Weak'
    else:
        trend_strength = 'Unknown'
    
    # Overall trend determination
    if short_trend == 'UP' and medium_trend == 'UP':
        overall_trend = 'BULLISH'
    elif short_trend == 'DOWN' and medium_trend == 'DOWN':
        overall_trend = 'BEARISH'
    else:
        overall_trend = 'SIDEWAYS'
    
    if ma_bullish:
        overall_trend = 'STRONGLY BULLISH'
    elif ma_bearish:
        overall_trend = 'STRONGLY BEARISH'
    
    return {
        'overall_trend': overall_trend,
        'short_term': short_trend,
        'medium_term': medium_trend,
        'strength': trend_strength,
        'ma_alignment': 'Bullish' if ma_bullish else ('Bearish' if ma_bearish else 'Mixed')
    }


def calculate_risk_metrics(df, risk_free_rate=0.05):
    """Calculate risk metrics"""
    if len(df) < 20:
        return {}
    
    returns = df['close'].pct_change().dropna()
    
    # Annual return
    annual_return = returns.mean() * 252
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5)
    
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Beta (using simple benchmark)
    beta = 1.0  # Simplified - would need market returns for actual calculation
    
    return {
        'annual_return': round(annual_return * 100, 2),
        'volatility': round(volatility * 100, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'var_95': round(var_95 * 100, 2),
        'beta': round(beta, 2)
    }
