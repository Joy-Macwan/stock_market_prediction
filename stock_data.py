"""
Stock Data Fetching Module using Bharat-SM-Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from BharatSMData import get_ohlc, get_stock_list, get_index_list
    BHARAT_SM_AVAILABLE = True
except ImportError:
    BHARAT_SM_AVAILABLE = False
    print("Warning: Bharat-SM-Data not installed. Using sample data.")

from config import NIFTY_50_STOCKS, INDICES


def get_available_stocks():
    """Get list of available stocks"""
    if BHARAT_SM_AVAILABLE:
        try:
            stocks = get_stock_list()
            return stocks if stocks else NIFTY_50_STOCKS
        except:
            return NIFTY_50_STOCKS
    return NIFTY_50_STOCKS


def get_available_indices():
    """Get list of available indices"""
    if BHARAT_SM_AVAILABLE:
        try:
            indices = get_index_list()
            return indices if indices else list(INDICES.keys())
        except:
            return list(INDICES.keys())
    return list(INDICES.keys())


def fetch_stock_data(symbol, start_date, end_date, is_index=False):
    """
    Fetch OHLC data for a stock or index
    
    Args:
        symbol: Stock symbol or index name
        start_date: Start date for data
        end_date: End date for data
        is_index: Set True for indices, futures, and options
    
    Returns:
        DataFrame with OHLC data
    """
    if BHARAT_SM_AVAILABLE:
        try:
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
            end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
            
            # Fetch data using Bharat-SM-Data
            # Pass is_index=True for indices, futures, and options
            df = get_ohlc(symbol, start_str, end_str, is_index=is_index)
            
            if df is not None and not df.empty:
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    return df
            
            # Fall back to sample data if fetch fails
            return generate_sample_data(symbol, start_date, end_date)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return generate_sample_data(symbol, start_date, end_date)
    else:
        return generate_sample_data(symbol, start_date, end_date)


def generate_sample_data(symbol, start_date, end_date):
    """Generate sample stock data for demonstration"""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    n = len(dates)
    if n == 0:
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        dates = dates[dates.dayofweek < 5]
        n = len(dates)
    
    # Generate realistic stock prices
    np.random.seed(hash(symbol) % 2**32)
    
    # Base price varies by symbol
    base_price = 100 + (hash(symbol) % 5000)
    
    # Generate returns with slight upward bias
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC data
    df = pd.DataFrame({
        'date': dates[:len(prices)],
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.03, len(prices))),
        'low': prices * (1 - np.random.uniform(0, 0.03, len(prices))),
        'close': prices,
        'volume': np.random.randint(100000, 10000000, len(prices))
    })
    
    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    df.set_index('date', inplace=True)
    
    return df


def get_live_price(symbol, is_index=False):
    """Get current live price for a symbol"""
    if BHARAT_SM_AVAILABLE:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            df = fetch_stock_data(symbol, start_date, end_date, is_index=is_index)
            if df is not None and not df.empty:
                return {
                    'symbol': symbol,
                    'price': df['close'].iloc[-1],
                    'open': df['open'].iloc[-1],
                    'high': df['high'].iloc[-1],
                    'low': df['low'].iloc[-1],
                    'change': df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0,
                    'change_pct': ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0,
                    'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0
                }
        except Exception as e:
            print(f"Error getting live price: {e}")
    
    # Return sample live price
    np.random.seed(hash(symbol) % 2**32)
    price = 100 + (hash(symbol) % 5000) + np.random.uniform(-50, 50)
    change = np.random.uniform(-50, 50)
    
    return {
        'symbol': symbol,
        'price': round(price, 2),
        'open': round(price - np.random.uniform(-10, 10), 2),
        'high': round(price + np.random.uniform(0, 20), 2),
        'low': round(price - np.random.uniform(0, 20), 2),
        'change': round(change, 2),
        'change_pct': round((change / price) * 100, 2),
        'volume': np.random.randint(100000, 10000000)
    }


def get_multiple_stocks_data(symbols, start_date, end_date, is_index=False):
    """Fetch data for multiple stocks"""
    data = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, start_date, end_date, is_index=is_index)
        if df is not None and not df.empty:
            data[symbol] = df
    return data


def calculate_returns(df, periods=[1, 5, 20, 60, 252]):
    """Calculate returns for different periods"""
    returns = {}
    if 'close' in df.columns:
        close = df['close']
        for period in periods:
            if len(close) > period:
                returns[f'{period}d_return'] = ((close.iloc[-1] / close.iloc[-period]) - 1) * 100
    return returns


def get_sector_data():
    """Get sector-wise stock classification"""
    sectors = {
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
        "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
        "Oil & Gas": ["RELIANCE", "ONGC", "BPCL", "GAIL", "IOC"],
        "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
        "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
        "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
        "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
        "Infrastructure": ["LT", "ADANIENT", "ADANIPORTS", "POWERGRID", "NTPC"],
        "Finance": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE"],
        "Consumer": ["TITAN", "ASIANPAINT", "ULTRACEMCO", "GRASIM"]
    }
    return sectors
