"""
Helper Utilities
================

Common utility functions for the application.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from datetime import datetime, timedelta


def format_currency(
    value: float,
    currency: str = "INR",
    decimals: int = 2
) -> str:
    """
    Format number as currency.
    
    Args:
        value: Numeric value
        currency: Currency code
        decimals: Decimal places
        
    Returns:
        Formatted currency string
    """
    symbols = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£"
    }
    
    symbol = symbols.get(currency, currency)
    
    if abs(value) >= 10000000:  # Crores
        return f"{symbol}{value/10000000:,.{decimals}f} Cr"
    elif abs(value) >= 100000:  # Lakhs
        return f"{symbol}{value/100000:,.{decimals}f} L"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def format_percentage(
    value: float,
    decimals: int = 2,
    show_sign: bool = True
) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Numeric value (0.1 = 10%)
        decimals: Decimal places
        show_sign: Show + for positive values
        
    Returns:
        Formatted percentage string
    """
    if show_sign:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def calculate_returns(
    prices: Union[List[float], np.ndarray, pd.Series],
    method: str = "simple"
) -> np.ndarray:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
        
    Returns:
        Returns array
    """
    prices = np.array(prices)
    
    if method == "simple":
        returns = np.diff(prices) / prices[:-1]
    elif method == "log":
        returns = np.diff(np.log(prices))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns


def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe ratio
    """
    returns = np.array(returns)
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if returns.std() == 0:
        return 0
    
    sharpe = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(
    prices: Union[List[float], np.ndarray, pd.Series]
) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown (negative value)
    """
    prices = np.array(prices)
    
    cumulative_max = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative_max) / cumulative_max
    
    return np.min(drawdown)


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        years: Number of years
        
    Returns:
        CAGR as percentage
    """
    if initial_value <= 0 or years <= 0:
        return 0
    
    cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100
    
    return cagr


def get_trading_days(
    start_date: datetime,
    end_date: datetime,
    holidays: Optional[List[datetime]] = None
) -> List[datetime]:
    """
    Get list of trading days between dates.
    
    Args:
        start_date: Start date
        end_date: End date
        holidays: List of holiday dates
        
    Returns:
        List of trading days
    """
    if holidays is None:
        holidays = []
    
    holidays_set = set(holidays)
    
    trading_days = []
    current = start_date
    
    while current <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5 and current not in holidays_set:
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


def normalize_data(
    data: np.ndarray,
    method: str = "minmax"
) -> tuple:
    """
    Normalize data.
    
    Args:
        data: Input data
        method: 'minmax', 'zscore', or 'robust'
        
    Returns:
        Tuple of (normalized_data, params)
    """
    if method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val}
    
    elif method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std}
    
    elif method == "robust":
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        normalized = (data - median) / iqr
        params = {'median': median, 'iqr': iqr}
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized, params


def denormalize_data(
    data: np.ndarray,
    params: dict,
    method: str = "minmax"
) -> np.ndarray:
    """
    Reverse normalization.
    
    Args:
        data: Normalized data
        params: Normalization parameters
        method: Normalization method
        
    Returns:
        Original scale data
    """
    if method == "minmax":
        return data * (params['max'] - params['min']) + params['min']
    
    elif method == "zscore":
        return data * params['std'] + params['mean']
    
    elif method == "robust":
        return data * params['iqr'] + params['median']
    
    else:
        raise ValueError(f"Unknown method: {method}")
