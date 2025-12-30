"""
Configuration file for Stock Market Prediction System
"""

# Application Settings
APP_NAME = "Stock Market Prediction & Wealth Management System"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Stock Market Analytics"

# Default Stock Lists
NIFTY_50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "HCLTECH", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TATAMOTORS", "POWERGRID", "NTPC", "M&M",
    "TECHM", "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV",
    "TATASTEEL", "ONGC", "JSWSTEEL", "HINDALCO", "GRASIM",
    "DIVISLAB", "BPCL", "BRITANNIA", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "INDUSINDBK", "APOLLOHOSP", "SBILIFE",
    "TATACONSUM", "UPL", "HDFCLIFE", "LTIM", "BAJAJ-AUTO"
]

# Market Indices
INDICES = {
    "NIFTY 50": "NIFTY 50",
    "NIFTY BANK": "NIFTY BANK",
    "NIFTY IT": "NIFTY IT",
    "NIFTY PHARMA": "NIFTY PHARMA",
    "NIFTY AUTO": "NIFTY AUTO",
    "NIFTY FMCG": "NIFTY FMCG",
    "NIFTY METAL": "NIFTY METAL",
    "NIFTY ENERGY": "NIFTY ENERGY",
    "NIFTY REALTY": "NIFTY REALTY",
    "NIFTY INFRA": "NIFTY INFRA",
    "SENSEX": "SENSEX"
}

# Time Periods for Analysis
TIME_PERIODS = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825
}

# Investment Return Options
RETURN_OPTIONS = [5, 7, 8, 10, 12, 15, 18, 20, 25, 30]

# Risk Profiles
RISK_PROFILES = {
    "Conservative": {"equity": 30, "debt": 50, "gold": 10, "cash": 10},
    "Moderate": {"equity": 50, "debt": 35, "gold": 10, "cash": 5},
    "Aggressive": {"equity": 70, "debt": 20, "gold": 5, "cash": 5},
    "Very Aggressive": {"equity": 85, "debt": 10, "gold": 3, "cash": 2}
}

# Technical Indicators Settings
TECHNICAL_SETTINGS = {
    "SMA_PERIODS": [20, 50, 200],
    "EMA_PERIODS": [12, 26, 50],
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BOLLINGER_PERIOD": 20,
    "BOLLINGER_STD": 2,
    "ATR_PERIOD": 14,
    "STOCH_PERIOD": 14
}

# Color Theme
COLORS = {
    "primary": "#1E88E5",
    "secondary": "#26A69A",
    "success": "#66BB6A",
    "danger": "#EF5350",
    "warning": "#FFA726",
    "info": "#42A5F5",
    "dark": "#424242",
    "light": "#F5F5F5",
    "profit": "#00C853",
    "loss": "#FF1744"
}

# Chart Settings
CHART_THEME = "plotly_dark"
CHART_HEIGHT = 500
CHART_WIDTH = None  # Auto
