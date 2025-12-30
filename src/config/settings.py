"""
Application Settings and Configuration
======================================

Centralized configuration management for the wealth management system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Application settings configuration."""
    
    # Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Stock Market Settings
    DEFAULT_MARKET: str = "NSE"  # NSE for Indian stocks
    DEFAULT_CURRENCY: str = "INR"
    
    # Indian Stock Indices
    INDIAN_INDICES: List[str] = field(default_factory=lambda: [
        "^NSEI",      # NIFTY 50
        "^BSESN",     # BSE SENSEX
        "^NSEBANK",   # NIFTY Bank
        "^CNXIT",     # NIFTY IT
    ])
    
    # Popular Indian Stocks (NSE symbols)
    POPULAR_STOCKS: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS"
    ])
    
    # Model Settings
    DEFAULT_LOOKBACK: int = 60  # Days to look back for prediction
    DEFAULT_FORECAST_DAYS: int = 30  # Days to forecast
    TRAIN_TEST_SPLIT: float = 0.8
    RANDOM_STATE: int = 42
    
    # Deep Learning Settings
    LSTM_UNITS: int = 128
    LSTM_LAYERS: int = 2
    DROPOUT_RATE: float = 0.2
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Technical Analysis Settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Risk Management
    MAX_PORTFOLIO_RISK: float = 0.02  # 2% max risk per trade
    STOP_LOSS_PERCENT: float = 0.05   # 5% stop loss
    TAKE_PROFIT_PERCENT: float = 0.15  # 15% take profit
    
    # API Settings
    YFINANCE_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "raw").mkdir(exist_ok=True)
        (self.DATA_DIR / "processed").mkdir(exist_ok=True)
        (self.DATA_DIR / "predictions").mkdir(exist_ok=True)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
