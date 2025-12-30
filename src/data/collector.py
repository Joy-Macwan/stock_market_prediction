"""
Stock Data Collector Module
===========================

Fetches historical and real-time stock data from various sources.
Uses subprocess to avoid TensorFlow/curl_cffi conflicts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from pathlib import Path
import logging
import subprocess
import sys
import json
import io
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config.settings import get_settings

console = Console()
logger = logging.getLogger(__name__)


def _fetch_stock_subprocess(symbol: str, period: str = "2y", interval: str = "1d", 
                            start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using subprocess to avoid TensorFlow/curl_cffi conflicts.
    
    This runs yfinance in a separate Python process where TensorFlow is not loaded.
    """
    code = f'''
import yfinance as yf
import json
import sys

try:
    ticker = yf.Ticker("{symbol}")
    {"df = ticker.history(start='" + start_date + "', end='" + end_date + "', interval='" + interval + "')" if start_date and end_date else "df = ticker.history(period='" + period + "', interval='" + interval + "')"}
    
    if df.empty:
        print(json.dumps({{"error": "empty", "length": 0}}))
    else:
        df = df.reset_index()
        df['Date'] = df['Date'].astype(str)
        print(json.dumps({{"length": len(df), "data": df.to_dict(orient='records')}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "length": 0}}))
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            logger.error(f"Subprocess error for {symbol}: {result.stderr}")
            return None
        
        output = result.stdout.strip()
        if not output:
            logger.warning(f"No output for {symbol}")
            return None
            
        data = json.loads(output)
        
        if data.get("error"):
            if data["error"] != "empty":
                logger.warning(f"Error fetching {symbol}: {data['error']}")
            return None
        
        if data["length"] == 0:
            return None
            
        df = pd.DataFrame(data["data"])
        df['Date'] = pd.to_datetime(df['Date'])
        return df
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout fetching {symbol}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in subprocess fetch for {symbol}: {e}")
        return None


class DataCollector:
    """
    Stock market data collector using Yahoo Finance API.
    
    Supports Indian stock market (NSE/BSE) and international markets.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Data period if dates not specified ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{period}_{interval}"
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            # Use subprocess to avoid TensorFlow/curl_cffi conflicts
            df = _fetch_stock_subprocess(
                symbol=symbol,
                period=period,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Rename date column
            if 'date' not in df.columns and 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            
            # Cache the data
            self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching stock data...", total=len(symbols))
            
            for symbol in symbols:
                try:
                    df = self.get_stock_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        period=period,
                        interval=interval
                    )
                    if not df.empty:
                        data[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                
                progress.update(task, advance=1)
        
        return data
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed stock information using subprocess.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        code = f'''
import yfinance as yf
import json

try:
    ticker = yf.Ticker("{symbol}")
    info = ticker.info
    result = {{
        'symbol': "{symbol}",
        'name': info.get('longName', info.get('shortName', 'N/A')),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'pb_ratio': info.get('priceToBook', 0),
        'dividend_yield': info.get('dividendYield', 0),
        'beta': info.get('beta', 0),
        '52_week_high': info.get('fiftyTwoWeekHigh', 0),
        '52_week_low': info.get('fiftyTwoWeekLow', 0),
        'avg_volume': info.get('averageVolume', 0),
        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
        'currency': info.get('currency', 'INR')
    }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{'symbol': "{symbol}", 'error': str(e)}}))
'''
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
            else:
                return {'symbol': symbol, 'error': result.stderr or 'Unknown error'}
                
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_index_data(self, index: str = "^NSEI", period: str = "2y") -> pd.DataFrame:
        """
        Fetch index data (NIFTY, SENSEX, etc.).
        
        Args:
            index: Index symbol (^NSEI for NIFTY 50, ^BSESN for SENSEX)
            period: Data period
            
        Returns:
            DataFrame with index data
        """
        return self.get_stock_data(symbol=index, period=period)
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = "csv") -> Path:
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            format: Output format ('csv', 'parquet', 'json')
            
        Returns:
            Path to saved file
        """
        output_dir = self.settings.DATA_DIR / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            filepath = output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == "parquet":
            filepath = output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        elif format == "json":
            filepath = output_dir / f"{filename}.json"
            df.to_json(filepath, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str, format: str = "csv") -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filename: Input filename
            format: File format
            
        Returns:
            DataFrame with loaded data
        """
        input_dir = self.settings.DATA_DIR / "raw"
        
        if format == "csv":
            filepath = input_dir / f"{filename}.csv"
            df = pd.read_csv(filepath, parse_dates=['date'])
        elif format == "parquet":
            filepath = input_dir / f"{filename}.parquet"
            df = pd.read_parquet(filepath)
        elif format == "json":
            filepath = input_dir / f"{filename}.json"
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return df
    
    def get_nifty50_stocks(self) -> List[str]:
        """Get list of NIFTY 50 stock symbols."""
        return self.settings.POPULAR_STOCKS.copy()
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Cache cleared")
