#!/usr/bin/env python3
"""
AI Wealth Manager - Main Entry Point
====================================

Run this file to start the CLI application:
    python wealth.py [command] [options]

Commands:
    info      - Display application information
    fetch     - Fetch stock data
    analyze   - Perform technical analysis
    predict   - Predict stock prices using ML/DL
    portfolio - Generate portfolio recommendations
    stocks    - List popular Indian stocks
    backtest  - Backtest trading strategies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.main import cli

if __name__ == '__main__':
    cli()
