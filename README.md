# AI Wealth Manager - Stock Market Prediction

🤖 **AI/ML-Driven Wealth Management System for Indian Stock Market**

A comprehensive CLI application for intelligent stock market analysis, prediction, and wealth management using Machine Learning and Deep Learning.

## 🌟 Features

- **📊 Stock Data Collection**: Fetch real-time and historical stock data from NSE/BSE via Yahoo Finance
- **📈 Technical Analysis**: Calculate 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **🧠 Deep Learning (LSTM)**: Advanced time series prediction using bidirectional LSTM networks
- **🌲 Random Forest**: Ensemble machine learning for robust predictions
- **⚡ XGBoost**: High-performance gradient boosting with hyperparameter optimization
- **🎯 Ensemble Models**: Combine multiple models for superior accuracy
- **💼 Portfolio Management**: Intelligent portfolio allocation and optimization
- **📉 Backtesting**: Test trading strategies on historical data
- **🎨 Rich CLI Interface**: Beautiful command-line interface with tables and progress bars

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Joy-Macwan/stock_market_prediction.git
cd stock_market_prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Display help and available commands
python wealth.py --help

# Show application info and features
python wealth.py info

# Fetch stock data
python wealth.py fetch -s RELIANCE.NS -p 1y

# Perform technical analysis
python wealth.py analyze -s TCS.NS

# Predict stock prices using different models
python wealth.py predict -s HDFCBANK.NS -m lstm -d 30
python wealth.py predict -s INFY.NS -m xgb -d 30
python wealth.py predict -s ICICIBANK.NS -m ensemble -d 30

# Generate portfolio recommendations
python wealth.py portfolio -s RELIANCE.NS -s TCS.NS -s HDFCBANK.NS -i 100000

# Backtest trading strategy
python wealth.py backtest -s RELIANCE.NS -i 100000 -p 1y

# List popular Indian stocks
python wealth.py stocks
```

## 📁 Project Structure

```text
stock_market_prediction/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Application configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py         # Stock data collection
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── technical_indicators.py  # Technical analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Base model class
│   │   ├── lstm_model.py        # LSTM deep learning model
│   │   ├── random_forest_model.py  # Random Forest model
│   │   ├── xgboost_model.py     # XGBoost model
│   │   ├── ensemble_model.py    # Ensemble model
│   │   └── model_evaluator.py   # Model evaluation utilities
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py              # CLI application
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       └── helpers.py           # Utility functions
├── data/                        # Data storage
│   ├── raw/                     # Raw downloaded data
│   ├── processed/               # Processed data
│   └── predictions/             # Model predictions
├── models/                      # Saved models
├── logs/                        # Application logs
├── wealth.py                    # Main entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🎯 Supported Models

| Model            | Type             | Best For                               |
| ---------------- | ---------------- | -------------------------------------- |
| **LSTM**         | Deep Learning    | Time series patterns, long-term trends |
| **Random Forest**| Machine Learning | Feature importance, robust predictions |
| **XGBoost**      | Gradient Boosting| High accuracy, fast training           |
| **Ensemble**     | Combined         | Best overall accuracy                  |

## 📊 Technical Indicators

The system calculates 50+ technical indicators including:

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Keltner Channel
- **Volume**: OBV, VWAP, Volume Ratio

## 🇮🇳 Supported Indian Stocks

The system supports all NSE/BSE listed stocks. Popular ones include:

- RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS
- SBIN.NS, BHARTIARTL.NS, KOTAKBANK.NS, ITC.NS, LT.NS
- And many more...

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is NOT financial advice.

- Past performance does not guarantee future results
- Stock market investments carry inherent risks
- Always do your own research before investing
- Consult a qualified financial advisor for investment decisions

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ for the Indian Stock Market Community
