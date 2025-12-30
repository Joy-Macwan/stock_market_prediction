# Stock Market Prediction & Wealth Management System

A comprehensive AI-powered stock market analysis, prediction, and wealth management system built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

### 📊 Stock Analysis
- Real-time stock data from Indian markets (NSE/BSE)
- Comprehensive price charts with candlestick patterns
- Support and resistance level calculation
- Trend analysis with multiple timeframes

### 🔮 Price Prediction
- Machine Learning models (Random Forest, Gradient Boosting, Linear Regression)
- Time Series forecasting (Exponential Smoothing, Moving Average)
- Ensemble predictions combining multiple models
- Confidence intervals for predictions
- Buy/Sell/Hold recommendations

### 📋 Technical Analysis
- 20+ Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA, EMA)
  - Stochastic Oscillator
  - ADX (Average Directional Index)
  - ATR (Average True Range)
  - And more...
- Trading signal generation
- Multi-indicator signal confirmation

### 💰 Wealth Calculator
- Future Value Calculator
- SIP (Systematic Investment Plan) Calculator
- Goal-Based Investment Planner
- Retirement Planning Calculator
- Lumpsum vs SIP Comparison
- Year-by-year investment growth schedule

### 📈 Portfolio Manager
- Add and track stock holdings
- Real-time portfolio valuation
- Profit/Loss calculation
- Asset allocation recommendations by risk profile
- Portfolio rebalancing suggestions

### 🎯 Stock Screener
- Filter stocks by price range
- Filter by returns percentage
- RSI-based filtering (oversold/overbought)
- Sector-wise filtering
- Top picks identification

### 📰 Market Overview
- Live market indices (NIFTY 50, BANK NIFTY, SENSEX, etc.)
- Sector-wise performance analysis
- Market heatmap visualization
- Top gainers and losers

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock_market_prediction.git
cd stock_market_prediction
```

2. **Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Start the application**
```bash
chmod +x run.sh
./run.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🎮 Usage

1. **Open your browser** and navigate to `http://localhost:8501`

2. **Select a module** from the sidebar:
   - 🏠 Dashboard - Quick market overview
   - 📊 Stock Analysis - Detailed stock analysis
   - 🔮 Price Prediction - AI-powered predictions
   - 💰 Wealth Calculator - Investment calculators
   - 📈 Portfolio Manager - Track your investments
   - 📋 Technical Analysis - Technical indicators
   - 🎯 Stock Screener - Filter stocks
   - 📰 Market Overview - Market summary

3. **For Stock Analysis**:
   - Select a stock from the dropdown
   - Choose time period
   - Click "Analyze Stock"

4. **For Price Prediction**:
   - Select a stock
   - Choose prediction days (7-90 days)
   - Click "Generate Prediction"

5. **For Wealth Calculator**:
   - Enter your investment amount
   - Set expected return rate
   - Choose investment period
   - View detailed projections

## 📁 Project Structure

```
stock_market_prediction/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── stock_data.py           # Stock data fetching module
├── technical_analysis.py   # Technical indicators
├── prediction_models.py    # ML prediction models
├── wealth_calculator.py    # Wealth calculation tools
├── styles.py               # Custom CSS styles
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── run.sh                 # Run script
└── README.md              # Documentation
```

## 🔧 Configuration

Edit `config.py` to customize:
- Default stock lists
- Technical indicator parameters
- Risk profiles
- Color themes
- Chart settings

## 📊 Data Source

This application uses the **Bharat-SM-Data** package for fetching Indian stock market data:
- NSE (National Stock Exchange)
- BSE (Bombay Stock Exchange)

> **Note**: Pass `is_index=True` when calling functions for Indices, Futures, and Options contracts.

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It is NOT financial advice. Always:
- Do your own research before investing
- Consult with a qualified financial advisor
- Understand the risks involved in stock market investments
- Past performance does not guarantee future results

## 🛠️ Technologies Used

- **Frontend**: Streamlit, HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Technical Analysis**: TA-Lib (ta)
- **Data Source**: Bharat-SM-Data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Bharat-SM-Data](https://github.com/Sampad-Hegde/Bharat-SM-Data) for Indian market data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive charts

---

**Built with ❤️ for Indian Stock Market Analysis**
