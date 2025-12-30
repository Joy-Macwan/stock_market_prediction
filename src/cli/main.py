"""
Wealth Management CLI Application
=================================

Command-line interface for AI-powered stock market prediction and wealth management.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich import box
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import get_settings
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.technical_indicators import TechnicalIndicators
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble_model import EnsembleModel
from src.models.model_evaluator import ModelEvaluator
from src.models.investment_advisor import InvestmentAdvisorModel, UserProfile

console = Console()
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗ ██╗    ██╗    ██╗███████╗ █████╗ ██╗  ████████╗██╗  ██╗           ║
║    ██╔══██╗██║    ██║    ██║██╔════╝██╔══██╗██║  ╚══██╔══╝██║  ██║           ║
║    ███████║██║    ██║ █╗ ██║█████╗  ███████║██║     ██║   ███████║           ║
║    ██╔══██║██║    ██║███╗██║██╔══╝  ██╔══██║██║     ██║   ██╔══██║           ║
║    ██║  ██║██║    ╚███╔███╔╝███████╗██║  ██║███████╗██║   ██║  ██║           ║
║    ╚═╝  ╚═╝╚═╝     ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝           ║
║                                                                              ║
║            🤖 AI-Powered Stock Market Prediction & Wealth Management         ║
║                      For Indian Stock Market (NSE/BSE)                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@click.group()
@click.version_option(version="1.0.0", prog_name="AI Wealth Manager")
def cli():
    """
    🤖 AI Wealth Manager - Stock Market Prediction CLI
    
    An intelligent command-line application for stock market analysis,
    prediction, and wealth management using Machine Learning and Deep Learning.
    """
    pass


@cli.command()
def info():
    """Display application information and available features."""
    print_banner()
    
    features_table = Table(title="📋 Available Features", box=box.ROUNDED)
    features_table.add_column("Feature", style="cyan", no_wrap=True)
    features_table.add_column("Description", style="white")
    
    features = [
        ("📊 Stock Data", "Fetch real-time and historical stock data from NSE/BSE"),
        ("📈 Technical Analysis", "Calculate 50+ technical indicators"),
        ("🧠 LSTM Model", "Deep learning for time series prediction"),
        ("🌲 Random Forest", "Ensemble ML for robust predictions"),
        ("⚡ XGBoost", "Gradient boosting for high performance"),
        ("🎯 Ensemble", "Combine multiple models for best accuracy"),
        ("💰 Portfolio", "Portfolio optimization and risk management"),
        ("📉 Backtesting", "Test strategies on historical data"),
    ]
    
    for feature, desc in features:
        features_table.add_row(feature, desc)
    
    console.print(features_table)
    console.print()


@cli.command()
@click.option('--symbol', '-s', default='RELIANCE.NS', help='Stock symbol (e.g., RELIANCE.NS)')
@click.option('--period', '-p', default='1y', help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)')
def fetch(symbol, period):
    """Fetch stock data for a given symbol."""
    console.print(f"\n[bold cyan]📊 Fetching data for {symbol}...[/bold cyan]")
    
    collector = DataCollector()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Downloading stock data...", total=None)
        
        try:
            df = collector.get_stock_data(symbol=symbol, period=period)
            progress.update(task, completed=True)
            
            if df.empty:
                console.print(f"[red]❌ No data found for {symbol}[/red]")
                return
            
            # Display stock info
            info = collector.get_stock_info(symbol)
            
            info_table = Table(title=f"📋 {info.get('name', symbol)} Information", box=box.ROUNDED)
            info_table.add_column("Metric", style="cyan")
            info_table.add_column("Value", style="green")
            
            info_rows = [
                ("Symbol", symbol),
                ("Sector", str(info.get('sector', 'N/A'))),
                ("Industry", str(info.get('industry', 'N/A'))),
                ("Current Price", f"₹{info.get('current_price', 0):,.2f}"),
                ("Market Cap", f"₹{info.get('market_cap', 0)/10000000:,.2f} Cr"),
                ("P/E Ratio", f"{info.get('pe_ratio', 0):.2f}"),
                ("52-Week High", f"₹{info.get('52_week_high', 0):,.2f}"),
                ("52-Week Low", f"₹{info.get('52_week_low', 0):,.2f}"),
            ]
            
            for metric, value in info_rows:
                info_table.add_row(metric, str(value))
            
            console.print(info_table)
            
            # Display recent data
            data_table = Table(title=f"📈 Recent Price Data (Last 10 Days)", box=box.ROUNDED)
            data_table.add_column("Date", style="cyan")
            data_table.add_column("Open", style="white")
            data_table.add_column("High", style="green")
            data_table.add_column("Low", style="red")
            data_table.add_column("Close", style="yellow")
            data_table.add_column("Volume", style="blue")
            
            recent_data = df.tail(10)
            for _, row in recent_data.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                data_table.add_row(
                    date_str,
                    f"₹{row['open']:,.2f}",
                    f"₹{row['high']:,.2f}",
                    f"₹{row['low']:,.2f}",
                    f"₹{row['close']:,.2f}",
                    f"{int(row['volume']):,}"
                )
            
            console.print(data_table)
            
            # Save data
            filepath = collector.save_data(df, f"{symbol.replace('.', '_')}_data")
            console.print(f"\n[green]✅ Data saved to {filepath}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            logger.error(f"Error fetching data: {e}")


@cli.command()
@click.option('--symbol', '-s', default='RELIANCE.NS', help='Stock symbol')
@click.option('--period', '-p', default='2y', help='Data period')
def analyze(symbol, period):
    """Perform technical analysis on a stock."""
    console.print(f"\n[bold cyan]📊 Analyzing {symbol}...[/bold cyan]")
    
    collector = DataCollector()
    indicators = TechnicalIndicators()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Fetch data
        task1 = progress.add_task("Fetching data...", total=None)
        df = collector.get_stock_data(symbol=symbol, period=period)
        progress.update(task1, completed=True)
        
        if df.empty:
            console.print(f"[red]❌ No data found for {symbol}[/red]")
            return
        
        # Calculate indicators
        task2 = progress.add_task("Calculating indicators...", total=None)
        df = indicators.add_all_indicators(df)
        df = indicators.get_signals(df)
        progress.update(task2, completed=True)
    
    # Get latest data
    latest = df.iloc[-1]
    
    # Technical Indicators Table
    ti_table = Table(title=f"📊 Technical Indicators for {symbol}", box=box.ROUNDED)
    ti_table.add_column("Indicator", style="cyan")
    ti_table.add_column("Value", style="white")
    ti_table.add_column("Signal", style="yellow")
    
    # Determine signals
    def get_signal_text(value, thresholds, labels):
        for threshold, label in zip(thresholds, labels):
            if value <= threshold:
                return label
        return labels[-1]
    
    indicators_data = [
        ("RSI (14)", f"{latest.get('rsi_14', 0):.2f}", 
         get_signal_text(latest.get('rsi_14', 50), [30, 70, 100], ["🟢 Oversold", "⚪ Neutral", "🔴 Overbought"])),
        ("MACD", f"{latest.get('macd', 0):.4f}", 
         "🟢 Bullish" if latest.get('macd', 0) > latest.get('macd_signal', 0) else "🔴 Bearish"),
        ("Stochastic %K", f"{latest.get('stoch_k', 0):.2f}",
         get_signal_text(latest.get('stoch_k', 50), [20, 80, 100], ["🟢 Oversold", "⚪ Neutral", "🔴 Overbought"])),
        ("ADX", f"{latest.get('adx', 0):.2f}",
         "Strong Trend" if latest.get('adx', 0) > 25 else "Weak Trend"),
        ("Bollinger %B", f"{latest.get('bb_percent', 0):.2f}",
         get_signal_text(latest.get('bb_percent', 0.5), [0.2, 0.8, 1.0], ["🟢 Near Lower", "⚪ Middle", "🔴 Near Upper"])),
        ("20-Day Volatility", f"{latest.get('volatility_20', 0)*100:.2f}%", ""),
    ]
    
    for ind, val, sig in indicators_data:
        ti_table.add_row(ind, val, sig)
    
    console.print(ti_table)
    
    # Moving Averages Table
    ma_table = Table(title="📈 Moving Averages", box=box.ROUNDED)
    ma_table.add_column("MA Type", style="cyan")
    ma_table.add_column("Value", style="white")
    ma_table.add_column("vs Price", style="yellow")
    
    current_price = latest['close']
    ma_data = [
        ("SMA 20", latest.get('sma_20', 0)),
        ("SMA 50", latest.get('sma_50', 0)),
        ("SMA 200", latest.get('sma_200', 0)),
        ("EMA 12", latest.get('ema_12', 0)),
        ("EMA 26", latest.get('ema_26', 0)),
    ]
    
    for ma_name, ma_val in ma_data:
        if ma_val > 0:
            diff_pct = ((current_price - ma_val) / ma_val) * 100
            signal = "🟢 Above" if current_price > ma_val else "🔴 Below"
            ma_table.add_row(ma_name, f"₹{ma_val:,.2f}", f"{signal} ({diff_pct:+.2f}%)")
    
    console.print(ma_table)
    
    # Overall Recommendation
    recommendation = latest.get('recommendation', 'HOLD')
    combined_signal = latest.get('combined_signal', 0)
    
    rec_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(recommendation, "white")
    
    console.print(Panel(
        f"[bold {rec_color}]{recommendation}[/bold {rec_color}]\n\n"
        f"Combined Signal Score: {combined_signal:.3f}\n"
        f"(-1 = Strong Sell, 0 = Neutral, +1 = Strong Buy)",
        title="🎯 Overall Recommendation",
        border_style=rec_color
    ))


@cli.command()
@click.option('--symbol', '-s', default='RELIANCE.NS', help='Stock symbol')
@click.option('--model', '-m', type=click.Choice(['lstm', 'rf', 'xgb', 'ensemble']), default='ensemble', help='Model type')
@click.option('--days', '-d', default=30, help='Days to predict')
@click.option('--train/--no-train', default=True, help='Train new model or use existing')
def predict(symbol, model, days, train):
    """Predict future stock prices using ML/DL models."""
    console.print(f"\n[bold cyan]🔮 Predicting {symbol} for next {days} days using {model.upper()} model...[/bold cyan]")
    
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    indicators = TechnicalIndicators()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        # Fetch data
        task1 = progress.add_task("Fetching historical data...", total=100)
        df = collector.get_stock_data(symbol=symbol, period='2y')
        progress.update(task1, completed=100)
        
        if df.empty:
            console.print(f"[red]❌ No data found for {symbol}[/red]")
            return
        
        # Prepare data
        task2 = progress.add_task("Preparing data...", total=100)
        df = preprocessor.clean_data(df)
        df = indicators.add_all_indicators(df)
        df = preprocessor.add_returns(df)
        df = preprocessor.add_volatility(df)
        df = df.dropna()
        progress.update(task2, completed=100)
        
        # Select features for ML models
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi_14', 'macd', 'bb_percent', 'atr', 'adx',
                       'sma_20', 'sma_50', 'ema_12', 'ema_26',
                       'daily_return', 'volatility_20']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if model == 'lstm':
            task3 = progress.add_task("Training LSTM model...", total=100)
            
            # Prepare LSTM data
            lstm_data = preprocessor.prepare_lstm_data(
                df, target_col='close', feature_cols=feature_cols,
                lookback=60, test_size=0.2
            )
            
            # Build and train LSTM
            lstm_model = LSTMModel(name=f"lstm_{symbol.replace('.', '_')}")
            input_shape = (lstm_data['X_train'].shape[1], lstm_data['X_train'].shape[2])
            lstm_model.build(input_shape)
            
            if train:
                lstm_model.train(
                    lstm_data['X_train'], lstm_data['y_train'],
                    lstm_data['X_test'], lstm_data['y_test'],
                    epochs=50, batch_size=32
                )
            
            progress.update(task3, completed=100)
            
            # Make predictions
            task4 = progress.add_task("Generating predictions...", total=100)
            predictions = lstm_model.predict(lstm_data['X_test'])
            
            # Inverse scale predictions
            predictions = preprocessor.inverse_scale(predictions.flatten(), 'close')
            actuals = preprocessor.inverse_scale(lstm_data['y_test'].flatten(), 'close')
            
            progress.update(task4, completed=100)
            
        else:
            # ML models (RF, XGB, Ensemble)
            task3 = progress.add_task(f"Training {model.upper()} model...", total=100)
            
            # Prepare ML data
            df_features = df[feature_cols].copy()
            df_features['target'] = df['close'].shift(-1)
            df_features = df_features.dropna()
            
            X = np.array(df_features[feature_cols].values)
            y = np.array(df_features['target'].values)
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if model == 'rf':
                ml_model = RandomForestModel(
                    name=f"rf_{symbol.replace('.', '_')}",
                    feature_names=feature_cols
                )
            elif model == 'xgb':
                ml_model = XGBoostModel(
                    name=f"xgb_{symbol.replace('.', '_')}",
                    feature_names=feature_cols
                )
            else:  # ensemble
                rf = RandomForestModel(name="rf_base", feature_names=feature_cols)
                xgb_m = XGBoostModel(name="xgb_base", feature_names=feature_cols)
                
                ml_model = EnsembleModel(
                    name=f"ensemble_{symbol.replace('.', '_')}",
                    strategy="weighted_average"
                )
                ml_model.add_model(rf, weight=0.4)
                ml_model.add_model(xgb_m, weight=0.6)
            
            ml_model.build()
            
            if train:
                ml_model.train(X_train, y_train, X_test, y_test)
            
            progress.update(task3, completed=100)
            
            # Make predictions
            task4 = progress.add_task("Generating predictions...", total=100)
            predictions = ml_model.predict(X_test)
            actuals = np.array(y_test)
            progress.update(task4, completed=100)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression(np.array(actuals), np.array(predictions), model)
    
    # Display metrics
    metrics_table = Table(title="📊 Model Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_display = [
        ("RMSE", f"₹{metrics['rmse']:.2f}"),
        ("MAE", f"₹{metrics['mae']:.2f}"),
        ("MAPE", f"{metrics['mape']:.2f}%"),
        ("R² Score", f"{metrics['r2']:.4f}"),
        ("Directional Accuracy", f"{metrics.get('directional_accuracy', 0):.2f}%"),
    ]
    
    for metric, value in metrics_display:
        metrics_table.add_row(metric, value)
    
    console.print(metrics_table)
    
    # Display predictions
    pred_table = Table(title=f"🔮 Price Predictions (Last 10 Days)", box=box.ROUNDED)
    pred_table.add_column("Day", style="cyan")
    pred_table.add_column("Actual", style="white")
    pred_table.add_column("Predicted", style="yellow")
    pred_table.add_column("Error", style="red")
    
    for i in range(min(10, len(predictions))):
        actual = actuals[-(10-i)] if len(actuals) > 10 else actuals[i]
        pred = predictions[-(10-i)] if len(predictions) > 10 else predictions[i]
        error = ((pred - actual) / actual) * 100
        
        pred_table.add_row(
            f"Day {i+1}",
            f"₹{actual:,.2f}",
            f"₹{pred:,.2f}",
            f"{error:+.2f}%"
        )
    
    console.print(pred_table)
    
    # Future prediction summary
    last_pred = predictions[-1]
    current_price = df['close'].iloc[-1]
    expected_change = ((last_pred - current_price) / current_price) * 100
    
    trend = "📈 BULLISH" if expected_change > 0 else "📉 BEARISH"
    trend_color = "green" if expected_change > 0 else "red"
    
    console.print(Panel(
        f"Current Price: ₹{current_price:,.2f}\n"
        f"Predicted Price: ₹{last_pred:,.2f}\n"
        f"Expected Change: {expected_change:+.2f}%\n\n"
        f"[bold {trend_color}]Trend: {trend}[/bold {trend_color}]",
        title="🎯 Prediction Summary",
        border_style=trend_color
    ))


@cli.command()
@click.option('--stocks', '-s', multiple=True, default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'], help='Stock symbols')
@click.option('--investment', '-i', default=100000, help='Total investment amount')
def portfolio(stocks, investment):
    """Generate portfolio recommendations."""
    console.print(f"\n[bold cyan]💼 Generating portfolio recommendations...[/bold cyan]")
    console.print(f"Investment Amount: ₹{investment:,.2f}")
    console.print(f"Stocks: {', '.join(stocks)}")
    
    collector = DataCollector()
    indicators = TechnicalIndicators()
    
    portfolio_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for symbol in stocks:
            task = progress.add_task(f"Analyzing {symbol}...", total=None)
            
            try:
                df = collector.get_stock_data(symbol=symbol, period='1y')
                if df.empty:
                    continue
                
                df = indicators.add_all_indicators(df)
                df = indicators.get_signals(df)
                
                latest = df.iloc[-1]
                info = collector.get_stock_info(symbol)
                
                # Calculate metrics
                returns_1y = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
                sharpe = returns_1y / volatility if volatility > 0 else 0
                
                portfolio_data.append({
                    'symbol': symbol,
                    'name': info.get('name', symbol),
                    'price': latest['close'],
                    'recommendation': latest.get('recommendation', 'HOLD'),
                    'signal': latest.get('combined_signal', 0),
                    'returns_1y': returns_1y,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'pe_ratio': info.get('pe_ratio', 0),
                    'sector': info.get('sector', 'N/A')
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
            
            progress.update(task, completed=True)
    
    if not portfolio_data:
        console.print("[red]❌ Could not analyze any stocks[/red]")
        return
    
    # Sort by signal strength and sharpe ratio
    portfolio_data.sort(key=lambda x: (x['signal'], x['sharpe']), reverse=True)
    
    # Calculate allocation (weighted by signal strength)
    total_signal = sum(max(0.1, d['signal'] + 1) for d in portfolio_data)
    for d in portfolio_data:
        weight = max(0.1, d['signal'] + 1) / total_signal
        d['allocation'] = weight * investment
        d['shares'] = int(d['allocation'] / d['price'])
        d['actual_investment'] = d['shares'] * d['price']
    
    # Display portfolio
    port_table = Table(title="💼 Recommended Portfolio Allocation", box=box.ROUNDED)
    port_table.add_column("Stock", style="cyan")
    port_table.add_column("Price", style="white")
    port_table.add_column("Signal", style="yellow")
    port_table.add_column("1Y Return", style="green")
    port_table.add_column("Volatility", style="red")
    port_table.add_column("Allocation", style="blue")
    port_table.add_column("Shares", style="magenta")
    
    for d in portfolio_data:
        signal_color = "green" if d['recommendation'] == 'BUY' else ("red" if d['recommendation'] == 'SELL' else "yellow")
        port_table.add_row(
            d['symbol'],
            f"₹{d['price']:,.2f}",
            f"[{signal_color}]{d['recommendation']}[/{signal_color}]",
            f"{d['returns_1y']:+.2f}%",
            f"{d['volatility']:.2f}%",
            f"₹{d['actual_investment']:,.2f}",
            str(d['shares'])
        )
    
    console.print(port_table)
    
    # Portfolio summary
    total_invested = sum(d['actual_investment'] for d in portfolio_data)
    avg_return = sum(d['returns_1y'] * d['actual_investment'] for d in portfolio_data) / total_invested
    avg_volatility = sum(d['volatility'] * d['actual_investment'] for d in portfolio_data) / total_invested
    
    console.print(Panel(
        f"Total Invested: ₹{total_invested:,.2f}\n"
        f"Cash Remaining: ₹{investment - total_invested:,.2f}\n"
        f"Weighted Avg Return (1Y): {avg_return:+.2f}%\n"
        f"Portfolio Volatility: {avg_volatility:.2f}%\n"
        f"Expected Portfolio Value (1Y): ₹{total_invested * (1 + avg_return/100):,.2f}",
        title="📊 Portfolio Summary",
        border_style="cyan"
    ))


@cli.command()
def stocks():
    """List popular Indian stocks."""
    console.print("\n[bold cyan]📋 Popular Indian Stocks (NSE)[/bold cyan]\n")
    
    stocks_table = Table(box=box.ROUNDED)
    stocks_table.add_column("Symbol", style="cyan")
    stocks_table.add_column("Company", style="white")
    stocks_table.add_column("Sector", style="yellow")
    
    popular = [
        ("RELIANCE.NS", "Reliance Industries", "Energy"),
        ("TCS.NS", "Tata Consultancy Services", "IT"),
        ("HDFCBANK.NS", "HDFC Bank", "Banking"),
        ("INFY.NS", "Infosys", "IT"),
        ("ICICIBANK.NS", "ICICI Bank", "Banking"),
        ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
        ("SBIN.NS", "State Bank of India", "Banking"),
        ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"),
        ("ITC.NS", "ITC Limited", "FMCG"),
        ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking"),
        ("LT.NS", "Larsen & Toubro", "Infrastructure"),
        ("AXISBANK.NS", "Axis Bank", "Banking"),
        ("ASIANPAINT.NS", "Asian Paints", "Paints"),
        ("MARUTI.NS", "Maruti Suzuki", "Auto"),
        ("TITAN.NS", "Titan Company", "Consumer"),
        ("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma"),
        ("BAJFINANCE.NS", "Bajaj Finance", "Finance"),
        ("WIPRO.NS", "Wipro", "IT"),
        ("HCLTECH.NS", "HCL Technologies", "IT"),
        ("ULTRACEMCO.NS", "UltraTech Cement", "Cement"),
    ]
    
    for symbol, name, sector in popular:
        stocks_table.add_row(symbol, name, sector)
    
    console.print(stocks_table)
    
    console.print("\n[dim]Use these symbols with other commands, e.g.:[/dim]")
    console.print("[green]  wealth fetch -s RELIANCE.NS[/green]")
    console.print("[green]  wealth analyze -s TCS.NS[/green]")
    console.print("[green]  wealth predict -s HDFCBANK.NS -m lstm[/green]")


@cli.command()
@click.option('--symbol', '-s', default='RELIANCE.NS', help='Stock symbol')
@click.option('--initial', '-i', default=100000, help='Initial investment')
@click.option('--period', '-p', default='1y', help='Backtest period')
def backtest(symbol, initial, period):
    """Backtest trading strategies on historical data."""
    console.print(f"\n[bold cyan]📈 Backtesting strategy on {symbol}...[/bold cyan]")
    
    collector = DataCollector()
    indicators = TechnicalIndicators()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)
        
        df = collector.get_stock_data(symbol=symbol, period=period)
        if df.empty:
            console.print(f"[red]❌ No data found for {symbol}[/red]")
            return
        
        df = indicators.add_all_indicators(df)
        df = indicators.get_signals(df)
        
        progress.update(task, completed=True)
    
    # Simple backtest based on signals
    cash = initial
    shares = 0
    trades = []
    portfolio_values = []
    
    for i, row in df.iterrows():
        # Record portfolio value
        portfolio_value = cash + (shares * row['close'])
        portfolio_values.append(portfolio_value)
        
        recommendation = row.get('recommendation', 'HOLD')
        
        if recommendation == 'BUY' and cash > row['close']:
            # Buy as many shares as possible
            buy_shares = int(cash * 0.95 / row['close'])  # Keep 5% cash
            if buy_shares > 0:
                cost = buy_shares * row['close']
                cash -= cost
                shares += buy_shares
                trades.append(('BUY', row['date'], row['close'], buy_shares))
        
        elif recommendation == 'SELL' and shares > 0:
            # Sell all shares
            revenue = shares * row['close']
            cash += revenue
            trades.append(('SELL', row['date'], row['close'], shares))
            shares = 0
    
    # Final portfolio value
    final_value = cash + (shares * df['close'].iloc[-1])
    total_return = ((final_value - initial) / initial) * 100
    
    # Buy and hold comparison
    buy_hold_shares = int(initial / df['close'].iloc[0])
    buy_hold_value = buy_hold_shares * df['close'].iloc[-1]
    buy_hold_return = ((buy_hold_value - initial) / initial) * 100
    
    # Display results
    results_table = Table(title="📊 Backtest Results", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Strategy", style="green")
    results_table.add_column("Buy & Hold", style="yellow")
    
    results_table.add_row("Initial Investment", f"₹{initial:,.2f}", f"₹{initial:,.2f}")
    results_table.add_row("Final Value", f"₹{final_value:,.2f}", f"₹{buy_hold_value:,.2f}")
    results_table.add_row("Total Return", f"{total_return:+.2f}%", f"{buy_hold_return:+.2f}%")
    results_table.add_row("Number of Trades", str(len(trades)), "1 (Buy)")
    
    console.print(results_table)
    
    # Display trades
    if trades:
        trades_table = Table(title="📝 Trade History (Last 10)", box=box.ROUNDED)
        trades_table.add_column("Action", style="cyan")
        trades_table.add_column("Date", style="white")
        trades_table.add_column("Price", style="yellow")
        trades_table.add_column("Shares", style="green")
        
        for action, date, price, qty in trades[-10:]:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
            action_style = "green" if action == 'BUY' else "red"
            trades_table.add_row(
                f"[{action_style}]{action}[/{action_style}]",
                date_str,
                f"₹{price:,.2f}",
                str(qty)
            )
        
        console.print(trades_table)
    
    # Performance comparison
    outperform = total_return > buy_hold_return
    perf_color = "green" if outperform else "red"
    perf_text = "OUTPERFORMED" if outperform else "UNDERPERFORMED"
    
    console.print(Panel(
        f"Strategy {perf_text} Buy & Hold by {abs(total_return - buy_hold_return):.2f}%",
        title="📈 Performance Summary",
        border_style=perf_color
    ))


@cli.command()
@click.option('-a', '--amount', required=True, type=float,
              help='Investment amount in INR (e.g., 100000)')
@click.option('-r', '--return-rate', required=True, type=float,
              help='Expected annual return percentage (e.g., 15 for 15%)')
@click.option('-y', '--years', required=True, type=int,
              help='Investment time horizon in years (e.g., 5)')
@click.option('-t', '--risk-tolerance', default='moderate',
              type=click.Choice(['conservative', 'moderate', 'aggressive']),
              help='Risk tolerance level')
@click.option('-n', '--num-stocks', default=8, type=int,
              help='Number of stocks to recommend (default: 8)')
def invest(amount: float, return_rate: float, years: int,
           risk_tolerance: str, num_stocks: int):
    """
    🎯 AI-powered personalized investment advisor.
    
    Analyzes your investment goals and recommends an optimal stock portfolio
    using Machine Learning and Monte Carlo simulations.
    
    Features:
    - Risk assessment (VaR, Volatility, Sharpe Ratio, Max Drawdown)
    - ML-based return prediction
    - Portfolio optimization using Modern Portfolio Theory
    - Sector diversification
    - Personalized recommendations
    
    Examples:
    
        wealth invest -a 100000 -r 15 -y 5
        
        wealth invest -a 500000 -r 20 -y 3 -t aggressive
        
        wealth invest -a 1000000 -r 12 -y 10 -t conservative -n 10
    """
    print_banner()
    
    # Validate inputs
    if amount <= 0:
        console.print("[red]❌ Investment amount must be positive[/red]")
        return
    
    if return_rate <= 0 or return_rate > 100:
        console.print("[red]❌ Return rate must be between 0 and 100%[/red]")
        return
    
    if years <= 0 or years > 30:
        console.print("[red]❌ Investment years must be between 1 and 30[/red]")
        return
    
    # Display user profile
    console.print(Panel(
        f"[cyan]💰 Investment Amount:[/cyan] ₹{amount:,.2f}\n"
        f"[cyan]📈 Target Return:[/cyan] {return_rate}% per year\n"
        f"[cyan]⏳ Time Horizon:[/cyan] {years} years\n"
        f"[cyan]🎚️ Risk Tolerance:[/cyan] {risk_tolerance.capitalize()}\n"
        f"[cyan]🎯 Target Value:[/cyan] ₹{amount * ((1 + return_rate/100) ** years):,.2f}",
        title="📋 Your Investment Profile",
        border_style="blue"
    ))
    
    # Create user profile
    user_profile = UserProfile(
        investment_amount=amount,
        expected_annual_return=return_rate,
        investment_years=years,
        risk_tolerance=risk_tolerance
    )
    
    # Initialize advisor
    advisor = InvestmentAdvisorModel()
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing stocks...", total=100)
        
        def update_progress(message: str, percent: float):
            progress.update(task, completed=percent, description=message)
        
        try:
            recommendation = advisor.get_personalized_recommendations(
                user_profile=user_profile,
                top_n=num_stocks,
                progress_callback=update_progress
            )
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            logger.error(f"Investment advisor error: {e}")
            return
    
    # Display Portfolio Summary
    console.print("\n")
    summary_table = Table(title="📊 Portfolio Summary", box=box.DOUBLE_EDGE)
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="green", justify="right")
    
    summary_table.add_row(
        "Expected Annual Return",
        f"{recommendation.total_expected_return:.2f}%"
    )
    summary_table.add_row(
        "Portfolio Risk Score",
        f"{recommendation.portfolio_risk_score:.1f}/100 ({recommendation.portfolio_risk_level.value})"
    )
    summary_table.add_row(
        "Expected Final Value",
        f"₹{recommendation.expected_final_value:,.2f}"
    )
    summary_table.add_row(
        "Best Case (95th %ile)",
        f"₹{recommendation.best_case_value:,.2f}"
    )
    summary_table.add_row(
        "Worst Case (5th %ile)",
        f"₹{recommendation.worst_case_value:,.2f}"
    )
    summary_table.add_row(
        "Probability of Profit",
        f"{recommendation.probability_of_target:.1f}%"
    )
    summary_table.add_row(
        "Diversification Score",
        f"{recommendation.diversification_score:.0f}/100"
    )
    
    console.print(summary_table)
    
    # Display Stock Recommendations
    console.print("\n")
    stocks_table = Table(title="🎯 Recommended Stock Portfolio", box=box.ROUNDED)
    stocks_table.add_column("#", style="dim", width=3)
    stocks_table.add_column("Stock", style="cyan", width=25)
    stocks_table.add_column("Sector", style="yellow", width=12)
    stocks_table.add_column("Allocation", style="green", justify="right")
    stocks_table.add_column("Amount (₹)", style="green", justify="right")
    stocks_table.add_column("Expected Return", style="magenta", justify="right")
    stocks_table.add_column("Risk", justify="center")
    stocks_table.add_column("Score", style="blue", justify="right")
    
    for i, stock in enumerate(recommendation.stocks, 1):
        allocation_amount = amount * stock.suggested_allocation / 100
        
        # Risk color coding
        risk_colors = {
            'Very Low': 'green',
            'Low': 'cyan',
            'Moderate': 'yellow',
            'High': 'red',
            'Very High': 'bright_red'
        }
        risk_color = risk_colors.get(stock.risk_level.value, 'white')
        
        stocks_table.add_row(
            str(i),
            f"{stock.company_name}\n[dim]{stock.symbol}[/dim]",
            stock.sector,
            f"{stock.suggested_allocation:.1f}%",
            f"₹{allocation_amount:,.0f}",
            f"{stock.predicted_return:+.1f}%",
            f"[{risk_color}]{stock.risk_level.value}[/{risk_color}]",
            f"{stock.recommendation_score:.0f}"
        )
    
    console.print(stocks_table)
    
    # Display Detailed Risk Analysis
    console.print("\n")
    risk_table = Table(title="📉 Risk Analysis", box=box.ROUNDED)
    risk_table.add_column("Stock", style="cyan", width=20)
    risk_table.add_column("Volatility", justify="right")
    risk_table.add_column("VaR (95%)", justify="right")
    risk_table.add_column("Max Drawdown", justify="right")
    risk_table.add_column("Sharpe Ratio", justify="right")
    risk_table.add_column("Beta", justify="right")
    
    for stock in recommendation.stocks:
        # Color code based on risk
        vol_color = "green" if stock.volatility < 25 else "yellow" if stock.volatility < 40 else "red"
        sharpe_color = "green" if stock.sharpe_ratio > 1 else "yellow" if stock.sharpe_ratio > 0.5 else "red"
        
        risk_table.add_row(
            stock.company_name[:18],
            f"[{vol_color}]{stock.volatility:.1f}%[/{vol_color}]",
            f"{stock.var_95:.1f}%",
            f"{stock.max_drawdown:.1f}%",
            f"[{sharpe_color}]{stock.sharpe_ratio:.2f}[/{sharpe_color}]",
            f"{stock.beta:.2f}"
        )
    
    console.print(risk_table)
    
    # Display Expected Returns by Stock
    console.print("\n")
    returns_table = Table(title="💰 Expected Value Projection", box=box.ROUNDED)
    returns_table.add_column("Stock", style="cyan", width=20)
    returns_table.add_column("Investment", justify="right")
    returns_table.add_column("Expected Value", justify="right", style="green")
    returns_table.add_column("Best Case", justify="right", style="cyan")
    returns_table.add_column("Worst Case", justify="right", style="red")
    
    total_invested = 0
    total_expected = 0
    total_best = 0
    total_worst = 0
    
    for stock in recommendation.stocks:
        investment = amount * stock.suggested_allocation / 100
        total_invested += investment
        total_expected += stock.expected_value
        total_best += stock.confidence_interval[1]
        total_worst += stock.confidence_interval[0]
        
        returns_table.add_row(
            stock.company_name[:18],
            f"₹{investment:,.0f}",
            f"₹{stock.expected_value:,.0f}",
            f"₹{stock.confidence_interval[1]:,.0f}",
            f"₹{stock.confidence_interval[0]:,.0f}"
        )
    
    returns_table.add_row(
        "[bold]TOTAL PORTFOLIO[/bold]",
        f"[bold]₹{total_invested:,.0f}[/bold]",
        f"[bold green]₹{total_expected:,.0f}[/bold green]",
        f"[bold cyan]₹{total_best:,.0f}[/bold cyan]",
        f"[bold red]₹{total_worst:,.0f}[/bold red]"
    )
    
    console.print(returns_table)
    
    # Display Personalized Recommendations
    console.print("\n")
    rec_panel_content = "\n".join(recommendation.recommendations)
    console.print(Panel(
        rec_panel_content,
        title="💡 Personalized Recommendations",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Display Investment Summary
    target_value = amount * ((1 + return_rate/100) ** years)
    expected_return_pct = ((recommendation.expected_final_value / amount) - 1) * 100
    
    summary_color = "green" if recommendation.expected_final_value >= target_value else "yellow"
    
    final_summary = f"""
[bold cyan]Investment Summary[/bold cyan]

📊 [bold]Your Investment:[/bold] ₹{amount:,.2f}
🎯 [bold]Your Target ({return_rate}% p.a. for {years} years):[/bold] ₹{target_value:,.2f}

💰 [bold]Expected Portfolio Value:[/bold] [{summary_color}]₹{recommendation.expected_final_value:,.2f}[/{summary_color}]
📈 [bold]Expected Total Return:[/bold] [{summary_color}]{expected_return_pct:.1f}%[/{summary_color}]
📉 [bold]Portfolio Risk Level:[/bold] {recommendation.portfolio_risk_level.value}

📊 [bold]Value Range (95% Confidence):[/bold]
   • Best Case: [green]₹{recommendation.best_case_value:,.2f}[/green]
   • Worst Case: [red]₹{recommendation.worst_case_value:,.2f}[/red]
"""
    
    console.print(Panel(
        final_summary,
        title="📋 Final Investment Summary",
        border_style=summary_color,
        padding=(1, 2)
    ))
    
    # Disclaimer
    console.print("\n[dim]⚠️ Disclaimer: This is AI-generated advice for educational purposes only. "
                  "Past performance does not guarantee future results. Please consult a certified "
                  "financial advisor before making investment decisions.[/dim]")


@cli.command()
def advisor():
    """
    🎯 Interactive AI Investment Advisor - Enter your details step by step.
    
    A professional AI-powered wealth management system that:
    - Takes your investment amount, target return, and time horizon
    - Lets you select risk tolerance (Very Safe to Risky)
    - Validates if your expectations are realistic
    - Provides personalized stock recommendations based on real-world data
    - Uses advanced ML/DL for high-accuracy predictions
    
    Simply run: wealth advisor
    """
    print_banner()
    
    console.print(Panel(
        "[bold cyan]Welcome to AI Investment Advisor![/bold cyan]\n\n"
        "I will help you create a personalized investment portfolio based on:\n"
        "• Your investment amount\n"
        "• Your target annual return\n"
        "• Your investment time horizon\n"
        "• Your risk tolerance\n\n"
        "[dim]Let's get started...[/dim]",
        title="🤖 AI Wealth Manager",
        border_style="blue"
    ))
    console.print()
    
    # ============ STEP 1: Investment Amount ============
    console.print("[bold cyan]━━━ Step 1: Investment Amount ━━━[/bold cyan]")
    while True:
        try:
            amount_input = console.input("[yellow]💰 Enter your investment amount (in ₹): [/yellow]")
            amount = float(amount_input.replace(',', '').replace('₹', '').strip())
            
            if amount <= 0:
                console.print("[red]❌ Amount must be positive. Please try again.[/red]")
                continue
            
            if amount < 10000:
                console.print("[yellow]⚠️ Minimum recommended investment is ₹10,000 for diversification.[/yellow]")
                confirm = console.input("[yellow]Continue anyway? (y/n): [/yellow]")
                if confirm.lower() != 'y':
                    continue
            
            if amount > 100000000:  # 10 Crore
                console.print("[yellow]⚠️ For investments above ₹10 Crore, please consult a professional wealth manager.[/yellow]")
            
            console.print(f"[green]✓ Investment Amount: ₹{amount:,.2f}[/green]\n")
            break
        except ValueError:
            console.print("[red]❌ Invalid amount. Please enter a number (e.g., 100000 or 1,00,000)[/red]")
    
    # ============ STEP 2: Target Return ============
    console.print("[bold cyan]━━━ Step 2: Expected Annual Return ━━━[/bold cyan]")
    console.print("[dim]Realistic return expectations based on Indian stock market:[/dim]")
    console.print("[dim]  • Conservative: 8-12% per year[/dim]")
    console.print("[dim]  • Moderate: 12-18% per year[/dim]")
    console.print("[dim]  • Aggressive: 18-25% per year[/dim]")
    console.print("[dim]  • Very Aggressive: 25%+ (High Risk)[/dim]\n")
    
    while True:
        try:
            return_input = console.input("[yellow]📈 Enter your expected annual return (%): [/yellow]")
            target_return = float(return_input.replace('%', '').strip())
            
            if target_return <= 0:
                console.print("[red]❌ Return must be positive. Please try again.[/red]")
                continue
            
            # Realistic return validation
            unrealistic = False
            warning_msg = ""
            
            if target_return > 50:
                unrealistic = True
                warning_msg = (
                    f"[bold red]⚠️ UNREALISTIC EXPECTATION![/bold red]\n\n"
                    f"[red]Your target of {target_return}% annual return is NOT achievable "
                    f"in the real world through legitimate stock market investing.[/red]\n\n"
                    f"[yellow]Real-world facts:[/yellow]\n"
                    f"• Warren Buffett's average: ~20% per year\n"
                    f"• NIFTY 50 historical average: ~12-15% per year\n"
                    f"• Best performing mutual funds: ~25-30% in exceptional years\n"
                    f"• Any promise of 50%+ returns is likely a SCAM!\n\n"
                    f"[cyan]Please enter a realistic target (recommended: 12-25%)[/cyan]"
                )
            elif target_return > 35:
                warning_msg = (
                    f"[bold yellow]⚠️ VERY HIGH EXPECTATION![/bold yellow]\n\n"
                    f"[yellow]{target_return}% annual return is extremely difficult to achieve consistently.[/yellow]\n"
                    f"Only the top 1% of investors achieve this, with HIGH RISK of losses.\n"
                    f"[dim]Proceed with caution.[/dim]"
                )
            elif target_return > 25:
                warning_msg = (
                    f"[yellow]⚠️ High target ({target_return}%). This requires aggressive investing "
                    f"with higher risk tolerance.[/yellow]"
                )
            
            if unrealistic:
                console.print(Panel(warning_msg, border_style="red"))
                continue
            elif warning_msg:
                console.print(Panel(warning_msg, border_style="yellow"))
                confirm = console.input("[yellow]Do you want to continue with this target? (y/n): [/yellow]")
                if confirm.lower() != 'y':
                    continue
            
            console.print(f"[green]✓ Target Annual Return: {target_return}%[/green]\n")
            break
        except ValueError:
            console.print("[red]❌ Invalid return. Please enter a number (e.g., 15 or 15%)[/red]")
    
    # ============ STEP 3: Time Horizon ============
    console.print("[bold cyan]━━━ Step 3: Investment Time Horizon ━━━[/bold cyan]")
    console.print("[dim]Recommended time horizons:[/dim]")
    console.print("[dim]  • Short-term: 1-3 years (Higher risk)[/dim]")
    console.print("[dim]  • Medium-term: 3-5 years (Moderate risk)[/dim]")
    console.print("[dim]  • Long-term: 5-10 years (Lower risk, better compounding)[/dim]")
    console.print("[dim]  • Very Long-term: 10+ years (Best for wealth creation)[/dim]\n")
    
    while True:
        try:
            years_input = console.input("[yellow]⏳ Enter investment duration (in years): [/yellow]")
            years = int(years_input.strip())
            
            if years <= 0:
                console.print("[red]❌ Duration must be at least 1 year.[/red]")
                continue
            
            if years > 30:
                console.print("[yellow]⚠️ Maximum planning horizon is 30 years.[/yellow]")
                years = 30
            
            if years < 3 and target_return > 15:
                console.print(Panel(
                    f"[yellow]⚠️ Warning: Expecting {target_return}% return in just {years} year(s) "
                    f"is very risky.\nShort-term market volatility can lead to significant losses.[/yellow]",
                    border_style="yellow"
                ))
            
            console.print(f"[green]✓ Investment Duration: {years} years[/green]\n")
            break
        except ValueError:
            console.print("[red]❌ Invalid duration. Please enter a whole number (e.g., 5)[/red]")
    
    # ============ STEP 4: Risk Tolerance ============
    console.print("[bold cyan]━━━ Step 4: Select Your Risk Tolerance ━━━[/bold cyan]\n")
    
    # Risk categories with realistic YOY returns based on Indian stock market
    # | Category                           | Risk Level  | Typical Stocks                      | Realistic YOY Return     |
    # | 1️⃣ Very Safe                       | Very Low    | Large-cap leaders, monopolies       | 8% – 10%                 |
    # | 2️⃣ Safe                            | Low         | Strong large-caps + stable mid-caps | 10% – 13%                |
    # | 3️⃣ Moderate                        | Medium      | Quality mid-caps, sector leaders    | 13% – 18%                |
    # | 4️⃣ Low Risk (Growth-Oriented)      | Medium–High | High-growth mid/small caps          | 18% – 25%                |
    # | 5️⃣ Risky                           | High        | Small-caps, cyclicals, turnarounds  | 25% – 40%+ (unstable)    |
    
    risk_options = [
        ("1", "very_safe", "🛡️  Very Safe", "Large-cap leaders, monopolies. YOY: 8-10%", "green"),
        ("2", "safe", "🔒 Safe", "Strong large-caps + stable mid-caps. YOY: 10-13%", "cyan"),
        ("3", "moderate", "⚖️  Moderate", "Quality mid-caps, sector leaders. YOY: 13-18%", "yellow"),
        ("4", "low_risk_growth", "📊 Low Risk (Growth)", "High-growth mid/small caps. YOY: 18-25%", "orange1"),
        ("5", "risky", "🔥 Risky", "Small-caps, cyclicals, turnarounds. YOY: 25-40%+ (unstable)", "red"),
    ]
    
    risk_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    risk_table.add_column("Option", style="bold", width=8)
    risk_table.add_column("Risk Level", width=22)
    risk_table.add_column("Description", width=55)
    
    for opt, key, name, desc, color in risk_options:
        risk_table.add_row(f"[{color}]{opt}[/{color}]", f"[{color}]{name}[/{color}]", desc)
    
    console.print(risk_table)
    console.print()
    
    while True:
        risk_input = console.input("[yellow]🎚️ Select risk tolerance (1-5): [/yellow]").strip()
        
        risk_map = {
            '1': 'very_safe',
            '2': 'safe', 
            '3': 'moderate',
            '4': 'low_risk_growth',
            '5': 'risky'
        }
        
        if risk_input in risk_map:
            risk_tolerance = risk_map[risk_input]
            risk_name = [r[2] for r in risk_options if r[1] == risk_tolerance][0]
            console.print(f"[green]✓ Risk Tolerance: {risk_name}[/green]\n")
            break
        else:
            console.print("[red]❌ Please enter a number between 1 and 5[/red]")
    
    # ============ Validate Return vs Risk Tolerance ============
    # Realistic YOY return limits per risk category
    risk_return_limits = {
        'very_safe': (8, 10),         # Large-cap leaders, monopolies
        'safe': (10, 13),             # Strong large-caps + stable mid-caps
        'moderate': (13, 18),         # Quality mid-caps, sector leaders
        'low_risk_growth': (18, 25),  # High-growth mid/small caps
        'risky': (25, 40)             # Small-caps, cyclicals, turnarounds
    }
    
    min_return, max_return = risk_return_limits[risk_tolerance]
    
    if target_return > max_return:
        console.print(Panel(
            f"[bold red]⚠️ MISMATCH DETECTED![/bold red]\n\n"
            f"Your target return of [bold]{target_return}%[/bold] is too high for "
            f"[bold]{risk_name}[/bold] risk profile.\n\n"
            f"[yellow]Realistic expectation for {risk_name}:[/yellow] {min_return}% - {max_return}%\n\n"
            f"[cyan]Options:[/cyan]\n"
            f"1. Lower your return expectation to {max_return}% or less\n"
            f"2. Increase your risk tolerance\n\n"
            f"[dim]Adjusting target to maximum realistic: {max_return}%[/dim]",
            title="⚠️ Reality Check",
            border_style="red"
        ))
        target_return = max_return
        console.print(f"[yellow]→ Adjusted Target Return: {target_return}%[/yellow]\n")
    
    # ============ Summary Before Processing ============
    target_value = amount * ((1 + target_return/100) ** years)
    
    console.print(Panel(
        f"[bold cyan]📋 Investment Summary[/bold cyan]\n\n"
        f"💰 [bold]Investment Amount:[/bold] ₹{amount:,.2f}\n"
        f"📈 [bold]Target Return:[/bold] {target_return}% per year\n"
        f"⏳ [bold]Time Horizon:[/bold] {years} years\n"
        f"🎚️ [bold]Risk Tolerance:[/bold] {risk_name}\n\n"
        f"🎯 [bold]Target Value in {years} years:[/bold] ₹{target_value:,.2f}",
        title="📊 Your Investment Profile",
        border_style="blue"
    ))
    
    confirm = console.input("\n[yellow]Proceed with analysis? (y/n): [/yellow]")
    if confirm.lower() != 'y':
        console.print("[dim]Analysis cancelled.[/dim]")
        return
    
    console.print()
    
    # ============ Run the AI Advisor ============
    # Use the risk tolerance directly - no mapping needed
    # The investment_advisor now supports the new risk categories directly:
    # very_safe, safe, moderate, low_risk_growth, risky
    
    # Create user profile with user inputs:
    # - investment_amount: User input (Step 1)
    # - expected_annual_return: User input YOY return % (Step 2)
    # - investment_years: User input investment span in years (Step 3)
    # - risk_tolerance: User input risk category (Step 4)
    user_profile = UserProfile(
        investment_amount=amount,
        expected_annual_return=target_return,
        investment_years=years,
        risk_tolerance=risk_tolerance  # Use directly - new categories supported
    )
    
    # Determine number of stocks based on amount and risk
    if amount < 50000:
        num_stocks = 4
    elif amount < 200000:
        num_stocks = 6
    elif amount < 500000:
        num_stocks = 8
    else:
        num_stocks = 10
    
    # Initialize advisor
    advisor_model = InvestmentAdvisorModel()
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("🔍 Analyzing market data...", total=100)
        
        def update_progress(message: str, percent: float):
            progress.update(task, completed=percent, description=message)
        
        try:
            recommendation = advisor_model.get_personalized_recommendations(
                user_profile=user_profile,
                top_n=num_stocks,
                progress_callback=update_progress
            )
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            logger.error(f"Investment advisor error: {e}")
            return
    
    # ============ Display Results ============
    console.print("\n")
    
    # Check if expected return meets target
    meets_target = recommendation.total_expected_return >= target_return
    
    # Portfolio Summary
    summary_table = Table(title="📊 AI Portfolio Analysis Results", box=box.DOUBLE_EDGE)
    summary_table.add_column("Metric", style="cyan", width=35)
    summary_table.add_column("Value", style="green", justify="right")
    
    return_color = "green" if meets_target else "yellow"
    summary_table.add_row(
        "Your Target Return",
        f"{target_return:.1f}%"
    )
    summary_table.add_row(
        "AI Predicted Portfolio Return",
        f"[{return_color}]{recommendation.total_expected_return:.2f}%[/{return_color}]"
    )
    summary_table.add_row(
        "Portfolio Risk Score",
        f"{recommendation.portfolio_risk_score:.1f}/100 ({recommendation.portfolio_risk_level.value})"
    )
    summary_table.add_row(
        "Your Investment",
        f"₹{amount:,.2f}"
    )
    summary_table.add_row(
        "Expected Value ({} years)".format(years),
        f"₹{recommendation.expected_final_value:,.2f}"
    )
    summary_table.add_row(
        "Best Case (95th percentile)",
        f"[cyan]₹{recommendation.best_case_value:,.2f}[/cyan]"
    )
    summary_table.add_row(
        "Worst Case (5th percentile)", 
        f"[red]₹{recommendation.worst_case_value:,.2f}[/red]"
    )
    summary_table.add_row(
        "Probability of Profit",
        f"{recommendation.probability_of_target:.1f}%"
    )
    summary_table.add_row(
        "Diversification Score",
        f"{recommendation.diversification_score:.0f}/100"
    )
    
    console.print(summary_table)
    
    # Target Achievement Status
    if meets_target:
        console.print(Panel(
            f"[bold green]✅ ACHIEVABLE![/bold green]\n\n"
            f"Your target of {target_return}% annual return is achievable based on our AI analysis.\n"
            f"Expected portfolio return: [bold]{recommendation.total_expected_return:.1f}%[/bold]",
            title="🎯 Target Status",
            border_style="green"
        ))
    else:
        gap = target_return - recommendation.total_expected_return
        console.print(Panel(
            f"[bold yellow]⚠️ CHALLENGING TARGET[/bold yellow]\n\n"
            f"Your target of {target_return}% may be difficult with {risk_name} risk profile.\n"
            f"AI predicted return: [bold]{recommendation.total_expected_return:.1f}%[/bold] "
            f"(Gap: {gap:.1f}%)\n\n"
            f"[cyan]Suggestions:[/cyan]\n"
            f"• Increase risk tolerance for higher potential returns\n"
            f"• Extend time horizon for better compounding\n"
            f"• Consider the recommended portfolio as realistic expectation",
            title="🎯 Target Status", 
            border_style="yellow"
        ))
    
    # Recommended Stocks Table
    console.print("\n")
    stocks_table = Table(title="🎯 Recommended Stock Portfolio (Real-World Picks)", box=box.ROUNDED)
    stocks_table.add_column("#", style="dim", width=3)
    stocks_table.add_column("Stock", style="cyan", width=25)
    stocks_table.add_column("Sector", style="yellow", width=14)
    stocks_table.add_column("Allocation", style="green", justify="right")
    stocks_table.add_column("Invest (₹)", style="green", justify="right")
    stocks_table.add_column("Exp. Return", style="magenta", justify="right")
    stocks_table.add_column("Risk", justify="center", width=12)
    stocks_table.add_column("AI Score", style="blue", justify="right")
    
    for i, stock in enumerate(recommendation.stocks, 1):
        allocation_amount = amount * stock.suggested_allocation / 100
        
        risk_colors = {
            'Very Low': 'bright_green',
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'orange1',
            'Very High': 'red'
        }
        risk_color = risk_colors.get(stock.risk_level.value, 'white')
        
        stocks_table.add_row(
            str(i),
            f"{stock.company_name}\n[dim]{stock.symbol}[/dim]",
            stock.sector,
            f"{stock.suggested_allocation:.1f}%",
            f"₹{allocation_amount:,.0f}",
            f"{stock.predicted_return:+.1f}%",
            f"[{risk_color}]{stock.risk_level.value}[/{risk_color}]",
            f"{stock.recommendation_score:.0f}/100"
        )
    
    console.print(stocks_table)
    
    # Risk Analysis Table
    console.print("\n")
    risk_table = Table(title="📉 Detailed Risk Analysis", box=box.ROUNDED)
    risk_table.add_column("Stock", style="cyan", width=20)
    risk_table.add_column("Volatility", justify="right")
    risk_table.add_column("VaR (95%)", justify="right")
    risk_table.add_column("Max Drawdown", justify="right")
    risk_table.add_column("Sharpe Ratio", justify="right")
    risk_table.add_column("Beta", justify="right")
    
    for stock in recommendation.stocks:
        vol_color = "green" if stock.volatility < 25 else "yellow" if stock.volatility < 40 else "red"
        sharpe_color = "green" if stock.sharpe_ratio > 1 else "yellow" if stock.sharpe_ratio > 0.5 else "red"
        
        risk_table.add_row(
            stock.company_name[:18],
            f"[{vol_color}]{stock.volatility:.1f}%[/{vol_color}]",
            f"{stock.var_95:.1f}%",
            f"{stock.max_drawdown:.1f}%",
            f"[{sharpe_color}]{stock.sharpe_ratio:.2f}[/{sharpe_color}]",
            f"{stock.beta:.2f}"
        )
    
    console.print(risk_table)
    
    # Expected Value Projection
    console.print("\n")
    projection_table = Table(title=f"💰 {years}-Year Value Projection", box=box.ROUNDED)
    projection_table.add_column("Stock", style="cyan", width=20)
    projection_table.add_column("Investment", justify="right")
    projection_table.add_column("Expected Value", justify="right", style="green")
    projection_table.add_column("Best Case", justify="right", style="cyan")
    projection_table.add_column("Worst Case", justify="right", style="red")
    
    total_invested = 0
    total_expected = 0
    total_best = 0
    total_worst = 0
    
    for stock in recommendation.stocks:
        investment = amount * stock.suggested_allocation / 100
        total_invested += investment
        total_expected += stock.expected_value
        total_best += stock.confidence_interval[1]
        total_worst += stock.confidence_interval[0]
        
        projection_table.add_row(
            stock.company_name[:18],
            f"₹{investment:,.0f}",
            f"₹{stock.expected_value:,.0f}",
            f"₹{stock.confidence_interval[1]:,.0f}",
            f"₹{stock.confidence_interval[0]:,.0f}"
        )
    
    projection_table.add_row(
        "[bold]TOTAL PORTFOLIO[/bold]",
        f"[bold]₹{total_invested:,.0f}[/bold]",
        f"[bold green]₹{total_expected:,.0f}[/bold green]",
        f"[bold cyan]₹{total_best:,.0f}[/bold cyan]",
        f"[bold red]₹{total_worst:,.0f}[/bold red]"
    )
    
    console.print(projection_table)
    
    # Personalized Recommendations
    console.print("\n")
    rec_content = "\n".join(recommendation.recommendations)
    console.print(Panel(
        rec_content,
        title="💡 AI Personalized Recommendations",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Final Summary
    expected_return_pct = ((recommendation.expected_final_value / amount) - 1) * 100
    summary_color = "green" if meets_target else "yellow"
    
    final_summary = f"""
[bold cyan]═══ FINAL INVESTMENT SUMMARY ═══[/bold cyan]

[bold]YOUR INPUT:[/bold]
  💰 Investment Amount: ₹{amount:,.2f}
  📈 Target Return: {target_return}% per year
  ⏳ Time Horizon: {years} years
  🎚️ Risk Profile: {risk_name}
  🎯 Target Value: ₹{target_value:,.2f}

[bold]AI PREDICTION:[/bold]
  📊 Expected Portfolio Return: [{summary_color}]{recommendation.total_expected_return:.1f}% per year[/{summary_color}]
  💰 Expected Final Value: [{summary_color}]₹{recommendation.expected_final_value:,.2f}[/{summary_color}]
  📈 Total Expected Gain: [{summary_color}]{expected_return_pct:.1f}%[/{summary_color}]
  
[bold]RISK ASSESSMENT:[/bold]
  📉 Portfolio Risk Level: {recommendation.portfolio_risk_level.value}
  ✅ Probability of Profit: {recommendation.probability_of_target:.1f}%
  
[bold]VALUE RANGE (95% Confidence):[/bold]
  🟢 Best Case: ₹{recommendation.best_case_value:,.2f}
  🔴 Worst Case: ₹{recommendation.worst_case_value:,.2f}
"""
    
    console.print(Panel(
        final_summary,
        title="📋 Complete Analysis Report",
        border_style=summary_color,
        padding=(1, 2)
    ))
    
    # Disclaimer
    console.print(Panel(
        "[bold red]⚠️ IMPORTANT DISCLAIMER[/bold red]\n\n"
        "• This is AI-generated advice for [bold]educational purposes only[/bold]\n"
        "• Past performance does NOT guarantee future results\n"
        "• Stock market investments are subject to market risks\n"
        "• Please consult a SEBI-registered financial advisor before investing\n"
        "• The predictions are based on historical data and ML models\n"
        "• Actual returns may vary significantly from predictions",
        title="⚠️ Disclaimer",
        border_style="red"
    ))


# Create app alias for direct execution
app = cli


if __name__ == '__main__':
    cli()
