"""
Stock Price Prediction Models
Machine Learning and Statistical models for stock price prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """Main class for stock price prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        df = df.copy()
        
        # Price-based features
        df['Returns'] = df['close'].pct_change()
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Price relative to moving averages
        df['Price_SMA20_Ratio'] = df['close'] / df['SMA_20']
        df['Price_SMA50_Ratio'] = df['close'] / df['SMA_50']
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Momentum
        df['Momentum_5'] = df['close'].diff(5)
        df['Momentum_10'] = df['close'].diff(10)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Day of week
        if isinstance(df.index, pd.DatetimeIndex):
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
        
        # Volume features if available
        if 'volume' in df.columns:
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
        
        # Target variable (next day's return)
        df['Target'] = df['close'].shift(-1)
        df['Target_Returns'] = df['Returns'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target', 'Target_Returns']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train_models(self, df, target_col='Target'):
        """Train multiple prediction models"""
        df = self.prepare_features(df)
        
        if len(df) < 100:
            return {"error": "Insufficient data for training (need at least 100 data points)"}
        
        self.feature_columns = self.get_feature_columns(df)
        
        X = df[self.feature_columns].values
        y = df[target_col].values
        
        # Scale features
        self.scalers['features'] = MinMaxScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        results = {}
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        self.models['Linear Regression'] = lr_model
        results['Linear Regression'] = self._evaluate_model(y_test, lr_pred)
        
        # Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)
        self.models['Ridge Regression'] = ridge_model
        results['Ridge Regression'] = self._evaluate_model(y_test, ridge_pred)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        self.models['Random Forest'] = rf_model
        results['Random Forest'] = self._evaluate_model(y_test, rf_pred)
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        self.models['Gradient Boosting'] = gb_model
        results['Gradient Boosting'] = self._evaluate_model(y_test, gb_pred)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['r2_score'])[0]
        results['best_model'] = best_model
        
        return results
    
    def _evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        return {
            'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            'mae': round(mean_absolute_error(y_true, y_pred), 2),
            'r2_score': round(r2_score(y_true, y_pred), 4),
            'mape': round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
        }
    
    def predict_future(self, df, days=30, model_name=None):
        """Predict future prices"""
        if not self.models:
            self.train_models(df)
        
        if model_name is None:
            model_name = 'Random Forest'
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        df = self.prepare_features(df)
        
        predictions = []
        current_data = df.copy()
        
        for i in range(days):
            # Prepare features for prediction
            X = current_data[self.feature_columns].iloc[-1:].values
            X_scaled = self.scalers['features'].transform(X)
            
            # Make prediction
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update current_data for next prediction (simplified)
            new_row = current_data.iloc[-1:].copy()
            new_row['close'] = pred
            new_row.index = new_row.index + timedelta(days=1)
            
            # Skip weekends
            while new_row.index.dayofweek[0] >= 5:
                new_row.index = new_row.index + timedelta(days=1)
            
            current_data = pd.concat([current_data, new_row])
            current_data = self.prepare_features(current_data.drop(columns=['Target', 'Target_Returns'], errors='ignore'))
        
        # Create prediction dataframe
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
        
        pred_df = pd.DataFrame({
            'Date': future_dates[:len(predictions)],
            'Predicted_Price': predictions
        })
        
        return pred_df
    
    def get_recommendation(self, df, predictions=None):
        """Generate buy/sell/hold recommendation"""
        if len(df) < 20:
            return {'recommendation': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        current_price = df['close'].iloc[-1]
        
        # Technical signals
        signals = []
        
        # RSI
        df_temp = self.prepare_features(df)
        if 'RSI' in df_temp.columns:
            rsi = df_temp['RSI'].iloc[-1]
            if rsi < 30:
                signals.append(('BUY', 'RSI indicates oversold'))
            elif rsi > 70:
                signals.append(('SELL', 'RSI indicates overbought'))
            else:
                signals.append(('HOLD', 'RSI is neutral'))
        
        # MACD
        if 'MACD' in df_temp.columns and 'MACD_Signal' in df_temp.columns:
            if df_temp['MACD'].iloc[-1] > df_temp['MACD_Signal'].iloc[-1]:
                signals.append(('BUY', 'MACD bullish crossover'))
            else:
                signals.append(('SELL', 'MACD bearish crossover'))
        
        # Moving average trend
        if 'SMA_20' in df_temp.columns and 'SMA_50' in df_temp.columns:
            if df_temp['SMA_20'].iloc[-1] > df_temp['SMA_50'].iloc[-1]:
                signals.append(('BUY', 'Short-term MA above long-term'))
            else:
                signals.append(('SELL', 'Short-term MA below long-term'))
        
        # Price momentum
        if len(df) >= 20:
            returns_20d = (current_price / df['close'].iloc[-20] - 1) * 100
            if returns_20d > 5:
                signals.append(('BUY', 'Strong positive momentum'))
            elif returns_20d < -5:
                signals.append(('SELL', 'Strong negative momentum'))
        
        # Prediction-based signal
        if predictions is not None and len(predictions) > 0:
            pred_price = predictions['Predicted_Price'].iloc[-1]
            pred_change = (pred_price / current_price - 1) * 100
            if pred_change > 5:
                signals.append(('BUY', f'Model predicts {pred_change:.1f}% increase'))
            elif pred_change < -5:
                signals.append(('SELL', f'Model predicts {pred_change:.1f}% decrease'))
        
        # Calculate final recommendation
        buy_signals = sum(1 for s in signals if s[0] == 'BUY')
        sell_signals = sum(1 for s in signals if s[0] == 'SELL')
        total_signals = len(signals)
        
        if buy_signals > sell_signals:
            recommendation = 'BUY'
            confidence = buy_signals / total_signals * 100 if total_signals > 0 else 50
        elif sell_signals > buy_signals:
            recommendation = 'SELL'
            confidence = sell_signals / total_signals * 100 if total_signals > 0 else 50
        else:
            recommendation = 'HOLD'
            confidence = 50
        
        reasons = [s[1] for s in signals]
        
        return {
            'recommendation': recommendation,
            'confidence': round(confidence, 1),
            'reasons': reasons,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'current_price': round(current_price, 2)
        }


class TimeSeriesPredictor:
    """Time series based prediction using statistical methods"""
    
    def __init__(self):
        self.model = None
    
    def exponential_smoothing_forecast(self, df, periods=30, alpha=0.3):
        """Simple exponential smoothing forecast"""
        prices = df['close'].values
        
        # Initialize
        forecast = [prices[0]]
        
        # Calculate smoothed values
        for i in range(1, len(prices)):
            forecast.append(alpha * prices[i] + (1 - alpha) * forecast[-1])
        
        # Forecast future periods
        future_forecast = []
        last_smoothed = forecast[-1]
        
        for _ in range(periods):
            future_forecast.append(last_smoothed)
        
        # Create forecast dataframe
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='B')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_forecast
        })
    
    def moving_average_forecast(self, df, periods=30, window=20):
        """Moving average based forecast"""
        prices = df['close'].values
        
        # Calculate moving average trend
        ma = pd.Series(prices).rolling(window=window).mean()
        
        # Trend calculation
        if len(ma.dropna()) >= 2:
            trend = (ma.iloc[-1] - ma.iloc[-window]) / window
        else:
            trend = 0
        
        # Forecast
        future_forecast = []
        last_price = prices[-1]
        
        for i in range(periods):
            next_price = last_price + trend * (i + 1)
            future_forecast.append(next_price)
        
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='B')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_forecast
        })
    
    def linear_regression_forecast(self, df, periods=30):
        """Linear regression based forecast"""
        prices = df['close'].values
        X = np.arange(len(prices)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, prices)
        
        # Forecast
        future_X = np.arange(len(prices), len(prices) + periods).reshape(-1, 1)
        future_forecast = model.predict(future_X)
        
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='B')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_forecast
        })


def ensemble_prediction(df, days=30):
    """Combine multiple prediction models"""
    ml_predictor = StockPredictor()
    ts_predictor = TimeSeriesPredictor()
    
    predictions = {}
    
    # ML-based prediction
    try:
        ml_predictor.train_models(df)
        ml_pred = ml_predictor.predict_future(df, days, 'Random Forest')
        if ml_pred is not None:
            predictions['ML_Random_Forest'] = ml_pred
    except Exception as e:
        print(f"ML prediction error: {e}")
    
    # Time series predictions
    try:
        predictions['Exp_Smoothing'] = ts_predictor.exponential_smoothing_forecast(df, days)
        predictions['Moving_Average'] = ts_predictor.moving_average_forecast(df, days)
        predictions['Linear_Regression'] = ts_predictor.linear_regression_forecast(df, days)
    except Exception as e:
        print(f"Time series prediction error: {e}")
    
    # Ensemble - average of all predictions
    if predictions:
        ensemble_df = pd.DataFrame()
        ensemble_df['Date'] = list(predictions.values())[0]['Date']
        
        pred_values = []
        for name, pred_df in predictions.items():
            pred_values.append(pred_df['Predicted_Price'].values)
        
        ensemble_df['Predicted_Price'] = np.mean(pred_values, axis=0)
        ensemble_df['Prediction_Std'] = np.std(pred_values, axis=0)
        ensemble_df['Upper_Bound'] = ensemble_df['Predicted_Price'] + 1.96 * ensemble_df['Prediction_Std']
        ensemble_df['Lower_Bound'] = ensemble_df['Predicted_Price'] - 1.96 * ensemble_df['Prediction_Std']
        
        predictions['Ensemble'] = ensemble_df
    
    return predictions
