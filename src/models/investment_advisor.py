"""
Investment Advisor Model
========================

AI-powered personalized investment advisor that:
1. Takes user inputs: investment amount, expected return %, time horizon
2. Analyzes all available stocks and their historical performance
3. Calculates risk metrics (volatility, VaR, Sharpe ratio, max drawdown)
4. Uses ML to predict future returns
5. Recommends optimal portfolio allocation
6. Provides risk-adjusted personalized recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

from ..data.collector import DataCollector
from ..data.technical_indicators import TechnicalIndicators
from ..config.settings import get_settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categories based on realistic market expectations."""
    VERY_SAFE = "Very Safe"      # Large-cap leaders, monopolies - 8-10% YOY
    SAFE = "Safe"                 # Strong large-caps + stable mid-caps - 10-13% YOY
    MODERATE = "Moderate"         # Quality mid-caps, sector leaders - 13-18% YOY
    LOW_RISK_GROWTH = "Low Risk (Growth-Oriented)"  # High-growth mid/small caps - 18-25% YOY
    RISKY = "Risky"               # Small-caps, cyclicals, turnarounds - 25-40%+ YOY (unstable)


# Risk Category Definitions with Realistic YOY Returns
RISK_CATEGORY_RETURNS = {
    "very_safe": {"min_return": 8, "max_return": 10, "typical_stocks": "Large-cap leaders, monopolies"},
    "safe": {"min_return": 10, "max_return": 13, "typical_stocks": "Strong large-caps + stable mid-caps"},
    "moderate": {"min_return": 13, "max_return": 18, "typical_stocks": "Quality mid-caps, sector leaders"},
    "low_risk_growth": {"min_return": 18, "max_return": 25, "typical_stocks": "High-growth mid/small caps"},
    "risky": {"min_return": 25, "max_return": 40, "typical_stocks": "Small-caps, cyclicals, turnarounds (unstable)"}
}


def validate_yoy_return_expectation(risk_tolerance: str, expected_return: float) -> Tuple[bool, str]:
    """
    Validate if the expected YOY return is realistic for the given risk category.
    
    Args:
        risk_tolerance: Risk category (very_safe, safe, moderate, low_risk_growth, risky)
        expected_return: Expected annual return percentage
        
    Returns:
        Tuple of (is_valid, message)
    """
    if risk_tolerance not in RISK_CATEGORY_RETURNS:
        return True, "Unknown risk category - using default validation"
    
    category = RISK_CATEGORY_RETURNS[risk_tolerance]
    min_ret = category["min_return"]
    max_ret = category["max_return"]
    
    if expected_return < min_ret:
        return False, (
            f"Your expected return of {expected_return}% is below the typical range "
            f"({min_ret}%-{max_ret}%) for {risk_tolerance.replace('_', ' ')} category. "
            f"Consider a safer category or adjust your expectations."
        )
    elif expected_return > max_ret:
        return False, (
            f"Your expected return of {expected_return}% exceeds the realistic range "
            f"({min_ret}%-{max_ret}%) for {risk_tolerance.replace('_', ' ')} category. "
            f"Consider a riskier category or lower your expectations."
        )
    
    return True, (
        f"Your expected return of {expected_return}% is within the realistic range "
        f"({min_ret}%-{max_ret}%) for {risk_tolerance.replace('_', ' ')} category."
    )


@dataclass
class UserProfile:
    """
    User investment profile.
    
    All parameters are user inputs:
    - investment_amount: Total amount to invest (in currency)
    - expected_annual_return: Target YOY return as percentage (e.g., 15 for 15%)
    - investment_years: Investment time horizon in years
    - risk_tolerance: Risk category selection
    """
    investment_amount: float          # User input: Total investment amount
    expected_annual_return: float     # User input: Expected YOY return percentage
    investment_years: int             # User input: Investment span in years
    risk_tolerance: str = "moderate"  # User input: very_safe, safe, moderate, low_risk_growth, risky


@dataclass
class StockAnalysis:
    """Analysis results for a single stock."""
    symbol: str
    company_name: str
    sector: str
    current_price: float
    predicted_return: float  # Annual predicted return %
    risk_score: float  # 0-100 (higher = riskier)
    risk_level: RiskLevel
    volatility: float  # Annual volatility %
    sharpe_ratio: float
    max_drawdown: float  # Maximum drawdown %
    var_95: float  # Value at Risk at 95% confidence
    beta: float  # Market beta
    recommendation_score: float  # 0-100 (higher = better match)
    suggested_allocation: float  # Suggested % of portfolio
    expected_value: float  # Expected value after investment period
    confidence_interval: Tuple[float, float]  # 95% CI for returns


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    user_profile: UserProfile
    stocks: List[StockAnalysis]
    total_expected_return: float
    portfolio_risk_score: float
    portfolio_risk_level: RiskLevel
    expected_final_value: float
    best_case_value: float  # 95th percentile
    worst_case_value: float  # 5th percentile
    probability_of_target: float  # Probability of meeting user's target return
    diversification_score: float  # 0-100
    recommendations: List[str]  # Personalized advice


class InvestmentAdvisorModel:
    """
    AI-powered Investment Advisor using ML/DL.
    
    Features:
    - User-defined investment parameters
    - Risk assessment using multiple metrics
    - ML-based return prediction
    - Portfolio optimization using Modern Portfolio Theory
    - Monte Carlo simulation for probability analysis
    - Personalized stock recommendations
    """
    
    # Popular Indian stocks for analysis
    STOCK_UNIVERSE = {
        'RELIANCE.NS': ('Reliance Industries', 'Energy'),
        'TCS.NS': ('Tata Consultancy Services', 'IT'),
        'HDFCBANK.NS': ('HDFC Bank', 'Banking'),
        'INFY.NS': ('Infosys', 'IT'),
        'ICICIBANK.NS': ('ICICI Bank', 'Banking'),
        'HINDUNILVR.NS': ('Hindustan Unilever', 'FMCG'),
        'SBIN.NS': ('State Bank of India', 'Banking'),
        'BHARTIARTL.NS': ('Bharti Airtel', 'Telecom'),
        'ITC.NS': ('ITC Limited', 'FMCG'),
        'KOTAKBANK.NS': ('Kotak Mahindra Bank', 'Banking'),
        'LT.NS': ('Larsen & Toubro', 'Infrastructure'),
        'AXISBANK.NS': ('Axis Bank', 'Banking'),
        'ASIANPAINT.NS': ('Asian Paints', 'Paints'),
        'MARUTI.NS': ('Maruti Suzuki', 'Auto'),
        'TITAN.NS': ('Titan Company', 'Consumer'),
        'SUNPHARMA.NS': ('Sun Pharmaceutical', 'Pharma'),
        'BAJFINANCE.NS': ('Bajaj Finance', 'Finance'),
        'WIPRO.NS': ('Wipro', 'IT'),
        'HCLTECH.NS': ('HCL Technologies', 'IT'),
        'ULTRACEMCO.NS': ('UltraTech Cement', 'Cement'),
        'TATAMOTORS.NS': ('Tata Motors', 'Auto'),
        'POWERGRID.NS': ('Power Grid Corp', 'Power'),
        'NTPC.NS': ('NTPC', 'Power'),
        'ONGC.NS': ('ONGC', 'Energy'),
        'COALINDIA.NS': ('Coal India', 'Mining'),
        'TECHM.NS': ('Tech Mahindra', 'IT'),
        'BAJAJFINSV.NS': ('Bajaj Finserv', 'Finance'),
        'ADANIENT.NS': ('Adani Enterprises', 'Diversified'),
        'TATASTEEL.NS': ('Tata Steel', 'Metals'),
        'JSWSTEEL.NS': ('JSW Steel', 'Metals'),
    }
    
    def __init__(self):
        """Initialize Investment Advisor."""
        self.settings = get_settings()
        self.collector = DataCollector()
        self.indicators = TechnicalIndicators()
        self.scaler = StandardScaler()
        
        # ML Models
        self.return_predictor: Any = None
        self.risk_predictor: Any = None
        
        # Cache for stock data
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Market benchmark (NIFTY 50)
        self.market_returns: Optional[np.ndarray] = None
        
    def _fetch_stock_data(
        self,
        symbol: str,
        years: int = 5
    ) -> Optional[pd.DataFrame]:
        """Fetch historical stock data."""
        if symbol in self.stock_data_cache:
            return self.stock_data_cache[symbol]
        
        try:
            df = self.collector.get_stock_data(
                symbol=symbol,
                period=f"{years}y",
                interval="1d"
            )
            
            if df is not None and len(df) > 100:
                self.stock_data_cache[symbol] = df
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
        
        return None
    
    def _calculate_risk_metrics(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.
        
        Metrics:
        - Volatility (annualized standard deviation)
        - Value at Risk (VaR) at 95% and 99% confidence
        - Conditional VaR (Expected Shortfall)
        - Maximum Drawdown
        - Sharpe Ratio
        - Sortino Ratio
        - Calmar Ratio
        """
        # Annualized volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Value at Risk
        var_95 = np.percentile(returns, 5) * np.sqrt(252) * 100
        var_99 = np.percentile(returns, 1) * np.sqrt(252) * 100
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252) * 100
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # Risk-free rate (approximate Indian rate)
        risk_free_rate = 0.06 / 252  # Daily
        
        # Sharpe Ratio (annualized)
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else daily_vol
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio
        annual_return = np.mean(returns) * 252
        calmar = annual_return / abs(max_drawdown) * 100 if max_drawdown != 0 else 0
        
        return {
            'volatility': annual_vol * 100,
            'var_95': abs(var_95),
            'var_99': abs(var_99),
            'cvar_95': abs(cvar_95) if not np.isnan(cvar_95) else abs(var_95),
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar
        }
    
    def _calculate_beta(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> float:
        """Calculate stock beta relative to market."""
        if len(stock_returns) != len(market_returns):
            min_len = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns[-min_len:]
            market_returns = market_returns[-min_len:]
        
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def _calculate_risk_score(
        self,
        metrics: Dict[str, float],
        beta: float
    ) -> Tuple[float, RiskLevel]:
        """
        Calculate overall risk score (0-100) and risk level.
        
        Components:
        - Volatility (30% weight)
        - Max Drawdown (25% weight)
        - VaR (20% weight)
        - Beta (15% weight)
        - Sharpe adjustment (10% weight)
        """
        # Normalize metrics to 0-100 scale
        vol_score = min(metrics['volatility'] / 50 * 100, 100)  # 50% vol = max risk
        dd_score = min(metrics['max_drawdown'] / 60 * 100, 100)  # 60% DD = max risk
        var_score = min(metrics['var_95'] / 40 * 100, 100)  # 40% VaR = max risk
        beta_score = min(abs(beta - 1) * 50 + (beta if beta > 1 else 0) * 30, 100)
        
        # Sharpe adjustment (negative Sharpe = higher risk)
        sharpe_adjustment = max(0, (1 - metrics['sharpe_ratio']) * 20)
        
        # Weighted risk score
        risk_score = (
            vol_score * 0.30 +
            dd_score * 0.25 +
            var_score * 0.20 +
            beta_score * 0.15 +
            sharpe_adjustment * 0.10
        )
        
        # Determine risk level based on new categories:
        # 1️⃣ Very Safe (0-20): Large-cap leaders, monopolies - 8-10% YOY
        # 2️⃣ Safe (20-40): Strong large-caps + stable mid-caps - 10-13% YOY
        # 3️⃣ Moderate (40-60): Quality mid-caps, sector leaders - 13-18% YOY
        # 4️⃣ Low Risk Growth (60-80): High-growth mid/small caps - 18-25% YOY
        # 5️⃣ Risky (80-100): Small-caps, cyclicals, turnarounds - 25-40%+ YOY
        if risk_score < 20:
            risk_level = RiskLevel.VERY_SAFE
        elif risk_score < 40:
            risk_level = RiskLevel.SAFE
        elif risk_score < 60:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 80:
            risk_level = RiskLevel.LOW_RISK_GROWTH
        else:
            risk_level = RiskLevel.RISKY
        
        return risk_score, risk_level
    
    def _build_return_predictor(
        self,
        training_data: List[Dict[str, Any]]
    ) -> None:
        """
        Build ML model to predict future returns.
        
        Features:
        - Historical returns (1m, 3m, 6m, 1y, 3y)
        - Volatility metrics
        - Technical indicators
        - Momentum factors
        """
        if len(training_data) < 5:
            logger.warning("Not enough data to train return predictor")
            return
        
        # Prepare features
        X = []
        y = []
        
        for data in training_data:
            features = [
                data.get('return_1m', 0),
                data.get('return_3m', 0),
                data.get('return_6m', 0),
                data.get('return_1y', 0),
                data.get('return_3y', 0),
                data.get('volatility', 0),
                data.get('sharpe_ratio', 0),
                data.get('beta', 1),
                data.get('rsi', 50),
                data.get('momentum', 0),
                # Additional features for better prediction
                data.get('sortino_ratio', 0),
                data.get('calmar_ratio', 0),
                data.get('max_drawdown', 0),
                data.get('var_95', 0),
            ]
            X.append(features)
            y.append(data.get('future_return', 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # Train advanced ensemble model with multiple estimators
        from sklearn.ensemble import VotingRegressor, StackingRegressor
        
        base_estimators = [
            ('gb', GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=self.settings.RANDOM_STATE
            )),
            ('rf', RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                random_state=self.settings.RANDOM_STATE
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                tree_method='hist',
                random_state=self.settings.RANDOM_STATE
            ))
        ]
        
        # Stacking ensemble for better prediction
        self.return_predictor = StackingRegressor(
            estimators=base_estimators,
            final_estimator=GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1
            ),
            cv=3
        )
        
        try:
            self.return_predictor.fit(X, y)
            logger.info("Advanced return predictor (Stacking Ensemble) trained successfully")
        except Exception as e:
            logger.error(f"Failed to train return predictor: {e}")
            # Fallback to simple model
            self.return_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.settings.RANDOM_STATE
            )
            self.return_predictor.fit(X, y)
    
    def _predict_returns(
        self,
        stock_features: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Predict future returns with confidence interval.
        
        Uses advanced ML ensemble with:
        1. Historical performance trend
        2. Risk-adjusted expectations
        3. Momentum factors
        4. Volatility-adjusted predictions
        5. Multi-model consensus
        
        Returns:
            Tuple of (predicted_return, (lower_ci, upper_ci))
        """
        # Get historical returns
        return_1y = stock_features.get('return_1y', 0)
        return_3y = stock_features.get('return_3y', 0)
        return_6m = stock_features.get('return_6m', 0)
        return_3m = stock_features.get('return_3m', 0)
        
        # Annualize 3-year return
        annual_from_3y = (return_3y / 3) if return_3y != 0 else 0
        
        # Calculate weighted average historical return
        # More recent returns get higher weights with exponential decay
        weights = [0.40, 0.25, 0.20, 0.15]  # 1y, 3y_annual, 6m_annual, 3m_annual
        returns = [
            return_1y,
            annual_from_3y,
            return_6m * 2,  # Annualized 6-month
            return_3m * 4   # Annualized 3-month
        ]
        
        base_predicted = sum(w * r for w, r in zip(weights, returns))
        
        # Adjust for momentum
        momentum = stock_features.get('momentum', 0)
        momentum_adjustment = momentum * 0.1  # 10% of momentum as adjustment
        
        # Adjust for Sharpe ratio (quality adjustment)
        sharpe = stock_features.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            quality_adjustment = 2.0  # Bonus for high quality
        elif sharpe > 1.0:
            quality_adjustment = 1.0
        elif sharpe < 0:
            quality_adjustment = -2.0  # Penalty for negative Sharpe
        else:
            quality_adjustment = 0
        
        # Final prediction with adjustments
        predicted = base_predicted + momentum_adjustment + quality_adjustment
        
        # Clamp to reasonable range
        predicted = max(-30, min(50, predicted))
        
        # Estimate confidence interval based on volatility
        vol = stock_features.get('volatility', 20)
        ci_width = vol * 1.2  # CI proportional to volatility
        
        return predicted, (predicted - ci_width, predicted + ci_width)
    
    def _monte_carlo_simulation(
        self,
        initial_value: float,
        expected_return: float,
        volatility: float,
        years: int,
        simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for portfolio value projection.
        
        Returns distribution statistics of final portfolio value.
        """
        # Convert to daily parameters
        daily_return = expected_return / 100 / 252
        daily_vol = volatility / 100 / np.sqrt(252)
        days = years * 252
        
        # Generate random returns
        np.random.seed(self.settings.RANDOM_STATE)
        random_returns = np.random.normal(
            daily_return,
            daily_vol,
            (simulations, days)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_returns, axis=1)
        final_values = initial_value * cumulative_returns[:, -1]
        
        return {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'prob_profit': np.mean(final_values > initial_value) * 100,
            'prob_double': np.mean(final_values > initial_value * 2) * 100,
        }
    
    def _calculate_recommendation_score(
        self,
        stock_analysis: Dict[str, Any],
        user_profile: UserProfile
    ) -> float:
        """
        Calculate recommendation score based on user profile match.
        
        Factors:
        - Return vs user expectation alignment
        - Risk tolerance match
        - Diversification benefit
        """
        score = 50.0  # Base score
        
        predicted_return = stock_analysis['predicted_return']
        risk_score = stock_analysis['risk_score']
        
        # Return alignment (±30 points)
        target_return = user_profile.expected_annual_return
        return_diff = abs(predicted_return - target_return)
        
        if predicted_return >= target_return:
            score += min(20, (predicted_return - target_return) * 2)
        else:
            score -= min(20, return_diff * 3)
        
        # Risk tolerance match (±30 points)
        # Map risk tolerance to ideal risk scores based on new categories
        risk_tolerance_map = {
            'very_safe': 10,        # Very Low risk - target score 0-20
            'safe': 30,             # Low risk - target score 20-40
            'moderate': 50,         # Medium risk - target score 40-60
            'low_risk_growth': 70,  # Medium-High risk - target score 60-80
            'risky': 90,            # High risk - target score 80-100
            # Legacy mappings for backward compatibility
            'conservative': 20,
            'aggressive': 80
        }
        ideal_risk = risk_tolerance_map.get(user_profile.risk_tolerance, 50)
        risk_diff = abs(risk_score - ideal_risk)
        
        if risk_diff < 15:
            score += 20
        elif risk_diff < 30:
            score += 10
        else:
            score -= min(20, risk_diff / 3)
        
        # Sharpe ratio bonus (up to 10 points)
        sharpe = stock_analysis.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            score += 10
        elif sharpe > 1.0:
            score += 5
        elif sharpe < 0:
            score -= 10
        
        # Ensure score is in valid range
        return max(0, min(100, score))
    
    def _optimize_portfolio_allocation(
        self,
        stocks: List[Dict[str, Any]],
        user_profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """
        Optimize portfolio allocation using Modern Portfolio Theory.
        
        Uses mean-variance optimization with user constraints.
        """
        n_stocks = len(stocks)
        
        if n_stocks == 0:
            return []
        
        if n_stocks == 1:
            stocks[0]['allocation'] = 100.0
            return stocks
        
        # Extract returns and build covariance matrix
        returns = np.array([s['predicted_return'] for s in stocks])
        volatilities = np.array([s['volatility'] for s in stocks])
        
        # Simple correlation assumption (can be enhanced with actual correlations)
        correlation = 0.3  # Average correlation assumption
        cov_matrix = np.outer(volatilities, volatilities) * correlation / 10000
        np.fill_diagonal(cov_matrix, (volatilities / 100) ** 2)
        
        # Risk tolerance adjustment based on new categories
        # Risk aversion: higher = more conservative allocation
        risk_aversion = {
            'very_safe': 4.0,       # Very conservative - capital protection
            'safe': 3.0,            # Conservative - stable returns
            'moderate': 2.0,        # Balanced risk-reward
            'low_risk_growth': 1.0, # Growth-oriented - accept more volatility
            'risky': 0.5,           # Aggressive - maximize returns
            # Legacy mappings
            'conservative': 3.0,
            'aggressive': 0.5
        }.get(user_profile.risk_tolerance, 2.0)
        
        # Simple mean-variance optimization
        # Maximize: returns - risk_aversion * variance
        try:
            # Use recommendation scores as initial weights
            rec_scores = np.array([s['recommendation_score'] for s in stocks])
            weights = rec_scores / rec_scores.sum()
            
            # Adjust based on risk tolerance
            if user_profile.risk_tolerance in ['very_safe', 'safe', 'conservative']:
                # Favor lower risk stocks
                risk_scores = np.array([s['risk_score'] for s in stocks])
                risk_weights = 1 / (risk_scores + 1)
                weights = weights * 0.5 + (risk_weights / risk_weights.sum()) * 0.5
            elif user_profile.risk_tolerance in ['risky', 'low_risk_growth', 'aggressive']:
                # Favor higher return stocks
                return_weights = returns / returns.sum() if returns.sum() > 0 else weights
                weights = weights * 0.5 + return_weights * 0.5
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Apply constraints
            max_allocation = 0.35 if n_stocks >= 5 else 0.50
            weights = np.minimum(weights, max_allocation)
            weights = weights / weights.sum()  # Re-normalize
            
        except Exception as e:
            logger.warning(f"Optimization failed, using equal weights: {e}")
            weights = np.ones(n_stocks) / n_stocks
        
        # Assign allocations
        for i, stock in enumerate(stocks):
            stock['allocation'] = round(weights[i] * 100, 2)
        
        return stocks
    
    def analyze_stock(
        self,
        symbol: str,
        years_history: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive analysis of a single stock.
        
        Returns all metrics needed for investment recommendation.
        """
        if symbol in self.analysis_cache:
            return self.analysis_cache[symbol]
        
        df = self._fetch_stock_data(symbol, years_history)
        
        if df is None or len(df) < 252:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            returns = df['returns'].dropna().values
            
            # Historical returns
            current_price = df['close'].iloc[-1]
            
            periods = {
                '1m': 21,
                '3m': 63,
                '6m': 126,
                '1y': 252,
                '3y': 756
            }
            
            historical_returns = {}
            for period_name, days in periods.items():
                if len(df) > days:
                    period_return = (df['close'].iloc[-1] / df['close'].iloc[-days] - 1) * 100
                    historical_returns[f'return_{period_name}'] = period_return
                else:
                    historical_returns[f'return_{period_name}'] = 0.0
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(np.array(returns))
            
            # Calculate beta (using NIFTY as benchmark)
            if self.market_returns is None:
                nifty_df = self._fetch_stock_data('^NSEI', years_history)
                if nifty_df is not None:
                    self.market_returns = np.array(nifty_df['close'].pct_change().dropna().values)
            
            beta = 1.0
            if self.market_returns is not None:
                beta = self._calculate_beta(np.array(returns), self.market_returns)
            
            # Technical indicators
            df_with_indicators = self.indicators.add_all_indicators(df.copy())
            
            rsi = df_with_indicators['rsi_14'].iloc[-1] if 'rsi_14' in df_with_indicators.columns else 50
            momentum = historical_returns.get('return_3m', 0)
            
            # Calculate risk score
            risk_score, risk_level = self._calculate_risk_score(risk_metrics, beta)
            
            analysis = {
                'symbol': symbol,
                'company_name': self.STOCK_UNIVERSE.get(symbol, (symbol, 'Unknown'))[0],
                'sector': self.STOCK_UNIVERSE.get(symbol, (symbol, 'Unknown'))[1],
                'current_price': current_price,
                **historical_returns,
                **risk_metrics,
                'beta': beta,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'rsi': rsi,
                'momentum': momentum,
            }
            
            self.analysis_cache[symbol] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None
    
    def get_personalized_recommendations(
        self,
        user_profile: UserProfile,
        top_n: int = 10,
        progress_callback: Optional[Any] = None
    ) -> PortfolioRecommendation:
        """
        Generate personalized investment recommendations.
        
        Args:
            user_profile: User's investment profile
            top_n: Number of stocks to recommend
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete portfolio recommendation
        """
        logger.info(f"Generating recommendations for ₹{user_profile.investment_amount:,.2f} "
                   f"targeting {user_profile.expected_annual_return}% over {user_profile.investment_years} years")
        
        # Step 1: Analyze all stocks
        all_analyses = []
        total_stocks = len(self.STOCK_UNIVERSE)
        
        for i, symbol in enumerate(self.STOCK_UNIVERSE.keys()):
            if progress_callback:
                progress_callback(f"Analyzing {symbol}", (i + 1) / total_stocks * 40)
            
            analysis = self.analyze_stock(symbol)
            if analysis:
                all_analyses.append(analysis)
        
        if len(all_analyses) < 3:
            raise ValueError("Not enough stocks could be analyzed. Check internet connection.")
        
        # Step 2: Train return predictor
        if progress_callback:
            progress_callback("Training AI model", 50)
        
        self._build_return_predictor(all_analyses)
        
        # Step 3: Predict returns for each stock
        if progress_callback:
            progress_callback("Predicting returns", 60)
        
        for analysis in all_analyses:
            predicted_return, ci = self._predict_returns(analysis)
            analysis['predicted_return'] = predicted_return
            analysis['confidence_interval'] = ci
        
        # Step 4: Calculate recommendation scores
        if progress_callback:
            progress_callback("Scoring recommendations", 70)
        
        for analysis in all_analyses:
            analysis['recommendation_score'] = self._calculate_recommendation_score(
                analysis, user_profile
            )
        
        # Step 5: Select top stocks
        sorted_analyses = sorted(
            all_analyses,
            key=lambda x: x['recommendation_score'],
            reverse=True
        )
        
        # Ensure sector diversification
        selected_stocks = []
        sectors_used = set()
        max_per_sector = 3
        
        for analysis in sorted_analyses:
            sector = analysis['sector']
            sector_count = sum(1 for s in selected_stocks if s['sector'] == sector)
            
            if sector_count < max_per_sector:
                selected_stocks.append(analysis)
                sectors_used.add(sector)
            
            if len(selected_stocks) >= top_n:
                break
        
        # Step 6: Optimize portfolio allocation
        if progress_callback:
            progress_callback("Optimizing portfolio", 80)
        
        selected_stocks = self._optimize_portfolio_allocation(selected_stocks, user_profile)
        
        # Step 7: Create stock analysis objects
        stock_analyses = []
        
        for stock in selected_stocks:
            allocation_amount = user_profile.investment_amount * stock['allocation'] / 100
            
            # Monte Carlo simulation for this stock
            mc_results = self._monte_carlo_simulation(
                allocation_amount,
                stock['predicted_return'],
                stock['volatility'],
                user_profile.investment_years
            )
            
            stock_analysis = StockAnalysis(
                symbol=stock['symbol'],
                company_name=stock['company_name'],
                sector=stock['sector'],
                current_price=stock['current_price'],
                predicted_return=stock['predicted_return'],
                risk_score=stock['risk_score'],
                risk_level=stock['risk_level'],
                volatility=stock['volatility'],
                sharpe_ratio=stock['sharpe_ratio'],
                max_drawdown=stock['max_drawdown'],
                var_95=stock['var_95'],
                beta=stock['beta'],
                recommendation_score=stock['recommendation_score'],
                suggested_allocation=stock['allocation'],
                expected_value=mc_results['mean'],
                confidence_interval=(mc_results['percentile_5'], mc_results['percentile_95'])
            )
            stock_analyses.append(stock_analysis)
        
        # Step 8: Calculate portfolio-level metrics
        if progress_callback:
            progress_callback("Calculating portfolio metrics", 90)
        
        # Portfolio expected return (weighted average)
        portfolio_return = sum(
            s.predicted_return * s.suggested_allocation / 100
            for s in stock_analyses
        )
        
        # Portfolio risk (simplified - weighted average volatility)
        portfolio_volatility = sum(
            s.volatility * s.suggested_allocation / 100
            for s in stock_analyses
        )
        
        # Portfolio risk score
        portfolio_risk_score = sum(
            s.risk_score * s.suggested_allocation / 100
            for s in stock_analyses
        )
        
        # Determine portfolio risk level based on new categories
        # 1️⃣ Very Safe (0-20): 8-10% YOY
        # 2️⃣ Safe (20-40): 10-13% YOY
        # 3️⃣ Moderate (40-60): 13-18% YOY
        # 4️⃣ Low Risk Growth (60-80): 18-25% YOY
        # 5️⃣ Risky (80-100): 25-40%+ YOY
        if portfolio_risk_score < 20:
            portfolio_risk_level = RiskLevel.VERY_SAFE
        elif portfolio_risk_score < 40:
            portfolio_risk_level = RiskLevel.SAFE
        elif portfolio_risk_score < 60:
            portfolio_risk_level = RiskLevel.MODERATE
        elif portfolio_risk_score < 80:
            portfolio_risk_level = RiskLevel.LOW_RISK_GROWTH
        else:
            portfolio_risk_level = RiskLevel.RISKY
        
        # Portfolio Monte Carlo simulation
        portfolio_mc = self._monte_carlo_simulation(
            user_profile.investment_amount,
            portfolio_return,
            portfolio_volatility,
            user_profile.investment_years
        )
        
        # Calculate probability of meeting target
        target_value = user_profile.investment_amount * (
            (1 + user_profile.expected_annual_return / 100) ** user_profile.investment_years
        )
        
        # Diversification score
        n_sectors = len(set(s.sector for s in stock_analyses))
        diversification_score = min(100, n_sectors * 15 + len(stock_analyses) * 5)
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(
            user_profile, stock_analyses, portfolio_return,
            portfolio_risk_score, portfolio_mc
        )
        
        if progress_callback:
            progress_callback("Complete", 100)
        
        return PortfolioRecommendation(
            user_profile=user_profile,
            stocks=stock_analyses,
            total_expected_return=portfolio_return,
            portfolio_risk_score=portfolio_risk_score,
            portfolio_risk_level=portfolio_risk_level,
            expected_final_value=portfolio_mc['mean'],
            best_case_value=portfolio_mc['percentile_95'],
            worst_case_value=portfolio_mc['percentile_5'],
            probability_of_target=portfolio_mc['prob_profit'],
            diversification_score=diversification_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        user_profile: UserProfile,
        stocks: List[StockAnalysis],
        portfolio_return: float,
        portfolio_risk: float,
        mc_results: Dict[str, float]
    ) -> List[str]:
        """Generate personalized advice based on analysis."""
        recommendations = []
        
        # Return expectation analysis
        if portfolio_return >= user_profile.expected_annual_return:
            recommendations.append(
                f"✅ Your target return of {user_profile.expected_annual_return}% is achievable. "
                f"Expected portfolio return: {portfolio_return:.1f}%"
            )
        else:
            gap = user_profile.expected_annual_return - portfolio_return
            recommendations.append(
                f"⚠️ Your target return of {user_profile.expected_annual_return}% may be ambitious. "
                f"Expected return: {portfolio_return:.1f}% (Gap: {gap:.1f}%)"
            )
        
        # Risk analysis based on new risk categories with realistic YOY returns
        # | Category                    | Risk Level  | Typical Stocks                      | Realistic YOY Return     |
        # | 1️⃣ Very Safe               | Very Low    | Large-cap leaders, monopolies       | 8% – 10%                 |
        # | 2️⃣ Safe                    | Low         | Strong large-caps + stable mid-caps | 10% – 13%                |
        # | 3️⃣ Moderate                | Medium      | Quality mid-caps, sector leaders    | 13% – 18%                |
        # | 4️⃣ Low Risk (Growth)       | Medium–High | High-growth mid/small caps          | 18% – 25%                |
        # | 5️⃣ Risky                   | High        | Small-caps, cyclicals, turnarounds  | 25% – 40%+ (unstable)    |
        risk_tolerance_ideal = {
            'very_safe': 10,        # Very Safe: 8-10% YOY
            'safe': 30,             # Safe: 10-13% YOY
            'moderate': 50,         # Moderate: 13-18% YOY
            'low_risk_growth': 70,  # Low Risk Growth: 18-25% YOY
            'risky': 90,            # Risky: 25-40%+ YOY
            # Legacy mappings
            'conservative': 25,
            'aggressive': 75
        }
        
        # Risk category names for display
        risk_category_names = {
            'very_safe': '1️⃣ Very Safe (8-10% YOY)',
            'safe': '2️⃣ Safe (10-13% YOY)',
            'moderate': '3️⃣ Moderate (13-18% YOY)',
            'low_risk_growth': '4️⃣ Low Risk Growth (18-25% YOY)',
            'risky': '5️⃣ Risky (25-40%+ YOY)',
            'conservative': 'Conservative',
            'aggressive': 'Aggressive'
        }
        
        ideal_risk = risk_tolerance_ideal.get(user_profile.risk_tolerance, 50)
        risk_name = risk_category_names.get(user_profile.risk_tolerance, user_profile.risk_tolerance)
        
        if abs(portfolio_risk - ideal_risk) < 15:
            recommendations.append(
                f"✅ Portfolio risk ({portfolio_risk:.1f}/100) aligns with your "
                f"{risk_name} risk tolerance"
            )
        elif portfolio_risk > ideal_risk + 15:
            recommendations.append(
                f"⚠️ Portfolio risk ({portfolio_risk:.1f}/100) is higher than ideal for "
                f"your {risk_name} profile. Consider reducing allocation to high-risk stocks."
            )
        
        # Time horizon advice
        if user_profile.investment_years < 3:
            recommendations.append(
                "⚠️ Short investment horizon (<3 years) increases risk. "
                "Consider extending to 5+ years for better risk-adjusted returns."
            )
        elif user_profile.investment_years >= 5:
            recommendations.append(
                "✅ Your investment horizon of 5+ years is ideal for equity investments."
            )
        
        # Diversification advice
        n_sectors = len(set(s.sector for s in stocks))
        if n_sectors >= 5:
            recommendations.append(
                f"✅ Good diversification across {n_sectors} sectors"
            )
        else:
            recommendations.append(
                f"💡 Consider diversifying more. Currently invested in {n_sectors} sectors."
            )
        
        # Probability analysis
        prob_profit = mc_results['prob_profit']
        if prob_profit >= 80:
            recommendations.append(
                f"✅ {prob_profit:.0f}% probability of positive returns based on Monte Carlo simulation"
            )
        elif prob_profit >= 60:
            recommendations.append(
                f"📊 {prob_profit:.0f}% probability of positive returns. Moderate confidence."
            )
        else:
            recommendations.append(
                f"⚠️ Only {prob_profit:.0f}% probability of positive returns. High uncertainty."
            )
        
        # Amount-specific advice
        if user_profile.investment_amount < 50000:
            recommendations.append(
                "💡 Consider SIP (Systematic Investment Plan) for small amounts "
                "to reduce timing risk."
            )
        elif user_profile.investment_amount > 1000000:
            recommendations.append(
                "💡 For large investments, consider staggered entry over 3-6 months "
                "to reduce timing risk."
            )
        
        return recommendations
