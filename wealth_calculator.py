"""
Wealth Management Calculator Module
Investment planning and portfolio management tools
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import RISK_PROFILES, RETURN_OPTIONS


class WealthCalculator:
    """Investment and wealth calculation tools"""
    
    @staticmethod
    def calculate_future_value(principal, rate, years, compound_frequency=12):
        """
        Calculate future value of investment
        
        Args:
            principal: Initial investment amount
            rate: Annual interest rate (as percentage, e.g., 10 for 10%)
            years: Investment duration in years
            compound_frequency: Number of times interest compounds per year
        
        Returns:
            Dictionary with future value and growth details
        """
        rate_decimal = rate / 100
        n = compound_frequency
        t = years
        
        # Compound interest formula: FV = P(1 + r/n)^(nt)
        future_value = principal * (1 + rate_decimal / n) ** (n * t)
        
        # Total interest earned
        total_interest = future_value - principal
        
        # Effective annual rate
        effective_rate = ((1 + rate_decimal / n) ** n - 1) * 100
        
        # Growth multiplier
        growth_multiplier = future_value / principal
        
        return {
            'principal': round(principal, 2),
            'rate': rate,
            'years': years,
            'future_value': round(future_value, 2),
            'total_interest': round(total_interest, 2),
            'effective_rate': round(effective_rate, 2),
            'growth_multiplier': round(growth_multiplier, 2)
        }
    
    @staticmethod
    def calculate_sip_returns(monthly_investment, rate, years):
        """
        Calculate SIP (Systematic Investment Plan) returns
        
        Args:
            monthly_investment: Monthly investment amount
            rate: Expected annual return rate (as percentage)
            years: Investment duration in years
        
        Returns:
            Dictionary with SIP calculation details
        """
        rate_decimal = rate / 100
        monthly_rate = rate_decimal / 12
        months = years * 12
        
        # SIP Future Value Formula: FV = P × ((1+r)^n - 1) / r × (1+r)
        if monthly_rate > 0:
            future_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        else:
            future_value = monthly_investment * months
        
        total_invested = monthly_investment * months
        wealth_gained = future_value - total_invested
        
        return {
            'monthly_investment': round(monthly_investment, 2),
            'rate': rate,
            'years': years,
            'total_months': months,
            'total_invested': round(total_invested, 2),
            'future_value': round(future_value, 2),
            'wealth_gained': round(wealth_gained, 2),
            'absolute_return': round((wealth_gained / total_invested) * 100, 2) if total_invested > 0 else 0
        }
    
    @staticmethod
    def calculate_lumpsum_vs_sip(total_amount, rate, years):
        """Compare lumpsum investment vs SIP"""
        # Lumpsum
        lumpsum = WealthCalculator.calculate_future_value(total_amount, rate, years)
        
        # SIP (dividing total amount equally over the period)
        monthly_sip = total_amount / (years * 12)
        sip = WealthCalculator.calculate_sip_returns(monthly_sip, rate, years)
        
        return {
            'lumpsum': lumpsum,
            'sip': sip,
            'difference': round(lumpsum['future_value'] - sip['future_value'], 2),
            'better_option': 'Lumpsum' if lumpsum['future_value'] > sip['future_value'] else 'SIP'
        }
    
    @staticmethod
    def calculate_goal_based_investment(target_amount, years, expected_rate):
        """
        Calculate required investment to reach a financial goal
        
        Args:
            target_amount: Target amount to achieve
            years: Time period in years
            expected_rate: Expected annual return rate
        
        Returns:
            Dictionary with required investment details
        """
        rate_decimal = expected_rate / 100
        
        # Required lumpsum
        required_lumpsum = target_amount / ((1 + rate_decimal) ** years)
        
        # Required monthly SIP
        monthly_rate = rate_decimal / 12
        months = years * 12
        
        if monthly_rate > 0:
            required_sip = target_amount / (((1 + monthly_rate) ** months - 1) / monthly_rate * (1 + monthly_rate))
        else:
            required_sip = target_amount / months
        
        return {
            'target_amount': round(target_amount, 2),
            'years': years,
            'expected_rate': expected_rate,
            'required_lumpsum': round(required_lumpsum, 2),
            'required_monthly_sip': round(required_sip, 2),
            'required_yearly_sip': round(required_sip * 12, 2)
        }
    
    @staticmethod
    def calculate_retirement_corpus(current_age, retirement_age, monthly_expense, inflation_rate=6, life_expectancy=85):
        """
        Calculate required retirement corpus
        
        Args:
            current_age: Current age
            retirement_age: Planned retirement age
            monthly_expense: Current monthly expense
            inflation_rate: Expected inflation rate
            life_expectancy: Expected life expectancy
        
        Returns:
            Dictionary with retirement planning details
        """
        years_to_retire = retirement_age - current_age
        years_in_retirement = life_expectancy - retirement_age
        inflation_decimal = inflation_rate / 100
        
        # Monthly expense at retirement (adjusted for inflation)
        expense_at_retirement = monthly_expense * ((1 + inflation_decimal) ** years_to_retire)
        
        # Annual expense at retirement
        annual_expense_at_retirement = expense_at_retirement * 12
        
        # Required corpus (considering inflation during retirement)
        # Using perpetuity formula with growth: C / (r - g)
        # Assuming 4% safe withdrawal rate
        withdrawal_rate = 0.04
        
        # Total corpus needed (simplified calculation)
        corpus_needed = annual_expense_at_retirement * years_in_retirement * 1.5  # 1.5x for buffer
        
        return {
            'current_age': current_age,
            'retirement_age': retirement_age,
            'years_to_retire': years_to_retire,
            'years_in_retirement': years_in_retirement,
            'current_monthly_expense': round(monthly_expense, 2),
            'expense_at_retirement': round(expense_at_retirement, 2),
            'annual_expense_at_retirement': round(annual_expense_at_retirement, 2),
            'corpus_needed': round(corpus_needed, 2)
        }
    
    @staticmethod
    def generate_investment_schedule(principal, rate, years, investment_type='lumpsum', monthly_addition=0):
        """
        Generate year-by-year investment growth schedule
        
        Args:
            principal: Initial investment
            rate: Annual return rate
            years: Investment duration
            investment_type: 'lumpsum' or 'sip'
            monthly_addition: Monthly addition (for SIP)
        
        Returns:
            DataFrame with yearly schedule
        """
        rate_decimal = rate / 100
        schedule = []
        
        current_value = principal
        total_invested = principal
        
        for year in range(1, years + 1):
            if investment_type == 'sip' and monthly_addition > 0:
                # Add monthly investments with compound interest
                yearly_addition = 0
                for month in range(12):
                    months_remaining = (years - year + 1) * 12 - month
                    yearly_addition += monthly_addition * ((1 + rate_decimal/12) ** (12 - month))
                current_value = current_value * (1 + rate_decimal) + yearly_addition
                total_invested += monthly_addition * 12
            else:
                current_value = current_value * (1 + rate_decimal)
            
            gains = current_value - total_invested
            
            schedule.append({
                'Year': year,
                'Total Invested': round(total_invested, 2),
                'Portfolio Value': round(current_value, 2),
                'Gains': round(gains, 2),
                'Returns %': round((gains / total_invested) * 100, 2) if total_invested > 0 else 0
            })
        
        return pd.DataFrame(schedule)


class PortfolioManager:
    """Portfolio management and allocation tools"""
    
    def __init__(self):
        self.risk_profiles = RISK_PROFILES
    
    def get_asset_allocation(self, risk_profile):
        """Get recommended asset allocation based on risk profile"""
        if risk_profile in self.risk_profiles:
            return self.risk_profiles[risk_profile]
        return self.risk_profiles['Moderate']
    
    def calculate_portfolio_value(self, holdings):
        """
        Calculate total portfolio value
        
        Args:
            holdings: List of dictionaries with 'symbol', 'quantity', 'current_price', 'buy_price'
        
        Returns:
            Dictionary with portfolio summary
        """
        total_invested = 0
        current_value = 0
        
        for holding in holdings:
            invested = holding['quantity'] * holding['buy_price']
            current = holding['quantity'] * holding['current_price']
            
            total_invested += invested
            current_value += current
        
        total_gains = current_value - total_invested
        returns_pct = (total_gains / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_invested': round(total_invested, 2),
            'current_value': round(current_value, 2),
            'total_gains': round(total_gains, 2),
            'returns_percent': round(returns_pct, 2),
            'is_profit': total_gains >= 0
        }
    
    def rebalance_portfolio(self, current_allocation, target_allocation, portfolio_value):
        """
        Calculate rebalancing requirements
        
        Args:
            current_allocation: Dict with current % allocation
            target_allocation: Dict with target % allocation
            portfolio_value: Total portfolio value
        
        Returns:
            Dictionary with rebalancing recommendations
        """
        rebalancing = {}
        
        for asset in target_allocation:
            current_pct = current_allocation.get(asset, 0)
            target_pct = target_allocation[asset]
            
            current_value = portfolio_value * (current_pct / 100)
            target_value = portfolio_value * (target_pct / 100)
            
            difference = target_value - current_value
            
            rebalancing[asset] = {
                'current_allocation': current_pct,
                'target_allocation': target_pct,
                'current_value': round(current_value, 2),
                'target_value': round(target_value, 2),
                'action': 'BUY' if difference > 0 else 'SELL' if difference < 0 else 'HOLD',
                'amount': round(abs(difference), 2)
            }
        
        return rebalancing
    
    def calculate_portfolio_risk(self, holdings, returns_data):
        """
        Calculate portfolio risk metrics
        
        Args:
            holdings: List of holdings with weights
            returns_data: DataFrame with returns for each holding
        
        Returns:
            Dictionary with risk metrics
        """
        if returns_data.empty:
            return {}
        
        # Calculate portfolio returns
        weights = np.array([h.get('weight', 1/len(holdings)) for h in holdings])
        
        # Portfolio return
        portfolio_return = np.sum(returns_data.mean() * weights) * 252
        
        # Portfolio volatility
        cov_matrix = returns_data.cov() * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe Ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'expected_return': round(portfolio_return * 100, 2),
            'volatility': round(portfolio_volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }


class TaxCalculator:
    """Tax calculation tools for investments"""
    
    @staticmethod
    def calculate_capital_gains_tax(buy_price, sell_price, quantity, holding_period_days, asset_type='equity'):
        """
        Calculate capital gains tax
        
        Args:
            buy_price: Purchase price per unit
            sell_price: Selling price per unit
            quantity: Number of units
            holding_period_days: Number of days held
            asset_type: 'equity', 'debt', 'gold'
        
        Returns:
            Dictionary with tax calculation
        """
        total_buy = buy_price * quantity
        total_sell = sell_price * quantity
        capital_gain = total_sell - total_buy
        
        # Determine if short-term or long-term
        if asset_type == 'equity':
            is_long_term = holding_period_days > 365
            stcg_rate = 15  # Short-term capital gains tax
            ltcg_rate = 10  # Long-term (above ₹1 lakh exemption)
            ltcg_exemption = 100000
        elif asset_type == 'debt':
            is_long_term = holding_period_days > 1095  # 3 years for debt
            stcg_rate = 30  # As per income slab (assuming highest)
            ltcg_rate = 20  # With indexation
            ltcg_exemption = 0
        else:  # gold and others
            is_long_term = holding_period_days > 1095
            stcg_rate = 30
            ltcg_rate = 20
            ltcg_exemption = 0
        
        if capital_gain <= 0:
            tax = 0
            tax_type = 'No Tax (Loss)'
        elif is_long_term:
            taxable_gain = max(0, capital_gain - ltcg_exemption)
            tax = taxable_gain * (ltcg_rate / 100)
            tax_type = 'LTCG'
        else:
            tax = capital_gain * (stcg_rate / 100)
            tax_type = 'STCG'
        
        net_gain = capital_gain - tax
        
        return {
            'buy_value': round(total_buy, 2),
            'sell_value': round(total_sell, 2),
            'capital_gain': round(capital_gain, 2),
            'holding_period_days': holding_period_days,
            'is_long_term': is_long_term,
            'tax_type': tax_type,
            'tax_rate': ltcg_rate if is_long_term else stcg_rate,
            'tax_amount': round(tax, 2),
            'net_gain_after_tax': round(net_gain, 2)
        }
    
    @staticmethod
    def calculate_dividend_tax(dividend_amount, income_slab='highest'):
        """Calculate tax on dividend income"""
        # Dividend is taxable as per income slab from AY 2021-22
        tax_rates = {
            'lowest': 5,
            'middle': 20,
            'highest': 30
        }
        
        rate = tax_rates.get(income_slab, 30)
        tax = dividend_amount * (rate / 100)
        
        return {
            'dividend_amount': round(dividend_amount, 2),
            'tax_rate': rate,
            'tax_amount': round(tax, 2),
            'net_dividend': round(dividend_amount - tax, 2)
        }


def calculate_cagr(initial_value, final_value, years):
    """Calculate Compound Annual Growth Rate"""
    if initial_value <= 0 or years <= 0:
        return 0
    
    cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100
    return round(cagr, 2)


def calculate_xirr(cashflows, dates):
    """
    Calculate XIRR (Extended Internal Rate of Return)
    
    Args:
        cashflows: List of cashflows (negative for investments, positive for returns)
        dates: List of dates corresponding to cashflows
    
    Returns:
        XIRR as percentage
    """
    if len(cashflows) != len(dates) or len(cashflows) < 2:
        return 0
    
    # Simple iterative XIRR calculation
    def npv(rate):
        total = 0
        for i, cf in enumerate(cashflows):
            days = (dates[i] - dates[0]).days
            total += cf / ((1 + rate) ** (days / 365))
        return total
    
    # Binary search for XIRR
    low, high = -0.99, 10
    
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
        
        if abs(npv(mid)) < 0.01:
            break
    
    return round(mid * 100, 2)
