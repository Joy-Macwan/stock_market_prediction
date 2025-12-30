"""
AI Investment Advisor - Streamlit Web Application
==================================================

Takes user inputs → Processes with ML models → Returns stock recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from src.models.investment_advisor import (
    InvestmentAdvisorModel,
    UserProfile,
    RISK_CATEGORY_RETURNS,
    RiskLevel,
    validate_yoy_return_expectation
)

# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="AI Wealth Manager 💰",
    page_icon="💰",
    layout="wide"
)

# =============== SIMPLE CSS ===============
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }

.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #ff006e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
    margin: 0.5rem 0;
}

.metric-card.green { background: linear-gradient(135deg, #11998e, #38ef7d); }
.metric-card.orange { background: linear-gradient(135deg, #f093fb, #f5576c); }
.metric-card.blue { background: linear-gradient(135deg, #4facfe, #00f2fe); }

.stock-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.stock-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: #00d4ff;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.spinner {
    width: 50px; height: 50px;
    border: 4px solid rgba(0, 212, 255, 0.2);
    border-top-color: #00d4ff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)


def format_inr(value):
    """Format as Indian Rupees."""
    if value >= 10000000:
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:
        return f"₹{value/100000:.2f} L"
    return f"₹{value:,.0f}"


def run_analysis(investment_amount, expected_return, investment_years, risk_tolerance, num_stocks, progress_bar, status_text):
    """
    Run the actual AI analysis and return recommendations.
    """
    try:
        # Step 1: Create user profile
        status_text.text("📋 Creating your investment profile...")
        progress_bar.progress(10)
        
        user_profile = UserProfile(
            investment_amount=float(investment_amount),
            expected_annual_return=float(expected_return),
            investment_years=int(investment_years),
            risk_tolerance=risk_tolerance
        )
        
        # Step 2: Initialize advisor
        status_text.text("🤖 Initializing AI Investment Advisor...")
        progress_bar.progress(20)
        advisor = InvestmentAdvisorModel()
        
        # Step 3: Define progress callback
        def update_progress(msg, pct):
            status_text.text(msg)
            progress_bar.progress(min(int(20 + pct * 0.7), 95))
        
        # Step 4: Get recommendations
        status_text.text("📊 Fetching market data for 30+ stocks...")
        progress_bar.progress(30)
        
        result = advisor.get_personalized_recommendations(
            user_profile=user_profile,
            top_n=num_stocks,
            progress_callback=update_progress
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return result, None
        
    except Exception as e:
        return None, str(e)


def display_results(result, investment_amount, expected_return, investment_years, risk_tolerance):
    """Display the analysis results."""
    
    # ========== SUMMARY BANNER ==========
    meets_target = result.total_expected_return >= expected_return
    
    if meets_target:
        st.success(f"🎉 **TARGET ACHIEVABLE!** Your {expected_return}% target is realistic. Expected: **{result.total_expected_return:.1f}%**")
    else:
        st.warning(f"⚠️ **AMBITIOUS TARGET** Your {expected_return}% target may be challenging. Expected: **{result.total_expected_return:.1f}%**")
    
    st.markdown("---")
    
    # ========== KEY METRICS ==========
    st.subheader("📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Expected Annual Return",
            value=f"{result.total_expected_return:.1f}%",
            delta=f"{result.total_expected_return - expected_return:.1f}%" if result.total_expected_return != expected_return else None
        )
    
    with col2:
        st.metric(
            label="Expected Final Value",
            value=format_inr(result.expected_final_value)
        )
    
    with col3:
        st.metric(
            label="Success Probability",
            value=f"{result.probability_of_target:.0f}%"
        )
    
    with col4:
        st.metric(
            label="Risk Score",
            value=f"{result.portfolio_risk_score:.0f}/100"
        )
    
    st.markdown("---")
    
    # ========== STOCK RECOMMENDATIONS ==========
    st.subheader("🎯 AI Recommended Stocks")
    
    # Create dataframe for display
    stocks_data = []
    for i, stock in enumerate(result.stocks, 1):
        allocation_amount = investment_amount * stock.suggested_allocation / 100
        stocks_data.append({
            "#": i,
            "Stock": stock.company_name,
            "Symbol": stock.symbol,
            "Sector": stock.sector,
            "Allocation %": f"{stock.suggested_allocation:.1f}%",
            "Amount": format_inr(allocation_amount),
            "Expected Return": f"+{stock.predicted_return:.1f}%",
            "Risk Level": stock.risk_level.value,
            "AI Score": f"{stock.recommendation_score:.0f}/100"
        })
    
    df = pd.DataFrame(stocks_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ========== CHARTS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Allocation")
        
        pie_data = pd.DataFrame([
            {"Stock": s.company_name, "Allocation": s.suggested_allocation}
            for s in result.stocks
        ])
        
        fig = px.pie(
            pie_data, 
            values='Allocation', 
            names='Stock',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Projected Growth")
        
        years = list(range(investment_years + 1))
        expected_values = [investment_amount * ((1 + result.total_expected_return/100) ** y) for y in years]
        best_case = [investment_amount * ((1 + (result.total_expected_return + 10)/100) ** y) for y in years]
        worst_case = [investment_amount * ((1 + max(0, result.total_expected_return - 15)/100) ** y) for y in years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=best_case, mode='lines', name='Best Case (+10%)', line=dict(color='#00ff88', dash='dash')))
        fig.add_trace(go.Scatter(x=years, y=expected_values, mode='lines+markers', name='Expected', line=dict(color='#00d4ff', width=3)))
        fig.add_trace(go.Scatter(x=years, y=worst_case, mode='lines', name='Worst Case (-15%)', line=dict(color='#ff6b6b', dash='dash')))
        
        fig.update_layout(
            xaxis_title="Years",
            yaxis_title="Portfolio Value (₹)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== RISK ANALYSIS ==========
    st.subheader("📊 Risk Analysis")
    
    risk_data = []
    for stock in result.stocks:
        risk_data.append({
            "Stock": stock.company_name,
            "Volatility": f"{stock.volatility:.1f}%",
            "VaR 95%": f"{stock.var_95:.1f}%",
            "Max Drawdown": f"{stock.max_drawdown:.1f}%",
            "Sharpe Ratio": f"{stock.sharpe_ratio:.2f}",
            "Beta": f"{stock.beta:.2f}"
        })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ========== AI INSIGHTS ==========
    st.subheader("💡 AI Insights & Recommendations")
    
    for rec in result.recommendations:
        if "✅" in rec:
            st.success(rec)
        elif "⚠️" in rec:
            st.warning(rec)
        else:
            st.info(rec)
    
    st.markdown("---")
    
    # ========== DISCLAIMER ==========
    st.warning("""
    ⚠️ **Disclaimer:** This is AI-generated advice for educational purposes only. 
    Past performance does NOT guarantee future results. Please consult a SEBI-registered 
    financial advisor before making investment decisions. Invest at your own risk.
    """)


def main():
    """Main application."""
    
    # ========== TITLE ==========
    st.markdown('<h1 class="big-title">💰 AI Wealth Manager</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>🚀 Enter your details → Get AI-powered stock recommendations</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== USER INPUTS ==========
    st.subheader("📝 Enter Your Investment Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount
        investment_amount = st.number_input(
            "💰 Investment Amount (₹)",
            min_value=10000,
            max_value=100000000,
            value=500000,
            step=10000,
            help="How much money do you want to invest?"
        )
        st.caption(f"Selected: **{format_inr(investment_amount)}**")
        
        # Duration
        investment_years = st.slider(
            "⏳ Investment Duration (Years)",
            min_value=1,
            max_value=20,
            value=5,
            help="How long do you plan to stay invested?"
        )
    
    with col2:
        # Expected Return
        expected_return = st.slider(
            "📈 Expected Annual Return (%)",
            min_value=8,
            max_value=40,
            value=15,
            help="What annual return do you expect?"
        )
        
        # Risk Tolerance
        risk_options = {
            "very_safe": "🛡️ Very Safe (8-10% YOY) - Large-cap Blue-chips",
            "safe": "🔒 Safe (10-13% YOY) - Large + Mid caps",
            "moderate": "⚖️ Moderate (13-18% YOY) - Quality Mid-caps",
            "low_risk_growth": "📊 Growth (18-25% YOY) - High-growth stocks",
            "risky": "🔥 Risky (25-40% YOY) - Small-caps, High volatility"
        }
        
        risk_tolerance = st.selectbox(
            "🎯 Risk Tolerance",
            options=list(risk_options.keys()),
            format_func=lambda x: risk_options[x],
            index=2,
            help="Select your risk appetite"
        )
    
    # Validate return expectation
    is_valid, validation_msg = validate_yoy_return_expectation(risk_tolerance, expected_return)
    if not is_valid:
        st.warning(f"⚠️ {validation_msg}")
    else:
        st.info(f"✅ {validation_msg}")
    
    # Number of stocks
    num_stocks = st.slider(
        "📊 Number of Stocks to Recommend",
        min_value=3,
        max_value=15,
        value=8 if investment_amount >= 500000 else 5,
        help="More stocks = better diversification"
    )
    
    # Target calculation
    target_value = investment_amount * ((1 + expected_return/100) ** investment_years)
    
    st.markdown("---")
    
    # Summary
    st.subheader("📋 Your Investment Summary")
    
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Investment", format_inr(investment_amount))
    with sum_col2:
        st.metric("Target Return", f"{expected_return}% YOY")
    with sum_col3:
        st.metric("Duration", f"{investment_years} Years")
    with sum_col4:
        st.metric("Target Value", format_inr(target_value))
    
    st.markdown("---")
    
    # ========== ANALYZE BUTTON ==========
    analyze_clicked = st.button("🚀 ANALYZE & GET RECOMMENDATIONS", type="primary", use_container_width=True)
    
    if analyze_clicked:
        st.markdown("---")
        
        # Loading section
        st.subheader("⏳ Analyzing Market Data...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Running AI analysis..."):
            result, error = run_analysis(
                investment_amount=investment_amount,
                expected_return=expected_return,
                investment_years=investment_years,
                risk_tolerance=risk_tolerance,
                num_stocks=num_stocks,
                progress_bar=progress_bar,
                status_text=status_text
            )
        
        if error:
            st.error(f"❌ Error during analysis: {error}")
            st.info("💡 **Tip:** Try again or check your internet connection.")
        else:
            # Clear loading elements
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            display_results(result, investment_amount, expected_return, investment_years, risk_tolerance)


if __name__ == "__main__":
    main()
