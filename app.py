"""
Stock Market Prediction & Wealth Management System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import (
    APP_NAME, NIFTY_50_STOCKS, INDICES, TIME_PERIODS, 
    RETURN_OPTIONS, RISK_PROFILES, COLORS, CHART_THEME
)
from stock_data import (
    fetch_stock_data, get_live_price, get_available_stocks,
    get_available_indices, get_sector_data, calculate_returns
)
from technical_analysis import (
    add_all_technical_indicators, generate_trading_signals,
    get_support_resistance, get_trend_analysis, calculate_risk_metrics
)
from prediction_models import StockPredictor, ensemble_prediction
from wealth_calculator import (
    WealthCalculator, PortfolioManager, TaxCalculator,
    calculate_cagr, calculate_xirr
)
from styles import (
    CUSTOM_CSS, get_metric_card_html, get_stock_card_html,
    get_recommendation_badge_html, get_info_box_html, get_alert_html
)

# Page Configuration
st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Stock Market Prediction System</h1>
        <p>AI-Powered Stock Analysis, Prediction & Wealth Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "📊 Stock Analysis", "🔮 Price Prediction", 
         "💰 Wealth Calculator", "📈 Portfolio Manager", "📋 Technical Analysis",
         "🎯 Stock Screener", "📰 Market Overview"]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Stock Analysis":
        show_stock_analysis()
    elif page == "🔮 Price Prediction":
        show_price_prediction()
    elif page == "💰 Wealth Calculator":
        show_wealth_calculator()
    elif page == "📈 Portfolio Manager":
        show_portfolio_manager()
    elif page == "📋 Technical Analysis":
        show_technical_analysis()
    elif page == "🎯 Stock Screener":
        show_stock_screener()
    elif page == "📰 Market Overview":
        show_market_overview()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>📈 Stock Market Prediction System | Built with ❤️ using Streamlit</p>
        <p style="font-size: 0.8rem; color: #666;">
            Disclaimer: This is for educational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_dashboard():
    """Display main dashboard"""
    st.subheader("📊 Market Dashboard")
    
    # Market indices overview
    col1, col2, col3, col4 = st.columns(4)
    
    indices_data = [
        ("NIFTY 50", True),
        ("NIFTY BANK", True),
        ("SENSEX", True),
        ("NIFTY IT", True)
    ]
    
    for col, (index_name, is_index) in zip([col1, col2, col3, col4], indices_data):
        with col:
            price_data = get_live_price(index_name, is_index=is_index)
            st.markdown(get_metric_card_html(
                index_name,
                price_data['price'],
                price_data['change'],
                price_data['change_pct'],
                prefix=""
            ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Gainers and Losers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Top Gainers")
        gainers_data = []
        for stock in NIFTY_50_STOCKS[:10]:
            price_data = get_live_price(stock)
            gainers_data.append({
                'Symbol': stock,
                'Price': price_data['price'],
                'Change': price_data['change'],
                'Change %': price_data['change_pct']
            })
        
        gainers_df = pd.DataFrame(gainers_data)
        gainers_df = gainers_df.sort_values('Change %', ascending=False).head(5)
        
        for _, row in gainers_df.iterrows():
            change_class = "positive" if row['Change %'] >= 0 else "negative"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                        border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #00d4ff; font-weight: 600;">{row['Symbol']}</span>
                <span style="color: #fff;">₹{row['Price']:,.2f}</span>
                <span class="{change_class}">{row['Change %']:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📉 Top Losers")
        losers_df = pd.DataFrame(gainers_data)
        losers_df = losers_df.sort_values('Change %', ascending=True).head(5)
        
        for _, row in losers_df.iterrows():
            change_class = "positive" if row['Change %'] >= 0 else "negative"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                        border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #00d4ff; font-weight: 600;">{row['Symbol']}</span>
                <span style="color: #fff;">₹{row['Price']:,.2f}</span>
                <span class="{change_class}">{row['Change %']:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stock Search
    st.markdown("### 🔍 Quick Stock Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stock = st.selectbox(
            "Select a stock to view quick analysis",
            NIFTY_50_STOCKS,
            key="dashboard_stock"
        )
    
    with col2:
        analyze_btn = st.button("Analyze", key="quick_analyze")
    
    if analyze_btn or selected_stock:
        with st.spinner("Fetching data..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            df = fetch_stock_data(selected_stock, start_date, end_date, is_index=False)
            
            if df is not None and not df.empty:
                price_data = get_live_price(selected_stock)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(get_metric_card_html(
                        "Current Price",
                        price_data['price'],
                        price_data['change'],
                        price_data['change_pct']
                    ), unsafe_allow_html=True)
                
                with col2:
                    returns = calculate_returns(df)
                    st.markdown(get_metric_card_html(
                        "1 Month Return",
                        returns.get('20d_return', 0),
                        prefix="",
                        suffix="%"
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(get_metric_card_html(
                        "52W High",
                        df['high'].max(),
                        prefix="₹"
                    ), unsafe_allow_html=True)
                
                with col4:
                    st.markdown(get_metric_card_html(
                        "52W Low",
                        df['low'].min(),
                        prefix="₹"
                    ), unsafe_allow_html=True)
                
                # Price Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=selected_stock
                ))
                
                fig.update_layout(
                    title=f"{selected_stock} Price Chart",
                    template=CHART_THEME,
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_stock_analysis():
    """Display detailed stock analysis"""
    st.subheader("📊 Stock Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock = st.selectbox(
            "Select Stock",
            NIFTY_50_STOCKS,
            key="analysis_stock"
        )
    
    with col2:
        period = st.selectbox(
            "Time Period",
            list(TIME_PERIODS.keys()),
            index=4,  # Default to 1 Year
            key="analysis_period"
        )
    
    with col3:
        is_index = st.checkbox("Is Index?", value=False)
    
    if st.button("📈 Analyze Stock", key="analyze_btn"):
        with st.spinner("Analyzing..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=TIME_PERIODS[period])
            
            df = fetch_stock_data(stock, start_date, end_date, is_index=is_index)
            
            if df is not None and not df.empty:
                # Add technical indicators
                df = add_all_technical_indicators(df)
                
                # Price Overview
                st.markdown("### 📈 Price Overview")
                
                price_data = get_live_price(stock, is_index=is_index)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(get_stock_card_html(
                        stock,
                        price_data['price'],
                        price_data['change'],
                        price_data['change_pct'],
                        price_data['high'],
                        price_data['low'],
                        price_data['volume']
                    ), unsafe_allow_html=True)
                
                with col2:
                    returns = calculate_returns(df)
                    st.metric("1D Return", f"{returns.get('1d_return', 0):.2f}%")
                    st.metric("5D Return", f"{returns.get('5d_return', 0):.2f}%")
                
                with col3:
                    st.metric("20D Return", f"{returns.get('20d_return', 0):.2f}%")
                    st.metric("60D Return", f"{returns.get('60d_return', 0):.2f}%")
                
                with col4:
                    st.metric("252D Return", f"{returns.get('252d_return', 0):.2f}%")
                    volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility", f"{volatility:.2f}%")
                
                st.markdown("---")
                
                # Candlestick Chart with indicators
                st.markdown("### 📊 Price Chart with Indicators")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=['Price', 'RSI', 'MACD']
                )
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ), row=1, col=1)
                
                # Moving Averages
                if 'SMA_20' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_20'],
                        name='SMA 20', line=dict(color='orange', width=1)
                    ), row=1, col=1)
                
                if 'SMA_50' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_50'],
                        name='SMA 50', line=dict(color='blue', width=1)
                    ), row=1, col=1)
                
                # RSI
                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['RSI'],
                        name='RSI', line=dict(color='purple', width=1)
                    ), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD'],
                        name='MACD', line=dict(color='blue', width=1)
                    ), row=3, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD_Signal'],
                        name='Signal', line=dict(color='orange', width=1)
                    ), row=3, col=1)
                
                fig.update_layout(
                    template=CHART_THEME,
                    height=700,
                    xaxis_rangeslider_visible=False,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Support and Resistance
                st.markdown("### 🎯 Support & Resistance Levels")
                
                sr_levels = get_support_resistance(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Resistance Levels**")
                    st.write(f"R3: ₹{sr_levels['resistance_3']:,.2f}")
                    st.write(f"R2: ₹{sr_levels['resistance_2']:,.2f}")
                    st.write(f"R1: ₹{sr_levels['resistance_1']:,.2f}")
                
                with col2:
                    st.markdown("**Support Levels**")
                    st.write(f"S1: ₹{sr_levels['support_1']:,.2f}")
                    st.write(f"S2: ₹{sr_levels['support_2']:,.2f}")
                    st.write(f"S3: ₹{sr_levels['support_3']:,.2f}")
                
                st.markdown(f"**Pivot Point**: ₹{sr_levels['pivot']:,.2f}")
                
                # Trend Analysis
                st.markdown("### 📈 Trend Analysis")
                
                trend = get_trend_analysis(df)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_color = "positive" if 'BULL' in trend['overall_trend'] else "negative"
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Overall Trend</h4>
                        <p class="{trend_color}" style="font-size: 1.5rem; font-weight: 700;">
                            {trend['overall_trend']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Short-term Trend", trend['short_term'])
                    st.metric("Medium-term Trend", trend['medium_term'])
                
                with col3:
                    st.metric("Trend Strength", trend['strength'])
                    st.metric("MA Alignment", trend['ma_alignment'])
                
                # Risk Metrics
                st.markdown("### ⚠️ Risk Metrics")
                
                risk = calculate_risk_metrics(df)
                
                if risk:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Annual Return", f"{risk['annual_return']}%")
                    with col2:
                        st.metric("Volatility", f"{risk['volatility']}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{risk['sharpe_ratio']}")
                    with col4:
                        st.metric("Max Drawdown", f"{risk['max_drawdown']}%")
            else:
                st.error("Unable to fetch data. Please try again.")


def show_price_prediction():
    """Display price prediction module"""
    st.subheader("🔮 Price Prediction")
    
    st.markdown("""
    <div class="info-box">
        <h4>AI-Powered Stock Price Prediction</h4>
        <p>Our prediction model uses multiple machine learning algorithms including Random Forest, 
        Gradient Boosting, and Time Series analysis to forecast future stock prices. 
        The ensemble approach combines predictions from different models for improved accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock = st.selectbox(
            "Select Stock",
            NIFTY_50_STOCKS,
            key="pred_stock"
        )
    
    with col2:
        prediction_days = st.slider(
            "Prediction Days",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
    
    with col3:
        is_index = st.checkbox("Is Index?", value=False, key="pred_is_index")
    
    if st.button("🔮 Generate Prediction", key="predict_btn"):
        with st.spinner("Training models and generating predictions..."):
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years of data
            
            df = fetch_stock_data(stock, start_date, end_date, is_index=is_index)
            
            if df is not None and not df.empty:
                # Initialize predictor
                predictor = StockPredictor()
                
                # Train models
                st.markdown("### 📊 Model Training Results")
                
                with st.expander("View Model Performance", expanded=True):
                    results = predictor.train_models(df)
                    
                    if 'error' not in results:
                        # Model comparison
                        model_data = []
                        for model_name, metrics in results.items():
                            if model_name != 'best_model' and isinstance(metrics, dict):
                                model_data.append({
                                    'Model': model_name,
                                    'RMSE': metrics['rmse'],
                                    'MAE': metrics['mae'],
                                    'R² Score': metrics['r2_score'],
                                    'MAPE (%)': metrics['mape']
                                })
                        
                        model_df = pd.DataFrame(model_data)
                        st.dataframe(model_df, use_container_width=True)
                        
                        st.success(f"✅ Best Model: **{results['best_model']}**")
                
                # Generate predictions
                st.markdown("### 📈 Price Predictions")
                
                predictions = ensemble_prediction(df, prediction_days)
                
                if 'Ensemble' in predictions:
                    ensemble_pred = predictions['Ensemble']
                    
                    # Current price and prediction summary
                    current_price = df['close'].iloc[-1]
                    final_pred = ensemble_pred['Predicted_Price'].iloc[-1]
                    pred_change = ((final_pred / current_price) - 1) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"₹{current_price:,.2f}")
                    
                    with col2:
                        st.metric(
                            f"Predicted ({prediction_days}D)",
                            f"₹{final_pred:,.2f}",
                            f"{pred_change:+.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Upper Bound",
                            f"₹{ensemble_pred['Upper_Bound'].iloc[-1]:,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Lower Bound",
                            f"₹{ensemble_pred['Lower_Bound'].iloc[-1]:,.2f}"
                        )
                    
                    # Prediction Chart
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=df['close'].iloc[-60:],
                        name='Historical',
                        line=dict(color='#00d4ff', width=2)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=ensemble_pred['Date'],
                        y=ensemble_pred['Predicted_Price'],
                        name='Predicted',
                        line=dict(color='#00ff88', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=ensemble_pred['Date'],
                        y=ensemble_pred['Upper_Bound'],
                        name='Upper Bound',
                        line=dict(color='rgba(0,255,136,0.3)', width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=ensemble_pred['Date'],
                        y=ensemble_pred['Lower_Bound'],
                        name='Lower Bound',
                        fill='tonexty',
                        line=dict(color='rgba(0,255,136,0.3)', width=0),
                        fillcolor='rgba(0,255,136,0.1)'
                    ))
                    
                    fig.update_layout(
                        title=f"{stock} Price Prediction",
                        template=CHART_THEME,
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Price (₹)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction Table
                    st.markdown("### 📋 Detailed Predictions")
                    
                    pred_display = ensemble_pred.copy()
                    pred_display['Date'] = pred_display['Date'].dt.strftime('%Y-%m-%d')
                    pred_display = pred_display.round(2)
                    
                    st.dataframe(pred_display, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendation")
                    
                    recommendation = predictor.get_recommendation(df, ensemble_pred)
                    
                    st.markdown(get_recommendation_badge_html(
                        recommendation['recommendation'],
                        recommendation['confidence']
                    ), unsafe_allow_html=True)
                    
                    st.markdown("**Analysis Factors:**")
                    for reason in recommendation['reasons']:
                        st.write(f"• {reason}")
                    
            else:
                st.error("Unable to fetch data. Please try again.")


def show_wealth_calculator():
    """Display wealth calculator module"""
    st.subheader("💰 Wealth Calculator")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Future Value", "💵 SIP Calculator", "🎯 Goal Planner",
        "👴 Retirement", "📊 Compare"
    ])
    
    with tab1:
        st.markdown("### Calculate Future Value of Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            principal = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                max_value=100000000,
                value=100000,
                step=10000
            )
            
            rate = st.select_slider(
                "Expected Annual Return (%)",
                options=RETURN_OPTIONS,
                value=12
            )
        
        with col2:
            years = st.slider(
                "Investment Period (Years)",
                min_value=1,
                max_value=40,
                value=10
            )
            
            compound_freq = st.selectbox(
                "Compounding Frequency",
                ["Monthly", "Quarterly", "Half-Yearly", "Yearly"],
                index=0
            )
        
        freq_map = {"Monthly": 12, "Quarterly": 4, "Half-Yearly": 2, "Yearly": 1}
        
        if st.button("Calculate Future Value", key="fv_btn"):
            result = WealthCalculator.calculate_future_value(
                principal, rate, years, freq_map[compound_freq]
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Future Value", f"₹{result['future_value']:,.2f}")
            
            with col2:
                st.metric("Interest Earned", f"₹{result['total_interest']:,.2f}")
            
            with col3:
                st.metric("Growth Multiple", f"{result['growth_multiplier']:.2f}x")
            
            # Growth chart
            schedule = WealthCalculator.generate_investment_schedule(principal, rate, years)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=schedule['Year'],
                y=schedule['Total Invested'],
                name='Invested',
                marker_color='#00d4ff'
            ))
            
            fig.add_trace(go.Bar(
                x=schedule['Year'],
                y=schedule['Gains'],
                name='Gains',
                marker_color='#00ff88'
            ))
            
            fig.update_layout(
                title="Investment Growth Over Time",
                template=CHART_THEME,
                barmode='stack',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### SIP Returns Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_sip = st.number_input(
                "Monthly SIP Amount (₹)",
                min_value=500,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            sip_rate = st.select_slider(
                "Expected Annual Return (%)",
                options=RETURN_OPTIONS,
                value=12,
                key="sip_rate"
            )
        
        with col2:
            sip_years = st.slider(
                "Investment Period (Years)",
                min_value=1,
                max_value=40,
                value=15,
                key="sip_years"
            )
        
        if st.button("Calculate SIP Returns", key="sip_btn"):
            result = WealthCalculator.calculate_sip_returns(monthly_sip, sip_rate, sip_years)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Investment", f"₹{result['total_invested']:,.2f}")
            
            with col2:
                st.metric("Future Value", f"₹{result['future_value']:,.2f}")
            
            with col3:
                st.metric("Wealth Gained", f"₹{result['wealth_gained']:,.2f}")
            
            with col4:
                st.metric("Total Returns", f"{result['absolute_return']:.1f}%")
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Invested Amount', 'Wealth Gained'],
                values=[result['total_invested'], result['wealth_gained']],
                hole=0.6,
                marker_colors=['#00d4ff', '#00ff88']
            )])
            
            fig.update_layout(
                title="Investment Breakdown",
                template=CHART_THEME,
                height=400,
                annotations=[dict(text=f"₹{result['future_value']:,.0f}", x=0.5, y=0.5, 
                                font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Goal-Based Investment Planner")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.number_input(
                "Target Amount (₹)",
                min_value=10000,
                max_value=100000000,
                value=1000000,
                step=50000
            )
            
            goal_years = st.slider(
                "Time to Goal (Years)",
                min_value=1,
                max_value=30,
                value=10,
                key="goal_years"
            )
        
        with col2:
            expected_return = st.select_slider(
                "Expected Annual Return (%)",
                options=RETURN_OPTIONS,
                value=12,
                key="goal_rate"
            )
        
        if st.button("Calculate Required Investment", key="goal_btn"):
            result = WealthCalculator.calculate_goal_based_investment(
                target, goal_years, expected_return
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(get_info_box_html(
                    "Required Lumpsum Investment",
                    f"₹{result['required_lumpsum']:,.2f}"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(get_info_box_html(
                    "Required Monthly SIP",
                    f"₹{result['required_monthly_sip']:,.2f}"
                ), unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Retirement Planning Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_age = st.number_input(
                "Current Age",
                min_value=18,
                max_value=60,
                value=30
            )
            
            retirement_age = st.number_input(
                "Retirement Age",
                min_value=45,
                max_value=70,
                value=60
            )
            
            monthly_expense = st.number_input(
                "Current Monthly Expense (₹)",
                min_value=10000,
                max_value=1000000,
                value=50000,
                step=5000
            )
        
        with col2:
            inflation = st.slider(
                "Expected Inflation (%)",
                min_value=3,
                max_value=10,
                value=6
            )
            
            life_expectancy = st.slider(
                "Life Expectancy",
                min_value=70,
                max_value=100,
                value=85
            )
        
        if st.button("Calculate Retirement Corpus", key="retire_btn"):
            result = WealthCalculator.calculate_retirement_corpus(
                current_age, retirement_age, monthly_expense, inflation, life_expectancy
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Years to Retirement", result['years_to_retire'])
                st.metric("Years in Retirement", result['years_in_retirement'])
            
            with col2:
                st.metric("Current Monthly Expense", f"₹{result['current_monthly_expense']:,.2f}")
                st.metric("Expense at Retirement", f"₹{result['expense_at_retirement']:,.2f}")
            
            with col3:
                st.metric("Required Corpus", f"₹{result['corpus_needed']:,.2f}")
            
            # Calculate SIP needed
            goal_result = WealthCalculator.calculate_goal_based_investment(
                result['corpus_needed'], result['years_to_retire'], 12
            )
            
            st.markdown(get_alert_html(
                f"💡 To accumulate this corpus, you need to invest approximately "
                f"₹{goal_result['required_monthly_sip']:,.2f} per month (assuming 12% returns)",
                "success"
            ), unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### Compare: Lumpsum vs SIP")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_amount = st.number_input(
                "Total Amount to Invest (₹)",
                min_value=10000,
                max_value=10000000,
                value=500000,
                step=50000
            )
        
        with col2:
            compare_rate = st.select_slider(
                "Expected Annual Return (%)",
                options=RETURN_OPTIONS,
                value=12,
                key="compare_rate"
            )
            
            compare_years = st.slider(
                "Investment Period (Years)",
                min_value=1,
                max_value=30,
                value=10,
                key="compare_years"
            )
        
        if st.button("Compare Options", key="compare_btn"):
            result = WealthCalculator.calculate_lumpsum_vs_sip(
                total_amount, compare_rate, compare_years
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Lumpsum Investment")
                st.metric("Future Value", f"₹{result['lumpsum']['future_value']:,.2f}")
                st.metric("Interest Earned", f"₹{result['lumpsum']['total_interest']:,.2f}")
            
            with col2:
                st.markdown("#### SIP Investment")
                st.metric("Future Value", f"₹{result['sip']['future_value']:,.2f}")
                st.metric("Wealth Gained", f"₹{result['sip']['wealth_gained']:,.2f}")
            
            st.markdown(get_alert_html(
                f"💡 **{result['better_option']}** generates ₹{abs(result['difference']):,.2f} more wealth!",
                "success"
            ), unsafe_allow_html=True)


def show_portfolio_manager():
    """Display portfolio management module"""
    st.subheader("📈 Portfolio Manager")
    
    tab1, tab2, tab3 = st.tabs(["📊 My Portfolio", "⚖️ Asset Allocation", "📈 Performance"])
    
    with tab1:
        st.markdown("### Add Holdings to Your Portfolio")
        
        # Initialize session state for portfolio
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            add_stock = st.selectbox("Select Stock", NIFTY_50_STOCKS, key="port_stock")
        
        with col2:
            add_qty = st.number_input("Quantity", min_value=1, value=10)
        
        with col3:
            add_price = st.number_input("Buy Price (₹)", min_value=1.0, value=100.0)
        
        with col4:
            st.write("")
            st.write("")
            if st.button("➕ Add"):
                current_price = get_live_price(add_stock)['price']
                st.session_state.portfolio.append({
                    'symbol': add_stock,
                    'quantity': add_qty,
                    'buy_price': add_price,
                    'current_price': current_price
                })
                st.success(f"Added {add_qty} shares of {add_stock}")
        
        if st.session_state.portfolio:
            st.markdown("### Current Holdings")
            
            holdings_data = []
            for holding in st.session_state.portfolio:
                invested = holding['quantity'] * holding['buy_price']
                current = holding['quantity'] * holding['current_price']
                gain = current - invested
                gain_pct = (gain / invested) * 100
                
                holdings_data.append({
                    'Symbol': holding['symbol'],
                    'Quantity': holding['quantity'],
                    'Buy Price': f"₹{holding['buy_price']:,.2f}",
                    'Current Price': f"₹{holding['current_price']:,.2f}",
                    'Invested': f"₹{invested:,.2f}",
                    'Current Value': f"₹{current:,.2f}",
                    'Gain/Loss': f"₹{gain:,.2f}",
                    'Return %': f"{gain_pct:+.2f}%"
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
            
            # Portfolio Summary
            pm = PortfolioManager()
            summary = pm.calculate_portfolio_value(st.session_state.portfolio)
            
            st.markdown("### Portfolio Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"₹{summary['total_invested']:,.2f}")
            
            with col2:
                st.metric("Current Value", f"₹{summary['current_value']:,.2f}")
            
            with col3:
                st.metric(
                    "Total Gain/Loss",
                    f"₹{summary['total_gains']:,.2f}",
                    f"{summary['returns_percent']:+.2f}%"
                )
            
            with col4:
                status = "🟢 Profit" if summary['is_profit'] else "🔴 Loss"
                st.metric("Status", status)
            
            # Portfolio Pie Chart
            if holdings_data:
                fig = go.Figure(data=[go.Pie(
                    labels=[h['Symbol'] for h in holdings_data],
                    values=[st.session_state.portfolio[i]['quantity'] * 
                           st.session_state.portfolio[i]['current_price'] 
                           for i in range(len(st.session_state.portfolio))],
                    hole=0.5
                )])
                
                fig.update_layout(
                    title="Portfolio Composition",
                    template=CHART_THEME,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio = []
                st.rerun()
    
    with tab2:
        st.markdown("### Recommended Asset Allocation")
        
        risk_profile = st.selectbox(
            "Select Your Risk Profile",
            list(RISK_PROFILES.keys())
        )
        
        pm = PortfolioManager()
        allocation = pm.get_asset_allocation(risk_profile)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {risk_profile} Investor")
            for asset, pct in allocation.items():
                st.write(f"**{asset.title()}**: {pct}%")
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.5,
                marker_colors=['#00d4ff', '#00ff88', '#ffa726', '#ff5252']
            )])
            
            fig.update_layout(
                title="Asset Allocation",
                template=CHART_THEME,
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Portfolio Performance Analysis")
        
        st.info("Add stocks to your portfolio in the 'My Portfolio' tab to see performance analysis.")


def show_technical_analysis():
    """Display technical analysis module"""
    st.subheader("📋 Technical Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock = st.selectbox(
            "Select Stock",
            NIFTY_50_STOCKS,
            key="ta_stock"
        )
    
    with col2:
        is_index = st.checkbox("Is Index?", value=False, key="ta_is_index")
    
    if st.button("🔍 Run Technical Analysis", key="ta_btn"):
        with st.spinner("Analyzing..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            df = fetch_stock_data(stock, start_date, end_date, is_index=is_index)
            
            if df is not None and not df.empty:
                df = add_all_technical_indicators(df)
                
                # Current Technical Values
                st.markdown("### 📊 Current Technical Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi:.2f}", rsi_status)
                
                with col2:
                    macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
                    macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
                    macd_status = "Bullish" if macd > macd_signal else "Bearish"
                    st.metric("MACD", f"{macd:.2f}", macd_status)
                
                with col3:
                    adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
                    adx_status = "Strong Trend" if adx > 25 else "Weak Trend"
                    st.metric("ADX", f"{adx:.2f}", adx_status)
                
                with col4:
                    vol = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 0
                    st.metric("Volatility", f"{vol:.2f}%")
                
                st.markdown("---")
                
                # Trading Signals
                st.markdown("### 📡 Trading Signals")
                
                signals = generate_trading_signals(df)
                
                col1, col2, col3 = st.columns(3)
                
                signal_value = signals['Signal'].iloc[-1] if 'Signal' in signals.columns else 0
                
                with col1:
                    if signal_value > 0:
                        st.markdown(get_recommendation_badge_html("BUY", 70), unsafe_allow_html=True)
                    elif signal_value < 0:
                        st.markdown(get_recommendation_badge_html("SELL", 70), unsafe_allow_html=True)
                    else:
                        st.markdown(get_recommendation_badge_html("HOLD", 50), unsafe_allow_html=True)
                
                with col2:
                    strength = signals['Signal_Strength'].iloc[-1] if 'Signal_Strength' in signals.columns else 0
                    st.metric("Signal Strength", f"{strength:.0f}/5")
                
                with col3:
                    combined = signals['Combined_Signal'].iloc[-1] if 'Combined_Signal' in signals.columns else 0
                    st.metric("Combined Score", f"{combined:.2f}")
                
                # Individual Signal Analysis
                st.markdown("### 📈 Individual Indicator Signals")
                
                signal_data = []
                
                if 'RSI' in df.columns:
                    rsi_val = df['RSI'].iloc[-1]
                    if rsi_val < 30:
                        signal_data.append(("RSI", "BUY", "Oversold condition"))
                    elif rsi_val > 70:
                        signal_data.append(("RSI", "SELL", "Overbought condition"))
                    else:
                        signal_data.append(("RSI", "NEUTRAL", "Normal range"))
                
                if 'MACD' in df.columns:
                    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                        signal_data.append(("MACD", "BUY", "Bullish crossover"))
                    else:
                        signal_data.append(("MACD", "SELL", "Bearish crossover"))
                
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                        signal_data.append(("Moving Avg", "BUY", "Golden crossover"))
                    else:
                        signal_data.append(("Moving Avg", "SELL", "Death crossover"))
                
                if 'BB_Upper' in df.columns:
                    if df['close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
                        signal_data.append(("Bollinger", "BUY", "Below lower band"))
                    elif df['close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                        signal_data.append(("Bollinger", "SELL", "Above upper band"))
                    else:
                        signal_data.append(("Bollinger", "NEUTRAL", "Within bands"))
                
                if 'Stoch_K' in df.columns:
                    stoch = df['Stoch_K'].iloc[-1]
                    if stoch < 20:
                        signal_data.append(("Stochastic", "BUY", "Oversold"))
                    elif stoch > 80:
                        signal_data.append(("Stochastic", "SELL", "Overbought"))
                    else:
                        signal_data.append(("Stochastic", "NEUTRAL", "Normal"))
                
                signal_df = pd.DataFrame(signal_data, columns=['Indicator', 'Signal', 'Reason'])
                st.dataframe(signal_df, use_container_width=True)
                
                # Charts
                st.markdown("### 📊 Technical Charts")
                
                tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
                
                with tab1:
                    if 'RSI' in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                        fig.add_hline(y=70, line_dash="dash", line_color="red")
                        fig.add_hline(y=30, line_dash="dash", line_color="green")
                        fig.update_layout(template=CHART_THEME, height=400, title="RSI (14)")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'MACD' in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
                        if 'MACD_Histogram' in df.columns:
                            colors = ['green' if v >= 0 else 'red' for v in df['MACD_Histogram']]
                            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], 
                                                name='Histogram', marker_color=colors))
                        fig.update_layout(template=CHART_THEME, height=400, title="MACD")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'BB_Upper' in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], 
                                                name='Upper Band', line=dict(dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], 
                                                name='Middle Band'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], 
                                                name='Lower Band', line=dict(dash='dash')))
                        fig.update_layout(template=CHART_THEME, height=400, title="Bollinger Bands")
                        st.plotly_chart(fig, use_container_width=True)


def show_stock_screener():
    """Display stock screener module"""
    st.subheader("🎯 Stock Screener")
    
    st.markdown("### Filter Stocks by Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = st.number_input("Min Price (₹)", value=0, min_value=0)
        max_price = st.number_input("Max Price (₹)", value=10000, min_value=0)
    
    with col2:
        min_return = st.slider("Min Return % (1M)", -50, 50, 0)
        rsi_filter = st.selectbox("RSI Filter", ["All", "Oversold (<30)", "Overbought (>70)"])
    
    with col3:
        sector_filter = st.multiselect(
            "Sectors",
            list(get_sector_data().keys()),
            default=list(get_sector_data().keys())
        )
    
    if st.button("🔍 Screen Stocks", key="screen_btn"):
        with st.spinner("Screening stocks..."):
            results = []
            sectors = get_sector_data()
            
            # Get stocks from selected sectors
            selected_stocks = []
            for sector in sector_filter:
                selected_stocks.extend(sectors.get(sector, []))
            
            selected_stocks = list(set(selected_stocks))  # Remove duplicates
            
            progress_bar = st.progress(0)
            
            for i, stock in enumerate(selected_stocks):
                try:
                    price_data = get_live_price(stock)
                    
                    # Apply filters
                    if not (min_price <= price_data['price'] <= max_price):
                        continue
                    
                    if price_data['change_pct'] < min_return:
                        continue
                    
                    # Fetch data for RSI
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    df = fetch_stock_data(stock, start_date, end_date)
                    
                    if df is not None and not df.empty:
                        df = add_all_technical_indicators(df)
                        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                        
                        # RSI filter
                        if rsi_filter == "Oversold (<30)" and rsi >= 30:
                            continue
                        elif rsi_filter == "Overbought (>70)" and rsi <= 70:
                            continue
                        
                        results.append({
                            'Symbol': stock,
                            'Price': price_data['price'],
                            'Change %': price_data['change_pct'],
                            'RSI': round(rsi, 2),
                            'Volume': price_data['volume']
                        })
                
                except Exception as e:
                    continue
                
                progress_bar.progress((i + 1) / len(selected_stocks))
            
            progress_bar.empty()
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Change %', ascending=False)
                
                st.markdown(f"### Found {len(results)} stocks matching criteria")
                
                st.dataframe(results_df, use_container_width=True)
                
                # Top picks
                st.markdown("### 🏆 Top 5 Picks")
                
                for _, row in results_df.head(5).iterrows():
                    change_class = "positive" if row['Change %'] >= 0 else "negative"
                    rsi_status = "🟢" if 30 <= row['RSI'] <= 70 else "🔴"
                    
                    st.markdown(f"""
                    <div class="stock-card">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <h4>{row['Symbol']}</h4>
                                <p style="color: #888;">RSI: {row['RSI']} {rsi_status}</p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.5rem; font-weight: 700;">₹{row['Price']:,.2f}</div>
                                <div class="{change_class}">{row['Change %']:+.2f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No stocks found matching the criteria. Try adjusting filters.")


def show_market_overview():
    """Display market overview"""
    st.subheader("📰 Market Overview")
    
    # Market Status
    st.markdown("### 📊 Indian Market Indices")
    
    indices_list = [
        ("NIFTY 50", True),
        ("NIFTY BANK", True),
        ("NIFTY IT", True),
        ("NIFTY PHARMA", True),
        ("NIFTY AUTO", True),
        ("NIFTY METAL", True)
    ]
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for i, (index_name, is_index) in enumerate(indices_list):
        with cols[i % 3]:
            price_data = get_live_price(index_name, is_index=is_index)
            st.markdown(get_metric_card_html(
                index_name,
                price_data['price'],
                price_data['change'],
                price_data['change_pct'],
                prefix=""
            ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sector Performance
    st.markdown("### 📈 Sector Performance")
    
    sectors = get_sector_data()
    sector_performance = []
    
    for sector, stocks in sectors.items():
        sector_returns = []
        for stock in stocks[:3]:  # Sample 3 stocks per sector
            try:
                price_data = get_live_price(stock)
                sector_returns.append(price_data['change_pct'])
            except:
                continue
        
        if sector_returns:
            avg_return = np.mean(sector_returns)
            sector_performance.append({
                'Sector': sector,
                'Avg Return %': round(avg_return, 2)
            })
    
    sector_df = pd.DataFrame(sector_performance)
    sector_df = sector_df.sort_values('Avg Return %', ascending=True)
    
    # Horizontal bar chart
    fig = go.Figure()
    
    colors = ['#00c853' if x >= 0 else '#ff1744' for x in sector_df['Avg Return %']]
    
    fig.add_trace(go.Bar(
        y=sector_df['Sector'],
        x=sector_df['Avg Return %'],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Sector-wise Performance",
        template=CHART_THEME,
        height=400,
        xaxis_title="Return %",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Market Heatmap
    st.markdown("### 🗺️ Market Heatmap")
    
    heatmap_data = []
    for stock in NIFTY_50_STOCKS[:25]:
        try:
            price_data = get_live_price(stock)
            heatmap_data.append({
                'Stock': stock,
                'Return': price_data['change_pct']
            })
        except:
            continue
    
    if heatmap_data:
        # Create a simple grid representation
        cols = st.columns(5)
        
        for i, data in enumerate(heatmap_data):
            with cols[i % 5]:
                change_class = "positive" if data['Return'] >= 0 else "negative"
                bg_color = f"rgba(0, 200, 83, {min(abs(data['Return'])/10, 0.5)})" if data['Return'] >= 0 else f"rgba(255, 23, 68, {min(abs(data['Return'])/10, 0.5)})"
                
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; 
                            text-align: center; margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; color: #fff;">{data['Stock']}</div>
                    <div class="{change_class}">{data['Return']:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
