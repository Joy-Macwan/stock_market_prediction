"""
Custom CSS Styles for Stock Market Prediction System
"""

CUSTOM_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        color: #00d4ff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    .main-header p {
        color: #b8b8b8;
        font-size: 1.1rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        color: #888;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }
    
    .metric-card .change {
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .positive {
        color: #00c853 !important;
    }
    
    .negative {
        color: #ff1744 !important;
    }
    
    .neutral {
        color: #ffa726 !important;
    }
    
    /* Stock Card */
    .stock-card {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stock-card:hover {
        border-left-color: #00ff88;
        background: linear-gradient(145deg, #252540, #1a1a2e);
    }
    
    .stock-card h4 {
        color: #00d4ff;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Recommendation Badge */
    .recommendation-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .buy-badge {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #000;
        box-shadow: 0 4px 20px rgba(0, 200, 83, 0.4);
    }
    
    .sell-badge {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: #fff;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.4);
    }
    
    .hold-badge {
        background: linear-gradient(135deg, #ffa726, #ffca28);
        color: #000;
        box-shadow: 0 4px 20px rgba(255, 167, 38, 0.4);
    }
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(145deg, #0f3460, #16213e);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #00d4ff;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #00d4ff;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-box p {
        color: #b8b8b8;
        line-height: 1.6;
    }
    
    /* Table Styles */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .styled-table th {
        background: linear-gradient(145deg, #0f3460, #16213e);
        color: #00d4ff;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
    
    .styled-table td {
        padding: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #fff;
    }
    
    .styled-table tr:hover {
        background: rgba(0, 212, 255, 0.05);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: #000 !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Input Styles */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #fff !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888 !important;
        font-weight: 500;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: #000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    /* Alert Boxes */
    .success-alert {
        background: linear-gradient(145deg, rgba(0, 200, 83, 0.1), rgba(0, 200, 83, 0.05));
        border: 1px solid #00c853;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #00c853;
    }
    
    .warning-alert {
        background: linear-gradient(145deg, rgba(255, 167, 38, 0.1), rgba(255, 167, 38, 0.05));
        border: 1px solid #ffa726;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #ffa726;
    }
    
    .danger-alert {
        background: linear-gradient(145deg, rgba(255, 23, 68, 0.1), rgba(255, 23, 68, 0.05));
        border: 1px solid #ff1744;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #ff1744;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(0, 212, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00b8e6;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-card .value {
            font-size: 1.5rem;
        }
    }
</style>
"""


def get_metric_card_html(title, value, change=None, change_pct=None, prefix="₹", suffix=""):
    """Generate HTML for a metric card"""
    change_class = "positive" if change and change >= 0 else "negative" if change else "neutral"
    change_symbol = "↑" if change and change >= 0 else "↓" if change else ""
    
    change_html = ""
    if change is not None:
        change_html = f"""
        <div class="change {change_class}">
            {change_symbol} {prefix}{abs(change):,.2f} ({change_pct:+.2f}%)
        </div>
        """
    
    return f"""
    <div class="metric-card fade-in">
        <h3>{title}</h3>
        <div class="value">{prefix}{value:,.2f}{suffix}</div>
        {change_html}
    </div>
    """


def get_stock_card_html(symbol, price, change, change_pct, high, low, volume):
    """Generate HTML for a stock info card"""
    change_class = "positive" if change >= 0 else "negative"
    change_symbol = "▲" if change >= 0 else "▼"
    
    return f"""
    <div class="stock-card fade-in">
        <h4>{symbol}</h4>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #fff;">₹{price:,.2f}</div>
                <div class="{change_class}" style="font-size: 1.1rem;">
                    {change_symbol} ₹{abs(change):,.2f} ({change_pct:+.2f}%)
                </div>
            </div>
            <div style="text-align: right; color: #888;">
                <div>H: ₹{high:,.2f}</div>
                <div>L: ₹{low:,.2f}</div>
                <div>Vol: {volume:,}</div>
            </div>
        </div>
    </div>
    """


def get_recommendation_badge_html(recommendation, confidence):
    """Generate HTML for recommendation badge"""
    badge_class = {
        'BUY': 'buy-badge',
        'SELL': 'sell-badge',
        'HOLD': 'hold-badge'
    }.get(recommendation.upper(), 'hold-badge')
    
    return f"""
    <div style="text-align: center; margin: 2rem 0;">
        <div class="recommendation-badge {badge_class} pulse">
            {recommendation}
        </div>
        <div style="margin-top: 1rem; color: #888;">
            Confidence: {confidence:.1f}%
        </div>
    </div>
    """


def get_info_box_html(title, content):
    """Generate HTML for info box"""
    return f"""
    <div class="info-box fade-in">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """


def get_alert_html(message, alert_type='success'):
    """Generate HTML for alert message"""
    return f"""
    <div class="{alert_type}-alert fade-in">
        {message}
    </div>
    """
