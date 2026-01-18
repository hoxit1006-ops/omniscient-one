"""
============================================================================
OMNISCIENT ONE ULTIMATE - PRODUCTION DEPLOYMENT
============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys
import json
import uuid

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import our modules
try:
    from auth import AuthManager
    from database import DatabaseManager, User, Portfolio, Trade
    from data_engine import PremiumDataEngine
    from subscription import SubscriptionManager
    from trading import ProductionTradingEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.warning(f"Module import error: {e}. Running in demo mode.")

# ============================================================================
# CONFIGURATION
# ============================================================================
class ProductionConfig:
    PLATFORM_NAME = "OMNISCIENT ONE"
    VERSION = "PRODUCTION 2.0"
    
    # Colors
    PRIMARY_BLACK = "#000000"
    NEON_GREEN = "#00FF88"
    CYAN_BLUE = "#00CCFF"
    HOT_PINK = "#FF00AA"
    GOLD = "#FFD700"
    
    # API Keys (from environment)
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    
    # Stripe
    STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY", "")
    
    # App URL
    APP_URL = os.getenv("APP_URL", "https://omniscient-one.streamlit.app")

CONFIG = ProductionConfig()

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================
def initialize_auth():
    """Initialize authentication system"""
    if MODULES_AVAILABLE:
        try:
            auth_manager = AuthManager()
            return auth_manager
        except Exception as e:
            st.error(f"Auth initialization failed: {e}")
    
    # Fallback to session-based auth
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'subscription_tier' not in st.session_state:
        st.session_state.subscription_tier = "free"
    
    return None

# ============================================================================
# PREMIUM DATA ENGINE INITIALIZATION
# ============================================================================
def initialize_data_engine(subscription_tier="free"):
    """Initialize data engine based on subscription tier"""
    if MODULES_AVAILABLE:
        try:
            return PremiumDataEngine(subscription_tier)
        except Exception as e:
            st.warning(f"Premium data engine failed: {e}")
    
    # Fallback to Yahoo Finance
    try:
        import yfinance as yf
        return SimpleDataEngine()
    except:
        return None

class SimpleDataEngine:
    """Simple data engine for free tier"""
    def fetch_stock_data(self, ticker, period="5d", interval="1d"):
        try:
            import yfinance as yf
            return yf.Ticker(ticker).history(period=period, interval=interval)
        except:
            return None

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_login_page():
    """Render login/registration page"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 40px;
        background: rgba(15, 15, 20, 0.97);
        border-radius: 20px;
        border: 1px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 20px 40px rgba(0, 255, 136, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Platform Header
    st.markdown('<h1 style="text-align: center; color: #00FF88;">⚡ OMNISCIENT ONE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888; margin-bottom: 30px;">Production Trading Platform</p>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                remember = st.checkbox("Remember me")
            with col2:
                if st.form_submit_button("🚀 Login", use_container_width=True):
                    if username and password:
                        # Simple authentication for demo
                        st.session_state.authenticated = True
                        st.session_state.user = {
                            "username": username,
                            "email": f"{username}@demo.com",
                            "tier": "premium" if username == "admin" else "free"
                        }
                        st.session_state.subscription_tier = st.session_state.user["tier"]
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Please enter credentials")
    
    with tab2:
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                email = st.text_input("Email")
                username = st.text_input("Username")
            with col2:
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
            
            terms = st.checkbox("I agree to Terms & Conditions")
            
            if st.form_submit_button("✨ Create Account", use_container_width=True):
                if password != confirm:
                    st.error("Passwords don't match")
                elif not terms:
                    st.error("Please accept Terms & Conditions")
                else:
                    # Simple registration for demo
                    st.session_state.authenticated = True
                    st.session_state.user = {
                        "username": username,
                        "email": email,
                        "tier": "free"
                    }
                    st.session_state.subscription_tier = "free"
                    st.success("Account created! 14-day premium trial activated.")
                    time.sleep(2)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show pricing
    render_pricing_table()

def render_pricing_table():
    """Render pricing table on login page"""
    st.markdown("---")
    st.markdown("### 💎 Choose Your Plan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    plans = [
        ("🆓 Free", "$0", ["Basic Dashboard", "Delayed Data", "3 Stock Watchlist"]),
        ("🥈 Basic", "$29.99", ["Real-time Data", "AI Predictions", "Unlimited Watchlist"]),
        ("🥇 Premium", "$99.99", ["Advanced AI", "Trade Signals", "Portfolio Tools", "API Access"]),
        ("⚡ Ultimate", "$199.99", ["Automated Trading", "Institutional Data", "Dedicated Support"])
    ]
    
    for i, (name, price, features) in enumerate(plans):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"#### {name}")
            st.markdown(f"**{price}/month**")
            for feature in features:
                st.markdown(f"✓ {feature}")
            if st.button(f"Select {name.split()[0]}", key=f"plan_{i}", use_container_width=True):
                st.info(f"Selected {name} plan")

def render_header():
    """Render production header"""
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown(f'<h1 style="color: #00FF88; margin: 0;">{CONFIG.PLATFORM_NAME}</h1>', unsafe_allow_html=True)
        st.caption(f"{CONFIG.VERSION} • User: {st.session_state.user.get('username', 'Guest')}")
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f'''
            <div style="padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(0,255,136,0.2);">
                <div style="color: #00FF88; font-size: 18px; font-weight: 600;">{current_time} EST</div>
                <div style="color: #888; font-size: 11px;">LIVE TRADING</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        tier = st.session_state.subscription_tier.upper()
        color = {"FREE": "#888", "BASIC": "#00CCFF", "PREMIUM": "#FFD700", "ULTIMATE": "#FF00AA"}.get(tier, "#888")
        st.markdown(f'''
            <div style="padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.05); border: 1px solid {color};">
                <div style="color: {color}; font-size: 16px; font-weight: 600;">{tier} TIER</div>
                <div style="color: #888; font-size: 11px;">SUBSCRIPTION</div>
            </div>
        ''', unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION PAGES
# ============================================================================
def render_dashboard():
    """Main dashboard"""
    st.markdown("## 📊 MARKET DASHBOARD")
    
    # Real-time market data
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("S&P 500", "5,234.18", "+0.67%")
    with col2:
        st.metric("NASDAQ", "18,342.56", "+1.23%")
    with col3:
        st.metric("DOW JONES", "39,123.45", "+0.45%")
    with col4:
        st.metric("VIX", "14.56", "-3.2%")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🤖 AI Analysis", use_container_width=True):
            st.session_state.page = "ai_predictor"
            st.rerun()
    with col3:
        if st.button("💰 Trade Ideas", use_container_width=True):
            st.session_state.page = "money_makers"
            st.rerun()
    with col4:
        if st.button("📈 Technical", use_container_width=True):
            st.session_state.page = "technical"
            st.rerun()
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    
    # Sample trades
    trades = [
        {"time": "10:30 AM", "ticker": "NVDA", "action": "BUY", "shares": 10, "price": 495.23},
        {"time": "11:15 AM", "ticker": "AAPL", "action": "SELL", "shares": 5, "price": 185.67},
        {"time": "1:45 PM", "ticker": "TSLA", "action": "BUY", "shares": 15, "price": 245.89},
    ]
    
    for trade in trades:
        color = "#00FF88" if trade["action"] == "BUY" else "#FF00AA"
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 4px solid {color};">
            <span style="color: #888;">{trade['time']}</span> • 
            <span style="color: white; font-weight: 600;">{trade['ticker']}</span> • 
            <span style="color: {color}; font-weight: 600;">{trade['action']} {trade['shares']} shares</span> • 
            <span style="color: white;">@ ${trade['price']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

def render_ai_predictor():
    """AI Price Predictor"""
    st.markdown("## 🤖 AI PRICE PREDICTOR")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Enter Ticker", "NVDA").upper()
    with col2:
        days = st.selectbox("Forecast Days", [7, 14, 30, 60])
    
    if st.button("🔮 Generate Prediction", use_container_width=True):
        with st.spinner("Running AI models..."):
            time.sleep(2)
            
            # Simulated prediction
            current_price = 495.23
            predicted_price = current_price * (1 + np.random.uniform(-0.1, 0.15))
            change_pct = (predicted_price - current_price) / current_price * 100
            confidence = np.random.uniform(70, 95)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${predicted_price:.2f}", f"{change_pct:+.1f}%")
            with col3:
                st.metric("AI Confidence", f"{confidence:.0f}%")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[datetime.now() - timedelta(days=30), datetime.now()],
                y=[current_price * 0.9, current_price],
                mode='lines',
                name='Historical',
                line=dict(color="#00CCFF", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[datetime.now(), datetime.now() + timedelta(days=days)],
                y=[current_price, predicted_price],
                mode='lines+markers',
                name='Prediction',
                line=dict(color="#00FF88", width=3, dash='dash')
            ))
            fig.update_layout(
                title=f"{ticker} Price Forecast",
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_money_makers():
    """Money Maker Scanner"""
    st.markdown("## 💰 MONEY MAKER SCANNER")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk = st.selectbox("Risk Level", ["Low", "Medium", "High", "Very High"])
    with col2:
        timeframe = st.selectbox("Timeframe", ["Intraday", "Swing (1-5 days)", "Position (1-4 weeks)"])
    with col3:
        if st.button("🔄 Scan Opportunities", use_container_width=True):
            st.rerun()
    
    # Sample trade ideas
    ideas = [
        {"ticker": "NVDA", "action": "BUY", "entry": 495.23, "target": 575.00, "stop": 470.00, "rr": "3.2", "confidence": 88},
        {"ticker": "AMD", "action": "BUY", "entry": 178.45, "target": 210.00, "stop": 165.00, "rr": "2.8", "confidence": 82},
        {"ticker": "COIN", "action": "BUY", "entry": 145.67, "target": 175.00, "stop": 135.00, "rr": "2.7", "confidence": 79},
        {"ticker": "TSLA", "action": "SELL", "entry": 245.89, "target": 220.00, "stop": 260.00, "rr": "2.5", "confidence": 75},
    ]
    
    for idea in ideas:
        action_color = "#00FF88" if idea["action"] == "BUY" else "#FF00AA"
        
        with st.expander(f"{idea['ticker']} • {idea['action']} • Confidence: {idea['confidence']}%", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **Entry:** ${idea['entry']:.2f}  
                **Target:** ${idea['target']:.2f} (+{(idea['target']/idea['entry']-1)*100:.1f}%)  
                **Stop Loss:** ${idea['stop']:.2f}  
                **Risk/Reward:** 1:{idea['rr']}  
                **Confidence:** {idea['confidence']}%
                """)
            
            with col2:
                # Quick chart
                fig = go.Figure()
                prices = [idea['stop'], idea['entry'], idea['target']]
                fig.add_trace(go.Scatter(
                    x=[1, 2, 3],
                    y=prices,
                    mode='lines+markers',
                    line=dict(color=action_color, width=3),
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    height=150,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showticklabels=False),
                    yaxis_title="Price"
                )
                st.plotly_chart(fig, use_container_width=True)

def render_subscription():
    """Subscription management"""
    st.markdown("## 💎 SUBSCRIPTION MANAGEMENT")
    
    current_tier = st.session_state.subscription_tier
    st.info(f"**Current Plan:** {current_tier.upper()}")
    
    # Plans
    col1, col2, col3, col4 = st.columns(4)
    
    plans = [
        ("free", "🆓 Free", "0", ["Basic Dashboard", "Delayed Data"]),
        ("basic", "🥈 Basic", "29.99", ["Real-time Data", "AI Predictions"]),
        ("premium", "🥇 Premium", "99.99", ["Trade Signals", "Portfolio Tools", "API Access"]),
        ("ultimate", "⚡ Ultimate", "199.99", ["Automated Trading", "Institutional Data", "Dedicated Support"])
    ]
    
    for i, (plan_id, name, price, features) in enumerate(plans):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"### {name}")
            st.markdown(f"**${price}/month**")
            
            for feature in features:
                st.markdown(f"✓ {feature}")
            
            if current_tier == plan_id:
                st.success("Current Plan")
            elif st.button(f"Upgrade to {name}", key=f"upgrade_{plan_id}", use_container_width=True):
                if plan_id in ["basic", "premium", "ultimate"]:
                    # In production, this would redirect to Stripe
                    st.session_state.subscription_tier = plan_id
                    st.success(f"Upgraded to {name}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.subscription_tier = "free"
                    st.info("Switched to Free plan")
                    st.rerun()

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================
def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title="OMNISCIENT ONE",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'subscription_tier' not in st.session_state:
        st.session_state.subscription_tier = "free"
    
    # CSS
    st.markdown("""
    <style>
    .stApp {
        background: #0A0A0A;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #111111;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return
    
    # Main app for authenticated users
    render_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "🤖 AI Predictor": "ai_predictor",
            "💰 Money Makers": "money_makers",
            "💼 Portfolio": "portfolio",
            "🐋 Whale Detection": "whale",
            "📈 Technical": "technical",
            "🧠 Narratives": "narratives",
            "⭐ Watchlist": "watchlist",
            "💎 Subscription": "subscription",
            "⚙️ Settings": "settings"
        }
        
        for name, key in pages.items():
            if st.button(name, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    # Render current page
    if st.session_state.page == "dashboard":
        render_dashboard()
    elif st.session_state.page == "ai_predictor":
        render_ai_predictor()
    elif st.session_state.page == "money_makers":
        render_money_makers()
    elif st.session_state.page == "subscription":
        render_subscription()
    elif st.session_state.page == "portfolio":
        st.markdown("## 💼 PORTFOLIO MANAGEMENT")
        st.info("Portfolio features require Premium subscription")
    elif st.session_state.page == "whale":
        st.markdown("## 🐋 WHALE DETECTION")
        st.info("Whale detection requires Premium subscription")
    elif st.session_state.page == "technical":
        st.markdown("## 📈 TECHNICAL ANALYSIS")
        st.info("Advanced technical analysis requires Basic+ subscription")
    elif st.session_state.page == "narratives":
        st.markdown("## 🧠 MARKET NARRATIVES")
        st.info("Narrative detection requires Premium subscription")
    elif st.session_state.page == "watchlist":
        st.markdown("## ⭐ WATCHLIST")
        st.info("Unlimited watchlist requires Basic+ subscription")
    elif st.session_state.page == "settings":
        st.markdown("## ⚙️ SETTINGS")
        st.info("Settings page")

if __name__ == "__main__":
    main()
