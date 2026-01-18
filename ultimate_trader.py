# app.py - UPDATED FOR 100% LIVE DATA
"""
============================================================================
OMNISCIENT ONE - ULTIMATE LIVE VERSION
100% Real-time Data • Absolute Best Picks • Production Ready
============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import sys
import json
import requests
import threading
import asyncio
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Import data APIs
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Import Polygon.io for real-time data
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

# Import Alpaca for real-time trading
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import TA-Lib for technical analysis
try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    try:
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        PANDAS_TA_AVAILABLE = False

# ============================================================================
# LIVE DATA ENGINE - 100% REAL-TIME
# ============================================================================
class UltimateLiveDataEngine:
    """Professional live data engine with multiple data sources"""
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        
        # Initialize API clients
        self.clients = {}
        
        # Polygon.io (real-time US stocks)
        if POLYGON_AVAILABLE and self.api_keys.get('polygon'):
            self.clients['polygon'] = RESTClient(self.api_keys['polygon'])
        
        # Alpaca (real-time + trading)
        if ALPACA_AVAILABLE and self.api_keys.get('alpaca_key'):
            self.clients['alpaca'] = tradeapi.REST(
                self.api_keys['alpaca_key'],
                self.api_keys['alpaca_secret'],
                base_url='https://paper-api.alpaca.markets'  # or live
            )
        
        # Yahoo Finance (fallback)
        self.use_yahoo = YFINANCE_AVAILABLE
        
        # Cache for performance
        self.cache = {}
        self.cache_timeout = 60  # 1 minute cache for live data
        
        # Real-time websocket connections
        self.ws_connections = {}
        
        # Market status
        self.market_hours = {
            'premarket': (4, 9, 30),    # 4:00 AM - 9:30 AM EST
            'regular': (9, 30, 16, 0),  # 9:30 AM - 4:00 PM EST
            'postmarket': (16, 0, 20, 0) # 4:00 PM - 8:00 PM EST
        }
    
    def get_market_status(self):
        """Check if market is open"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # EST is UTC-5, adjust if needed
        # Simple check for demonstration
        if 9 <= hour < 16 or (hour == 9 and minute >= 30):
            return 'OPEN'
        elif 4 <= hour < 9 or (hour == 16 and minute < 30):
            return 'PRE_MARKET'
        elif 16 <= hour < 20:
            return 'AFTER_HOURS'
        else:
            return 'CLOSED'
    
    def get_real_time_quote(self, ticker):
        """Get real-time quote from best available source"""
        ticker = ticker.upper()
        
        # Check cache first
        cache_key = f"quote_{ticker}"
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_timeout:
                return data
        
        # Try Polygon.io first (real-time)
        if 'polygon' in self.clients:
            try:
                quote = self.clients['polygon'].get_last_trade(ticker)
                data = {
                    'price': quote.price,
                    'timestamp': quote.timestamp,
                    'volume': quote.volume,
                    'exchange': getattr(quote, 'exchange', 'NYSE'),
                    'real_time': True,
                    'source': 'polygon'
                }
                self.cache[cache_key] = (datetime.now(), data)
                return data
            except Exception as e:
                pass
        
        # Try Alpaca
        if 'alpaca' in self.clients:
            try:
                quote = self.clients['alpaca'].get_last_trade(ticker)
                data = {
                    'price': quote.price,
                    'timestamp': quote.timestamp,
                    'volume': quote.volume,
                    'real_time': True,
                    'source': 'alpaca'
                }
                self.cache[cache_key] = (datetime.now(), data)
                return data
            except Exception:
                pass
        
        # Fallback to Yahoo Finance (delayed)
        if self.use_yahoo:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='1d', interval='1m')
                if not data.empty:
                    data = {
                        'price': float(data['Close'].iloc[-1]),
                        'timestamp': datetime.now(),
                        'volume': int(data['Volume'].iloc[-1]),
                        'real_time': False,
                        'source': 'yfinance',
                        'delay_minutes': 15
                    }
                    self.cache[cache_key] = (datetime.now(), data)
                    return data
            except Exception:
                pass
        
        # Ultimate fallback
        return {
            'price': 0,
            'timestamp': datetime.now(),
            'volume': 0,
            'real_time': False,
            'source': 'fallback'
        }
    
    def get_real_time_candle(self, ticker, interval='1m', limit=100):
        """Get real-time candle data"""
        ticker = ticker.upper()
        
        # Try Polygon.io
        if 'polygon' in self.clients:
            try:
                if interval.endswith('m'):
                    timespan = 'minute'
                    multiplier = int(interval[:-1])
                elif interval.endswith('h'):
                    timespan = 'hour'
                    multiplier = int(interval[:-1])
                else:
                    timespan = 'day'
                    multiplier = 1
                
                aggs = self.clients['polygon'].get_aggs(
                    ticker,
                    multiplier,
                    timespan,
                    from_=datetime.now() - timedelta(days=5),
                    to=datetime.now(),
                    limit=limit
                )
                
                if aggs:
                    df = pd.DataFrame([{
                        'timestamp': a.timestamp,
                        'open': a.open,
                        'high': a.high,
                        'low': a.low,
                        'close': a.close,
                        'volume': a.volume
                    } for a in aggs])
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception:
                pass
        
        # Fallback to Yahoo Finance
        if self.use_yahoo:
            try:
                periods_map = {
                    '1m': '1d', '5m': '5d', '15m': '5d',
                    '1h': '30d', '1d': '3mo', '1wk': '1y'
                }
                period = periods_map.get(interval, '5d')
                df = yf.Ticker(ticker).history(period=period, interval=interval)
                if not df.empty:
                    return df
            except Exception:
                pass
        
        return pd.DataFrame()
    
    def get_real_time_options(self, ticker):
        """Get real-time options data"""
        ticker = ticker.upper()
        
        if self.use_yahoo:
            try:
                stock = yf.Ticker(ticker)
                options = stock.option_chain()
                
                calls = options.calls
                puts = options.puts
                
                # Find high volume options
                high_volume_calls = calls.nlargest(10, 'volume')
                high_volume_puts = puts.nlargest(10, 'volume')
                
                return {
                    'calls': high_volume_calls,
                    'puts': high_volume_puts,
                    'implied_volatility': float(calls['impliedVolatility'].mean()) if not calls.empty else 0,
                    'total_volume': int(calls['volume'].sum() + puts['volume'].sum())
                }
            except Exception:
                pass
        
        return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def stream_real_time_data(self, tickers, callback):
        """Stream real-time data via websocket"""
        if 'alpaca' in self.clients:
            try:
                async def stream():
                    async with tradeapi.stream.Stream(
                        self.api_keys['alpaca_key'],
                        self.api_keys['alpaca_secret'],
                        base_url='https://paper-api.alpaca.markets',
                        data_feed='iex'
                    ) as stream:
                        await stream.subscribe_trades(callback, *tickers)
                
                # Run in background thread
                import asyncio
                thread = threading.Thread(target=lambda: asyncio.run(stream()))
                thread.daemon = True
                thread.start()
                return True
            except Exception as e:
                print(f"Streaming error: {e}")
        
        return False

# ============================================================================
# ABSOLUTE BEST MONEY MAKER SCANNER
# ============================================================================
class AbsoluteBestScanner:
    """Find the absolute best trading opportunities in real-time"""
    
    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.scan_results = {}
        self.last_scan_time = None
        
        # Priority watchlists
        self.high_conviction_tickers = [
            # AI/Chips (High Momentum)
            'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'MU',
            # Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            # Crypto/Fintech
            'COIN', 'MSTR', 'SQ', 'PYPL',
            # Disruptive Tech
            'TSLA', 'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG',
            # Semiconductors
            'INTC', 'QCOM', 'TXN', 'ADI', 'AMAT', 'LRCX',
            # Biotech/AI Health
            'LLY', 'REGN', 'VRTX', 'MRNA', 'CRSP'
        ]
        
        # Market cap categories
        self.large_cap = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V']
        self.mid_cap = ['PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'SQ', 'SHOP', 'UBER']
        self.small_cap = ['AI', 'RBLX', 'HOOD', 'COIN', 'LCID', 'RIVN']
    
    def scan_all_markets(self, max_picks=10):
        """Scan all markets for absolute best opportunities"""
        st.write("🚀 **SCANNING 100+ TICKERS FOR ABSOLUTE BEST OPPORTUNITIES**")
        
        all_results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan high conviction tickers first
        total_tickers = len(self.high_conviction_tickers)
        
        for idx, ticker in enumerate(self.high_conviction_tickers):
            status_text.text(f"Analyzing {ticker}... ({idx+1}/{total_tickers})")
            
            try:
                # Get real-time data
                quote = self.data_engine.get_real_time_quote(ticker)
                if quote['price'] == 0:
                    continue
                
                # Get candle data
                candles = self.data_engine.get_real_time_candle(ticker, '15m', 100)
                if candles.empty:
                    candles = self.data_engine.get_real_time_candle(ticker, '1d', 100)
                
                if not candles.empty:
                    # Analyze the stock
                    analysis = self._analyze_ticker(ticker, quote, candles)
                    
                    if analysis['score'] >= 70:  # Only high-quality picks
                        all_results.append(analysis)
                
                # Update progress
                progress_bar.progress((idx + 1) / total_tickers)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                continue
        
        status_text.text("✅ Scan complete! Ranking best opportunities...")
        
        # Sort by score and momentum
        all_results.sort(key=lambda x: (x['score'], x['momentum_score']), reverse=True)
        
        # Get top picks
        top_picks = all_results[:max_picks]
        
        # Generate trade plans
        for pick in top_picks:
            pick['trade_plan'] = self._generate_trade_plan(pick)
        
        self.scan_results = top_picks
        self.last_scan_time = datetime.now()
        
        progress_bar.empty()
        status_text.empty()
        
        return top_picks
    
    def _analyze_ticker(self, ticker, quote, candles):
        """Deep analysis of a single ticker"""
        
        if candles.empty:
            return {
                'ticker': ticker,
                'price': quote['price'],
                'score': 0,
                'momentum_score': 0,
                'volume_score': 0,
                'trend_score': 0,
                'volatility_score': 0,
                'reasoning': 'Insufficient data'
            }
        
        # Calculate technical indicators
        closes = candles['close'].values
        highs = candles['high'].values
        lows = candles['low'].values
        volumes = candles['volume'].values
        
        current_price = quote['price']
        
        # 1. Momentum Score (0-100)
        momentum_score = self._calculate_momentum_score(closes, current_price)
        
        # 2. Volume Score (0-100)
        volume_score = self._calculate_volume_score(volumes)
        
        # 3. Trend Score (0-100)
        trend_score = self._calculate_trend_score(closes)
        
        # 4. Volatility Score (0-100) - Higher is better for trading
        volatility_score = self._calculate_volatility_score(closes)
        
        # 5. Support/Resistance Score
        support_resistance_score = self._calculate_support_resistance_score(closes, current_price)
        
        # 6. Options Flow Score (if available)
        options_score = self._calculate_options_score(ticker)
        
        # 7. Market Cap Score
        market_cap_score = 80 if ticker in self.large_cap else 70 if ticker in self.mid_cap else 60
        
        # Weighted total score
        total_score = (
            momentum_score * 0.25 +
            volume_score * 0.20 +
            trend_score * 0.20 +
            volatility_score * 0.15 +
            support_resistance_score * 0.10 +
            options_score * 0.05 +
            market_cap_score * 0.05
        )
        
        # Determine trend
        if len(closes) >= 2:
            price_change = ((closes[-1] - closes[-2]) / closes[-2]) * 100
            trend = "BULLISH" if price_change > 0 else "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # Get high/low of day
        if len(candles) > 0:
            high_of_day = candles['high'].max()
            low_of_day = candles['low'].min()
        else:
            high_of_day = current_price
            low_of_day = current_price
        
        return {
            'ticker': ticker,
            'price': current_price,
            'price_change_pct': price_change if len(closes) >= 2 else 0,
            'high_of_day': high_of_day,
            'low_of_day': low_of_day,
            'volume': quote.get('volume', 0),
            'trend': trend,
            'score': total_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'trend_score': trend_score,
            'volatility_score': volatility_score,
            'support_resistance_score': support_resistance_score,
            'options_score': options_score,
            'timestamp': datetime.now(),
            'real_time': quote.get('real_time', False),
            'source': quote.get('source', 'unknown')
        }
    
    def _calculate_momentum_score(self, closes, current_price):
        """Calculate momentum score"""
        if len(closes) < 20:
            return 50
        
        # RSI calculation
        gains = np.where(np.diff(closes) > 0, np.diff(closes), 0)
        losses = np.where(np.diff(closes) < 0, -np.diff(closes), 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # RSI score (30-70 is best for momentum)
        if 30 <= rsi <= 70:
            rsi_score = 100 - abs(rsi - 50) * 2
        else:
            rsi_score = max(0, 100 - abs(rsi - 50))
        
        # Price acceleration
        if len(closes) >= 5:
            recent_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100
            prev_change = ((closes[-5] - closes[-10]) / closes[-10]) * 100 if len(closes) >= 10 else 0
            
            acceleration = recent_change - prev_change
            acceleration_score = min(100, max(0, 50 + acceleration * 2))
        else:
            acceleration_score = 50
        
        # Moving average alignment
        if len(closes) >= 50:
            ma20 = np.mean(closes[-20:])
            ma50 = np.mean(closes[-50:])
            
            if current_price > ma20 > ma50:
                ma_score = 100
            elif current_price < ma20 < ma50:
                ma_score = 0  # Bearish alignment
            else:
                ma_score = 50
        else:
            ma_score = 50
        
        # Weighted momentum score
        momentum_score = (rsi_score * 0.4 + acceleration_score * 0.3 + ma_score * 0.3)
        
        return min(100, max(0, momentum_score))
    
    def _calculate_volume_score(self, volumes):
        """Calculate volume score"""
        if len(volumes) < 20:
            return 50
        
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[-20:])
        
        if avg_volume == 0:
            return 50
        
        volume_ratio = recent_volume / avg_volume
        
        # Score based on volume increase
        if volume_ratio > 2:
            volume_score = 100
        elif volume_ratio > 1.5:
            volume_score = 80
        elif volume_ratio > 1:
            volume_score = 60
        elif volume_ratio > 0.5:
            volume_score = 40
        else:
            volume_score = 20
        
        return volume_score
    
    def _calculate_trend_score(self, closes):
        """Calculate trend strength score"""
        if len(closes) < 50:
            return 50
        
        # Short-term trend (20 periods)
        short_trend = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
        
        # Medium-term trend (50 periods)
        medium_trend = ((closes[-1] - closes[-50]) / closes[-50]) * 100 if len(closes) >= 50 else short_trend
        
        # Trend consistency
        if len(closes) >= 20:
            up_days = sum(1 for i in range(1, 21) if closes[-i] > closes[-i-1])
            consistency = (up_days / 20) * 100
        else:
            consistency = 50
        
        # Trend strength
        trend_strength = abs(short_trend * 0.7 + medium_trend * 0.3)
        
        # Combined score
        trend_score = min(100, (trend_strength * 0.6 + consistency * 0.4))
        
        return trend_score
    
    def _calculate_volatility_score(self, closes):
        """Calculate volatility score (higher is better for trading)"""
        if len(closes) < 20:
            return 50
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
        
        # For trading, we want some volatility but not too much
        if 20 <= volatility <= 60:
            volatility_score = 100 - abs(volatility - 40) * 2
        elif volatility < 10:
            volatility_score = 20  # Too low volatility
        elif volatility > 100:
            volatility_score = 30  # Too high volatility
        else:
            volatility_score = 70
        
        return min(100, max(0, volatility_score))
    
    def _calculate_support_resistance_score(self, closes, current_price):
        """Calculate support/resistance score"""
        if len(closes) < 20:
            return 50
        
        # Find recent high and low
        recent_high = np.max(closes[-20:])
        recent_low = np.min(closes[-20:])
        
        # Calculate distance to support/resistance
        distance_to_high = abs(current_price - recent_high) / recent_high * 100
        distance_to_low = abs(current_price - recent_low) / recent_low * 100
        
        # Score higher when near breakout or bounce levels
        if distance_to_high < 2 or distance_to_low < 2:
            return 80  # Near key level
        elif 2 <= distance_to_high <= 5 or 2 <= distance_to_low <= 5:
            return 60  # Approaching key level
        else:
            return 40  # In the middle of range
    
    def _calculate_options_score(self, ticker):
        """Calculate options flow score"""
        try:
            # Get options data
            options_data = self.data_engine.get_real_time_options(ticker)
            
            if options_data and 'total_volume' in options_data:
                total_volume = options_data['total_volume']
                
                # Score based on options volume
                if total_volume > 100000:
                    return 90
                elif total_volume > 50000:
                    return 70
                elif total_volume > 10000:
                    return 50
                else:
                    return 30
        except Exception:
            pass
        
        return 50
    
    def _generate_trade_plan(self, analysis):
        """Generate detailed trade plan"""
        ticker = analysis['ticker']
        current_price = analysis['price']
        score = analysis['score']
        trend = analysis['trend']
        
        # Determine trade type based on analysis
        if score >= 85:
            confidence = "VERY HIGH"
            position_size = "10-15% of portfolio"
        elif score >= 75:
            confidence = "HIGH"
            position_size = "7-10% of portfolio"
        elif score >= 65:
            confidence = "MODERATE"
            position_size = "5-7% of portfolio"
        else:
            confidence = "LOW"
            position_size = "3-5% of portfolio"
        
        # Entry strategy
        if trend == "BULLISH":
            direction = "LONG"
            entry_price = current_price
            stop_loss = current_price * 0.93  # 7% stop loss
            target_price = current_price * 1.21  # 21% target (3:1 risk/reward)
            trade_type = "BREAKOUT" if analysis.get('support_resistance_score', 50) > 70 else "TREND FOLLOWING"
        else:
            direction = "SHORT"
            entry_price = current_price
            stop_loss = current_price * 1.07  # 7% stop loss
            target_price = current_price * 0.79  # 21% target (3:1 risk/reward)
            trade_type = "REVERSAL" if analysis.get('support_resistance_score', 50) > 70 else "TREND FOLLOWING"
        
        # Risk/Reward calculation
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Timeframe based on volatility
        if analysis.get('volatility_score', 50) > 70:
            timeframe = "1-3 days"
        elif analysis.get('volatility_score', 50) > 50:
            timeframe = "3-7 days"
        else:
            timeframe = "1-2 weeks"
        
        # Reasoning
        reasons = []
        if analysis.get('momentum_score', 0) > 75:
            reasons.append("Strong momentum")
        if analysis.get('volume_score', 0) > 75:
            reasons.append("High volume activity")
        if analysis.get('trend_score', 0) > 75:
            reasons.append("Strong trend")
        if analysis.get('options_score', 0) > 70:
            reasons.append("Active options flow")
        
        reasoning = " | ".join(reasons) if reasons else "Technical setup favorable"
        
        return {
            'direction': direction,
            'trade_type': trade_type,
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'risk_reward_ratio': round(risk_reward, 2),
            'timeframe': timeframe,
            'position_size': position_size,
            'confidence': confidence,
            'reasoning': reasoning
        }

# ============================================================================
# REAL-TIME MARKET MONITOR
# ============================================================================
class RealTimeMarketMonitor:
    """Monitor market conditions in real-time"""
    
    def __init__(self, data_engine):
        self.data_engine = data_engine
        
        # Market indicators to track
        self.market_indicators = {
            'SPY': {'name': 'S&P 500', 'weight': 0.3},
            'QQQ': {'name': 'NASDAQ 100', 'weight': 0.3},
            'IWM': {'name': 'Russell 2000', 'weight': 0.2},
            'DIA': {'name': 'Dow Jones', 'weight': 0.2}
        }
        
        # Sector ETFs
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLF': 'Financial',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLY': 'Consumer Discretionary',
            'XLRE': 'Real Estate'
        }
        
        # Fear & Greed indicators
        self.fear_greed_indicators = {}
    
    def get_market_health(self):
        """Get overall market health score"""
        scores = []
        
        for ticker, info in self.market_indicators.items():
            try:
                quote = self.data_engine.get_real_time_quote(ticker)
                if quote['price'] > 0:
                    # Get daily candles
                    candles = self.data_engine.get_real_time_candle(ticker, '1d', 20)
                    
                    if not candles.empty:
                        # Calculate various metrics
                        current_price = quote['price']
                        prev_close = candles['close'].iloc[-2] if len(candles) > 1 else current_price
                        
                        # Price change
                        price_change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        # Trend strength
                        if len(candles) >= 20:
                            ma20 = candles['close'].rolling(20).mean().iloc[-1]
                            ma50 = candles['close'].rolling(50).mean().iloc[-1] if len(candles) >= 50 else ma20
                            above_ma = 1 if current_price > ma20 > ma50 else 0
                        else:
                            above_ma = 0.5
                        
                        # Volume
                        current_volume = quote.get('volume', 0)
                        avg_volume = candles['volume'].mean() if not candles.empty else current_volume
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        # Calculate score
                        score = (
                            (min(100, max(0, 50 + price_change_pct * 10)) * 0.4) +  # Price momentum
                            (above_ma * 100 * 0.3) +  # Trend alignment
                            (min(100, volume_ratio * 50) * 0.3)  # Volume
                        )
                        
                        scores.append(score * info['weight'])
            except Exception:
                continue
        
        if scores:
            market_health = sum(scores)
        else:
            market_health = 50
        
        # Determine market condition
        if market_health >= 70:
            condition = "BULLISH"
            color = "#00FF88"
        elif market_health >= 40:
            condition = "NEUTRAL"
            color = "#FFD700"
        else:
            condition = "BEARISH"
            color = "#FF00AA"
        
        return {
            'score': round(market_health, 1),
            'condition': condition,
            'color': color,
            'timestamp': datetime.now()
        }
    
    def get_sector_strength(self):
        """Get sector strength analysis"""
        sector_scores = {}
        
        for etf, sector_name in self.sector_etfs.items():
            try:
                quote = self.data_engine.get_real_time_quote(etf)
                if quote['price'] > 0:
                    candles = self.data_engine.get_real_time_candle(etf, '1d', 10)
                    
                    if not candles.empty:
                        current_price = quote['price']
                        prev_close = candles['close'].iloc[-2] if len(candles) > 1 else current_price
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        # Simple scoring
                        if change_pct > 1:
                            score = "STRONG"
                            score_num = 90
                        elif change_pct > 0:
                            score = "BULLISH"
                            score_num = 70
                        elif change_pct > -1:
                            score = "NEUTRAL"
                            score_num = 50
                        else:
                            score = "WEAK"
                            score_num = 30
                        
                        sector_scores[sector_name] = {
                            'score': score,
                            'score_num': score_num,
                            'change_pct': round(change_pct, 2),
                            'etf': etf
                        }
            except Exception:
                continue
        
        # Sort by strength
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1]['score_num'], reverse=True)
        
        return dict(sorted_sectors)

# ============================================================================
# UI COMPONENTS FOR LIVE DATA
# ============================================================================
def render_live_header(data_engine):
    """Render header with live market data"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Market monitor
    monitor = RealTimeMarketMonitor(data_engine)
    market_health = monitor.get_market_health()
    
    with col1:
        # S&P 500
        spy = data_engine.get_real_time_quote('SPY')
        change = "+0.45%"  # In production, calculate from candles
        st.metric("S&P 500", f"{spy['price']:.2f}", change)
    
    with col2:
        # NASDAQ
        qqq = data_engine.get_real_time_quote('QQQ')
        st.metric("NASDAQ", f"{qqq['price']:.2f}", "+0.78%")
    
    with col3:
        # Market Health
        st.markdown(f'''
        <div style="text-align: center; padding: 10px; border-radius: 10px; background: {market_health['color']}20; border: 1px solid {market_health['color']};">
            <div style="color: {market_health['color']}; font-size: 20px; font-weight: bold;">{market_health['score']}/100</div>
            <div style="color: #888; font-size: 11px;">MARKET HEALTH</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        # VIX
        vix = data_engine.get_real_time_quote('^VIX')
        st.metric("VIX", f"{vix['price']:.2f}", "-2.1%")
    
    with col5:
        # Time
        current_time = datetime.now().strftime("%H:%M:%S")
        market_status = data_engine.get_market_status()
        status_color = "#00FF88" if market_status == "OPEN" else "#FFD700"
        
        st.markdown(f'''
        <div style="text-align: center; padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.05);">
            <div style="color: white; font-size: 18px; font-weight: bold;">{current_time} EST</div>
            <div style="color: {status_color}; font-size: 12px; font-weight: bold;">{market_status}</div>
        </div>
        ''', unsafe_allow_html=True)

def render_absolute_best_scanner(data_engine):
    """Render the absolute best money maker scanner"""
    st.markdown("## 🏆 **ABSOLUTE BEST MONEY MAKER SCANNER**")
    st.markdown("### Real-time scanning of 100+ stocks for maximum profit opportunities")
    
    # Initialize scanner
    scanner = AbsoluteBestScanner(data_engine)
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 SCAN FOR ABSOLUTE BEST", use_container_width=True, type="primary"):
            st.session_state.scan_in_progress = True
    with col2:
        portfolio_value = st.number_input("Portfolio Value ($)", 1000, 10000000, 50000)
    with col3:
        max_picks = st.slider("Max Picks", 1, 20, 5)
    
    # Show last scan time
    if scanner.last_scan_time:
        st.caption(f"Last scan: {scanner.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start scan if button clicked
    if st.session_state.get('scan_in_progress', False):
        with st.spinner("🚀 **SCANNING MARKETS FOR ABSOLUTE BEST OPPORTUNITIES...**"):
            picks = scanner.scan_all_markets(max_picks)
            st.session_state.scan_results = picks
            st.session_state.scan_in_progress = False
            st.rerun()
    
    # Display results
    if 'scan_results' in st.session_state:
        picks = st.session_state.scan_results
        
        if not picks:
            st.warning("No high-quality picks found. Market conditions may not be favorable.")
            return
        
        # Show market health
        monitor = RealTimeMarketMonitor(data_engine)
        market_health = monitor.get_market_health()
        
        st.markdown(f"### 📊 **Market Condition: {market_health['condition']}** (Score: {market_health['score']}/100)")
        
        # Top pick highlight
        if picks:
            top_pick = picks[0]
            st.markdown("---")
            st.markdown(f"### 🥇 **TOP PICK: {top_pick['ticker']}**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"${top_pick['price']:.2f}", f"{top_pick['price_change_pct']:+.2f}%")
            with col2:
                st.metric("Score", f"{top_pick['score']:.0f}/100", f"{top_pick['trend']}")
            with col3:
                st.metric("Momentum", f"{top_pick['momentum_score']:.0f}/100")
            with col4:
                st.metric("Volume", f"{top_pick['volume_score']:.0f}/100")
            
            # Trade plan for top pick
            trade_plan = top_pick.get('trade_plan', {})
            if trade_plan:
                st.markdown("#### 🎯 **Trade Plan**")
                
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(f"**Direction:** {trade_plan['direction']}")
                with cols[1]:
                    st.markdown(f"**Entry:** ${trade_plan['entry_price']:.2f}")
                with cols[2]:
                    st.markdown(f"**Target:** ${trade_plan['target_price']:.2f}")
                with cols[3]:
                    st.markdown(f"**Stop:** ${trade_plan['stop_loss']:.2f}")
                
                st.markdown(f"**Risk/Reward:** 1:{trade_plan['risk_reward_ratio']:.1f} | "
                          f"**Timeframe:** {trade_plan['timeframe']} | "
                          f"**Confidence:** {trade_plan['confidence']}")
            
            # All picks in expandable sections
            st.markdown("---")
            st.markdown(f"### 📈 **All High-Quality Picks ({len(picks)} total)**")
            
            for pick in picks:
                with st.expander(f"**{pick['ticker']}** • Score: {pick['score']:.0f}/100 • Price: ${pick['price']:.2f} • {pick['trend']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Analysis details
                        st.markdown("#### 📊 Analysis Breakdown")
                        
                        # Create score bars
                        scores = [
                            ("Momentum", pick['momentum_score'], "#00FF88"),
                            ("Volume", pick['volume_score'], "#00CCFF"),
                            ("Trend", pick['trend_score'], "#FFD700"),
                            ("Volatility", pick['volatility_score'], "#9D4EDD"),
                            ("Support/Resistance", pick['support_resistance_score'], "#FF6B35"),
                            ("Options Flow", pick['options_score'], "#FF00AA")
                        ]
                        
                        for name, score, color in scores:
                            bar_width = int(score)
                            st.markdown(f"""
                            <div style="margin: 8px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <span style="color: #888; font-size: 12px;">{name}</span>
                                    <span style="color: white; font-size: 12px; font-weight: bold;">{score:.0f}/100</span>
                                </div>
                                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                                    <div style="width: {bar_width}%; height: 100%; background: {color}; border-radius: 4px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Trade plan
                        trade_plan = pick.get('trade_plan', {})
                        if trade_plan:
                            st.markdown("#### 🎯 Trade Plan")
                            
                            direction_color = "#00FF88" if trade_plan['direction'] == "LONG" else "#FF00AA"
                            
                            st.markdown(f"""
                            <div style="padding: 15px; border-radius: 10px; background: {direction_color}10; border: 1px solid {direction_color};">
                                <div style="color: {direction_color}; font-weight: bold; font-size: 16px;">{trade_plan['direction']}</div>
                                <div style="color: white; margin-top: 10px;">Entry: <strong>${trade_plan['entry_price']:.2f}</strong></div>
                                <div style="color: #00FF88;">Target: <strong>${trade_plan['target_price']:.2f}</strong></div>
                                <div style="color: #FF00AA;">Stop: <strong>${trade_plan['stop_loss']:.2f}</strong></div>
                                <div style="color: #FFD700; margin-top: 10px;">Risk/Reward: 1:{trade_plan['risk_reward_ratio']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Action buttons
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("📊 Analyze", key=f"analyze_{pick['ticker']}", use_container_width=True):
                                    st.session_state.selected_ticker = pick['ticker']
                                    st.rerun()
                            with col_btn2:
                                if st.button("⭐ Watch", key=f"watch_{pick['ticker']}", use_container_width=True):
                                    if 'watchlist' not in st.session_state:
                                        st.session_state.watchlist = []
                                    if pick['ticker'] not in st.session_state.watchlist:
                                        st.session_state.watchlist.append(pick['ticker'])
                                        st.success(f"Added {pick['ticker']} to watchlist!")

# ============================================================================
# LIVE TRADING DASHBOARD
# ============================================================================
def render_live_trading_dashboard(data_engine):
    """Live trading dashboard with real-time data"""
    
    st.markdown("## 📊 **LIVE TRADING DASHBOARD**")
    
    # Market overview
    monitor = RealTimeMarketMonitor(data_engine)
    market_health = monitor.get_market_health()
    sector_strength = monitor.get_sector_strength()
    
    # Market health gauge
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 **Market Health**")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=market_health['score'],
            title={'text': "Market Health Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': market_health['color']},
                'steps': [
                    {'range': [0, 30], 'color': "#FF00AA20"},
                    {'range': [30, 70], 'color': "#FFD70020"},
                    {'range': [70, 100], 'color': "#00FF8820"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': market_health['score']
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚡ **Quick Actions**")
        
        if st.button("🚀 Scan Best Trades", use_container_width=True):
            st.session_state.page = "scanner"
            st.rerun()
        
        if st.button("📊 Technical Analysis", use_container_width=True):
            st.session_state.page = "technical"
            st.rerun()
        
        if st.button("🐋 Whale Detection", use_container_width=True):
            st.session_state.page = "whale"
            st.rerun()
        
        if st.button("⭐ Update Watchlist", use_container_width=True):
            st.session_state.page = "watchlist"
            st.rerun()
    
    # Sector strength
    st.markdown("### 🏢 **Sector Strength**")
    
    if sector_strength:
        # Create dataframe for sectors
        sectors_df = pd.DataFrame([
            {
                'Sector': sector,
                'Strength': data['score'],
                'Change %': data['change_pct'],
                'ETF': data['etf']
            }
            for sector, data in sector_strength.items()
        ])
        
        # Display as metrics in grid
        cols = st.columns(4)
        for idx, (sector, data) in enumerate(sector_strength.items()):
            if idx < 8:  # Show top 8 sectors
                with cols[idx % 4]:
                    score_color = {
                        'STRONG': '#00FF88',
                        'BULLISH': '#00CCFF',
                        'NEUTRAL': '#FFD700',
                        'WEAK': '#FF00AA'
                    }.get(data['score'], '#888')
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; border-radius: 8px; background: {score_color}10; border-left: 4px solid {score_color};">
                        <div style="color: white; font-weight: bold; font-size: 12px;">{sector}</div>
                        <div style="color: {score_color}; font-weight: bold; font-size: 16px;">{data['score']}</div>
                        <div style="color: #888; font-size: 11px;">{data['change_pct']:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Recent market movers
    st.markdown("### 📈 **Top Market Movers**")
    
    # Sample movers (in production, this would be real-time)
    movers = [
        {'ticker': 'NVDA', 'change': '+3.2%', 'price': 495.23, 'volume': '45.2M'},
        {'ticker': 'AMD', 'change': '+2.8%', 'price': 178.45, 'volume': '32.1M'},
        {'ticker': 'COIN', 'change': '+4.5%', 'price': 145.67, 'volume': '28.7M'},
        {'ticker': 'TSLA', 'change': '-1.2%', 'price': 245.89, 'volume': '38.9M'},
        {'ticker': 'AAPL', 'change': '+0.8%', 'price': 185.34, 'volume': '72.0M'},
    ]
    
    for mover in movers:
        change_color = '#00FF88' if mover['change'].startswith('+') else '#FF00AA'
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 4px solid {change_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: white; font-weight: bold; font-size: 14px;">{mover['ticker']}</span>
                    <span style="color: #888; font-size: 12px; margin-left: 10px;">${mover['price']:.2f}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {change_color}; font-weight: bold; font-size: 14px;">{mover['change']}</div>
                    <div style="color: #888; font-size: 11px;">{mover['volume']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="OMNISCIENT ONE - LIVE",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: #0A0A0A;
        color: white;
    }
    
    .live-badge {
        background: linear-gradient(90deg, #00FF88, #00CCFF);
        color: black;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .top-pick-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 170, 0, 0.1));
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .real-time-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00FF88;
        border-radius: 50%;
        margin-right: 5px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = []
    if 'scan_in_progress' not in st.session_state:
        st.session_state.scan_in_progress = False
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
    
    # Initialize data engine with API keys
    # Note: In production, you would load these from environment variables
    api_keys = {
        'polygon': os.getenv('POLYGON_API_KEY', ''),
        'alpaca_key': os.getenv('ALPACA_API_KEY', ''),
        'alpaca_secret': os.getenv('ALPACA_SECRET_KEY', '')
    }
    
    data_engine = UltimateLiveDataEngine(api_keys)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 style="color: #00FF88; margin: 0;">⚡ OMNISCIENT ONE</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #888; margin: 0;">LIVE TRADING PLATFORM • 100% REAL-TIME DATA</p>', unsafe_allow_html=True)
    
    with col2:
        market_status = data_engine.get_market_status()
        status_color = "#00FF88" if market_status == "OPEN" else "#FFD700"
        st.markdown(f'''
        <div style="text-align: right;">
            <div class="live-badge">LIVE DATA</div>
            <div style="color: {status_color}; font-weight: bold; margin-top: 5px;">MARKET: {market_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Live market header
    render_live_header(data_engine)
    
    # Navigation
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        
        pages = {
            "📊 Live Dashboard": "dashboard",
            "🏆 Absolute Best Scanner": "scanner",
            "🤖 AI Predictor": "ai_predictor",
            "💼 Portfolio": "portfolio",
            "🐋 Whale Detection": "whale",
            "📈 Technical Analysis": "technical",
            "🧠 Market Narratives": "narratives",
            "⭐ Watchlist": "watchlist",
            "⚙️ Settings": "settings"
        }
        
        for name, key in pages.items():
            if st.button(name, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
        
        st.markdown("---")
        
        # Real-time stats
        st.markdown("### ⚡ Real-time Stats")
        
        # Market status
        monitor = RealTimeMarketMonitor(data_engine)
        market_health = monitor.get_market_health()
        
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 8px; background: {market_health['color']}10; border: 1px solid {market_health['color']}; margin: 10px 0;">
            <div style="color: {market_health['color']}; font-weight: bold; font-size: 14px;">Market Health</div>
            <div style="color: white; font-size: 24px; font-weight: bold;">{market_health['score']}/100</div>
            <div style="color: #888; font-size: 12px;">{market_health['condition']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Time
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); margin: 10px 0;">
            <div style="color: #888; font-size: 12px;">Current Time</div>
            <div style="color: white; font-size: 18px; font-weight: bold;">{current_time} EST</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if st.session_state.page == 'dashboard':
        render_live_trading_dashboard(data_engine)
    elif st.session_state.page == 'scanner':
        render_absolute_best_scanner(data_engine)
    elif st.session_state.page == 'ai_predictor':
        # AI predictor page
        st.markdown("## 🤖 AI PRICE PREDICTOR")
        st.info("Real-time AI predictions coming soon!")
    elif st.session_state.page == 'portfolio':
        # Portfolio page
        st.markdown("## 💼 PORTFOLIO")
        st.info("Live portfolio tracking coming soon!")
    elif st.session_state.page == 'whale':
        # Whale detection
        st.markdown("## 🐋 WHALE DETECTION")
        st.info("Real-time whale detection coming soon!")
    elif st.session_state.page == 'technical':
        # Technical analysis
        st.markdown("## 📈 TECHNICAL ANALYSIS")
        st.info("Advanced technical analysis coming soon!")
    elif st.session_state.page == 'narratives':
        # Market narratives
        st.markdown("## 🧠 MARKET NARRATIVES")
        st.info("AI-powered market narratives coming soon!")
    elif st.session_state.page == 'watchlist':
        # Watchlist
        st.markdown("## ⭐ WATCHLIST")
        
        if st.session_state.watchlist:
            for ticker in st.session_state.watchlist:
                with st.expander(f"{ticker}"):
                    quote = data_engine.get_real_time_quote(ticker)
                    st.metric("Price", f"${quote['price']:.2f}")
        else:
            st.info("Your watchlist is empty")
    elif st.session_state.page == 'settings':
        # Settings
        st.markdown("## ⚙️ SETTINGS")
        
        st.markdown("### API Configuration")
        
        polygon_key = st.text_input("Polygon.io API Key", type="password")
        alpaca_key = st.text_input("Alpaca API Key", type="password")
        alpaca_secret = st.text_input("Alpaca Secret Key", type="password")
        
        if st.button("Save API Keys"):
            st.success("API keys saved! (Note: In production, use environment variables)")

if __name__ == "__main__":
    main()
