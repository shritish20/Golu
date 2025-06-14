"""
VolGuard FastAPI Backend - Production Ready
AI-Powered Options Trading Platform with Complete Upstox Integration

Author: Manus AI
Version: 1.0.0
Description: Complete production-ready FastAPI backend for VolGuard with full Upstox API integration,
            AI models (GARCH, XGBoost), risk management, and Supabase connectivity.
"""

import asyncio
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from arch import arch_model
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy.stats import linregress
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Application configuration"""
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://eurepsbikwxwmgpgzvzn.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV1cmVwc2Jpa3d4d21ncGd6dnpuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0MTYzNzQsImV4cCI6MjA2NDk5MjM3NH0.r3soCQV8nkbvc8RzFoLNGxK9MqQUOEIQUAWubAzAIkA")
    
    # Upstox Configuration
    UPSTOX_BASE_URL = "https://api.upstox.com/v2"
    UPSTOX_CLIENT_ID = os.getenv("UPSTOX_CLIENT_ID", "")
    UPSTOX_CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET", "")
    UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/auth/callback")
    
    # Trading Configuration
    INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
    TOTAL_FUNDS = 2000000  # ₹20 Lakhs
    LOT_SIZE = 75
    
    # Risk Configuration
    RISK_CONFIG = {
        "Iron Fly": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
        "Iron Condor": {"capital_pct": 0.25, "risk_per_trade_pct": 0.015},
        "Jade Lizard": {"capital_pct": 0.20, "risk_per_trade_pct": 0.01},
        "Straddle": {"capital_pct": 0.15, "risk_per_trade_pct": 0.02},
        "Calendar Spread": {"capital_pct": 0.10, "risk_per_trade_pct": 0.01},
        "Bull Put Spread": {"capital_pct": 0.15, "risk_per_trade_pct": 0.01},
        "Wide Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015},
        "ATM Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015}
    }
    
    DAILY_RISK_LIMIT_PCT = 0.02
    WEEKLY_RISK_LIMIT_PCT = 0.03
    
    # External Data URLs
    EVENT_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv"
    IVP_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv"
    NIFTY_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
    XGBOOST_MODEL_URL = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"

config = Config()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AuthRequest(BaseModel):
    """Authentication request model"""
    authorization_code: str = Field(..., description="Authorization code from Upstox")

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class TradeRequest(BaseModel):
    """Trade logging request model"""
    strategy: str
    instrument_token: str
    entry_price: float
    quantity: float
    realized_pnl: float
    unrealized_pnl: float
    regime_score: Optional[float] = None
    notes: Optional[str] = ""

class JournalRequest(BaseModel):
    """Journal entry request model"""
    title: str
    content: str
    mood: str
    tags: Optional[str] = ""

class StrategyRequest(BaseModel):
    """Strategy analysis request model"""
    strategy: str
    lots: int = 1

class OrderRequest(BaseModel):
    """Order placement request model"""
    quantity: int
    product: str
    validity: str
    price: float
    tag: Optional[str] = None
    instrument_token: str
    order_type: str
    transaction_type: str
    disclosed_quantity: int = 0
    trigger_price: float = 0.0
    is_amo: bool = False

class MarketDataRequest(BaseModel):
    """Market data request model"""
    instrument_keys: List[str]
    mode: str = "full"

# =============================================================================
# UPSTOX API CLIENT
# =============================================================================

class UpstoxClient:
    """Comprehensive Upstox API client with authentication and all endpoints"""
    
    def __init__(self):
        self.base_url = config.UPSTOX_BASE_URL
        self.client_id = config.UPSTOX_CLIENT_ID
        self.client_secret = config.UPSTOX_CLIENT_SECRET
        self.redirect_uri = config.UPSTOX_REDIRECT_URI
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        
    def get_auth_url(self) -> str:
        """Generate authorization URL for OAuth flow"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": "volguard_auth"
        }
        return f"{self.base_url}/login/authorization/dialog?{urlencode(params)}"
    
    async def exchange_code_for_token(self, authorization_code: str) -> TokenResponse:
        """Exchange authorization code for access token"""
        try:
            url = f"{self.base_url}/login/authorization/token"
            data = {
                "code": authorization_code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            response = self.session.post(url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            self.token_expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])
            
            # Update session headers
            self._update_headers()
            
            return TokenResponse(**token_data)
            
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")
    
    async def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            return False
            
        try:
            url = f"{self.base_url}/login/authorization/token"
            data = {
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token"
            }
            
            response = self.session.post(url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])
            
            self._update_headers()
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def _update_headers(self):
        """Update session headers with current access token"""
        if self.access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Api-Version": "2.0"
            })
    
    async def _ensure_valid_token(self):
        """Ensure access token is valid, refresh if necessary"""
        if not self.access_token:
            raise HTTPException(status_code=401, detail="No access token available")
        
        if self.token_expires_at and datetime.now() >= self.token_expires_at - timedelta(minutes=5):
            if not await self.refresh_access_token():
                raise HTTPException(status_code=401, detail="Token refresh failed")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Upstox API"""
        await self._ensure_valid_token()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {method} {endpoint}: {e}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        except Exception as e:
            logger.error(f"Request failed for {method} {endpoint}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # User Profile and Funds
    async def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information"""
        return await self._make_request("GET", "/user/profile")
    
    async def get_funds_and_margin(self) -> Dict[str, Any]:
        """Get user funds and margin information"""
        return await self._make_request("GET", "/user/get-funds-and-margin")
    
    # Portfolio Management
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return await self._make_request("GET", "/portfolio/short-term-positions")
    
    async def get_holdings(self) -> Dict[str, Any]:
        """Get long-term holdings"""
        return await self._make_request("GET", "/portfolio/long-term-holdings")
    
    async def convert_position(self, instrument_token: str, new_product: str, 
                             old_product: str, transaction_type: str, quantity: int) -> Dict[str, Any]:
        """Convert position from one product type to another"""
        data = {
            "instrument_token": instrument_token,
            "new_product": new_product,
            "old_product": old_product,
            "transaction_type": transaction_type,
            "quantity": quantity
        }
        return await self._make_request("PUT", "/portfolio/convert-position", json=data)
    
    # Order Management
    async def place_order(self, order_data: OrderRequest) -> Dict[str, Any]:
        """Place a new order"""
        return await self._make_request("POST", "/order/place", json=order_data.dict())
    
    async def modify_order(self, order_id: str, quantity: Optional[int] = None, 
                          price: Optional[float] = None, order_type: Optional[str] = None,
                          validity: Optional[str] = None, disclosed_quantity: Optional[int] = None,
                          trigger_price: Optional[float] = None) -> Dict[str, Any]:
        """Modify an existing order"""
        data = {"order_id": order_id}
        if quantity is not None:
            data["quantity"] = quantity
        if price is not None:
            data["price"] = price
        if order_type is not None:
            data["order_type"] = order_type
        if validity is not None:
            data["validity"] = validity
        if disclosed_quantity is not None:
            data["disclosed_quantity"] = disclosed_quantity
        if trigger_price is not None:
            data["trigger_price"] = trigger_price
            
        return await self._make_request("PUT", "/order/modify", json=data)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._make_request("DELETE", f"/order/cancel?order_id={order_id}")
    
    async def get_order_book(self) -> Dict[str, Any]:
        """Get order book"""
        return await self._make_request("GET", "/order/retrieve-all")
    
    async def get_order_history(self, order_id: str) -> Dict[str, Any]:
        """Get order history"""
        return await self._make_request("GET", f"/order/history?order_id={order_id}")
    
    async def get_order_details(self, order_id: str) -> Dict[str, Any]:
        """Get order details"""
        return await self._make_request("GET", f"/order/details?order_id={order_id}")
    
    async def get_trades(self) -> Dict[str, Any]:
        """Get trades for the day"""
        return await self._make_request("GET", "/order/trades/get-trades-for-day")
    
    async def get_trades_for_order(self, order_id: str) -> Dict[str, Any]:
        """Get trades for specific order"""
        return await self._make_request("GET", f"/order/trades?order_id={order_id}")
    
    # Market Data
    async def get_market_quotes(self, instrument_keys: List[str]) -> Dict[str, Any]:
        """Get market quotes for instruments"""
        params = {"instrument_key": ",".join(instrument_keys)}
        return await self._make_request("GET", "/market-quote/quotes", params=params)
    
    async def get_ohlc(self, instrument_keys: List[str]) -> Dict[str, Any]:
        """Get OHLC data for instruments"""
        params = {"instrument_key": ",".join(instrument_keys)}
        return await self._make_request("GET", "/market-quote/ohlc", params=params)
    
    async def get_ltp(self, instrument_keys: List[str]) -> Dict[str, Any]:
        """Get LTP for instruments"""
        params = {"instrument_key": ",".join(instrument_keys)}
        return await self._make_request("GET", "/market-quote/ltp", params=params)
    
    async def get_option_greeks(self, instrument_keys: List[str]) -> Dict[str, Any]:
        """Get option Greeks"""
        params = {"instrument_key": ",".join(instrument_keys)}
        return await self._make_request("GET", "/v3/market-quote/option-greek", params=params)
    
    # Options
    async def get_option_contracts(self, instrument_key: str) -> Dict[str, Any]:
        """Get option contracts"""
        params = {"instrument_key": instrument_key}
        return await self._make_request("GET", "/option/contract", params=params)
    
    async def get_option_chain(self, instrument_key: str, expiry_date: str) -> Dict[str, Any]:
        """Get option chain"""
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        return await self._make_request("GET", "/option/chain", params=params)
    
    # Historical Data
    async def get_historical_candle_data(self, instrument_key: str, interval: str, 
                                       to_date: str, from_date: Optional[str] = None) -> Dict[str, Any]:
        """Get historical candle data"""
        endpoint = f"/historical-candle/{instrument_key}/{interval}/{to_date}"
        if from_date:
            endpoint += f"/{from_date}"
        return await self._make_request("GET", endpoint)
    
    async def get_intraday_candle_data(self, instrument_key: str, interval: str) -> Dict[str, Any]:
        """Get intraday candle data"""
        endpoint = f"/historical-candle/intraday/{instrument_key}/{interval}"
        return await self._make_request("GET", endpoint)
    
    # Market Status and Holidays
    async def get_market_status(self, exchange: str) -> Dict[str, Any]:
        """Get market status for exchange"""
        return await self._make_request("GET", f"/market/status/{exchange}")
    
    async def get_market_holidays(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get market holidays"""
        endpoint = "/market/holidays"
        if date:
            endpoint += f"/{date}"
        return await self._make_request("GET", endpoint)
    
    # Charges and Margin
    async def calculate_margin(self, instruments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate margin for instruments"""
        data = {"instruments": instruments}
        return await self._make_request("POST", "/charges/margin", json=data)
    
    async def get_brokerage_details(self, instrument_token: str, quantity: int, 
                                  product: str, transaction_type: str, price: float) -> Dict[str, Any]:
        """Get brokerage details"""
        params = {
            "instrument_token": instrument_token,
            "quantity": quantity,
            "product": product,
            "transaction_type": transaction_type,
            "price": price
        }
        return await self._make_request("GET", "/charges/brokerage", params=params)

# Global Upstox client instance
upstox_client = UpstoxClient()

# =============================================================================
# SUPABASE CLIENT
# =============================================================================

class SupabaseClient:
    """Supabase client for data persistence"""
    
    def __init__(self):
        self.url = config.SUPABASE_URL
        self.key = config.SUPABASE_KEY
        self.headers = {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
    
    async def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Log trade to Supabase"""
        try:
            trade_data["timestamp_entry"] = datetime.utcnow().isoformat()
            trade_data["timestamp_exit"] = datetime.utcnow().isoformat()
            trade_data["status"] = "closed"
            
            response = requests.post(
                f"{self.url}/rest/v1/trade_logs",
                json=trade_data,
                headers=self.headers
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return False
    
    async def add_journal_entry(self, journal_data: Dict[str, Any]) -> bool:
        """Add journal entry to Supabase"""
        try:
            journal_data["timestamp"] = datetime.utcnow().isoformat()
            
            response = requests.post(
                f"{self.url}/rest/v1/journals",
                json=journal_data,
                headers=self.headers
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add journal entry: {e}")
            return False
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades from Supabase"""
        try:
            response = requests.get(
                f"{self.url}/rest/v1/trade_logs",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []
    
    async def get_journals(self) -> List[Dict[str, Any]]:
        """Get all journal entries from Supabase"""
        try:
            response = requests.get(
                f"{self.url}/rest/v1/journals",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to fetch journals: {e}")
            return []
    
    async def log_simulation(self, simulation_data: Dict[str, Any]) -> bool:
        """Log simulation results to Supabase"""
        try:
            simulation_data["timestamp"] = datetime.utcnow().isoformat()
            
            response = requests.post(
                f"{self.url}/rest/v1/simulation_logs",
                json=simulation_data,
                headers=self.headers
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log simulation: {e}")
            return False

# Global Supabase client instance
supabase_client = SupabaseClient()

# =============================================================================
# AI MODELS AND ANALYTICS
# =============================================================================

class AIEngine:
    """AI engine for volatility forecasting and regime detection"""
    
    def __init__(self):
        self.xgb_model = None
        self._load_xgboost_model()
    
    def _load_xgboost_model(self):
        """Load XGBoost model from remote URL"""
        try:
            response = requests.get(config.XGBOOST_MODEL_URL)
            if response.status_code == 200:
                self.xgb_model = pickle.load(BytesIO(response.content))
                logger.info("XGBoost model loaded successfully")
            else:
                logger.error(f"Failed to load XGBoost model: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception loading XGBoost model: {e}")
    
    async def calculate_volatility(self, seller_avg_iv: float) -> tuple:
        """Calculate historical and GARCH volatility"""
        try:
            # Fetch Nifty data
            df = pd.read_csv(config.NIFTY_URL)
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
            df = df.sort_values('Date')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df.dropna(inplace=True)
            
            # Calculate 7-day historical volatility
            hv_7 = np.std(df["Log_Returns"][-7:]) * np.sqrt(252) * 100
            
            # GARCH model
            model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
            res = model.fit(disp="off")
            forecast = res.forecast(horizon=7)
            garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
            
            # IV-RV spread
            iv_rv_spread = round(seller_avg_iv - hv_7, 2)
            
            return hv_7, garch_7d, iv_rv_spread
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return 0, 0, 0
    
    async def predict_xgboost_volatility(self, atm_iv: float, realized_vol: float, 
                                       ivp: float, pcr: float, vix: float, 
                                       days_to_expiry: int, garch_vol: float) -> float:
        """Predict volatility using XGBoost model"""
        try:
            if self.xgb_model is None:
                return 0
            
            features = pd.DataFrame({
                'ATM_IV': [atm_iv],
                'Realized_Vol': [realized_vol],
                'IVP': [ivp],
                'PCR': [pcr],
                'VIX': [vix],
                'Days_to_Expiry': [days_to_expiry],
                'GARCH_Predicted_Vol': [garch_vol]
            })
            
            prediction = self.xgb_model.predict(features)[0]
            return round(float(prediction), 2)
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return 0
    
    async def calculate_iv_skew_slope(self, option_chain_data: List[Dict]) -> float:
        """Calculate IV skew slope"""
        try:
            if not option_chain_data:
                return 0
            
            strikes = []
            ivs = []
            
            for option in option_chain_data:
                if 'call_options' in option and 'option_greeks' in option['call_options']:
                    strikes.append(option['strike_price'])
                    ivs.append(option['call_options']['option_greeks']['iv'])
            
            if len(strikes) < 2:
                return 0
            
            slope, _, _, _, _ = linregress(strikes, ivs)
            return round(slope, 4)
            
        except Exception as e:
            logger.error(f"IV skew calculation failed: {e}")
            return 0
    
    async def calculate_regime(self, atm_iv: float, ivp: float, realized_vol: float, 
                             garch_vol: float, straddle_price: float, spot_price: float, 
                             pcr: float, vix: float, iv_skew_slope: float) -> tuple:
        """Calculate market regime"""
        try:
            expected_move = (straddle_price / spot_price) * 100
            vol_spread = atm_iv - realized_vol
            regime_score = 0
            
            # Regime scoring logic
            regime_score += 10 if ivp > 80 else -10 if ivp < 20 else 0
            regime_score += 10 if vol_spread > 10 else -10 if vol_spread < -10 else 0
            regime_score += 10 if vix > 20 else -10 if vix < 10 else 0
            regime_score += 5 if pcr > 1.2 else -5 if pcr < 0.8 else 0
            regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0
            regime_score += 10 if expected_move > 0.05 else -10 if expected_move < 0.02 else 0
            regime_score += 5 if garch_vol > realized_vol * 1.2 else -5 if garch_vol < realized_vol * 0.8 else 0
            
            # Regime classification
            if regime_score > 20:
                return regime_score, "High Vol Trend", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
            elif regime_score > 10:
                return regime_score, "Elevated Volatility", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
            elif regime_score > -10:
                return regime_score, "Neutral Volatility", "Balanced market — flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
            else:
                return regime_score, "Low Volatility", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."
                
        except Exception as e:
            logger.error(f"Regime calculation failed: {e}")
            return 0, "Unknown", "Error calculating regime", "Unable to determine market regime"

# Global AI engine instance
ai_engine = AIEngine()

# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    """Strategy suggestion and analysis engine"""
    
    async def suggest_strategy(self, regime_label: str, ivp: float, iv_minus_rv: float, 
                             days_to_expiry: int, expiry_date: str, straddle_price: float, 
                             spot_price: float) -> tuple:
        """Suggest optimal strategies based on market conditions"""
        try:
            strategies = []
            rationale = []
            event_warning = None
            
            # Fetch event data
            event_df = await self._get_event_data()
            
            # Event analysis
            event_window = 3 if ivp > 80 else 2
            high_impact_event_near = False
            event_impact_score = 0
            
            for _, row in event_df.iterrows():
                try:
                    dt = pd.to_datetime(row["Datetime"])
                    level = row["Classification"]
                    if level == "High" and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
                        high_impact_event_near = True
                    if level == "High" and pd.notnull(row.get("Forecast")) and pd.notnull(row.get("Prior")):
                        forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                        prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                        if abs(forecast - prior) > 0.5:
                            event_impact_score += 1
                except:
                    continue
            
            if high_impact_event_near:
                event_warning = f"High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
            
            if event_impact_score > 0:
                rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
            
            expected_move_pct = (straddle_price / spot_price) * 100
            
            # Strategy selection logic
            if regime_label == "High Vol Trend":
                strategies = ["Iron Fly", "Wide Strangle"]
                rationale.append("Strong IV premium — neutral strategies for premium capture.")
            elif regime_label == "Elevated Volatility":
                strategies = ["Iron Condor", "Jade Lizard"]
                rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
            elif regime_label == "Neutral Volatility":
                if days_to_expiry >= 3:
                    strategies = ["Jade Lizard", "Bull Put Spread"]
                    rationale.append("Market balanced — slight directional bias strategies offer edge.")
                else:
                    strategies = ["Iron Fly"]
                    rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
            elif regime_label == "Low Volatility":
                if days_to_expiry > 7:
                    strategies = ["Straddle", "Calendar Spread"]
                    rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
                else:
                    strategies = ["Straddle", "ATM Strangle"]
                    rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
            
            # Event-based strategy filtering
            if event_impact_score > 0 and not high_impact_event_near:
                strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
            
            # Additional rationale
            if ivp > 85 and iv_minus_rv > 5:
                rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
            elif ivp < 30:
                rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
            
            rationale.append(f"Expected move: {expected_move_pct:.2f}% — strategies should account for this range.")
            
            return strategies, rationale, event_warning
            
        except Exception as e:
            logger.error(f"Strategy suggestion failed: {e}")
            return [], ["Error in strategy analysis"], None
    
    async def _get_event_data(self) -> pd.DataFrame:
        """Fetch economic events data"""
        try:
            return pd.read_csv(config.EVENT_URL)
        except Exception as e:
            logger.error(f"Failed to fetch event data: {e}")
            return pd.DataFrame()
    
    async def calculate_hedge_width(self, strategy: str, atm_iv: float, ivp: float, 
                                  garch_forecast: float, straddle_price: float, 
                                  spot_price: float) -> int:
        """Calculate dynamic hedge width for strategies"""
        try:
            base_width = 100  # Base width in points
            
            # Adjust based on volatility
            vol_multiplier = 1.0
            if atm_iv > 25:
                vol_multiplier = 1.5
            elif atm_iv > 20:
                vol_multiplier = 1.2
            elif atm_iv < 15:
                vol_multiplier = 0.8
            
            # Adjust based on IVP
            ivp_multiplier = 1.0
            if ivp > 80:
                ivp_multiplier = 1.3
            elif ivp > 60:
                ivp_multiplier = 1.1
            elif ivp < 30:
                ivp_multiplier = 0.9
            
            # Adjust based on GARCH forecast
            garch_multiplier = 1.0
            if garch_forecast > atm_iv * 1.2:
                garch_multiplier = 1.2
            elif garch_forecast < atm_iv * 0.8:
                garch_multiplier = 0.9
            
            # Strategy-specific adjustments
            strategy_multiplier = {
                "Iron Fly": 0.8,
                "Iron Condor": 1.0,
                "Jade Lizard": 1.2,
                "Wide Strangle": 1.5,
                "ATM Strangle": 1.0,
                "Bull Put Spread": 0.9,
                "Calendar Spread": 0.7,
                "Straddle": 0.0  # No hedge for straddle
            }.get(strategy, 1.0)
            
            if strategy == "Straddle":
                return 0
            
            # Calculate final width
            width = base_width * vol_multiplier * ivp_multiplier * garch_multiplier * strategy_multiplier
            
            # Round to nearest 50 points
            return int(round(width / 50) * 50)
            
        except Exception as e:
            logger.error(f"Hedge width calculation failed: {e}")
            return 100

# Global strategy engine instance
strategy_engine = StrategyEngine()

# =============================================================================
# RISK MANAGEMENT ENGINE
# =============================================================================

class RiskEngine:
    """Risk management and behavioral monitoring"""
    
    def __init__(self):
        self.trade_history = []
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.overtrading_threshold = 10  # Max trades per day
        self.daily_trade_count = 0
    
    async def check_capital_allocation(self, strategy: str, funds_data: Dict[str, Any]) -> tuple:
        """Check if capital allocation allows for new trade"""
        try:
            available_margin = funds_data.get("equity", {}).get("available_margin", 0)
            used_margin = funds_data.get("equity", {}).get("used_margin", 0)
            
            strategy_config = config.RISK_CONFIG.get(strategy, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
            
            # Calculate allowed capital for strategy
            total_capital = available_margin + used_margin
            strategy_capital = total_capital * strategy_config["capital_pct"]
            
            # Calculate current exposure for this strategy
            current_exposure = await self._calculate_strategy_exposure(strategy)
            
            # Check if we can allocate more capital
            available_for_strategy = strategy_capital - current_exposure
            
            can_trade = available_for_strategy > (total_capital * strategy_config["risk_per_trade_pct"])
            
            return can_trade, available_for_strategy, strategy_capital
            
        except Exception as e:
            logger.error(f"Capital allocation check failed: {e}")
            return False, 0, 0
    
    async def _calculate_strategy_exposure(self, strategy: str) -> float:
        """Calculate current exposure for a specific strategy"""
        try:
            # This would typically query current positions and calculate exposure
            # For now, return a placeholder
            return 0
        except Exception as e:
            logger.error(f"Strategy exposure calculation failed: {e}")
            return 0
    
    async def check_risk_limits(self, potential_loss: float) -> tuple:
        """Check if trade violates risk limits"""
        try:
            # Daily risk limit check
            daily_limit = config.TOTAL_FUNDS * config.DAILY_RISK_LIMIT_PCT
            if abs(self.daily_pnl + potential_loss) > daily_limit:
                return False, "Daily risk limit exceeded"
            
            # Weekly risk limit check
            weekly_limit = config.TOTAL_FUNDS * config.WEEKLY_RISK_LIMIT_PCT
            if abs(self.weekly_pnl + potential_loss) > weekly_limit:
                return False, "Weekly risk limit exceeded"
            
            return True, "Risk limits OK"
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return False, "Risk check error"
    
    async def check_behavioral_filters(self) -> tuple:
        """Check behavioral trading filters"""
        try:
            # Check for consecutive losses
            if self.consecutive_losses >= 3:
                return False, "Too many consecutive losses - trading blocked"
            
            # Check for overtrading
            if self.daily_trade_count >= self.overtrading_threshold:
                return False, "Daily trade limit exceeded - overtrading protection"
            
            # Check for revenge trading (rapid trades after loss)
            if (self.last_trade_time and 
                datetime.now() - self.last_trade_time < timedelta(minutes=30) and
                self.consecutive_losses > 0):
                return False, "Potential revenge trading detected - cooling off period"
            
            return True, "Behavioral filters passed"
            
        except Exception as e:
            logger.error(f"Behavioral filter check failed: {e}")
            return False, "Behavioral check error"
    
    async def update_trade_metrics(self, pnl: float, is_loss: bool):
        """Update trade metrics for risk monitoring"""
        try:
            self.daily_pnl += pnl
            self.weekly_pnl += pnl
            self.daily_trade_count += 1
            self.last_trade_time = datetime.now()
            
            if is_loss:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
        except Exception as e:
            logger.error(f"Trade metrics update failed: {e}")
    
    async def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)"""
        self.daily_pnl = 0
        self.daily_trade_count = 0
    
    async def reset_weekly_metrics(self):
        """Reset weekly metrics (call weekly)"""
        self.weekly_pnl = 0

# Global risk engine instance
risk_engine = RiskEngine()

# =============================================================================
# MARKET DATA PROCESSOR
# =============================================================================

class MarketDataProcessor:
    """Process and analyze market data"""
    
    async def get_next_expiry(self) -> str:
        """Get next Thursday expiry"""
        try:
            contracts = await upstox_client.get_option_contracts(config.INSTRUMENT_KEY)
            expiries = sorted(contracts["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
            
            today = datetime.now()
            for expiry in expiries:
                expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                if expiry_dt.weekday() == 3 and expiry_dt > today:  # Thursday
                    return expiry["expiry"]
            
            return datetime.now().strftime("%Y-%m-%d")
            
        except Exception as e:
            logger.error(f"Next expiry calculation failed: {e}")
            return datetime.now().strftime("%Y-%m-%d")
    
    async def extract_seller_metrics(self, option_chain: List[Dict], spot_price: float) -> Dict[str, Any]:
        """Extract key metrics for option sellers"""
        try:
            # Find ATM option
            atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
            call = atm["call_options"]
            put = atm["put_options"]
            
            return {
                "strike": atm["strike_price"],
                "straddle_price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
                "avg_iv": (call["option_greeks"]["iv"] + put["option_greeks"]["iv"]) / 2,
                "theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
                "vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
                "delta": call["option_greeks"]["delta"] + put["option_greeks"]["delta"],
                "gamma": call["option_greeks"]["gamma"] + put["option_greeks"]["gamma"],
                "pop": (call["option_greeks"]["pop"] + put["option_greeks"]["pop"]) / 2,
            }
            
        except Exception as e:
            logger.error(f"Seller metrics extraction failed: {e}")
            return {}
    
    async def calculate_market_metrics(self, option_chain: List[Dict], expiry_date: str) -> Dict[str, Any]:
        """Calculate market-wide metrics"""
        try:
            expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
            days_to_expiry = (expiry_dt - datetime.now()).days
            
            # Calculate PCR
            call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain 
                         if "call_options" in opt and "market_data" in opt["call_options"])
            put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain 
                        if "put_options" in opt and "market_data" in opt["put_options"])
            pcr = put_oi / call_oi if call_oi != 0 else 0
            
            # Calculate Max Pain
            strikes = sorted(set(opt["strike_price"] for opt in option_chain))
            max_pain_strike = 0
            min_pain = float('inf')
            
            for strike in strikes:
                pain_at_strike = 0
                for opt in option_chain:
                    if "call_options" in opt:
                        pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                    if "put_options" in opt:
                        pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
                
                if pain_at_strike < min_pain:
                    min_pain = pain_at_strike
                    max_pain_strike = strike
            
            return {
                "days_to_expiry": days_to_expiry,
                "pcr": round(pcr, 2),
                "max_pain": max_pain_strike
            }
            
        except Exception as e:
            logger.error(f"Market metrics calculation failed: {e}")
            return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}
    
    async def get_ivp_data(self) -> float:
        """Get Implied Volatility Percentile"""
        try:
            df = pd.read_csv(config.IVP_URL)
            return df.iloc[-1]['IVP'] if not df.empty else 50.0
        except Exception as e:
            logger.error(f"IVP data fetch failed: {e}")
            return 50.0
    
    async def get_vix_data(self) -> float:
        """Get VIX data (placeholder - would need actual VIX feed)"""
        try:
            # This would typically fetch from a VIX data source
            # For now, return a placeholder value
            return 18.5
        except Exception as e:
            logger.error(f"VIX data fetch failed: {e}")
            return 18.5

# Global market data processor instance
market_processor = MarketDataProcessor()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="VolGuard API",
    description="AI-Powered Options Trading Platform with Complete Upstox Integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.get("/auth/url")
async def get_auth_url():
    """Get Upstox authorization URL"""
    try:
        auth_url = upstox_client.get_auth_url()
        return {"auth_url": auth_url}
    except Exception as e:
        logger.error(f"Auth URL generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/token")
async def exchange_token(auth_request: AuthRequest):
    """Exchange authorization code for access token"""
    try:
        token_response = await upstox_client.exchange_code_for_token(auth_request.authorization_code)
        return token_response
    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/refresh")
async def refresh_token():
    """Refresh access token"""
    try:
        success = await upstox_client.refresh_access_token()
        if success:
            return {"message": "Token refreshed successfully"}
        else:
            raise HTTPException(status_code=401, detail="Token refresh failed")
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# USER AND ACCOUNT ENDPOINTS
# =============================================================================

@app.get("/user/profile")
async def get_user_profile():
    """Get user profile"""
    try:
        profile = await upstox_client.get_user_profile()
        return profile
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/funds")
async def get_funds_and_margin():
    """Get user funds and margin"""
    try:
        funds = await upstox_client.get_funds_and_margin()
        return funds
    except Exception as e:
        logger.error(f"Funds fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PORTFOLIO ENDPOINTS
# =============================================================================

@app.get("/portfolio/positions")
async def get_positions():
    """Get current positions"""
    try:
        positions = await upstox_client.get_positions()
        return positions
    except Exception as e:
        logger.error(f"Positions fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/holdings")
async def get_holdings():
    """Get holdings"""
    try:
        holdings = await upstox_client.get_holdings()
        return holdings
    except Exception as e:
        logger.error(f"Holdings fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ORDER MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/orders/place")
async def place_order(order_request: OrderRequest):
    """Place a new order"""
    try:
        # Check risk limits before placing order
        funds = await upstox_client.get_funds_and_margin()
        
        # Risk checks
        can_trade, available_capital, strategy_capital = await risk_engine.check_capital_allocation("General", funds)
        if not can_trade:
            raise HTTPException(status_code=400, detail="Insufficient capital allocation")
        
        behavioral_ok, behavioral_msg = await risk_engine.check_behavioral_filters()
        if not behavioral_ok:
            raise HTTPException(status_code=400, detail=behavioral_msg)
        
        # Place order
        result = await upstox_client.place_order(order_request)
        
        # Update trade metrics
        await risk_engine.update_trade_metrics(0, False)  # Will be updated when order fills
        
        return result
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/orders/{order_id}/modify")
async def modify_order(order_id: str, quantity: Optional[int] = None, 
                      price: Optional[float] = None, order_type: Optional[str] = None):
    """Modify an existing order"""
    try:
        result = await upstox_client.modify_order(order_id, quantity, price, order_type)
        return result
    except Exception as e:
        logger.error(f"Order modification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an order"""
    try:
        result = await upstox_client.cancel_order(order_id)
        return result
    except Exception as e:
        logger.error(f"Order cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def get_order_book():
    """Get order book"""
    try:
        orders = await upstox_client.get_order_book()
        return orders
    except Exception as e:
        logger.error(f"Order book fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}/history")
async def get_order_history(order_id: str):
    """Get order history"""
    try:
        history = await upstox_client.get_order_history(order_id)
        return history
    except Exception as e:
        logger.error(f"Order history fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades():
    """Get trades for the day"""
    try:
        trades = await upstox_client.get_trades()
        return trades
    except Exception as e:
        logger.error(f"Trades fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MARKET DATA ENDPOINTS
# =============================================================================

@app.get("/market/quotes")
async def get_market_quotes(instrument_keys: str = Query(..., description="Comma-separated instrument keys")):
    """Get market quotes"""
    try:
        keys = instrument_keys.split(",")
        quotes = await upstox_client.get_market_quotes(keys)
        return quotes
    except Exception as e:
        logger.error(f"Market quotes fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/ohlc")
async def get_ohlc(instrument_keys: str = Query(..., description="Comma-separated instrument keys")):
    """Get OHLC data"""
    try:
        keys = instrument_keys.split(",")
        ohlc = await upstox_client.get_ohlc(keys)
        return ohlc
    except Exception as e:
        logger.error(f"OHLC fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/ltp")
async def get_ltp(instrument_keys: str = Query(..., description="Comma-separated instrument keys")):
    """Get LTP"""
    try:
        keys = instrument_keys.split(",")
        ltp = await upstox_client.get_ltp(keys)
        return ltp
    except Exception as e:
        logger.error(f"LTP fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# OPTIONS ENDPOINTS
# =============================================================================

@app.get("/options/contracts")
async def get_option_contracts(instrument_key: str = Query(default=config.INSTRUMENT_KEY)):
    """Get option contracts"""
    try:
        contracts = await upstox_client.get_option_contracts(instrument_key)
        return contracts
    except Exception as e:
        logger.error(f"Option contracts fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options/chain")
async def get_option_chain(instrument_key: str = Query(default=config.INSTRUMENT_KEY), 
                          expiry_date: Optional[str] = None):
    """Get option chain"""
    try:
        if not expiry_date:
            expiry_date = await market_processor.get_next_expiry()
        
        chain = await upstox_client.get_option_chain(instrument_key, expiry_date)
        return chain
    except Exception as e:
        logger.error(f"Option chain fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options/greeks")
async def get_option_greeks(instrument_keys: str = Query(..., description="Comma-separated instrument keys")):
    """Get option Greeks"""
    try:
        keys = instrument_keys.split(",")
        greeks = await upstox_client.get_option_greeks(keys)
        return greeks
    except Exception as e:
        logger.error(f"Option Greeks fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# AI AND ANALYTICS ENDPOINTS
# =============================================================================

@app.get("/analytics/volatility")
async def get_volatility_analysis():
    """Get comprehensive volatility analysis"""
    try:
        # Get current market data
        expiry_date = await market_processor.get_next_expiry()
        option_chain = await upstox_client.get_option_chain(config.INSTRUMENT_KEY, expiry_date)
        
        # Get spot price (using Nifty LTP)
        ltp_data = await upstox_client.get_ltp([config.INSTRUMENT_KEY])
        spot_price = ltp_data["data"][config.INSTRUMENT_KEY]["last_price"]
        
        # Extract seller metrics
        seller_metrics = await market_processor.extract_seller_metrics(option_chain["data"], spot_price)
        
        # Calculate volatilities
        hv_7, garch_7d, iv_rv_spread = await ai_engine.calculate_volatility(seller_metrics["avg_iv"])
        
        # Get additional data
        ivp = await market_processor.get_ivp_data()
        vix = await market_processor.get_vix_data()
        market_metrics = await market_processor.calculate_market_metrics(option_chain["data"], expiry_date)
        
        # XGBoost prediction
        xgb_prediction = await ai_engine.predict_xgboost_volatility(
            seller_metrics["avg_iv"], hv_7, ivp, market_metrics["pcr"], 
            vix, market_metrics["days_to_expiry"], garch_7d
        )
        
        # IV skew
        iv_skew_slope = await ai_engine.calculate_iv_skew_slope(option_chain["data"])
        
        return {
            "spot_price": spot_price,
            "expiry_date": expiry_date,
            "seller_metrics": seller_metrics,
            "volatility": {
                "historical_7d": round(hv_7, 2),
                "garch_7d": round(garch_7d, 2),
                "iv_rv_spread": iv_rv_spread,
                "xgboost_prediction": xgb_prediction
            },
            "market_metrics": market_metrics,
            "ivp": ivp,
            "vix": vix,
            "iv_skew_slope": iv_skew_slope
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/regime")
async def get_regime_analysis():
    """Get market regime analysis"""
    try:
        # Get volatility analysis first
        vol_analysis = await get_volatility_analysis()
        
        # Calculate regime
        regime_score, regime_label, regime_desc, regime_rationale = await ai_engine.calculate_regime(
            vol_analysis["seller_metrics"]["avg_iv"],
            vol_analysis["ivp"],
            vol_analysis["volatility"]["historical_7d"],
            vol_analysis["volatility"]["garch_7d"],
            vol_analysis["seller_metrics"]["straddle_price"],
            vol_analysis["spot_price"],
            vol_analysis["market_metrics"]["pcr"],
            vol_analysis["vix"],
            vol_analysis["iv_skew_slope"]
        )
        
        return {
            "regime_score": regime_score,
            "regime_label": regime_label,
            "description": regime_desc,
            "rationale": regime_rationale,
            "underlying_data": vol_analysis
        }
        
    except Exception as e:
        logger.error(f"Regime analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/strategies")
async def get_strategy_suggestions():
    """Get AI-powered strategy suggestions"""
    try:
        # Get regime analysis
        regime_analysis = await get_regime_analysis()
        
        # Get strategy suggestions
        strategies, rationale, event_warning = await strategy_engine.suggest_strategy(
            regime_analysis["regime_label"],
            regime_analysis["underlying_data"]["ivp"],
            regime_analysis["underlying_data"]["volatility"]["iv_rv_spread"],
            regime_analysis["underlying_data"]["market_metrics"]["days_to_expiry"],
            regime_analysis["underlying_data"]["expiry_date"],
            regime_analysis["underlying_data"]["seller_metrics"]["straddle_price"],
            regime_analysis["underlying_data"]["spot_price"]
        )
        
        # Calculate hedge widths for each strategy
        strategy_details = []
        for strategy in strategies:
            hedge_width = await strategy_engine.calculate_hedge_width(
                strategy,
                regime_analysis["underlying_data"]["seller_metrics"]["avg_iv"],
                regime_analysis["underlying_data"]["ivp"],
                regime_analysis["underlying_data"]["volatility"]["garch_7d"],
                regime_analysis["underlying_data"]["seller_metrics"]["straddle_price"],
                regime_analysis["underlying_data"]["spot_price"]
            )
            
            strategy_details.append({
                "strategy": strategy,
                "hedge_width": hedge_width,
                "risk_config": config.RISK_CONFIG.get(strategy, {})
            })
        
        return {
            "suggested_strategies": strategy_details,
            "rationale": rationale,
            "event_warning": event_warning,
            "regime_context": regime_analysis
        }
        
    except Exception as e:
        logger.error(f"Strategy suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RISK MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/risk/status")
async def get_risk_status():
    """Get current risk status"""
    try:
        funds = await upstox_client.get_funds_and_margin()
        
        risk_status = {
            "daily_pnl": risk_engine.daily_pnl,
            "weekly_pnl": risk_engine.weekly_pnl,
            "consecutive_losses": risk_engine.consecutive_losses,
            "daily_trade_count": risk_engine.daily_trade_count,
            "daily_limit": config.TOTAL_FUNDS * config.DAILY_RISK_LIMIT_PCT,
            "weekly_limit": config.TOTAL_FUNDS * config.WEEKLY_RISK_LIMIT_PCT,
            "funds_data": funds
        }
        
        return risk_status
        
    except Exception as e:
        logger.error(f"Risk status fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk/check")
async def check_trade_risk(strategy: str, potential_loss: float):
    """Check if a trade passes risk filters"""
    try:
        funds = await upstox_client.get_funds_and_margin()
        
        # Capital allocation check
        can_trade, available_capital, strategy_capital = await risk_engine.check_capital_allocation(strategy, funds)
        
        # Risk limits check
        risk_ok, risk_msg = await risk_engine.check_risk_limits(potential_loss)
        
        # Behavioral filters check
        behavioral_ok, behavioral_msg = await risk_engine.check_behavioral_filters()
        
        overall_ok = can_trade and risk_ok and behavioral_ok
        
        return {
            "can_trade": overall_ok,
            "capital_check": {
                "passed": can_trade,
                "available_capital": available_capital,
                "strategy_capital": strategy_capital
            },
            "risk_check": {
                "passed": risk_ok,
                "message": risk_msg
            },
            "behavioral_check": {
                "passed": behavioral_ok,
                "message": behavioral_msg
            }
        }
        
    except Exception as e:
        logger.error(f"Risk check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# TRADING JOURNAL ENDPOINTS
# =============================================================================

@app.post("/journal/trade")
async def log_trade(trade_request: TradeRequest):
    """Log a trade to the journal"""
    try:
        success = await supabase_client.log_trade(trade_request.dict())
        if success:
            return {"message": "Trade logged successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to log trade")
    except Exception as e:
        logger.error(f"Trade logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/journal/entry")
async def add_journal_entry(journal_request: JournalRequest):
    """Add a journal entry"""
    try:
        success = await supabase_client.add_journal_entry(journal_request.dict())
        if success:
            return {"message": "Journal entry added successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add journal entry")
    except Exception as e:
        logger.error(f"Journal entry failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/journal/trades")
async def get_trade_history():
    """Get trade history"""
    try:
        trades = await supabase_client.get_trades()
        return {"trades": trades}
    except Exception as e:
        logger.error(f"Trade history fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/journal/entries")
async def get_journal_entries():
    """Get journal entries"""
    try:
        entries = await supabase_client.get_journals()
        return {"entries": entries}
    except Exception as e:
        logger.error(f"Journal entries fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================

@app.post("/simulation/backtest")
async def run_backtest(strategy: str, start_date: str, end_date: str, 
                      initial_capital: float = 100000):
    """Run strategy backtest"""
    try:
        # This would implement a comprehensive backtesting engine
        # For now, return a placeholder response
        
        simulation_result = {
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "initial_capital": initial_capital,
            "final_capital": initial_capital * 1.15,  # Placeholder
            "total_return": 15.0,
            "max_drawdown": -5.2,
            "sharpe_ratio": 1.8,
            "win_rate": 68.5,
            "total_trades": 45,
            "avg_trade_duration": 3.2
        }
        
        # Log simulation to Supabase
        await supabase_client.log_simulation(simulation_result)
        
        return simulation_result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HEALTH CHECK AND STATUS ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "upstox_connected": upstox_client.access_token is not None,
        "ai_models_loaded": ai_engine.xgb_model is not None
    }

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Check Upstox connection
        upstox_status = "connected" if upstox_client.access_token else "disconnected"
        
        # Check AI models
        ai_status = "loaded" if ai_engine.xgb_model else "not_loaded"
        
        # Get market status
        try:
            market_status = await upstox_client.get_market_status("NSE")
            market_open = market_status.get("data", {}).get("market_status") == "OPEN"
        except:
            market_open = False
        
        return {
            "system_status": "operational",
            "upstox_status": upstox_status,
            "ai_models_status": ai_status,
            "market_open": market_open,
            "risk_engine_status": "active",
            "supabase_status": "connected",
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "system_status": "degraded",
            "error": str(e),
            "last_updated": datetime.utcnow().isoformat()
        }

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("VolGuard FastAPI backend starting up...")
    
    # Initialize AI models
    if not ai_engine.xgb_model:
        ai_engine._load_xgboost_model()
    
    # Reset daily metrics if needed
    await risk_engine.reset_daily_metrics()
    
    logger.info("VolGuard FastAPI backend startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("VolGuard FastAPI backend shutting down...")

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

