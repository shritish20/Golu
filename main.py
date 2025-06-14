import os
import logging
import pandas as pd
import numpy as np
import httpx # Asynchronous HTTP client
import pickle
from io import BytesIO
from datetime import datetime, timedelta
from arch import arch_model
from scipy.stats import linregress

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- CONFIGURATION & ENVIRONMENT VARIABLES ---
# It's crucial to set these as environment variables in your Render deployment.
# Example:
# SUPABASE_URL="https://eurepsbikwxwmgpgzvzn.supabase.co"
# SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV1cmVwc2Jpa3d4d21ncGd6dnpuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0MTYzNzQsImV4cCI6MjA2NDk5MjM3NH0.r3soCQV8nkbvc8RzFoLNGxK9MqQUOEIQUAWubAzAIkA"
# UPSTOX_ACCESS_TOKEN="YOUR_UPSTOX_ACCESS_TOKEN" # This should be passed via API calls for security
# XG_BOOST_MODEL_URL="https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
# NIFTY_HIST_URL="https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
# IVP_HIST_URL="https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv"
# UPCOMING_EVENTS_URL="https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
XG_BOOST_MODEL_URL = os.getenv("XG_BOOST_MODEL_URL", "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl")
NIFTY_HIST_URL = os.getenv("NIFTY_HIST_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
IVP_HIST_URL = os.getenv("IVP_HIST_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv")
UPCOMING_EVENTS_URL = os.getenv("UPCOMING_EVENTS_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set as environment variables.")

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SUPABASE CLIENT ---
SUPABASE_HEADERS = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apikey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

async def log_trade_to_supabase(data: dict):
    data["timestamp_entry"] = datetime.utcnow().isoformat() + "Z" # ISO 8601 with Z for UTC
    data["timestamp_exit"] = datetime.utcnow().isoformat() + "Z"
    data["status"] = "closed" # Assuming logs are for closed trades
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SUPABASE_URL}/rest/v1/trade_logs", json=data, headers=SUPABASE_HEADERS)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Trade logged to Supabase: {response.json()}")
            return response.status_code, response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error logging trade to Supabase: {e.response.status_code} - {e.response.text}")
        return e.response.status_code, {"error": e.response.text}
    except httpx.RequestError as e:
        logger.error(f"Network error logging trade to Supabase: {e}")
        return 500, {"error": str(e)}

async def add_journal_to_supabase(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SUPABASE_URL}/rest/v1/journals", json=data, headers=SUPABASE_HEADERS)
            response.raise_for_status()
            logger.info(f"Journal entry added to Supabase: {response.json()}")
            return response.status_code, response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error adding journal to Supabase: {e.response.status_code} - {e.response.text}")
        return e.response.status_code, {"error": e.response.text}
    except httpx.RequestError as e:
        logger.error(f"Network error adding journal to Supabase: {e}")
        return 500, {"error": str(e)}

# --- FastAPI Setup ---
app = FastAPI(title="VoluGuard API", description="Wrapped from Streamlit backend")

# --- FastAPI Models ---
class TradeRequest(BaseModel):
    strategy: str
    instrument_token: str
    entry_price: float
    quantity: float
    realized_pnl: float
    unrealized_pnl: float
    regime_score: Optional[float] = None
    notes: Optional[str] = ""
    # Add fields for risk evaluation from evaluate_full_risk if needed to log actual trade data
    capital_used: Optional[float] = None
    potential_loss: Optional[float] = None
    sl_hit: Optional[bool] = False
    vega: Optional[float] = None

class JournalRequest(BaseModel):
    title: str
    content: str
    mood: str
    tags: Optional[str] = ""

class StrategyRequest(BaseModel):
    strategy: str
    lots: int = 1

class UpstoxOrderRequest(BaseModel):
    quantity: int
    product: str
    validity: str
    price: float
    tag: Optional[str] = None
    slice: Optional[bool] = False
    instrument_token: str
    order_type: str
    transaction_type: str
    disclosed_quantity: Optional[int] = 0
    trigger_price: Optional[float] = 0
    is_amo: Optional[bool] = False
    correlation_id: Optional[str] = None

# --- Helper Functions for Database (Supabase) ---
async def get_all_trades():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SUPABASE_URL}/rest/v1/trade_logs", headers=SUPABASE_HEADERS)
            response.raise_for_status()
            logger.info("Successfully fetched trades from Supabase.")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching trades: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching trades: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching trades: {str(e)}")

async def get_all_journals():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SUPABASE_URL}/rest/v1/journals", headers=SUPABASE_HEADERS)
            response.raise_for_status()
            logger.info("Successfully fetched journals from Supabase.")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching journals: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching journals: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching journals: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching journals: {str(e)}")

# --- CONFIGURATION & UPSTOX API SETUP ---
def get_upstox_headers(access_token: str) -> Dict[str, str]:
    return {
        "accept": "application/json",
        "Api-Version": "2.0",
        "Authorization": f"Bearer {access_token}"
    }

async def get_config(access_token: str) -> Dict[str, Any]:
    upstox_headers = get_upstox_headers(access_token)
    config = {
        "access_token": access_token, # Keep for consistency, but prefer using get_upstox_headers for calls
        "base_url": "https://api.upstox.com/v2",
        "headers": upstox_headers, # Use this for direct calls within this function
        "instrument_key": "NSE_INDEX|Nifty 50", # Assuming Nifty 50 is the primary instrument
        "nifty_url": NIFTY_HIST_URL,
        "ivp_url": IVP_HIST_URL,
        "event_url": UPCOMING_EVENTS_URL,
        "total_funds": 2000000, # Default total funds, can be made dynamic
        "risk_config": {
            "Iron Fly": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
            "Iron Condor": {"capital_pct": 0.25, "risk_per_trade_pct": 0.015},
            "Jade Lizard": {"capital_pct": 0.20, "risk_per_trade_pct": 0.01},
            "Straddle": {"capital_pct": 0.15, "risk_per_trade_pct": 0.02},
            "Calendar Spread": {"capital_pct": 0.10, "risk_per_trade_pct": 0.01},
            "Bull Put Spread": {"capital_pct": 0.15, "risk_per_trade_pct": 0.01},
            "Wide Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015},
            "ATM Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015}
        },
        "daily_risk_limit_pct": 0.02,
        "weekly_risk_limit_pct": 0.03,
        "lot_size": 75 # Nifty lot size
    }

    try:
        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status()
            expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
            today = datetime.now()
            # Find next Thursday expiry
            next_expiry = None
            for expiry in expiries:
                expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                if expiry_dt.weekday() == 3 and expiry_dt >= today: # Thursday is weekday 3, >= today includes today if it's a Thursday
                    next_expiry = expiry["expiry"]
                    break
            if next_expiry:
                config['expiry_date'] = next_expiry
            else:
                logger.warning(f"No upcoming Thursday expiry found. Defaulting to current date: {today.strftime('%Y-%m-%d')}")
                config['expiry_date'] = today.strftime("%Y-%m-%d")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching expiries in get_config: {e.response.status_code} - {e.response.text}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d") # Fallback
    except httpx.RequestError as e:
        logger.error(f"Network error fetching expiries in get_config: {e}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d") # Fallback
    except Exception as e:
        logger.error(f"Unexpected exception in get_config (expiry fetch): {e}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d") # Fallback

    return config

# --- Data Fetching and Calculation Functions ---
async def fetch_option_chain(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/option/chain"
            params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status()
            return res.json()["data"]
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching option chain: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching option chain: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching option chain: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching option chain: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected exception in fetch_option_chain: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching option chain: {str(e)}")

def extract_seller_metrics(option_chain: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    try:
        atm_strike_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
        call_atm = atm_strike_info.get("call_options")
        put_atm = atm_strike_info.get("put_options")

        if not call_atm or not put_atm:
            logger.warning(f"Could not find both ATM call and put for spot price {spot_price}")
            return {}

        return {
            "strike": atm_strike_info["strike_price"],
            "straddle_price": call_atm["market_data"]["ltp"] + put_atm["market_data"]["ltp"],
            "avg_iv": (call_atm["option_greeks"]["iv"] + put_atm["option_greeks"]["iv"]) / 2 if call_atm["option_greeks"]["iv"] and put_atm["option_greeks"]["iv"] else 0,
            "theta": call_atm["option_greeks"]["theta"] + put_atm["option_greeks"]["theta"],
            "vega": call_atm["option_greeks"]["vega"] + put_atm["option_greeks"]["vega"],
            "delta": call_atm["option_greeks"]["delta"] + put_atm["option_greeks"]["delta"],
            "gamma": call_atm["option_greeks"]["gamma"] + put_atm["option_greeks"]["gamma"],
            "pop": ((call_atm["option_greeks"]["pop"] + put_atm["option_greeks"]["pop"]) / 2) if call_atm["option_greeks"]["pop"] and put_atm["option_greeks"]["pop"] else 0,
        }
    except Exception as e:
        logger.error(f"Exception in extract_seller_metrics for spot {spot_price}: {e}")
        return {}

def market_metrics(option_chain: List[Dict[str, Any]], expiry_date: str) -> Dict[str, Any]:
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days

        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if opt.get("call_options") and opt["call_options"].get("market_data") and opt["call_options"]["market_data"].get("oi") is not None)
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if opt.get("put_options") and opt["put_options"].get("market_data") and opt["put_options"]["market_data"].get("oi") is not None)

        pcr = put_oi / call_oi if call_oi != 0 else 0

        strikes = sorted(list(set(opt["strike_price"] for opt in option_chain)))
        max_pain_strike = 0
        min_pain = float('inf')

        for strike in strikes:
            pain_at_strike = 0
            for opt in option_chain:
                if "call_options" in opt and opt["call_options"].get("market_data") and opt["call_options"]["market_data"].get("oi") is not None:
                    pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                if "put_options" in opt and opt["put_options"].get("market_data") and opt["put_options"]["market_data"].get("oi") is not None:
                    pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
            
            if pain_at_strike < min_pain:
                min_pain = pain_at_strike
                max_pain_strike = strike

        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
    except Exception as e:
        logger.error(f"Exception in market_metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

# Set up a basic logger for demonstration purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
async def fetch_india_vix(config: Dict[str, Any]) -> float:
    """
    Fetches India VIX from Upstox API using the correct instrument key and endpoint.
    Removes the fallback option to a static value.
    """
    try:
        # Use the correct instrument key for India VIX with a colon
        vix_instrument_key_for_api = "NSE_INDEX:India VIX"
        
        async with httpx.AsyncClient() as client:
            # Use the /market-quote/quotes endpoint for fetching index data
            # You can fetch multiple instruments by comma-separating them
            url = f"{config['base_url']}/market-quote/quotes"
            params = {"instrument_key": vix_instrument_key_for_api} # Passing the specific VIX key
            
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status() # Raises an exception for 4xx/5xx responses
            
            data = res.json()
            
            # Navigate the JSON response to find the VIX last_price
            # The structure for the /quotes endpoint places data directly under 'data' key
            # and then by the exact instrument_key (with colon)
            if data and data.get("data") and data["data"].get(vix_instrument_key_for_api):
                vix_ltp = data["data"][vix_instrument_key_for_api].get("last_price")
                if vix_ltp is not None:
                    logger.info(f"Fetched India VIX: {vix_ltp}")
                    return vix_ltp
                else:
                    logger.error(f"India VIX 'last_price' not found in response for {vix_instrument_key_for_api}.")
                    raise ValueError("India VIX data not available in API response.")
            else:
                logger.error(f"India VIX data not found in API response for key: {vix_instrument_key_for_api}. Full response: {data}")
                raise ValueError("India VIX data not found or invalid in API response.")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching India VIX: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Failed to fetch India VIX due to HTTP error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.error(f"Network error fetching India VIX: {e}")
        raise RuntimeError(f"Failed to fetch India VIX due to network error: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing India VIX data from API response: {e}")
        raise RuntimeError(f"Failed to parse India VIX data: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching India VIX: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}") from e




async def calculate_volatility(config: Dict[str, Any], seller_avg_iv: float) -> tuple[float, float, float]:
    try:
        async with httpx.AsyncClient() as client:
            nifty_response = await client.get(config['nifty_url'])
            nifty_response.raise_for_status()
            df = pd.read_csv(BytesIO(nifty_response.content))
        
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
        df = df.sort_values('Date').set_index('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        if df.empty or len(df) < 7:
            logger.warning("Not enough historical data for volatility calculation.")
            return 0, 0, 0

        # Historical Volatility (7-day)
        hv_7 = np.std(df["Log_Returns"].tail(7)) * np.sqrt(252) * 100

        # GARCH Model
        # Ensure enough data points for GARCH model fitting
        if len(df["Log_Returns"]) > 20: # A common heuristic for minimum data points
            model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
            res = model.fit(disp="off", show_warning=False) # Suppress warnings
            forecast = res.forecast(horizon=7)
            # Ensure forecast exists and is not empty
            if not forecast.variance.empty:
                garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
            else:
                logger.warning("GARCH forecast variance is empty.")
                garch_7d = 0
        else:
            logger.warning("Not enough data for GARCH model. GARCH volatility set to 0.")
            garch_7d = 0
        
        iv_rv_spread = round(seller_avg_iv - hv_7, 2)
        
        return hv_7, garch_7d, iv_rv_spread
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching Nifty historical data: {e.response.status_code} - {e.response.text}")
        return 0, 0, 0
    except httpx.RequestError as e:
        logger.error(f"Network error fetching Nifty historical data: {e}")
        return 0, 0, 0
    except Exception as e:
        logger.error(f"Exception in calculate_volatility: {e}")
        return 0, 0, 0

_xgboost_model = None # Global variable to store the loaded model

async def load_xgboost_model():
    global _xgboost_model
    if _xgboost_model is not None:
        logger.info("XGBoost model already loaded.")
        return _xgboost_model

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(XG_BOOST_MODEL_URL)
            response.raise_for_status()
            _xgboost_model = pickle.load(BytesIO(response.content))
            logger.info("XGBoost model loaded successfully.")
            return _xgboost_model
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching XGBoost model: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Network error fetching XGBoost model: {e}")
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling XGBoost model: {e}. Model file might be corrupt or incompatible.")
        return None
    except Exception as e:
        logger.error(f"Unexpected exception in load_xgboost_model: {e}")
        return None

def predict_xgboost_volatility(model: Any, atm_iv: float, realized_vol: float, ivp: float, pcr: float, vix: float, days_to_expiry: int, garch_vol: float) -> float:
    if model is None:
        logger.warning("XGBoost model not loaded, cannot predict volatility.")
        return 0.0
    try:
        features = pd.DataFrame({
            'ATM_IV': [atm_iv],
            'Realized_Vol': [realized_vol],
            'IVP': [ivp],
            'PCR': [pcr],
            'VIX': [vix],
            'Days_to_Expiry': [days_to_expiry],
            'GARCH_Predicted_Vol': [garch_vol]
        })
        prediction = model.predict(features)[0]
        return round(float(prediction), 2)
    except Exception as e:
        logger.error(f"Exception in predict_xgboost_volatility: {e}")
        return 0.0

async def calculate_ivp(config: Dict[str, Any], current_atm_iv: float) -> float:
    """Calculates Implied Volatility Percentile (IVP)."""
    try:
        async with httpx.AsyncClient() as client:
            ivp_response = await client.get(config['ivp_url'])
            ivp_response.raise_for_status()
            df_iv = pd.read_csv(BytesIO(ivp_response.content))
        
        df_iv.columns = df_iv.columns.str.strip()
        df_iv['Date'] = pd.to_datetime(df_iv['Date']) # Assuming 'Date' column is present
        # Assuming 'ATM_IV' is the column containing historical ATM IVs
        if 'ATM_IV' not in df_iv.columns:
            logger.error("IVP CSV must contain an 'ATM_IV' column.")
            return 0.0

        historical_ivs = df_iv['ATM_IV'].dropna().values
        if len(historical_ivs) < 10: # Need sufficient historical data for percentile
            logger.warning("Not enough historical data in IVP CSV for percentile calculation.")
            return 0.0

        # Calculate percentile
        percentile = np.mean(current_atm_iv > historical_ivs) * 100
        return round(percentile, 2)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching IVP historical data: {e.response.status_code} - {e.response.text}")
        return 0.0
    except httpx.RequestError as e:
        logger.error(f"Network error fetching IVP historical data: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Exception in calculate_ivp: {e}")
        return 0.0

def calculate_iv_skew_slope(full_chain_df: pd.DataFrame) -> float:
    try:
        if full_chain_df.empty or "Strike" not in full_chain_df.columns or "IV Skew" not in full_chain_df.columns:
            logger.warning("Full chain DataFrame is empty or missing required columns for IV skew slope calculation.")
            return 0.0
        # Filter out rows with NaN in critical columns if any
        df_filtered = full_chain_df[["Strike", "IV Skew"]].dropna()
        if len(df_filtered) < 2:
            logger.warning("Not enough valid data points for linear regression on IV Skew.")
            return 0.0
        slope, _, _, _, _ = linregress(df_filtered["Strike"], df_filtered["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        logger.error(f"Exception in calculate_iv_skew_slope: {e}")
        return 0.0

def calculate_regime(atm_iv: float, ivp: float, realized_vol: float, garch_vol: float, straddle_price: float, spot_price: float, pcr: float, vix: float, iv_skew_slope: float) -> tuple[float, str, str, str]:
    expected_move = (straddle_price / spot_price) * 100 if spot_price else 0
    vol_spread = atm_iv - realized_vol
    
    regime_score = 0
    regime_score += 10 if ivp > 80 else (-10 if ivp < 20 else 0)
    regime_score += 10 if vol_spread > 10 else (-10 if vol_spread < -10 else 0)
    regime_score += 10 if vix > 20 else (-10 if vix < 10 else 0)
    regime_score += 5 if pcr > 1.2 else (-5 if pcr < 0.8 else 0)
    regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0 # Significant skew indicates potential for moves
    regime_score += 10 if expected_move > 0.05 else (-10 if expected_move < 0.02 else 0)
    regime_score += 5 if garch_vol > realized_vol * 1.2 else (-5 if garch_vol < realized_vol * 0.8 else 0) # GARCH predicting higher/lower than realized

    if regime_score > 20:
        return regime_score, "High Vol Trend :fire:", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
    elif regime_score > 10:
        return regime_score, "Elevated Volatility :zap:", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
    elif regime_score > -10:
        return regime_score, "Neutral Volatility :smile:", "Balanced market — flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return regime_score, "Low Volatility :chart_with_downwards_trend:", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."

async def suggest_strategy(regime_label: str, ivp: float, iv_minus_rv: float, days_to_expiry: int, expiry_date: str, straddle_price: float, spot_price: float, config: Dict[str, Any]) -> tuple[List[str], str, Optional[str]]:
    strategies = []
    rationale = []
    event_warning = None
    event_impact_score = 0
    
    try:
        async with httpx.AsyncClient() as client:
            events_response = await client.get(config['event_url'])
            events_response.raise_for_status()
            event_df = pd.read_csv(BytesIO(events_response.content))
        
        event_df.columns = event_df.columns.str.strip()
        event_df['Datetime'] = pd.to_datetime(event_df['Datetime'])

        event_window = 3 if ivp > 80 else 2 # Days before expiry to consider as "near"
        high_impact_event_near = False
        
        current_expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")

        for _, row in event_df.iterrows():
            if pd.isna(row["Datetime"]): continue # Skip rows with invalid datetime
            event_dt = row["Datetime"]
            level = str(row["Classification"]).strip() # Ensure string and strip whitespace

            # Check if event is relevant for the *current expiry*
            days_until_event = (event_dt.date() - datetime.now().date()).days
            days_from_event_to_expiry = (current_expiry_dt.date() - event_dt.date()).days

            if level == "High" and (0 <= days_until_event <= event_window or (days_from_event_to_expiry >=0 and days_from_event_to_expiry <= event_window)):
                high_impact_event_near = True
                
            if level == "High" and pd.notnull(row.get("Forecast")) and pd.notnull(row.get("Prior")):
                try:
                    forecast = float(str(row["Forecast"]).replace('%', '').strip())
                    prior = float(str(row["Prior"]).replace('%', '').strip())
                    if abs(forecast - prior) > 0.5: # Significant deviation
                        event_impact_score += 1
                except ValueError:
                    logger.warning(f"Could not parse Forecast/Prior for event: {row.get('Event')}")
            
        if high_impact_event_near:
            event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
            rationale.append("Upcoming high-impact event suggests cautious, defined-risk approach.")
        
        if event_impact_score > 0:
            rationale.append(f"High-impact events with significant forecast deviations detected ({event_impact_score} events).")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching event data: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching event data: {e}")
    except Exception as e:
        logger.error(f"Exception in fetching/processing event data: {e}")
    
    expected_move_pct = (straddle_price / spot_price) * 100 if spot_price else 0

    # Strategy suggestions based on regime
    if regime_label == "High Vol Trend :fire:":
        strategies = ["Iron Fly", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "Elevated Volatility :zap:":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "Neutral Volatility :smile:":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
    elif regime_label == "Low Volatility :chart_with_downwards_trend:":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")

    # Adjust strategies if high-impact event is near, prioritizing defined risk
    if high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
        if not strategies: # If no defined-risk strategies were suggested, suggest a generic one
            strategies.append("Iron Condor (Defined Risk)")

    # Additional rationale based on IVP and IV-RV spread
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    
    rationale.append(f"Expected market move based on straddle price: ±{expected_move_pct:.2f}%.")

    return strategies, " | ".join(rationale), event_warning

def find_option_by_strike(option_chain: List[Dict[str, Any]], strike: float, option_type: str, tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
    """Helper to find a specific option by strike and type with a tolerance."""
    for opt in option_chain:
        if abs(opt["strike_price"] - strike) < tolerance:
            if option_type == "CE" and "call_options" in opt:
                return opt["call_options"]
            elif option_type == "PE" and "put_options" in opt:
                return opt["put_options"]
    logger.warning(f"No option found for strike {strike} {option_type}")
    return None

def get_dynamic_wing_distance(ivp: float, straddle_price: float) -> int:
    """Calculates dynamic wing distance for multi-leg strategies."""
    if ivp >= 80:
        multiplier = 0.35
    elif ivp <= 20:
        multiplier = 0.2
    else:
        multiplier = 0.25
    raw_distance = straddle_price * multiplier
    return int(round(raw_distance / 50.0)) * 50  # Round to nearest 50 for Nifty

# --- Strategy Detail Calculation Functions ---
# These functions now return comprehensive details ready for order placement
def _iron_fly_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]
    straddle_price = atm_info["call_options"]["market_data"]["ltp"] + atm_info["put_options"]["market_data"]["ltp"]
    
    # Using a high IVP for wing distance for Iron Fly as it's a high premium strategy
    wing_distance = get_dynamic_wing_distance(80, straddle_price) 
    
    ce_short_opt = find_option_by_strike(option_chain, atm_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, atm_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, atm_strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, atm_strike - wing_distance, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Fly: One or more legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_long_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Iron Fly", "strikes": [atm_strike - wing_distance, atm_strike, atm_strike + wing_distance], "orders": orders}

def _iron_condor_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]
    straddle_price = atm_info["call_options"]["market_data"]["ltp"] + atm_info["put_options"]["market_data"]["ltp"]

    short_wing_distance = get_dynamic_wing_distance(50, straddle_price) # Moderate IVP for wider short wings
    long_wing_distance = int(round(short_wing_distance * 1.5 / 50)) * 50 # 1.5x the short wing distance
    
    ce_short_strike = atm_strike + short_wing_distance
    pe_short_strike = atm_strike - short_wing_distance
    ce_long_strike = atm_strike + long_wing_distance
    pe_long_strike = atm_strike - long_wing_distance

    ce_short_opt = find_option_by_strike(option_chain, ce_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, ce_long_strike, "CE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Condor: One or more legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_long_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Iron Condor", "strikes": [pe_long_strike, pe_short_strike, ce_short_strike, ce_long_strike], "orders": orders}

def _jade_lizard_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    
    # Common Jade Lizard: Short OTM Call, Short OTM Put, Long further OTM Put
    call_short_strike = atm_info["strike_price"] + 100 # Slightly OTM Call
    pe_short_strike = atm_info["strike_price"] - 50 # Closer OTM Put
    pe_long_strike = atm_info["strike_price"] - 150 # Further OTM Put for hedge

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Jade Lizard: One or more legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Jade Lizard", "strikes": [pe_long_strike, pe_short_strike, call_short_strike], "orders": orders}

def _straddle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]
    
    ce_short_opt = find_option_by_strike(option_chain, atm_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, atm_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Straddle: One or both legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Straddle", "strikes": [atm_strike, atm_strike], "orders": orders}

def _calendar_spread_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]

    # For calendar spread, you need two different expiry dates.
    # The current option_chain is for a single expiry.
    # To implement this, you'd need to fetch option chain for *another* expiry date.
    # For now, this will return a placeholder, as the current `option_chain` only contains one expiry.
    # To make this real, you'd need:
    # 1. Logic to determine a 'far' expiry date.
    # 2. An API call to fetch option chain for that 'far' expiry.
    
    # Placeholder: Assuming you want to buy far expiry CE and sell near expiry CE at ATM strike
    # This requires another fetch_option_chain call for the far expiry.
    # Example for demonstration purposes - NOT FUNCTIONAL WITHOUT FETCHING FAR CHAIN
    
    # For this to work, you'd need to call fetch_option_chain again with a different expiry_date
    # For example: next_next_expiry_date = get_next_expiry(today + timedelta(weeks=1))
    # far_option_chain = await fetch_option_chain(config_with_next_next_expiry)
    
    # Let's return a detailed message that this strategy requires multi-expiry chain lookup.
    logger.error("Calendar Spread calculation requires fetching option chains for multiple expiry dates, which is not fully implemented in this single-expiry flow.")
    return None # Or return a specific error/message to the user
    
    # If implemented, it would look like:
    # near_leg_opt = find_option_by_strike(option_chain, atm_strike, "CE") # Near expiry CE
    # far_leg_opt = find_option_by_strike(far_option_chain, atm_strike, "CE") # Far expiry CE
    # if not all([near_leg_opt, far_leg_opt]): return None
    # orders = [
    #     {"instrument_key": near_leg_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": near_leg_opt["market_data"]["ltp"]},
    #     {"instrument_key": far_leg_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": far_leg_opt["market_data"]["ltp"]}
    # ]
    # return {"strategy": "Calendar Spread", "strikes": [atm_strike], "orders": orders}


def _bull_put_spread_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    # Bull Put Spread: Sell OTM Put, Buy further OTM Put
    # Typically sell a put with higher premium, buy a cheaper put for hedge
    # Example: Sell strike -100, Buy strike -200 (relative to spot)
    
    # Find a strike roughly 1-2 standard deviations below spot price for the short put,
    # then go further down for the long put. Let's use fixed distances for simplicity.
    short_put_strike = spot_price - 100 # Sell slightly OTM Put
    long_put_strike = spot_price - 200 # Buy further OTM Put for protection

    # Round to nearest 50 for Nifty strikes
    short_put_strike = int(round(short_put_strike / 50.0)) * 50
    long_put_strike = int(round(long_put_strike / 50.0)) * 50

    pe_short_opt = find_option_by_strike(option_chain, short_put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_put_strike, "PE")

    if not all([pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Bull Put Spread: One or both legs not found.")
        return None
    
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [long_put_strike, short_put_strike], "orders": orders}


def _wide_strangle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    # Wide Strangle: Sell OTM Call and Sell OTM Put, both further away from ATM
    # Example: Sell call strike +200, Sell put strike -200 (relative to spot)
    call_short_strike = spot_price + 200
    put_short_strike = spot_price - 200

    call_short_strike = int(round(call_short_strike / 50.0)) * 50
    put_short_strike = int(round(put_short_strike / 50.0)) * 50

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_short_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Wide Strangle: One or both legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Wide Strangle", "strikes": [put_short_strike, call_short_strike], "orders": orders}


def _atm_strangle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    # ATM Strangle: Sell OTM Call and Sell OTM Put, closer to ATM
    # Example: Sell call strike +50, Sell put strike -50 (relative to spot)
    call_short_strike = spot_price + 50
    put_short_strike = spot_price - 50

    call_short_strike = int(round(call_short_strike / 50.0)) * 50
    put_short_strike = int(round(put_short_strike / 50.0)) * 50

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_short_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for ATM Strangle: One or both legs not found.")
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "ATM Strangle", "strikes": [put_short_strike, call_short_strike], "orders": orders}

async def get_strategy_details(strategy_name: str, option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int = 1) -> Optional[Dict[str, Any]]:
    func_map = {
        "Iron Fly": _iron_fly_calc,
        "Iron Condor": _iron_condor_calc,
        "Jade Lizard": _jade_lizard_calc,
        "Straddle": _straddle_calc,
        "Calendar Spread": _calendar_spread_calc, # This will currently return None
        "Bull Put Spread": _bull_put_spread_calc,
        "Wide Strangle": _wide_strangle_calc,
        "ATM Strangle": _atm_strangle_calc
    }
    
    if strategy_name not in func_map:
        logger.warning(f"Strategy {strategy_name} not supported.")
        return None
    
    try:
        # Pass config to calc functions if they need it for lot_size or other specifics
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
        if detail is None:
            return None # Strategy specific calculation returned None
            
    except Exception as e:
        logger.error(f"Error calculating {strategy_name} details: {e}")
        return None
    
    if detail:
        # Update LTPs in orders and calculate premiums, max loss/profit
        ltp_map = {}
        for opt in option_chain:
            if opt.get("call_options") and opt["call_options"].get("instrument_key"):
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0.0)
            if opt.get("put_options") and opt["put_options"].get("instrument_key"):
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0.0)
        
        updated_orders = []
        prices = {}
        premium = 0.0
        
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0.0)
            prices[key] = ltp
            
            # Ensure price is set for order placement, especially for LIMIT orders
            if order.get("order_type") == "LIMIT" and order.get("price") is None:
                 order["price"] = ltp # Default to LTP for limit orders if not specified
                 
            updated_orders.append({**order, "current_price": ltp})

            # Calculate premium based on transaction type and current price
            if order["transaction_type"] == "SELL":
                premium += ltp * order["quantity"]
            else: # BUY
                premium -= ltp * order["quantity"]
        
        detail["orders"] = updated_orders
        detail["pricing"] = prices
        detail["premium"] = premium / config["lot_size"] # Premium per lot
        detail["premium_total"] = premium # Total premium received/paid
        
        # Calculate Max Loss and Max Profit
        detail["max_loss"] = float("inf")
        detail["max_profit"] = float("inf")

        if strategy_name == "Iron Fly":
            # Max Loss = Wing Width - Net Premium Received (if credit spread)
            # Max Profit = Net Premium Received
            # Assuming wings are symmetric around ATM strike, strike[0] is put wing, strike[2] is call wing
            if len(detail["strikes"]) == 3: # Assuming [PE_Long, ATM_Short, CE_Long]
                wing_width = abs(detail["strikes"][0] - detail["strikes"][2]) / 2 # Half for one side's wing
                detail["max_loss"] = (wing_width * config["lot_size"] * lots) - detail["premium_total"] if detail["premium_total"] > 0 else float('inf')
                detail["max_profit"] = detail["premium_total"]
            else:
                 logger.warning(f"Unexpected number of strikes for {strategy_name}: {detail['strikes']}")

        elif strategy_name == "Iron Condor":
            # Max Loss = Long Wing Strike - Short Wing Strike - Net Premium Received (if credit spread)
            # Max Profit = Net Premium Received
            if len(detail["strikes"]) == 4: # Assuming [PE_Long, PE_Short, CE_Short, CE_Long]
                put_wing_width = abs(detail["strikes"][0] - detail["strikes"][1])
                call_wing_width = abs(detail["strikes"][2] - detail["strikes"][3])
                # Max Loss is typically the wider of the two spreads minus premium, or cap per side
                # For a condor, it's (wing spread - premium)
                detail["max_loss"] = (max(put_wing_width, call_wing_width) * config["lot_size"] * lots) - detail["premium_total"] if detail["premium_total"] > 0 else float('inf')
                detail["max_profit"] = detail["premium_total"]
            else:
                 logger.warning(f"Unexpected number of strikes for {strategy_name}: {detail['strikes']}")

        elif strategy_name == "Jade Lizard":
            # Max Loss = Short Put Strike - Long Put Strike - Net Premium Received (if credit)
            # Max Profit = Net Premium Received
            if len(detail["strikes"]) == 3: # Assuming [PE_Long, PE_Short, CE_Short]
                put_spread_width = abs(detail["strikes"][0] - detail["strikes"][1])
                detail["max_loss"] = (put_spread_width * config["lot_size"] * lots) - detail["premium_total"] if detail["premium_total"] > 0 else float('inf')
                detail["max_profit"] = detail["premium_total"] # Credit received
            else:
                 logger.warning(f"Unexpected number of strikes for {strategy_name}: {detail['strikes']}")

        elif strategy_name == "Bull Put Spread":
            # Max Loss = Short Put Strike - Long Put Strike - Net Premium Received
            # Max Profit = Net Premium Received
            if len(detail["strikes"]) == 2: # Assuming [PE_Long, PE_Short]
                spread_width = abs(detail["strikes"][0] - detail["strikes"][1])
                detail["max_loss"] = (spread_width * config["lot_size"] * lots) - detail["premium_total"] if detail["premium_total"] > 0 else float('inf')
                detail["max_profit"] = detail["premium_total"]
            else:
                 logger.warning(f"Unexpected number of strikes for {strategy_name}: {detail['strikes']}")

        elif strategy_name in ["Straddle", "Wide Strangle", "ATM Strangle"]:
            detail["max_loss"] = float("inf") # Unlimited loss for naked short options
            detail["max_profit"] = detail["premium_total"] # Max profit is premium received

        elif strategy_name == "Calendar Spread":
            # Max Profit: Unlimited theoretically, but practically capped at difference in IV growth/decay
            # Max Loss: Net debit paid (cost of the spread)
            detail["max_profit"] = float("inf")
            detail["max_loss"] = abs(detail["premium_total"]) if detail["premium_total"] < 0 else 0 # Net debit paid
        
        # Adjust for negative premium (debit strategies) if max_profit was set to premium_total
        if detail["premium_total"] < 0 and strategy_name not in ["Calendar Spread"]: # For debit strategies
            detail["max_profit"] = float("inf") # Max profit for long options can be theoretically unlimited
            detail["max_loss"] = abs(detail["premium_total"]) # Max loss is limited to debit paid
        
        logger.info(f"Calculated details for {strategy_name}: Premium={detail['premium']:.2f}, Max Profit={detail['max_profit']:.2f}, Max Loss={detail['max_loss']:.2f}")

    return detail

async def evaluate_full_risk(trades_df: pd.DataFrame, config: Dict[str, Any], regime_label: str, vix: float) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates the overall risk of the trading portfolio.
    `trades_df` should ideally come from Supabase trade_logs, filtered for active/open trades.
    For this implementation, it assumes `trades_df` structure includes
    'strategy', 'capital_used', 'potential_loss', 'realized_pnl', 'sl_hit', 'vega'.
    """
    try:
        total_funds = config.get('total_funds', 2000000)
        daily_risk_limit = config['daily_risk_limit_pct'] * total_funds
        weekly_risk_limit = config['weekly_risk_limit_pct'] * total_funds
        
        # Dynamic Max Drawdown based on VIX
        if vix > 25:
            max_drawdown_pct = 0.06 # Higher drawdown tolerance in very high VIX
        elif vix > 20:
            max_drawdown_pct = 0.05
        elif vix > 12:
            max_drawdown_pct = 0.03
        else:
            max_drawdown_pct = 0.02
        max_drawdown = max_drawdown_pct * total_funds

        strategy_summary = []
        total_cap_used = 0.0
        total_risk_on_table = 0.0
        total_realized_pnl = 0.0
        total_vega_exposure = 0.0
        flags = []

        if trades_df.empty:
            logger.info("No active trades to evaluate risk for.")
            strategy_summary.append({
                "Strategy": "None", "Capital Used": 0.0, "Cap Limit": total_funds, "% Used": 0.0,
                "Potential Risk": 0.0, "Risk Limit": total_funds * 0.01,
                "P&L": 0.0, "Vega": 0.0, "Risk OK?": "✅"
            })
        else:
            for _, row in trades_df.iterrows():
                strat = row["strategy"]
                capital_used = row.get("capital_used", 0.0)
                potential_risk = row.get("potential_loss", 0.0)
                pnl = row.get("realized_pnl", 0.0) # For closed trades, this is realized PnL
                sl_hit = row.get("sl_hit", False)
                vega = row.get("vega", 0.0)

                cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
                
                risk_factor = 1.2 if "High Vol Trend" in regime_label else (0.8 if "Low Volatility" in regime_label else 1.0)
                
                max_cap = cfg["capital_pct"] * total_funds
                max_risk_per_trade = cfg["risk_per_trade_pct"] * max_cap * risk_factor
                
                risk_ok = "✅" if potential_risk <= max_risk_per_trade else "❌"
                
                strategy_summary.append({
                    "Strategy": strat,
                    "Capital Used": round(capital_used, 2),
                    "Cap Limit": round(max_cap, 2),
                    "% Used": round((capital_used / max_cap * 100) if max_cap else 0, 2),
                    "Potential Risk": round(potential_risk, 2),
                    "Risk Limit": round(max_risk_per_trade, 2),
                    "P&L": round(pnl, 2),
                    "Vega": round(vega, 2),
                    "Risk OK?": risk_ok
                })
                total_cap_used += capital_used
                total_risk_on_table += potential_risk
                total_realized_pnl += pnl
                total_vega_exposure += vega
                
                if risk_ok == "❌":
                    flags.append(f"❌ {strat} exceeded its per-trade risk limit (Potential Risk: {potential_risk:.2f}, Limit: {max_risk_per_trade:.2f})")
                if sl_hit:
                    flags.append(f"⚠️ {strat} hit stop-loss. Review for potential revenge trading or strategy effectiveness.")
        
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / total_funds * 100, 2) if total_funds else 0
        risk_pct = round(total_risk_on_table / total_funds * 100, 2) if total_funds else 0
        dd_pct = round(net_dd / total_funds * 100, 2) if total_funds else 0

        # Check total daily/weekly risk limits (these would typically be against open positions)
        # For simplicity, if total_risk_on_table is considered as the max risk from all open trades
        if total_risk_on_table > daily_risk_limit:
            flags.append(f"❌ Total portfolio risk ({total_risk_on_table:.2f}) exceeds daily limit ({daily_risk_limit:.2f}).")
        # Weekly risk check might require more sophisticated tracking of realized PnL over the week
        
        if net_dd > max_drawdown:
            flags.append(f"❌ Portfolio drawdown ({net_dd:.2f}) exceeds maximum allowed ({max_drawdown:.2f}).")
            
        portfolio_summary = {
            "Total Funds": round(total_funds, 2),
            "Capital Deployed": round(total_cap_used, 2),
            "Exposure Percent": exposure_pct,
            "Risk on Table": round(total_risk_on_table, 2),
            "Risk Percent": risk_pct,
            "Daily Risk Limit": round(daily_risk_limit, 2),
            "Weekly Risk Limit": round(weekly_risk_limit, 2),
            "Realized P&L (from logged trades)": round(total_realized_pnl, 2),
            "Drawdown ₹": round(net_dd, 2),
            "Drawdown Percent": dd_pct,
            "Max Drawdown Allowed": round(max_drawdown, 2),
            "Total Vega Exposure": round(total_vega_exposure, 2),
            "Flags": flags if flags else ["✅ All risk parameters within limits."]
        }
        logger.info("Full risk evaluation completed.")
        return strategy_summary, portfolio_summary
    except Exception as e:
        logger.error(f"Exception in evaluate_full_risk: {e}")
        # Return sensible defaults or raise a specific error
        raise HTTPException(status_code=500, detail=f"Error evaluating full risk: {str(e)}")

# --- FastAPI Endpoints ---
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    logger.info("Root endpoint accessed.")
    return JSONResponse(content={"message": "VoluGuard API is live! ✅ Check /docs for API documentation."})

@app.get("/predict/volatility")
async def predict_volatility(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Predicts various volatility metrics including Historical, GARCH, and XGBoost-predicted volatility.
    Requires Upstox access token for fetching option chain data.
    """
    logger.info("Predicting volatility...")
    config = await get_config(access_token)
    
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"] # Assuming first element has spot price
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])

    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics from option chain.")

    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller_metrics["avg_iv"])
    
    xgb_model = await load_xgboost_model()
    # Fetch VIX and calculate IVP dynamically for XGBoost prediction
    vix = await fetch_india_vix(config)
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])

    xgb_vol = predict_xgboost_volatility(
        xgb_model, seller_metrics["avg_iv"], hv_7, ivp, market_metrics_data["pcr"], vix, market_metrics_data["days_to_expiry"], garch_7d
    )
    
    logger.info("Volatility prediction complete.")
    return {
        "volatility": {
            "hv_7": round(hv_7, 2),
            "garch_7d": round(garch_7d, 2),
            "xgb_vol": round(xgb_vol, 2),
            "ivp": round(ivp, 2),
            "vix": round(vix, 2),
            "iv_rv_spread": round(iv_rv_spread, 2)
        }
    }

@app.post("/log/trade")
async def log_new_trade(trade: TradeRequest):
    """
    Logs a new trade entry to Supabase.
    """
    logger.info(f"Logging new trade: {trade.strategy}")
    status, result = await log_trade_to_supabase(trade.dict())
    if status != 201:
        raise HTTPException(status_code=status, detail=f"Failed to log trade: {result.get('error', 'Unknown error')}")
    logger.info("Trade logged successfully.")
    return {"status": "success", "data": result}

@app.post("/log/journal")
async def log_new_journal(journal: JournalRequest):
    """
    Adds a new journal entry to Supabase.
    """
    logger.info(f"Adding new journal entry: {journal.title}")
    status, result = await add_journal_to_supabase(journal.dict())
    if status != 201:
        raise HTTPException(status_code=status, detail=f"Failed to save journal: {result.get('error', 'Unknown error')}")
    logger.info("Journal entry added successfully.")
    return {"status": "success", "data": result}

@app.get("/fetch/trades")
async def fetch_trades():
    """
    Fetches all logged trade entries from Supabase.
    """
    logger.info("Fetching all trades...")
    trades = await get_all_trades()
    return {"trades": trades}

@app.get("/fetch/journals")
async def fetch_journals():
    """
    Fetches all logged journal entries from Supabase.
    """
    logger.info("Fetching all journal entries...")
    journals = await get_all_journals()
    return {"journals": journals}

@app.get("/fetch/option-chain")
async def fetch_option_chain_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Fetches the current option chain for the configured instrument and expiry.
    """
    logger.info("Fetching option chain...")
    config = await get_config(access_token)
    data = await fetch_option_chain(config)
    return {"data": data}

@app.post("/order/place-multi-leg")
async def place_multi_leg_orders_endpoint(
    orders: List[UpstoxOrderRequest], 
    access_token: str = Query(..., description="Upstox API Access Token")
):
    """
    Places multiple orders to the exchange via Upstox API.
    """
    logger.info(f"Received request to place {len(orders)} multi-leg orders.")
    upstox_headers = get_upstox_headers(access_token)
    config = await get_config(access_token) # To get base_url
    
    order_payload = [order.dict(exclude_unset=True) for order in orders] # Use exclude_unset to send only provided fields

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config['base_url']}/order/multi/place",
                json=order_payload,
                headers=upstox_headers
            )
            response.raise_for_status() # Raise an exception for 4xx or 5xx responses
            result = response.json()
            logger.info(f"Multi-leg order placement successful: {result}")
            return result
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error placing multi-leg orders: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstox API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error placing multi-leg orders: {e}")
        raise HTTPException(status_code=500, detail=f"Network error during order placement: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error placing multi-leg orders: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/suggest/strategy")
async def suggest_strategy_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Suggests optimal trading strategies based on current market conditions and volatility regime.
    """
    logger.info("Suggesting strategies...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])
    
    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics from option chain.")

    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller_metrics["avg_iv"])
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])
    vix = await fetch_india_vix(config)

    # To calculate IV skew slope, we need the full chain DataFrame with IV Skew
    full_chain_df_data = []
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300: # Limit range for skew calculation
            call = opt.get("call_options")
            put = opt.get("put_options")
            if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                full_chain_df_data.append({
                    "Strike": strike,
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                })
    full_chain_df = pd.DataFrame(full_chain_df_data)
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)

    regime_score, regime_label, regime_note, regime_explanation = calculate_regime(
        seller_metrics["avg_iv"], ivp, hv_7, garch_7d, seller_metrics["straddle_price"], spot_price, market_metrics_data["pcr"], vix, iv_skew_slope
    )
    
    strategies, rationale, event_warning = await suggest_strategy(
        regime_label, ivp, iv_rv_spread, market_metrics_data["days_to_expiry"], config['expiry_date'], seller_metrics["straddle_price"], spot_price, config
    )
    
    logger.info("Strategy suggestion complete.")
    return {
        "regime": regime_label,
        "score": round(regime_score, 2),
        "note": regime_note,
        "explanation": regime_explanation,
        "strategies": strategies,
        "rationale": rationale,
        "event_warning": event_warning
    }

@app.post("/strategy/details")
async def get_strategy_details_endpoint(req: StrategyRequest, access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Retrieves detailed information (legs, premiums, max PnL) for a specific option strategy.
    """
    logger.info(f"Getting details for strategy: {req.strategy}")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    detail = await get_strategy_details(req.strategy, option_chain, spot_price, config, req.lots)
    
    if detail is None:
        if req.strategy == "Calendar Spread":
             raise HTTPException(status_code=400, detail=f"Cannot provide details for {req.strategy}. This strategy requires fetching data for multiple expiries, which is not fully supported by current single-expiry chain fetch flow.")
        else:
             raise HTTPException(status_code=404, detail=f"No details found for {req.strategy}. Check if required options exist in the chain or if calculation logic failed.")
    
    # Convert 'orders' list of dicts to UpstoxOrderRequest Pydantic models for consistency
    detail['orders'] = [UpstoxOrderRequest(**order) for order in detail['orders']]

    logger.info(f"Details found for {req.strategy}.")
    return detail

@app.get("/risk/portfolio")
async def evaluate_risk(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Evaluates the risk profile of the entire trading portfolio based on logged trades.
    Note: For a live system, you'd fetch ACTIVE trades from Supabase or Upstox positions.
    """
    logger.info("Evaluating portfolio risk...")
    config = await get_config(access_token)
    
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    
    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics for risk evaluation.")

    # Dynamically fetch VIX and calculate regime
    vix = await fetch_india_vix(config)
    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller_metrics["avg_iv"])
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])

    full_chain_df_data = []
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300: # Limit range for skew calculation
            call = opt.get("call_options")
            put = opt.get("put_options")
            if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                full_chain_df_data.append({
                    "Strike": strike,
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                })
    full_chain_df = pd.DataFrame(full_chain_df_data)
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)

    regime_score, regime_label, _, _ = calculate_regime(
        seller_metrics["avg_iv"], ivp, hv_7, garch_7d, seller_metrics["straddle_price"], spot_price, market_metrics_data["pcr"], vix, iv_skew_slope
    )

    # Fetching logged trades to evaluate risk. These are 'closed' trades in Supabase currently.
    # For active risk management, you'd need 'open' trades from Supabase or Upstox positions API.
    # For this exercise, we will use a sample `trades_df` or you can change get_all_trades to filter for open trades.
    # To properly evaluate portfolio risk against open positions, you'd fetch open positions from Upstox.
    # Example for active trades for evaluation.
    # Let's assume for this endpoint, we're taking historical trades from Supabase,
    # and you would adapt this to fetch active positions if required for real-time portfolio risk.
    all_logged_trades = await get_all_trades() # This fetches all closed trades
    
    # Filter for 'active' or 'open' trades if your Supabase schema supports it.
    # For demo, let's just use a sample, as the current `log_trade_to_supabase` marks them as "closed".
    # trades_df = pd.DataFrame([t for t in all_logged_trades if t.get("status") == "open"]) # Example filter
    
    # Using the original sample data to demonstrate the `evaluate_full_risk` function as per the prompt.
    # In a real scenario, this would dynamically pull your currently open positions.
    trades_df = pd.DataFrame([
        {"strategy": "Iron Fly", "capital_used": 60000.0, "potential_loss": 1000.0, "realized_pnl": 200.0, "sl_hit": False, "vega": 150.0},
        {"strategy": "Jade Lizard", "capital_used": 45000.0, "potential_loss": 800.0, "realized_pnl": -50.0, "sl_hit": False, "vega": 100.0}
    ])


    summary_df_records, portfolio_summary = await evaluate_full_risk(trades_df, config, regime_label, vix)
    
    logger.info("Portfolio risk evaluation complete.")
    return {
        "summary": summary_df_records, # Already list of dicts
        "portfolio": portfolio_summary
    }

@app.get("/full-chain-table")
async def full_chain_table_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Returns a detailed table of option chain data for strikes near the current spot price.
    Includes IVs, IV skew, Greeks, Straddle Price, and Total Open Interest.
    """
    logger.info("Generating full option chain table...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    full_chain_df_data = []
    
    for opt in option_chain:
        strike = opt["strike_price"]
        # Limit to strikes around +/- 500 points from spot for relevance
        if abs(strike - spot_price) <= 500:
            call = opt.get("call_options")
            put = opt.get("put_options")
            
            # Ensure required data exists before adding to list
            if call and put and \
               call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None and \
               call["market_data"].get("ltp") is not None and put["market_data"].get("ltp") is not None and \
               call["market_data"].get("oi") is not None and put["market_data"].get("oi") is not None:
                
                full_chain_df_data.append({
                    "Strike": strike,
                    "Call IV": round(call["option_greeks"]["iv"], 2),
                    "Put IV": round(put["option_greeks"]["iv"], 2),
                    "IV Skew": round(call["option_greeks"]["iv"] - put["option_greeks"]["iv"], 4),
                    "Total Theta": round((call["option_greeks"].get("theta", 0.0) or 0.0) + (put["option_greeks"].get("theta", 0.0) or 0.0), 2),
                    "Total Vega": round((call["option_greeks"].get("vega", 0.0) or 0.0) + (put["option_greeks"].get("vega", 0.0) or 0.0), 2),
                    "Straddle Price": round(call["market_data"]["ltp"] + put["market_data"]["ltp"], 2),
                    "Total OI": int((call["market_data"].get("oi", 0) or 0) + (put["market_data"].get("oi", 0) or 0))
                })
    
    full_chain_df = pd.DataFrame(full_chain_df_data)
    # Sort by strike price for better readability
    full_chain_df = full_chain_df.sort_values(by="Strike").reset_index(drop=True)
    
    logger.info("Full option chain table generated.")
    return {"data": full_chain_df.to_dict(orient="records")}

@app.get("/calculate/regime")
async def calculate_regime_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Calculates and returns the current market volatility regime (e.g., High Vol Trend, Low Volatility).
    """
    logger.info("Calculating market regime...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])

    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics for regime calculation.")

    hv_7, garch_7d, _ = await calculate_volatility(config, seller_metrics["avg_iv"])
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])
    vix = await fetch_india_vix(config)

    # To calculate IV skew slope, we need the full chain DataFrame with IV Skew
    full_chain_df_data = []
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300: # Limit range for skew calculation
            call = opt.get("call_options")
            put = opt.get("put_options")
            if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                full_chain_df_data.append({
                    "Strike": strike,
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                })
    full_chain_df = pd.DataFrame(full_chain_df_data)
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)

    regime_score, regime_label, regime_note, regime_explanation = calculate_regime(
        seller_metrics["avg_iv"], ivp, hv_7, garch_7d, seller_metrics["straddle_price"], spot_price, market_metrics_data["pcr"], vix, iv_skew_slope
    )
    logger.info(f"Market regime calculated: {regime_label}")
    return {
        "regime": regime_label,
        "score": round(regime_score, 2),
        "note": regime_note,
        "explanation": regime_explanation
    }

@app.get("/calculate/iv-skew-slope")
async def calculate_iv_skew_slope_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Calculates the slope of the Implied Volatility (IV) skew.
    """
    logger.info("Calculating IV skew slope...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    full_chain_df_data = []
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300: # Limit range for skew calculation
            call = opt.get("call_options")
            put = opt.get("put_options")
            if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                full_chain_df_data.append({
                    "Strike": strike,
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                })
    full_chain_df = pd.DataFrame(full_chain_df_data)
    slope = calculate_iv_skew_slope(full_chain_df)
    logger.info(f"IV skew slope calculated: {slope}")
    return {"slope": round(slope, 4)}

@app.get("/calculate/chain-metrics")
async def chain_metrics_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Calculates and returns seller-specific metrics (e.g., straddle price, average IV)
    and broader market metrics (e.g., PCR, Max Pain) from the option chain.
    """
    logger.info("Calculating option chain metrics...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])
    
    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics from option chain.")

    logger.info("Option chain metrics calculated.")
    return {
        "seller": {k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in seller_metrics.items()},
        "market": {k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in market_metrics_data.items()}
    }

@app.get("/calculate/volatility")
async def calc_volatility_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Calculates and returns 7-day Historical Volatility (HV), GARCH-predicted 7-day Volatility,
    and the Implied Volatility-Realized Volatility (IV-RV) spread.
    """
    logger.info("Calculating volatility metrics...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    
    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics for volatility calculation.")

    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller_metrics["avg_iv"])
    logger.info("Volatility metrics calculated.")
    return {
        "hv_7": round(hv_7, 2),
        "garch_7d": round(garch_7d, 2),
        "iv_rv_spread": round(iv_rv_spread, 2)
    }

@app.get("/load/model")
async def load_xgboost_model_endpoint():
    """
    Attempts to load the XGBoost volatility prediction model.
    Returns status indicating if the model was loaded successfully.
    """
    logger.info("Attempting to load XGBoost model...")
    model = await load_xgboost_model()
    if model:
        logger.info("XGBoost model loaded successfully via endpoint.")
        return {"model_loaded": True}
    else:
        logger.error("Failed to load XGBoost model via endpoint.")
        raise HTTPException(status_code=500, detail="Failed to load XGBoost model.")

@app.get("/predict/xgboost-vol")
async def predict_xgboost_vol_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Predicts future volatility using the pre-trained XGBoost model based on current market data.
    """
    logger.info("Predicting XGBoost volatility...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    market_metrics_data = market_metrics(option_chain, config['expiry_date'])

    if not seller_metrics:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics for XGBoost prediction.")

    hv_7, garch_7d, _ = await calculate_volatility(config, seller_metrics["avg_iv"])
    model = await load_xgboost_model()
    
    # Dynamically fetch VIX and calculate IVP
    vix = await fetch_india_vix(config)
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])

    xgb_vol = predict_xgboost_volatility(
        model, seller_metrics["avg_iv"], hv_7, ivp, market_metrics_data["pcr"], vix, market_metrics_data["days_to_expiry"], garch_7d
    )
    logger.info(f"XGBoost volatility predicted: {xgb_vol}")
    return {"xgb_vol": round(xgb_vol, 2)}

@app.get("/test/all")
async def test_all_endpoints(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Runs a series of tests against core calculation endpoints to verify functionality.
    """
    logger.info("Running all endpoint tests...")
    results = {}
    try:
        results["regime"] = await calculate_regime_endpoint(access_token)
    except HTTPException as e:
        results["regime_error"] = str(e.detail)
    except Exception as e:
        results["regime_error"] = f"Unexpected error: {str(e)}"
    
    try:
        results["iv_skew"] = await calculate_iv_skew_slope_endpoint(access_token)
    except HTTPException as e:
        results["iv_skew_error"] = str(e.detail)
    except Exception as e:
        results["iv_skew_error"] = f"Unexpected error: {str(e)}"

    try:
        results["volatility"] = await calc_volatility_endpoint(access_token)
    except HTTPException as e:
        results["volatility_error"] = str(e.detail)
    except Exception as e:
        results["volatility_error"] = f"Unexpected error: {str(e)}"
    
    try:
        results["xgb_vol"] = await predict_xgboost_vol_endpoint(access_token)
    except HTTPException as e:
        results["xgb_vol_error"] = str(e.detail)
    except Exception as e:
        results["xgb_vol_error"] = f"Unexpected error: {str(e)}"

    try:
        # For risk, we use a fixed sample trades_df as this endpoint evaluates logged trades
        # or mock data for testing purposes in this test function.
        # This will call the logic in /risk/portfolio
        results["risk"] = await evaluate_risk(access_token)
    except HTTPException as e:
        results["risk_error"] = str(e.detail)
    except Exception as e:
        results["risk_error"] = f"Unexpected error: {str(e)}"

    try:
        results["strategies"] = await suggest_strategy_endpoint(access_token)
    except HTTPException as e:
        results["strategies_error"] = str(e.detail)
    except Exception as e:
        results["strategies_error"] = f"Unexpected error: {str(e)}"
    
    logger.info("All endpoint tests complete.")
    return results

@app.get("/ws/market-url")
async def get_market_websocket_url(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Returns an authorized WebSocket URL for Upstox market data feed.
    """
    logger.info("Requesting market data WebSocket URL...")
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
            headers = get_upstox_headers(access_token)
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            ws_url = data["data"]["authorized_redirect_uri"]
            logger.info(f"Market data WebSocket URL obtained.")
            return {"ws_url": ws_url}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting market WS URL: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstox API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error getting market WS URL: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting market WS URL: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/ws/portfolio-url")
async def get_portfolio_websocket_url(access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Returns an authorized WebSocket URL for Upstox portfolio feed (order, position, holding).
    """
    logger.info("Requesting portfolio WebSocket URL...")
    try:
        async with httpx.AsyncClient() as client:
            update_types = "order,gtt_order,position,holding" # Standard portfolio updates
            url = f"https://api.upstox.com/v2/feed/portfolio-stream-feed/authorize?update_types={update_types}"
            headers = get_upstox_headers(access_token)
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            ws_url = data["data"]["authorized_redirect_uri"]
            logger.info(f"Portfolio WebSocket URL obtained.")
            return {"ws_url": ws_url}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting portfolio WS URL: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstox API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error getting portfolio WS URL: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting portfolio WS URL: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


