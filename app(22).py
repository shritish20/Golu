from functools import lru_cache
import httpx
from fastapi import FastAPI, Depends, HTTPException, Query, Security
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from arch import arch_model
from scipy.stats import linregress
import xgboost as xgb
import pickle
from io import BytesIO
import base64
import os
from time import time

# --- Authentication ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def verify_token(token: str = Depends(oauth2_scheme)):
    # In a real application, you would validate the token against an authentication service
    # For now, we'll just check if it's not empty
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- SUPABASE CLIENT ---
async def log_trade_to_supabase(data: dict):
    data["timestamp_entry"] = datetime.utcnow().isoformat()
    data["timestamp_exit"] = datetime.utcnow().isoformat()
    data["status"] = "closed"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{settings.SUPABASE_URL}/rest/v1/trade_logs", json=data, headers=settings.HEADERS)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.status_code, response.json()
        except httpx.RequestError as e:
            logger.error(f"Error logging trade to Supabase: {e}")
            return 500, {"error": str(e)}

async def add_journal_to_supabase(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{settings.SUPABASE_URL}/rest/v1/journals", json=data, headers=settings.HEADERS)
            response.raise_for_status()
            return response.status_code, response.json()
        except httpx.RequestError as e:
            logger.error(f"Error adding journal entry to Supabase: {e}")
            return 500, {"error": str(e)}

class Settings:
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")
    HEADERS: Dict[str, str] = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    EVENT_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv"
    IVP_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv"
    NIFTY_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
    XGBOOST_MODEL_URL: str = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"

settings = Settings()
app = FastAPI(title="VoluGuard API", description="Wrapped from Streamlit backend", dependencies=[Depends(verify_token)])

# --- FastAPI Models ---
class TradeRequest(BaseModel):
    strategy: str = Field(..., min_length=1, max_length=50)
    instrument_token: str = Field(..., min_length=1, max_length=100)
    entry_price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    realized_pnl: float
    unrealized_pnl: float
    regime_score: Optional[float] = None
    notes: Optional[str] = Field("", max_length=500)

class JournalRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=2000)
    mood: str = Field(..., min_length=1, max_length=50)
    tags: Optional[str] = Field("", max_length=200)

class StrategyRequest(BaseModel):
    strategy: str = Field(..., min_length=1, max_length=50)
    lots: int = Field(1, gt=0)

# --- Helper Functions for Database ---
async def log_trade(trade_data):
    status, result = await log_trade_to_supabase(trade_data)
    if status == 201:
        logger.info("Trade logged successfully.")
        return True
    else:
        logger.error(f"Error logging trade to Supabase: {result}")
        return False

async def add_journal_entry(entry_data):
    status, result = await add_journal_to_supabase(entry_data)
    if status == 201:
        logger.info("Journal entry added successfully.")
        return True
    else:
        logger.error(f"Error adding journal entry to Supabase: {result}")
        return False

async def get_all_trades():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{settings.SUPABASE_URL}/rest/v1/trade_logs", headers=settings.HEADERS)
            response.raise_for_status()
            logger.info("Trades fetched successfully.")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error fetching trades: {e}")
            return []

async def get_all_journals():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{settings.SUPABASE_URL}/rest/v1/journals", headers=settings.HEADERS)
            response.raise_for_status()
            logger.info("Journals fetched successfully.")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error fetching journals: {e}")
            return []    config["expiry_date"] = await get_next_expiry_internal()
    return config

# --- Data Fetching and Calculation Functions ---
@lru_cache(maxsize=128)
async def fetch_option_chain(config):
    async with httpx.AsyncClient() as client:
        try:
            url = f"{config["base_url"]}/option/chain"
            params = {"instrument_key": config["instrument_key"], "expiry_date": config["expiry_date"]}
            res = await client.get(url, headers=config["headers"], params=params)
            res.raise_for_status()
            return res.json()["data"]
        except httpx.RequestError as e:
            logger.error(f"Exception in fetch_option_chain: {e}")
            return []

@lru_cache(maxsize=128)
async def calculate_volatility(config, seller_avg_iv):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(config["nifty_url"])
            response.raise_for_status()
            df = pd.read_csv(BytesIO(response.content))
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
        df = df.sort_values("Date")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df.dropna(inplace=True)
        hv_7 = np.std(df["Log_Returns"][-7:]) * np.sqrt(252) * 100
        model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=7)
        garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
        iv_rv_spread = round(seller_avg_iv - hv_7, 2)
        return hv_7, garch_7d, iv_rv_spread
    except Exception as e:
        logger.error(f"Exception in calculate_volatility: {e}")
        return 0, 0, 0

@lru_cache(maxsize=1)
async def load_xgboost_model():
    try:
        model_url = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
        async with httpx.AsyncClient() as client:
            response = await client.get(model_url)
            response.raise_for_status()
            model = pickle.load(BytesIO(response.content))
            return model
        logger.error(f"Error fetching XGBoost model: {response.status_code} - {response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Exception in load_xgboost_model: {e}")
        return None
async def predict_xgboost_volatility(model, atm_iv, realized_vol, ivp, pcr, vix, days_to_expiry, garch_vol):
    try:
        features = pd.DataFrame({
            "ATM_IV": [atm_iv],
            "Realized_Vol": [realized_vol],
            "IVP": [ivp],
            "PCR": [pcr],
            "VIX": [vix],
            "Days_to_Expiry": [days_to_expiry],
            "GARCH_Predicted_Vol": [garch_vol]
        })
        if model is not None:
            prediction = model.predict(features)[0]
            return round(float(prediction), 2)
        return 0
    except Exception as e:
        logger.error(f"Exception in predict_xgboost_volatility: {e}")
        return 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty:
            return 0
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        logger.error(f"Exception in calculate_iv_skew_slope: {e}")
        return 0

def calculate_regime(atm_iv, ivp, realized_vol, garch_vol, straddle_price, spot_price, pcr, vix, iv_skew_slope):
    expected_move = (straddle_price / spot_price) * 100
    vol_spread = atm_iv - realized_vol
    regime_score = 0
    regime_score += 10 if ivp > 80 else -10 if ivp < 20 else 0
    regime_score += 10 if vol_spread > 10 else -10 if vol_spread < -10 else 0
    regime_score += 10 if vix > 20 else -10 if vix < 10 else 0
    regime_score += 5 if pcr > 1.2 else -5 if pcr < 0.8 else 0
    regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0
    regime_score += 10 if expected_move > 0.05 else -10 if expected_move < 0.02 else 0
    regime_score += 5 if garch_vol > realized_vol * 1.2 else -5 if garch_vol < realized_vol * 0.8 else 0
    if regime_score > 20:
        return regime_score, ":fire: High Vol Trend", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
    elif regime_score > 10:
        return regime_score, ":zap: Elevated Volatility", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
    elif regime_score > -10:
        return regime_score, ":smile: Neutral Volatility", "Balanced market — flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return regime_score, ":chart_with_downwards_trend: Low Volatility", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."

async def suggest_strategy(regime_label, ivp, iv_minus_rv, days_to_expiry, event_df, expiry_date, straddle_price, spot_price):
    strategies = []
    rationale = []
    event_warning = None
    event_window = 3 if ivp > 80 else 2
    high_impact_event_near = False
    event_impact_score = 0
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(event_df) # event_df is actually a URL here
            response.raise_for_status()
            event_df = pd.read_csv(BytesIO(response.content))
        except httpx.RequestError as e:
            logger.error(f"Error fetching event data: {e}")
            event_df = pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

    for _, row in event_df.iterrows():
        try:
            dt = pd.to_datetime(row["Datetime"])
            level = row["Classification"]
            if level == "High" and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
                high_impact_event_near = True
            if level == "High" and pd.notnull(row["Forecast"]) and pd.notnull(row["Prior"]):
                forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                if abs(forecast - prior) > 0.5:
                    event_impact_score += 1
        except Exception as e:
            logger.warning(f"Error processing event row: {e}")
            continue
    if high_impact_event_near:
        event_warning = f":warning: High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    expected_move_pct = (straddle_price / spot_price) * 100
    if regime_label == ":fire: High Vol Trend":
        strategies = ["Iron Fly", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == ":zap: Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == ":smile: Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
    elif regime_label == ":chart_with_downwards_trend: Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

def evaluate_full_risk(trades_df, config, regime_label, vix):
    try:
        total_funds = config.get("total_funds", 2000000)
        daily_risk_limit = config["daily_risk_limit_pct"] * total_funds
        weekly_risk_limit = config["weekly_risk_limit_pct"] * total_funds
        max_drawdown_pct = 0.05 if vix > 20 else 0.03 if vix > 12 else 0.02
        max_drawdown = max_drawdown_pct * total_funds
        strategy_summary = []
        total_cap_used = total_risk_used = total_realized_pnl = total_vega = 0
        flags = []
        if trades_df.empty:
            strategy_summary.append({
                "Strategy": "None", "Capital Used": 0, "Cap Limit": total_funds, "% Used": 0,
                "Potential Risk": 0, "Risk Limit": total_funds * 0.01,
                "P&L": 0, "Vega": 0, "Risk OK?": ":white_check_mark:"
            })
        else:
            for _, row in trades_df.iterrows():
                strat = row["strategy"]
                capital_used = row["capital_used"]
                potential_risk = row["potential_loss"]
                pnl = row["realized_pnl"]
                sl_hit = row["sl_hit"]
                vega = row["vega"]
                cfg = config["risk_config"].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
                risk_factor = 1.2 if regime_label == ":fire: High Vol Trend" else 0.8 if regime_label == ":chart_with_downwards_trend: Low Volatility" else 1.0
                max_cap = cfg["capital_pct"] * total_funds
                max_risk = cfg["risk_per_trade_pct"] * max_cap * risk_factor
                risk_ok = potential_risk <= max_risk
                strategy_summary.append({
                    "Strategy": strat,
                    "Capital Used": capital_used,
                    "Cap Limit": round(max_cap),
                    "% Used": round(capital_used / max_cap * 100, 2) if max_cap else 0,
                    "Potential Risk": potential_risk,
                    "Risk Limit": round(max_risk),
                    "P&L": pnl,
                    "Vega": vega,
                    "Risk OK?": ":white_check_mark:" if risk_ok else ":x:"
                })
                total_cap_used += capital_used
                total_risk_used += potential_risk
                total_realized_pnl += pnl
                total_vega += vega
                if not risk_ok:
                    flags.append(f":x: {strat} exceeded risk limit")
                if sl_hit:
                    flags.append(f":warning: {strat} shows possible revenge trading")
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / total_funds * 100, 2) if total_funds else 0
        risk_pct = round(total_risk_used / total_funds * 100, 2) if total_funds else 0
        dd_pct = round(net_dd / total_funds * 100, 2) if total_funds else 0
        portfolio_summary = {
            "Total Funds": total_funds,
            "Capital Deployed": total_cap_used,
            "Exposure Percent": exposure_pct,
            "Risk on Table": total_risk_used,
            "Risk Percent": risk_pct,
            "Daily Risk Limit": daily_risk_limit,
            "Weekly Risk Limit": weekly_risk_limit,
            "Realized P&L": total_realized_pnl,
            "Drawdown ₹": net_dd,
            "Drawdown Percent": dd_pct,
            "Max Drawdown Allowed": max_drawdown,
            "Flags": flags
        }
        return pd.DataFrame(strategy_summary), portfolio_summary
    except Exception as e:
        logger.error(f"Exception in evaluate_full_risk: {e}")
        return pd.DataFrame([{ "Strategy": "None", "Capital Used": 0, "Cap Limit": 2000000, "% Used": 0, "Potential Risk": 0, "Risk Limit": 20000, "P&L": 0, "Vega": 0, "Risk OK?": ":white_check_mark:" }]), {
            "Total Funds": 2000000,
            "Capital Deployed": 0,
            "Exposure Percent": 0,
            "Risk on Table": 0,
            "Risk Percent": 0,
            "Daily Risk Limit": 40000,
            "Weekly Risk Limit": 60000,
            "Realized P&L": 0,
            "Drawdown ₹": 0,
            "Drawdown Percent": 0,
            "Max Drawdown Allowed": 40000,
            "Flags": []
        }

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        logger.warning(f"No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        logger.error(f"Exception in find_option_by_strike: {e}")
        return None

def get_dynamic_wing_distance(ivp, straddle_price):
    if ivp >= 80:
        multiplier = 0.35
    elif ivp <= 20:
        multiplier = 0.2
    else:
        multiplier = 0.25
    raw_distance = straddle_price * multiplier
    return int(round(raw_distance / 50.0)) * 50  # Round to nearest 50 for Nifty

def _iron_fly_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    wing_distance = get_dynamic_wing_distance(80, straddle_price)
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing_distance, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Fly.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Fly", "strikes": [strike + wing_distance, strike, strike - wing_distance], "orders": orders}

def _iron_condor_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    short_wing_distance = get_dynamic_wing_distance(80, straddle_price)
    long_wing_distance = int(round(short_wing_distance * 1.5 / 50)) * 50
    ce_short_opt = find_option_by_strike(option_chain, strike + short_wing_distance, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike - short_wing_distance, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + long_wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - long_wing_distance, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Condor.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Condor", "strikes": [strike + long_wing_distance, strike + short_wing_distance, strike - short_wing_distance, strike - long_wing_distance], "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    put_long_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Jade Lizard.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Jade Lizard", "strikes": [call_strike, put_strike, put_long_strike], "orders": orders}

def _straddle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Straddle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Straddle", "strikes": [strike, strike], "orders": orders}

def _calendar_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    near_leg = find_option_by_strike(option_chain, strike, "CE")
    far_leg = find_option_by_strike(option_chain, strike, "CE")
    if not all([near_leg, far_leg]):
        logger.error("Invalid options for Calendar Spread.")
        return None
    orders = [
        {"instrument_key": near_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": far_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Calendar Spread", "strikes": [strike, strike], "orders": orders}

def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Bull Put Spread.")
        return None
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike], "orders": orders}

def _wide_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 100
    put_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Wide Strangle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike], "orders": orders}

def _atm_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for ATM Strangle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike], "orders": orders}

def get_strategy_details(strategy_name, option_chain, spot_price, config, lots=1):
    func_map = {
        "Iron Fly": _iron_fly_calc,
        "Iron Condor": _iron_condor_calc,
        "Jade Lizard": _jade_lizard_calc,
        "Straddle": _straddle_calc,
        "Calendar Spread": _calendar_spread_calc,
        "Bull Put Spread": _bull_put_spread_calc,
        "Wide Strangle": _wide_strangle_calc,
        "ATM Strangle": _atm_strangle_calc
    }
    if strategy_name not in func_map:
        logger.warning(f"Strategy {strategy_name} not supported.")
        return None
    try:
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
    except Exception as e:
        logger.error(f"Error calculating {strategy_name} details: {e}")
        return None
    if detail:
        ltp_map = {}
        for opt in option_chain:
            if "call_options" in opt and "market_data" in opt["call_options"]:
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0)
            if "put_options" in opt and "market_data" in opt["put_options"]:
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0)
        updated_orders = []
        prices = {}
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0)
            prices[key] = ltp
            updated_orders.append({**order, "current_price": ltp})
        detail["orders"] = updated_orders
        detail["pricing"] = prices
        premium = 0
        for order in detail["orders"]:
            price = order["current_price"]
            qty = order["quantity"]
            if order["transaction_type"] == "SELL":
                premium += price * qty
            else:
                premium -= price * qty
        detail["premium"] = premium / config["lot_size"]
        detail["premium_total"] = premium
        wing_width = 0
        if strategy_name == "Iron Fly":
            wing_width = abs(detail["strikes"][0] - detail["strikes"][2])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float("inf")
        elif strategy_name == "Iron Condor":
            wing_width = abs(detail["strikes"][2] - detail["strikes"][0])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float("inf")
        elif strategy_name == "Jade Lizard":
            wing_width = abs(detail["strikes"][1] - detail["strikes"][2])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float("inf")
        elif strategy_name == "Bull Put Spread":
            wing_width = abs(detail["strikes"][0] - detail["strikes"][1])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float("inf")
        elif strategy_name in ["Straddle", "Wide Strangle", "ATM Strangle"]:
            detail["max_loss"] = float("inf")
        elif strategy_name == "Calendar Spread":
            detail["max_loss"] = detail["premium"]
            detail["max_profit"] = float("inf")
        detail["max_profit"] = detail["premium_total"] if strategy_name != "Calendar Spread" else float("inf")
    return detail

# --- FastAPI Endpoints ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to VoluGuard FastAPI Backend"}

@app.get("/predict/volatility")
async def predict_volatility(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for volatility prediction.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller["avg_iv"])
    xgb_model = await load_xgboost_model()
    xgb_vol = predict_xgboost_volatility(
        xgb_model, seller["avg_iv"], hv_7, 80, market["pcr"], 20, market["days_to_expiry"], garch_7d
    )
    return {
        "volatility": {
            "hv_7": hv_7,
            "garch_7d": garch_7d,
            "xgb_vol": xgb_vol
        }
    }

@app.post("/log/trade")
async def log_new_trade(trade: TradeRequest):
    success = await log_trade(trade.dict())
    if not success:
        raise HTTPException(status_code=500, detail="Failed to log trade")
    return {"status": "success"}

@app.post("/log/journal")
async def log_new_journal(journal: JournalRequest):
    success = await add_journal_entry(journal.dict())
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save journal")
    return {"status": "success"}

@app.get("/fetch/trades")
async def fetch_trades():
    return {"trades": await get_all_trades()}

@app.get("/fetch/journals")
async def fetch_journals():
    return {"journals": await get_all_journals()}

@app.get("/fetch/option-chain")
async def fetch_option_chain_endpoint(access_token: str):
    config = get_config(access_token)
    data = await fetch_option_chain(config)
    return {"data": data}

@app.post("/order/place-multi-leg")
async def place_multi_leg_orders_endpoint(orders: List[Dict], access_token: str):
    config = get_config(access_token)
    # Stub for now
    return {"status": "Order received", "orders": orders}

@app.get("/suggest/strategy")
async def suggest_strategy_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for strategy suggestion.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    ivp = 80  # Placeholder, should be calculated dynamically
    vix = 20  # Placeholder, should be fetched dynamically
    iv_rv_spread = round(seller["avg_iv"] - 18, 2)  # Placeholder, 18 is a magic number
    iv_skew_slope = 0.0005  # Placeholder, should be calculated dynamically
    regime_score, regime_label, regime_note, regime_explanation = calculate_regime(
        seller["avg_iv"], ivp, 18, 22, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope
    )
    event_df = config["event_url"] # This should be fetched from config["event_url"]
    strategies, rationale, event_warning = await suggest_strategy(
        regime_label, ivp, iv_rv_spread, market["days_to_expiry"], event_df, config["expiry_date"], seller["straddle_price"], spot_price
    )
    return {
        "regime": regime_label,
        "score": regime_score,
        "note": regime_note,
        "explanation": regime_explanation,
        "strategies": strategies,
        "rationale": rationale,
        "event_warning": event_warning
    }

@app.post("/strategy/details")
async def get_strategy_details_endpoint(req: StrategyRequest, access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error(f"Failed to fetch option chain for strategy details for {req.strategy}.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    detail = get_strategy_details(req.strategy, option_chain, spot_price, config, req.lots)
    if not detail:
        logger.error(f"No details found for strategy {req.strategy}.")
        raise HTTPException(status_code=404, detail=f"No details found for {req.strategy}")
    return detail

@app.get("/risk/portfolio")
async def evaluate_risk(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for risk portfolio evaluation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    ivp = 80  # Placeholder
    vix = 20  # Placeholder
    iv_rv_spread = round(seller["avg_iv"] - 18, 2)  # Placeholder
    iv_skew_slope = 0.0005  # Placeholder
    regime_score, regime_label, _, _ = calculate_regime(
        seller["avg_iv"], ivp, 18, 22, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope
    )
    trades_df = pd.DataFrame([
        {"strategy": "Iron Fly", "capital_used": 60000, "potential_loss": 1000, "realized_pnl": 200, "sl_hit": False, "vega": 150}
    ]) # This should be fetched from Supabase
    summary_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime_label, vix)
    return {
        "summary": summary_df.to_dict(orient="records"),
        "portfolio": portfolio_summary
    }

@app.get("/full-chain-table")
async def full_chain_table_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for full chain table.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    full_chain_df = pd.DataFrame()
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300:
            call = opt["call_options"]
            put = opt["put_options"]
            full_chain_df = pd.concat([full_chain_df, pd.DataFrame([{
                "Strike": strike,
                "Call IV": call["option_greeks"]["iv"],
                "Put IV": put["option_greeks"]["iv"],
                "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"],
                "Total Theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
                "Total Vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
                "Straddle Price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
                "Total OI": call["market_data"]["oi"] + put["market_data"]["oi"]
            }])])
    return {"data": full_chain_df.to_dict(orient="records")}

@app.get("/calculate/regime")
async def calculate_regime_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for regime calculation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    ivp = 80  # Placeholder
    vix = 20  # Placeholder
    iv_rv_spread = round(seller["avg_iv"] - 18, 2)  # Placeholder
    iv_skew_slope = 0.0005  # Placeholder
    regime_score, regime_label, regime_note, regime_explanation = calculate_regime(
        seller["avg_iv"], ivp, 18, 22, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope
    )
    return {
        "regime": regime_label,
        "score": regime_score,
        "note": regime_note,
        "explanation": regime_explanation
    }

@app.get("/calculate/iv-skew-slope")
async def calculate_iv_skew_slope_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for IV skew slope calculation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    full_chain_df = pd.DataFrame()
    for opt in option_chain:
        strike = opt["strike_price"]
        if abs(strike - spot_price) <= 300:
            call = opt["call_options"]
            put = opt["put_options"]
            full_chain_df = pd.concat([full_chain_df, pd.DataFrame([{
                "Strike": strike,
                "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
            }])])
    slope = calculate_iv_skew_slope(full_chain_df)
    return {"slope": slope}

@app.get("/calculate/chain-metrics")
async def chain_metrics_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for chain metrics calculation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    return {
        "seller": seller,
        "market": market
    }

@app.get("/calculate/volatility")
async def calc_volatility_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for volatility calculation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller["avg_iv"])
    return {
        "hv_7": hv_7,
        "garch_7d": garch_7d,
        "iv_rv_spread": iv_rv_spread
    }

@app.get("/load/model")
async def load_xgboost_model_endpoint():
    model = await load_xgboost_model()
    if model:
        logger.info("XGBoost model loaded successfully.")
        return {"model_loaded": True}
    logger.error("Failed to load XGBoost model.")
    return {"model_loaded": False}

@app.get("/predict/xgboost-vol")
async def predict_xgboost_vol_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for XGBoost volatility prediction.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    model = await load_xgboost_model()
    xgb_vol = predict_xgboost_volatility(
        model, seller["avg_iv"], 18, 80, market["pcr"], 20, market["days_to_expiry"], 22
    )
    return {"xgb_vol": xgb_vol}

@app.get("/calculate/risk")
async def calculate_risk_endpoint(access_token: str):
    config = get_config(access_token)
    option_chain = await fetch_option_chain(config)
    if not option_chain:
        logger.error("Failed to fetch option chain for risk calculation.")
        raise HTTPException(status_code=400, detail="Failed to fetch option chain")
    spot_price = option_chain[0]["underlying_spot_price"]
    seller = extract_seller_metrics(option_chain, spot_price)
    market = market_metrics(option_chain, config["expiry_date"])
    ivp = 80  # Placeholder
    vix = 20  # Placeholder
    iv_rv_spread = round(seller["avg_iv"] - 18, 2)  # Placeholder
    iv_skew_slope = 0.0005  # Placeholder
    regime_score, regime_label, _, _ = calculate_regime(
        seller["avg_iv"], ivp, 18, 22, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope
    )
    trades_df = pd.DataFrame([
        {"strategy": "Iron Fly", "capital_used": 60000, "potential_loss": 1000, "realized_pnl": 200, "sl_hit": False, "vega": 150}
    ]) # This should be fetched from Supabase
    summary_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime_label, vix)
    return {
        "risk_summary": summary_df.to_dict(orient="records"),
        "portfolio": portfolio_summary
    }

@app.get("/test/all")
async def test_all_endpoints(access_token: str):
    # Run all endpoints to verify coverage
    regime = await calculate_regime_endpoint(access_token)
    iv_skew = await calculate_iv_skew_slope_endpoint(access_token)
    vol = await calc_volatility_endpoint(access_token)
    xgb_vol = await predict_xgboost_vol_endpoint(access_token)
    risk = await calculate_risk_endpoint(access_token)
    strategies = await suggest_strategy_endpoint(access_token)
    return {
        "regime": regime,
        "iv_skew": iv_skew,
        "volatility": vol,
        "xgb_vol": xgb_vol,
        "risk": risk,
        "strategies": strategies
}