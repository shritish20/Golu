import streamlit as st
import pandas as pd
import requests
import os

def evaluate_full_risk(trades_df, config, regime_label, vix):
    try:
        total_funds = config.get('total_funds', 2000000) or 2000000
        daily_risk_limit = config['daily_risk_limit_pct'] * total_funds
        weekly_risk_limit = config['weekly_risk_limit_pct'] * total_funds
        max_drawdown_pct = 0.05 if vix > 20 else 0.03 if vix > 12 else 0.02
        max_drawdown = max_drawdown_pct * total_funds
        strategy_summary = []
        total_cap_used = total_risk_used = total_realized_pnl = total_vega = 0
        flags = []
        if trades_df.empty:
            strategy_summary.append({
                "Strategy": "None",
                "Capital Used": 0,
                "Cap Limit": total_funds,
                "% Used": 0,
                "Potential Risk": 0,
                "Risk Limit": total_funds * 0.01,
                "P&L": 0,
                "Vega": 0,
                "Risk OK?": ":white_check_mark:"
            })
        else:
            for _, row in trades_df.iterrows():
                strat = row["strategy"]
                capital_used = row["capital_used"]
                potential_risk = row["potential_loss"]
                pnl = row["realized_pnl"]
                sl_hit = row["sl_hit"]
                vega = row["vega"]
                cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
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
        st.error(f":warning: Exception in evaluate_full_risk: {e}")
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

def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"{config['base_url']}/portfolio/short-term-positions"
        res_positions = requests.get(url_positions, headers=config['headers'])
        url_trades = f"{config['base_url']}/order/trades/get-trades-for-day"
        res_trades = requests.get(url_trades, headers=config['headers'])
        positions = []
        trades = []
        if res_positions.status_code == 200:
            positions = res_positions.json().get("data", [])
        else:
            st.warning(f":warning: Error fetching positions: {res_positions.status_code} - {res_positions.text}")
        if res_trades.status_code == 200:
            trades = res_trades.json().get("data", [])
        else:
            st.warning(f":warning: Error fetching trades: {res_trades.status_code} - {res_trades.text}")
        trade_counts = {}
        for trade in trades:
            instrument = trade.get("instrument_token", "")
            strat = "Straddle" if "NIFTY" in instrument and ("CE" in instrument or "PE" in instrument) else "Unknown"
            trade_counts[strat] = trade_counts.get(strat, 0) + 1
        trades_df_list = []
        for pos in positions:
            instrument = pos.get("instrument_token", "")
            strategy = "Unknown"
            if pos.get("product") == "D":
                if pos.get("quantity") < 0 and pos.get("average_price") > 0:
                    if "CE" in instrument or "PE" in instrument:
                        strategy = "Straddle"
                    else:
                        strategy = "Iron Condor"
            capital = pos["quantity"] * pos["average_price"]
            trades_df_list.append({
                "strategy": strategy,
                "capital_used": abs(capital),
                "potential_loss": abs(capital * 0.1),
                "realized_pnl": pos["pnl"],
                "trades_today": trade_counts.get(strategy, 0),
                "sl_hit": pos["pnl"] < -abs(capital * 0.05),
                "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0,
                "instrument_token": instrument
            })
        trades_df = pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame()
        save_trade_data(trades_df)
        return trades_df
    except Exception as e:
        st.error(f":warning: Exception in fetch_trade_data: {e}")
        return pd.DataFrame()

def save_trade_data(trades_df, filename="trade_history.csv"):
    try:
        if not trades_df.empty:
            trades_df.to_csv(filename, mode='a', index=False, header=not os.path.exists(filename))
    except Exception as e:
        st.warning(f":warning: Error saving trade data: {e}")
