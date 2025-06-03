import streamlit as st
import requests
from time import time

def calculate_strategy_margin(config, strategy_details):
    try:
        instruments = [
            {
                "instrument_key": order["instrument_key"],
                "quantity": abs(order["quantity"]),
                "transaction_type": order["transaction_type"],
                "product": "D"
            }
            for order in strategy_details["orders"]
        ]
        url = f"{config['base_url']}/charges/margin"
        res = requests.post(url, headers=config['headers'], json={"instruments": instruments})
        if res.status_code == 200:
            data = res.json().get("data", {})
            if isinstance(data, list):
                total_margin = sum(item.get("total_margin", 0) for item in data)
            elif isinstance(data, dict):
                margins = data.get("margins", [])
                if isinstance(margins, list):
                    total_margin = sum(item.get("total_margin", 0) for item in margins)
                else:
                    total_margin = data.get("required_margin", 0) or data.get("final_margin", 0)
            return round(total_margin, 2)
        else:
            st.warning(f":warning: Failed to calculate margin: {res.status_code} - {res.text}")
            return 0
    except Exception as e:
        st.warning(f":warning: Error calculating strategy margin: {e}")
        return 0

def place_multi_leg_orders(config, orders):
    try:
        sorted_orders = sorted(orders, key=lambda x: 0 if x["transaction_type"] == "BUY" else 1)
        payload = []
        for idx, order in enumerate(sorted_orders):
            correlation_id = f"s{idx}_{int(time()) % 1000000}"
            payload.append({
                "quantity": abs(order["quantity"]),
                "product": "D",
                "validity": order.get("validity", "DAY"),
                "price": order.get("current_price", 0),
                "tag": f"{order['instrument_key']}_leg_{idx}",
                "slice": False,
                "instrument_token": order["instrument_key"],
                "order_type": order.get("order_type", "MARKET"),
                "transaction_type": order["transaction_type"],
                "disclosed_quantity": 0,
                "trigger_price": 0,
                "is_amo": False,
                "correlation_id": correlation_id
            })
        url = f"{config['base_url']}/order/multi/place"
        res = requests.post(url, headers=config['headers'], json=payload)
        if res.status_code == 200:
            return True
        else:
            st.error(f":x: Failed to place multi-leg order: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f":warning: Error placing multi-leg order: {e}")
        return False

def create_gtt_order(config, instrument_token, trigger_price, transaction_type="SELL", tag="SL"):
    try:
        url = f"{config['base_url'].replace('v2', 'v3')}/order/gtt/place"
        payload = {
            "type": "SINGLE",
            "quantity": config["lot_size"],
            "product": "D",
            "rules": [{
                "strategy": "ENTRY",
                "trigger_type": "ABOVE" if transaction_type == "SELL" else "BELOW",
                "trigger_price": trigger_price
            }],
            "instrument_token": instrument_token,
            "transaction_type": transaction_type,
            "tag": tag
        }
        res = requests.post(url, headers=config['headers'], json=payload)
        if res.status_code == 200:
            return True
        else:
            st.warning(f":warning: GTT failed: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f":warning: Error creating GTT: {e}")
        return False

def exit_all_positions(config):
    try:
        url = f"{config['base_url']}/order/positions/exit?segment=EQ"
        res = requests.post(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                order_ids = data.get("data", {}).get("order_ids", [])
                return order_ids
            st.error(f":x: Unexpected response status: {data}")
            return []
        elif res.status_code == 400:
            errors = res.json().get("errors", [])
            for error in errors:
                if error.get("errorCode") == "UDAPI1108":
                    return []
            st.error(":x: Exit failed due to unknown reason.")
            return []
        st.error(f":x: Error exiting positions: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        st.error(f":warning: Exception in exit_all_positions: {e}")
        return []

def logout(config):
    try:
        url = f"{config['base_url']}/logout"
        res = requests.delete(url, headers=config['headers'])
        if res.status_code == 200:
            st.session_state.access_token = ""
            st.session_state.logged_in = False
            st.cache_data.clear()
            return True
        st.error(f":x: Logout failed: {res.status_code} - {res.text}")
        return False
    except Exception as e:
        st.error(f":warning: Exception in logout: {e}")
        return False
