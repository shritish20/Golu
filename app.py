import streamlit as st
from backend.config import get_config
from backend.data_fetch import fetch_option_chain, get_indices_quotes, load_upcoming_events, load_ivp, load_xgboost_model, fetch_trade_data, get_funds_and_margin
from backend.analytics import extract_seller_metrics, full_chain_table, market_metrics, calculate_volatility, calculate_iv_skew_slope, calculate_regime, suggest_strategy, predict_xgboost_volatility, calculate_sharpe_ratio
from backend.strategies import get_strategy_details
from backend.risk_management import evaluate_full_risk
from backend.order_management import place_multi_leg_orders, create_gtt_order, exit_all_positions, logout
from utils.plotting import plot_vol_comparison, plot_chain_analysis, plot_allocation_pie, plot_drawdown_trend, plot_margin_gauge

# Streamlit configuration
st.set_page_config(page_title="Volguard - Your Trading Copilot", layout="wide", initial_sidebar_state="expanded")
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=2 * 60 * 1000, key="refresh")
st.info("⏳ Auto-refreshing every 2 minutes to fetch updated market data.")

# CSS styling
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; color: white; }
    .metric-box { background-color: #1A1C24; border-radius: 8px; padding: 15px; margin-bottom: 10px; color: white; }
    .metric-box h3 { color: #6495ED; font-size: 1.1em; margin-bottom: 5px; }
    .metric-box h4 { color: #6495ED; font-size: 0.9em; margin-bottom: 5px; }
    .metric-box .value { font-size: 1.8em; font-weight: bold; color: #00BFFF; }
    </style>
    """, unsafe_allow_html=True
)

# Session state initialization
if 'access_token' not in st.session_state:
    st.session_state.access_token = ""
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Sidebar login
access_token = st.sidebar.text_input("Enter Upstox Access Token", type="password", value=st.session_state.access_token)
if st.sidebar.button("Login"):
    if access_token:
        config = get_config(access_token)
        test_url = f"{config['base_url']}/user/profile"
        try:
            import requests
            res = requests.get(test_url, headers=config['headers'])
            if res.status_code == 200:
                st.session_state.access_token = access_token
                st.session_state.logged_in = True
                st.sidebar.success(":white_check_mark: Logged in successfully!")
            else:
                st.sidebar.error(f":x: Invalid token: {res.status_code} - {res.text}")
        except Exception as e:
            st.sidebar.error(f":warning: Error validating token: {e}")
    else:
        st.sidebar.error(":x: Please enter an access token.")

if st.session_state.logged_in and st.sidebar.button("Logout"):
    config = get_config(st.session_state.access_token)
    logout(config)
    st.experimental_rerun()

# Main UI
if st.session_state.logged_in and access_token:
    config = get_config(st.session_state.access_token)
    config['total_funds'] = get_funds_and_margin(config)['total_funds']

    @st.cache_data(show_spinner="Analyzing market data...")
    def load_all_data(config):
        xgb_model = load_xgboost_model()
        option_chain = fetch_option_chain(config)
        if not option_chain:
            st.error(":x: Failed to fetch option chain data.")
            return tuple([None]*27)
        spot_price = option_chain[0]["underlying_spot_price"]
        vix, nifty = get_indices_quotes(config)
        if not vix or not nifty:
            st.error(":x: Failed to fetch India VIX or Nifty 50 data.")
            return tuple([None]*27)
        seller = extract_seller_metrics(option_chain, spot_price)
        if not seller:
            st.error(":x: Failed to extract seller metrics.")
            return tuple([None]*27)
        full_chain_df = full_chain_table(option_chain, spot_price)
        market = market_metrics(option_chain, config['expiry_date'])
        ivp = load_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller["avg_iv"])
        xgb_vol = predict_xgboost_volatility(xgb_model, seller["avg_iv"], hv_7, ivp, market["pcr"], vix, market["days_to_expiry"], garch_7d)
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        regime_score, regime, regime_note, regime_explanation = calculate_regime(
            seller["avg_iv"], ivp, hv_7, garch_7d, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope)
        event_df = load_upcoming_events(config)
        strategies, strategy_rationale, event_warning = suggest_strategy(
            regime, ivp, iv_rv_spread, market['days_to_expiry'], event_df, config['expiry_date'], seller["straddle_price"], spot_price)
        strategy_details = [detail for strat in strategies if (detail := get_strategy_details(strat, option_chain, spot_price, config, lots=1)) is not None]
        trades_df = fetch_trade_data(config, full_chain_df)
        strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime, vix)
        funds_data = get_funds_and_margin(config)
        sharpe_ratio = calculate_sharpe_ratio()
        return (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio)

    data = load_all_data(config)
    if data[0] is None:
        st.stop()

    (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio) = data

    st.markdown("<h1 style='text-align: center;'>Market Insights Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>Nifty 50 Spot</h3><div class='value'>₹{nifty:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>India VIX</h3><div class='value'>{vix:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>ATM Strike</h3><div class='value'>{seller['strike']:.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>Straddle Price</h3><div class='value'>₹{seller['straddle_price']:.2f}</div></div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"<div class='metric-box'><h3>ATM IV</h3><div class='value'>{seller['avg_iv']:.2f}%</div></div>", unsafe_allow_html=True)
    with col6:
        st.markdown(f"<div class='metric-box'><h3>IVP</h3><div class='value'>{ivp}%</div></div>", unsafe_allow_html=True)
    with col7:
        st.markdown(f"<div class='metric-box'><h3>Days to Expiry</h3><div class='value'>{market['days_to_expiry']}</div></div>", unsafe_allow_html=True)
    with col8:
        st.markdown(f"<div class='metric-box'><h3>PCR</h3><div class='value'>{market['pcr']:.2f}</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Option Chain Analysis", "Strategy Suggestions", "Risk & Portfolio", "Place Orders", "Risk Management Dashboard"])

    with tab1:
        st.subheader("Volatility Landscape")
        plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol)
        st.markdown(f"<div class='metric-box'><h4>IV - RV Spread:</h4> {iv_rv_spread:+.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>IV Skew Slope:</h4> {iv_skew_slope:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Current Expiry:</h4> {config['expiry_date']}</div>", unsafe_allow_html=True)

        st.subheader("XGBoost Volatility Prediction Inputs")
        xgb_inputs = pd.DataFrame({
            "Feature": ["ATM_IV", "Realized_Vol", "IVP", "PCR", "VIX", "Days_to_Expiry", "GARCH_Predicted_Vol"],
            "Value": [seller["avg_iv"], hv_7, ivp, market["pcr"], vix, market["days_to_expiry"], garch_7d]
        })
        st.dataframe(xgb_inputs.style.format({"Value": "{:.2f}"}).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)

        st.subheader("Breakeven & Max Pain")
        st.markdown(f"<div class='metric-box'><h4>Breakeven Range:</h4> {seller['strike'] - seller['straddle_price']:.0f} – {seller['strike'] + seller['straddle_price']:.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Max Pain:</h4> {market['max_pain']:.0f}</div>", unsafe_allow_html=True)

        st.subheader("Greeks at ATM")
        st.markdown(f"<div class='metric-box'><h4>Delta</h4>{seller['delta']:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Theta</h4>₹{seller['theta']:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Vega</h4>₹{seller['vega']:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Gamma</h4>{seller['gamma']:.6f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>POP</h4>{seller['pop']:.2f}%</div>", unsafe_allow_html=True)

        st.subheader("Upcoming Events")
        if not event_df.empty:
            st.dataframe(event_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
            if event_warning:
                st.warning(event_warning)
        else:
            st.info("No upcoming events before expiry.")

    with tab2:
        st.subheader("Option Chain Analysis")
        plot_chain_analysis(full_chain_df)
        st.subheader("ATM ±300 Chain Table")
        st.dataframe(full_chain_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        st.subheader("Theta/Vega Ranking")
        eff_df = full_chain_df.copy()
        eff_df["Theta/Vega"] = eff_df["Total Theta"] / eff_df["Total Vega"]
        eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False)
        st.dataframe(eff_df.style.format(precision=2).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)

    with tab3:
        st.markdown(f"<div class='metric-box'><h3>Regime: {regime}</h3><p style='color: #6495ED;'>Score: {regime_score:.2f}</p><p>{regime_note}</p><p><i>{regime_explanation}</i></p></div>", unsafe_allow_html=True)
        st.subheader("Recommended Strategies")
        if strategies:
            st.success(f"**Suggested Strategies:** {', '.join(strategies)}")
            st.info(f"**Rationale:** {strategy_rationale}")
            if event_warning:
                st.warning(event_warning)
            for strat in strategies:
                st.markdown(f"### {strat}")
                detail = get_strategy_details(strat, option_chain, spot_price, config, lots=1)
                if detail:
                    order_df = pd.DataFrame({
                        "Instrument": [order["instrument_key"] for order in detail["orders"]],
                        "Type": [order["transaction_type"] for order in detail["orders"]],
                        "Quantity": [config["lot_size"] for _ in detail["orders"]],
                        "Price": [order.get("current_price", 0) for order in detail["orders"]]
                    })
                    st.dataframe(order_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                    from backend.order_management import calculate_strategy_margin
                    margin = calculate_strategy_margin(config, detail)
                    st.markdown(f"<div class='metric-box'><h4>Estimated Margin Required</h4> ₹{margin:.2f}</div>", unsafe_allow_html=True)
                    lots = st.number_input(f"Lots for {strat}", min_value=1, value=1, step=1, key=f"lots_{strat}")
                    if st.button(f"Place {strat} Order", key=f"place_{strat}"):
                        updated_detail = get_strategy_details(strat, option_chain, spot_price, config, lots=lots)
                        if updated_detail:
                            success = place_multi_leg_orders(config, updated_detail["orders"])
                            if success:
                                st.success(f":white_check_mark: Placed {strat} order with {lots} lots!")
                            else:
                                st.error(f":x: Failed to place {strat} order.")
                        else:
                            st.error(f":x: Unable to generate order details for {strat}.")
                else:
                    st.error(f":x: Error: No details found for {strat}.")
        else:
            st.info("No strategies suggested for current market conditions.")

    with tab4:
        st.subheader("Portfolio Summary")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.markdown(f"<div class='metric-box'><h3>Available Capital</h3><div class='value'>₹{funds_data['available_margin']:.2f}</div></div>", unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"<div class='metric-box'><h3>Used Margin</h3><div class='value'>₹{funds_data['used_margin']:.2f}</div></div>", unsafe_allow_html=True)
        with col_p3:
            st.markdown(f"<div class='metric-box'><h3>Exposure %</h3><div class='value'>{portfolio_summary.get('Exposure Percent', 0):.2f}%</div></div>", unsafe_allow_html=True)
        with col_p4:
            st.markdown(f"<div class='metric-box'><h3>Sharpe Ratio</h3><div class='value'>{sharpe_ratio:.2f}</div></div>", unsafe_allow_html=True)

        st.subheader("Capital Allocation")
        plot_allocation_pie(strategy_df, config)

        st.subheader("Drawdown Trend")
        plot_drawdown_trend(portfolio_summary)

        st.subheader("Margin Utilization")
        plot_margin_gauge(funds_data)

        st.subheader("Strategy Risk Summary")
        if not strategy_df.empty:
            st.dataframe(strategy_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
            if portfolio_summary.get("Flags"):
                st.warning(f":warning: Risk Alerts: {' | '.join(portfolio_summary['Flags'])}")
        else:
            st.info("No active strategies to display.")

    with tab5:
        all_strategies = ["Iron Fly", "Iron Condor", "Jade Lizard", "Straddle", "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"]
        st.subheader("📥 Manual Multi-Leg Order Placement")
        selected_strategy = st.selectbox("Select Strategy", all_strategies, key="manual_strategy")
        lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="manual_lots")
        sl_percentage = st.slider("Stop Loss %", 0.0, 50.0, 10.0, 0.5, key="manual_sl_pct")
        order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
        validity = st.radio("Order Validity", ["DAY", "IOC"], horizontal=True)

        if selected_strategy:
            detail = get_strategy_details(selected_strategy, option_chain, spot_price, config, lots=lots)
            if detail:
                st.write("🧮 Strategy Legs (Editable)")
                updated_orders = []
                for idx, order in enumerate(detail["orders"]):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.markdown(f"**Leg {idx + 1}**")
                    with col2:
                        qty = st.number_input(f"Qty {idx+1}", min_value=1, value=order["quantity"], step=1, key=f"qty_{idx}")
                    with col3:
                        tx_type = st.selectbox(f"Type {idx+1}", ["BUY", "SELL"], index=0 if order["transaction_type"] == "BUY" else 1, key=f"tx_{idx}")
                    with col4:
                        price = st.number_input(f"Price {idx+1}", min_value=0.0, value=order.get("current_price", 0.0), step=0.05, key=f"price_{idx}")
                    with col5:
                        instr = order["instrument_key"]
                        st.code(instr, language="text")
                    updated_orders.append({
                        "instrument_key": instr,
                        "quantity": qty,
                        "transaction_type": tx_type,
                        "order_type": order_type,
                        "validity": validity,
                        "product": "D",
                        "current_price": price
                    })
                st.markdown("---")
                from backend.order_management import calculate_strategy_margin
                margin = calculate_strategy_margin(config, detail) * lots
                st.markdown(f"💰 **Estimated Margin:** ₹{margin:,.2f}")
                st.markdown(f"💵 **Premium Collected:** ₹{detail['premium_total'] * lots:,.2f}")
                st.markdown(f"🔻 **Max Loss:** ₹{detail['max_loss'] * lots:,.2f}")
                st.markdown(f"🟢 **Max Profit:** ₹{detail['max_profit'] * lots:,.2f}")
                if st.button("🚀 Place Multi-Leg Order"):
                    import time
                    payload = []
                    for idx, leg in enumerate(updated_orders):
                        correlation_id = f"mleg_{idx}_{int(time.time()) % 100000}"
                        payload.append({
                            "quantity": leg["quantity"],
                            "product": "D",
                            "validity": leg["validity"],
                            "price": leg["current_price"],
                            "tag": f"{leg['instrument_key']}_leg_{idx}",
                            "slice": False,
                            "instrument_token": leg["instrument_key"],
                            "order_type": leg["order_type"],
                            "transaction_type": leg["transaction_type"],
                            "disclosed_quantity": 0,
                            "trigger_price": 0,
                            "is_amo": False,
                            "correlation_id": correlation_id
                        })
                    success = place_multi_leg_orders(config, updated_orders)
                    if success:
                        for leg in updated_orders:
                            if leg["transaction_type"] == "SELL":
                                sl_price = leg["current_price"] * (1 + sl_percentage / 100)
                                create_gtt_order(config, leg["instrument_key"], sl_price, "BUY", tag=f"SL_{selected_strategy}")
                        st.success(f"✅ Multi-leg order placed successfully! 🛡️ SL orders placed at {sl_percentage}% above sell price.")
                    else:
                        st.error(f"❌ Failed to place order.")

    with tab6:
        st.subheader("Risk Management Dashboard")
        st.markdown("<h4>Portfolio Risk Metrics</h4>", unsafe_allow_html=True)
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='metric-box'><h3>Total Risk %</h3><div class='value'>{portfolio_summary['Risk Percent']:.2f}%</div></div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='metric-box'><h3>Daily Risk Limit</h3><div class='value'>₹{portfolio_summary['Daily Risk Limit']:.2f}</div></div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='metric-box'><h3>Weekly Risk Limit</h3><div class='value'>₹{portfolio_summary['Weekly Risk Limit']:.2f}</div></div>", unsafe_allow_html=True)

        st.subheader("Drawdown Analysis")
        plot_drawdown_trend(portfolio_summary)
        st.markdown(f"<div class='metric-box'><h4>Current Drawdown</h4> ₹{portfolio_summary['Drawdown ₹']:.2f} ({portfolio_summary['Drawdown Percent']:.2f}%)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Max Drawdown Allowed</h4> ₹{portfolio_summary['Max Drawdown Allowed']:.2f}</div>", unsafe_allow_html=True)

        st.subheader("Risk Flags")
        if portfolio_summary.get("Flags"):
            for flag in portfolio_summary["Flags"]:
                st.warning(flag)
        else:
            st.info("No risk flags raised.")

        st.subheader("Strategy Risk Details")
        if not strategy_df.empty:
            st.dataframe(strategy_df[["Strategy", "Capital Used", "% Used", "Potential Risk", "Risk OK?"]].style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        else:
            st.info("No active strategies to display.")

else:
    st.markdown("<h1 style='text-align: center;'>Volguard - Your Trading Copilot</h1>", unsafe_allow_html=True)
    st.info("Please enter your Upstox Access Token in the sidebar to access the dashboard.")
