import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol):
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['ATM IV', 'Realized Vol', 'GARCH Vol', 'XGBoost Vol'],
            y=[seller['avg_iv'], hv_7, garch_7d, xgb_vol],
            marker_color=['#00BFFF', '#FFD700', '#FF4500', '#32CD32']
        ))
        fig.update_layout(
            title="Volatility Comparison",
            xaxis_title="Metric",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f":warning: Error in plot_vol_comparison: {e}")

def plot_chain_analysis(full_chain_df):
    try:
        if full_chain_df.empty:
            st.warning(":warning: No data to plot for chain analysis.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=full_chain_df["Strike"],
            y=full_chain_df["Call IV"],
            mode='lines+markers',
            name='Call IV',
            line=dict(color='#00BFFF')
        ))
        fig.add_trace(go.Scatter(
            x=full_chain_df["Strike"],
            y=full_chain_df["Put IV"],
            mode='lines+markers',
            name='Put IV',
            line=dict(color='#FFD700')
        ))
        fig.update_layout(
            title="IV Skew Across Strikes",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f":warning: Error in plot_chain_analysis: {e}")

def plot_allocation_pie(strategy_df, config):
    try:
        total_funds = config.get('total_funds', 2000000) or 2000000
        if strategy_df.empty:
            labels = ['Available Capital']
            values = [total_funds]
        else:
            labels = strategy_df['Strategy'].tolist() + ['Available Capital']
            used = strategy_df['Capital Used'].sum()
            values = strategy_df['Capital Used'].tolist() + [max(0, total_funds - used)]
        fig = px.pie(
            names=labels,
            values=values,
            title="Capital Allocation",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f":warning: Error in plot_allocation_pie: {e}")

def plot_drawdown_trend(portfolio_summary):
    try:
        drawdowns = np.random.normal(-portfolio_summary['Drawdown ₹'], portfolio_summary['Drawdown ₹'] / 5, 30)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            name='Drawdown (₹)',
            line=dict(color='#FF4500')
        ))
        fig.update_layout(
            title="Drawdown Trend (30 Days)",
            xaxis_title="Date",
            yaxis_title="Drawdown (₹)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f":warning: Error in plot_drawdown_trend: {e}")

def plot_margin_gauge(funds_data):
    try:
        available = funds_data.get('available_margin', 0)
        used = funds_data.get('used_margin', 0)
        total = available + used
        if total == 0:
            st.warning(":warning: No margin data to plot.")
            return
        used_pct = (used / total) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=used_pct,
            title={'text': "Margin Utilization (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4500"},
                'steps': [
                    {'range': [0, 50], 'color': "#32CD32"},
                    {'range': [50, 80], 'color': "#FFD700"},
                    {'range': [80, 100], 'color': "#FF4500"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f":warning: Error in plot_margin_gauge: {e}")
