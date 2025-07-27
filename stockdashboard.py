import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide") # Use wide layout for better space utilization
st.title("Quantitative Finance Performance Dashboard")

# --- Load Data (replace with your actual data loading) ---
@st.cache_data # Cache data to avoid re-loading on every interaction
def load_data():
    # Example dummy data
    data = {
        'Date': pd.to_datetime(pd.date_range(start='2020-01-01', periods=500, freq='B')),
        'Strategy_Equity': (1000 * (1 + np.random.randn(500) * 0.005 + 0.0002).cumprod()).round(2),
        'Benchmark_Equity': (1000 * (1 + np.random.randn(500) * 0.004 + 0.0001).cumprod()).round(2)
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df['Strategy_Returns'] = df['Strategy_Equity'].pct_change()
    df['Benchmark_Returns'] = df['Benchmark_Equity'].pct_change()
    return df

df = load_data()

# --- Date Range Selector ---
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

# Filter data based on selected dates
filtered_df = df.loc[start_date:end_date]

# --- Calculate KPIs ---
# Ensure filtered_df is not empty before calculating KPIs
if not filtered_df.empty:
    strategy_total_return = (filtered_df['Strategy_Equity'].iloc[-1] / filtered_df['Strategy_Equity'].iloc[0] - 1) * 100
    benchmark_total_return = (filtered_df['Benchmark_Equity'].iloc[-1] / filtered_df['Benchmark_Equity'].iloc[0] - 1) * 100

    # Annualized return (simple approximation for daily data)
    num_trading_days = len(filtered_df)
    annualized_strategy_return = (1 + strategy_total_return / 100)**(252 / num_trading_days) - 1 if num_trading_days > 0 else 0
    annualized_benchmark_return = (1 + benchmark_total_return / 100)**(252 / num_trading_days) - 1 if num_trading_days > 0 else 0

    strategy_volatility = filtered_df['Strategy_Returns'].std() * np.sqrt(252) * 100
    benchmark_volatility = filtered_df['Benchmark_Returns'].std() * np.sqrt(252) * 100

    # Assuming risk-free rate is 0 for simplicity in Sharpe calculation
    sharpe_ratio_strategy = (annualized_strategy_return - 0) / (strategy_volatility / 100) if strategy_volatility > 0 else 0
    sharpe_ratio_benchmark = (annualized_benchmark_return - 0) / (benchmark_volatility / 100) if benchmark_volatility > 0 else 0

    # Calculate Max Drawdown
    roll_max = filtered_df['Strategy_Equity'].expanding(min_periods=1).max()
    daily_drawdown = (filtered_df['Strategy_Equity'] / roll_max) - 1.0
    max_drawdown = daily_drawdown.min() * 100

    # --- Display KPIs ---
    st.header("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strategy Total Return", f"{strategy_total_return:.2f}%")
        st.metric("Strategy Annualized Return", f"{annualized_strategy_return:.2%}")
    with col2:
        st.metric("Benchmark Total Return", f"{benchmark_total_return:.2f}%")
        st.metric("Benchmark Annualized Return", f"{annualized_benchmark_return:.2%}")
    with col3:
        st.metric("Strategy Volatility", f"{strategy_volatility:.2f}%")
        st.metric("Strategy Sharpe Ratio", f"{sharpe_ratio_strategy:.2f}")
    with col4:
        st.metric("Benchmark Volatility", f"{benchmark_volatility:.2f}%")
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    # --- Create Charts ---
    st.header("Performance Charts")

    # Equity Curve
    fig_equity = px.line(filtered_df[['Strategy_Equity', 'Benchmark_Equity']],
                         title='Strategy vs. Benchmark Equity Curve',
                         labels={'value': 'Equity Value', 'Date': 'Date'},
                         line_group='variable',
                         color_discrete_map={'Strategy_Equity': 'blue', 'Benchmark_Equity': 'orange'})
    fig_equity.update_layout(hovermode="x unified")
    st.plotly_chart(fig_equity, use_container_width=True)

    # Daily Returns Histogram
    st.subheader("Daily Returns Distribution")
    fig_hist = px.histogram(filtered_df[['Strategy_Returns', 'Benchmark_Returns']].melt(var_name='Type', value_name='Returns'),
                            x='Returns',
                            color='Type',
                            barmode='overlay',
                            nbins=50,
                            title='Daily Returns Distribution',
                            opacity=0.7)
    fig_hist.update_layout(xaxis_title="Daily Returns", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Drawdown Chart
    st.subheader("Strategy Drawdown")
    fig_drawdown = px.line(daily_drawdown * 100,
                           title='Strategy Daily Drawdown',
                           labels={'value': 'Drawdown (%)', 'Date': 'Date'})
    fig_drawdown.update_layout(hovermode="x unified", yaxis_range=[-100, 0]) # Drawdown is negative
    st.plotly_chart(fig_drawdown, use_container_width=True)

else:
    st.warning("No data available for the selected date range.")

st.markdown("---")
st.markdown("Dashboard created using Streamlit and Plotly.")