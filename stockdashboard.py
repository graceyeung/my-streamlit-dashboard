import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf # Import the yfinance library

st.set_page_config(layout="wide") # Use wide layout for better space utilization
st.title("Quantitative Finance Performance Dashboard")

# --- Load Real Data ---
@st.cache_data # Cache data to avoid re-loading on every interaction
def load_real_data(ticker_strategy='AAPL', ticker_benchmark='SPY', start_date='2020-01-01', end_date='2023-12-31'):
    """
    Fetches real historical stock data and calculates equity curves.
    """
    try:
        # Fetch data for the strategy ticker (e.g., a single stock or an asset you 'trade')
        # We'll use Adjusted Close price as the equity value
        strategy_data = yf.download(ticker_strategy, start=start_date, end=end_date)['Adj Close']
        strategy_df = pd.DataFrame(strategy_data).rename(columns={'Adj Close': 'Strategy_Equity'})

        # Fetch data for the benchmark ticker (e.g., S&P 500 ETF)
        benchmark_data = yf.download(ticker_benchmark, start=start_date, end=end_date)['Adj Close']
        benchmark_df = pd.DataFrame(benchmark_data).rename(columns={'Adj Close': 'Benchmark_Equity'})

        # Combine into a single DataFrame, handling potential missing dates
        # Use a common index by joining
        df = pd.merge(strategy_df, benchmark_df, left_index=True, right_index=True, how='inner')

        # Normalize equity curves to start at 1000 for easier comparison
        df['Strategy_Equity'] = (df['Strategy_Equity'] / df['Strategy_Equity'].iloc[0]) * 1000
        df['Benchmark_Equity'] = (df['Benchmark_Equity'] / df['Benchmark_Equity'].iloc[0]) * 1000

        df['Strategy_Returns'] = df['Strategy_Equity'].pct_change()
        df['Benchmark_Returns'] = df['Benchmark_Equity'].pct_change()

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}. Please check ticker symbols and date range.")
        return pd.DataFrame() # Return empty DataFrame on error

# --- User Inputs for Tickers and Date Range ---
st.sidebar.header("Data Selection")
strategy_ticker = st.sidebar.text_input("Strategy Ticker (e.g., AAPL, MSFT)", "AAPL")
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (e.g., SPY, QQQ)", "SPY")

# Default dates for the date input widgets
default_start_date = pd.to_datetime('2020-01-01').date()
default_end_date = pd.to_datetime('2023-12-31').date() # Data might not be available up to current date

st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)


# Load data based on user inputs
df = load_real_data(strategy_ticker, benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


# --- Check if DataFrame is empty before proceeding ---
if df.empty:
    st.warning("No data available for the selected tickers or date range. Please adjust your selections.")
else:
    # --- Filter data based on selected dates (redundant now as yfinance downloads for range) ---
    # Kept for consistency, but `load_real_data` already handles date filtering.
    # This filter ensures UI input drives the data retrieval directly.
    filtered_df = df.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

    if not filtered_df.empty:
        # --- Calculate KPIs ---
        # Ensure filtered_df is not empty before calculating KPIs
        strategy_total_return = (filtered_df['Strategy_Equity'].iloc[-1] / filtered_df['Strategy_Equity'].iloc[0] - 1) * 100
        benchmark_total_return = (filtered_df['Benchmark_Equity'].iloc[-1] / filtered_df['Benchmark_Equity'].iloc[0] - 1) * 100

        # Annualized return (simple approximation for daily data)
        # Using timedelta to get number of years for more accurate annualization
        num_years = (filtered_df.index[-1] - filtered_df.index[0]).days / 365.25
        if num_years == 0: # Prevent division by zero if only one day selected
            annualized_strategy_return = strategy_total_return
            annualized_benchmark_return = benchmark_total_return
        else:
            annualized_strategy_return = ((1 + strategy_total_return / 100)**(1 / num_years) - 1) * 100
            annualized_benchmark_return = ((1 + benchmark_total_return / 100)**(1 / num_years) - 1) * 100


        strategy_volatility = filtered_df['Strategy_Returns'].std() * np.sqrt(252) * 100
        benchmark_volatility = filtered_df['Benchmark_Returns'].std() * np.sqrt(252) * 100

        # Assuming risk-free rate is 0 for simplicity in Sharpe calculation
        sharpe_ratio_strategy = (annualized_strategy_return / 100 - 0) / (strategy_volatility / 100) if strategy_volatility > 0 else 0
        sharpe_ratio_benchmark = (annualized_benchmark_return / 100 - 0) / (benchmark_volatility / 100) if benchmark_volatility > 0 else 0

        # Calculate Max Drawdown
        roll_max = filtered_df['Strategy_Equity'].expanding(min_periods=1).max()
        daily_drawdown = (filtered_df['Strategy_Equity'] / roll_max) - 1.0
        max_drawdown = daily_drawdown.min() * 100

        # --- Display KPIs ---
        st.header("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{strategy_ticker} Total Return", f"{strategy_total_return:.2f}%")
            st.metric(f"{strategy_ticker} Annualized Return", f"{annualized_strategy_return:.2f}%")
        with col2:
            st.metric(f"{benchmark_ticker} Total Return", f"{benchmark_total_return:.2f}%")
            st.metric(f"{benchmark_ticker} Annualized Return", f"{annualized_benchmark_return:.2f}%")
        with col3:
            st.metric(f"{strategy_ticker} Volatility", f"{strategy_volatility:.2f}%")
            st.metric(f"{strategy_ticker} Sharpe Ratio", f"{sharpe_ratio_strategy:.2f}")
        with col4:
            st.metric(f"{benchmark_ticker} Volatility", f"{benchmark_volatility:.2f}%")
            st.metric(f"{strategy_ticker} Max Drawdown", f"{max_drawdown:.2f}%")

        # --- Create Charts ---
        st.header("Performance Charts")

        # Equity Curve
        fig_equity = px.line(filtered_df[['Strategy_Equity', 'Benchmark_Equity']],
                             title=f'{strategy_ticker} vs. {benchmark_ticker} Equity Curve (Normalized to 1000)',
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
        st.subheader(f"{strategy_ticker} Drawdown")
        fig_drawdown = px.line(daily_drawdown * 100,
                               title=f'{strategy_ticker} Daily Drawdown',
                               labels={'value': 'Drawdown (%)', 'Date': 'Date'})
        fig_drawdown.update_layout(hovermode="x unified", yaxis_range=[-100, 0]) # Drawdown is negative
        st.plotly_chart(fig_drawdown, use_container_width=True)

    else:
        st.warning("No data available for the selected date range after filtering. Please select a valid period.")


st.markdown("---")
st.markdown("Dashboard created using Streamlit, Plotly, and yfinance.")