import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Money Freedom - Portfolio Backtesting",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 Money Freedom</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Quantitative Portfolio Backtesting & Rebalancing Platform</p>', unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    """Fetch historical price data"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers[0]]
        data = data.ffill().bfill()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_returns(prices):
    """Calculate returns from prices"""
    return prices.pct_change().fillna(0)

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_momentum(prices, lookback):
    """Calculate momentum score"""
    return prices.pct_change(lookback).fillna(0)

def equal_weight(n_assets):
    """Equal weight strategy"""
    return np.ones(n_assets) / n_assets

def inverse_volatility_weight(returns, lookback):
    """Inverse volatility weighting"""
    vol = returns.tail(lookback).std()
    inv_vol = 1 / vol
    return (inv_vol / inv_vol.sum()).values

def momentum_weight(prices, lookback):
    """Momentum-based weighting"""
    momentum = calculate_momentum(prices, lookback).iloc[-1]
    momentum = momentum.clip(lower=0)
    if momentum.sum() == 0:
        return equal_weight(len(momentum))
    return (momentum / momentum.sum()).values

def risk_parity_weight(returns, lookback):
    """Risk Parity weighting"""
    cov_matrix = returns.tail(lookback).cov()
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / inv_vol.sum()
    return weights

def min_variance_weight(returns, lookback):
    """Minimum Variance Optimization"""
    cov_matrix = returns.tail(lookback).cov().values
    n = len(cov_matrix)
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.ones(n) / n
    
    result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_weights

def max_sharpe_weight(returns, lookback, risk_free_rate=0.02):
    """Maximum Sharpe Ratio Optimization"""
    mean_returns = returns.tail(lookback).mean() * 252
    cov_matrix = returns.tail(lookback).cov().values * 252
    n = len(mean_returns)
    
    def neg_sharpe(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(portfolio_return - risk_free_rate) / portfolio_vol
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.ones(n) / n
    
    result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_weights

def backtest_strategy(prices, returns, strategy, lookback, rebalance_freq, 
                     initial_capital, transaction_fee, management_fee, investment_type='lump_sum', dca_amount=0):
    """Main backtesting engine"""
    
    portfolio_value = [initial_capital]
    weights_history = []
    trade_log = []
    cash = initial_capital
    holdings = np.zeros(len(prices.columns))
    
    # Rebalance frequency mapping
    freq_map = {
        'Daily': 1,
        'Weekly': 5,
        'Monthly': 21,
        'Quarterly': 63,
        'Yearly': 252
    }
    rebalance_days = freq_map.get(rebalance_freq, 21)
    
    for i in range(lookback, len(prices)):
        current_prices = prices.iloc[i].values
        
        # DCA: Add funds
        if investment_type == 'DCA' and i > lookback:
            cash += dca_amount
        
        # Rebalancing logic
        if i % rebalance_days == 0:
            # Calculate weights based on strategy
            if strategy == 'Equal Weight':
                weights = equal_weight(len(prices.columns))
            elif strategy == 'Inverse Volatility':
                weights = inverse_volatility_weight(returns.iloc[:i], lookback)
            elif strategy == 'Momentum':
                weights = momentum_weight(prices.iloc[:i], lookback)
            elif strategy == 'Risk Parity':
                weights = risk_parity_weight(returns.iloc[:i], lookback)
            elif strategy == 'Minimum Variance':
                weights = min_variance_weight(returns.iloc[:i], lookback)
            elif strategy == 'Maximum Sharpe':
                weights = max_sharpe_weight(returns.iloc[:i], lookback)
            else:
                weights = equal_weight(len(prices.columns))
            
            weights_history.append({
                'date': prices.index[i],
                'weights': weights.copy()
            })
            
            # Calculate target holdings
            total_value = cash + np.sum(holdings * current_prices)
            target_value = total_value * weights
            target_holdings = target_value / current_prices
            
            # Execute trades
            trades = target_holdings - holdings
            transaction_costs = np.sum(np.abs(trades * current_prices)) * transaction_fee
            
            # Update holdings and cash
            holdings = target_holdings
            cash = total_value - np.sum(holdings * current_prices) - transaction_costs
            
            # Log trades
            for j, ticker in enumerate(prices.columns):
                if abs(trades[j]) > 0.001:
                    trade_log.append({
                        'Date': prices.index[i],
                        'Ticker': ticker,
                        'Action': 'Buy' if trades[j] > 0 else 'Sell',
                        'Shares': abs(trades[j]),
                        'Price': current_prices[j],
                        'Value': abs(trades[j] * current_prices[j])
                    })
        
        # Calculate portfolio value
        total_value = cash + np.sum(holdings * current_prices)
        
        # Apply management fee (annualized)
        if i > lookback:
            daily_mgmt_fee = management_fee / 252
            total_value *= (1 - daily_mgmt_fee)
        
        portfolio_value.append(total_value)
    
    return portfolio_value, weights_history, trade_log

def calculate_metrics(portfolio_values, initial_capital):
    """Calculate performance metrics"""
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # CAGR
    total_return = portfolio_values[-1] / initial_capital
    n_years = len(portfolio_values) / 252
    cagr = (total_return ** (1 / n_years) - 1) * 100
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Final Value': portfolio_values[-1]
    }

def monte_carlo_simulation(returns, initial_value, n_simulations=1000, n_days=252):
    """Monte Carlo simulation for future projections"""
    mean_return = returns.mean()
    std_return = returns.std()
    
    simulations = []
    for _ in range(n_simulations):
        simulation = [initial_value]
        for _ in range(n_days):
            daily_return = np.random.normal(mean_return, std_return)
            simulation.append(simulation[-1] * (1 + daily_return))
        simulations.append(simulation)
    
    return np.array(simulations)

# Sidebar - Input Parameters
st.sidebar.header("📊 Portfolio Configuration")

# Tickers Input
tickers_input = st.sidebar.text_input(
    "Enter Tickers (comma-separated)",
    "SPY,QQQ,IWM,TLT,GLD",
    help="Enter stock/ETF tickers separated by commas"
)
tickers = [t.strip().upper() for t in tickers_input.split(',')]

# Date Range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*5))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Investment Parameters
st.sidebar.subheader("💵 Investment Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)

investment_type = st.sidebar.radio("Investment Type", ['Lump Sum', 'DCA'])
dca_amount = 0
if investment_type == 'DCA':
    dca_amount = st.sidebar.number_input("Monthly DCA Amount ($)", min_value=0, value=500, step=100)

# Strategy Selection
st.sidebar.subheader("🎯 Strategy Settings")
strategy = st.sidebar.selectbox(
    "Weighting Strategy",
    ['Equal Weight', 'Inverse Volatility', 'Momentum', 'Risk Parity', 
     'Minimum Variance', 'Maximum Sharpe']
)

lookback = st.sidebar.slider("Lookback Period (days)", 20, 252, 60)
rebalance_freq = st.sidebar.selectbox(
    "Rebalance Frequency",
    ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
)

# Fees
st.sidebar.subheader("💸 Fees")
transaction_fee = st.sidebar.number_input("Transaction Fee (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100
management_fee = st.sidebar.number_input("Management Fee (% p.a.)", min_value=0.0, max_value=5.0, value=0.5, step=0.1) / 100

# Run Backtest Button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

# Main Content
if run_backtest:
    with st.spinner("Fetching data and running backtest..."):
        # Fetch data
        prices = fetch_data(tickers, start_date, end_date)
        
        if prices is not None and not prices.empty:
            returns = calculate_returns(prices)
            
            # Run backtest
            portfolio_values, weights_history, trade_log = backtest_strategy(
                prices, returns, strategy, lookback, rebalance_freq,
                initial_capital, transaction_fee, management_fee,
                investment_type.lower().replace(' ', '_'), dca_amount
            )
            
            # Calculate metrics
            metrics = calculate_metrics(portfolio_values, initial_capital)
            
            # Display Metrics
            st.header("📈 Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("CAGR", f"{metrics['CAGR']:.2f}%")
            with col2:
                st.metric("Volatility", f"{metrics['Volatility']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")
            with col5:
                st.metric("Final Value", f"${metrics['Final Value']:,.2f}")
            
            # Equity Curve
            st.header("📊 Equity Curve")
            fig = go.Figure()
            dates = prices.index[lookback:]
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values[1:],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown Chart
            st.header("📉 Drawdown Analysis")
            portfolio_series = pd.Series(portfolio_values[1:], index=dates)
            returns_series = portfolio_series.pct_change().dropna()
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ))
            fig.update_layout(
                title='Portfolio Drawdown Over Time',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Asset Allocation
            if weights_history:
                st.header("🎯 Asset Allocation Over Time")
                weights_df = pd.DataFrame([
                    {'Date': w['date'], **{tickers[i]: w['weights'][i] for i in range(len(tickers))}}
                    for w in weights_history
                ]).set_index('Date')
                
                fig = go.Figure()
                for ticker in tickers:
                    fig.add_trace(go.Scatter(
                        x=weights_df.index,
                        y=weights_df[ticker] * 100,
                        mode='lines',
                        name=ticker,
                        stackgroup='one'
                    ))
                fig.update_layout(
                    title='Portfolio Allocation Over Time',
                    xaxis_title='Date',
                    yaxis_title='Allocation (%)',
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Monte Carlo Simulation
            st.header("🎲 Monte Carlo Simulation (1 Year Ahead)")
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = monte_carlo_simulation(returns_series, portfolio_values[-1], n_simulations=1000, n_days=252)
                
                fig = go.Figure()
                
                # Plot sample paths
                for i in range(min(50, len(mc_results))):
                    fig.add_trace(go.Scatter(
                        y=mc_results[i],
                        mode='lines',
                        line=dict(color='lightblue', width=0.5),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Plot percentiles
                percentiles = np.percentile(mc_results, [5, 50, 95], axis=0)
                fig.add_trace(go.Scatter(
                    y=percentiles[1],
                    mode='lines',
                    name='Median (50th)',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=percentiles[2],
                    mode='lines',
                    name='95th Percentile',
                    line=dict(color='green', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    y=percentiles[0],
                    mode='lines',
                    name='5th Percentile',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Monte Carlo Simulation - 1 Year Projection',
                    xaxis_title='Trading Days',
                    yaxis_title='Portfolio Value ($)',
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                final_values = mc_results[:, -1]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Value (Median)", f"${np.median(final_values):,.2f}")
                with col2:
                    st.metric("95th Percentile", f"${np.percentile(final_values, 95):,.2f}")
                with col3:
                    st.metric("5th Percentile", f"${np.percentile(final_values, 5):,.2f}")
            
            # Trade Log
            if trade_log:
                st.header("📋 Trade Log")
                trade_df = pd.DataFrame(trade_log)
                st.dataframe(trade_df, use_container_width=True)
                
                # Download button
                csv = trade_df.to_csv(index=False)
                st.download_button(
                    label="Download Trade Log",
                    data=csv,
                    file_name="trade_log.csv",
                    mime="text/csv"
                )
        else:
            st.error("Unable to fetch data. Please check your tickers and try again.")
else:
    # Welcome screen
    st.info("👈 Configure your portfolio settings in the sidebar and click 'Run Backtest' to begin!")
    
    st.markdown("""
    ## 🎯 Features
    
    ### Weighting Strategies
    - **Equal Weight**: Distribute capital equally across all assets
    - **Inverse Volatility**: Weight inversely proportional to volatility
    - **Momentum**: Weight based on recent price momentum
    - **Risk Parity**: Equal risk contribution from each asset
    - **Minimum Variance**: Optimize for lowest portfolio variance
    - **Maximum Sharpe**: Optimize for highest risk-adjusted returns
    
    ### Analysis Tools
    - 📊 Equity curve and drawdown analysis
    - 🎯 Dynamic asset allocation visualization
    - 🎲 Monte Carlo simulation for future projections
    - 📋 Detailed trade logs
    - 📈 Comprehensive performance metrics
    
    ### Getting Started
    1. Enter your asset tickers (e.g., SPY, QQQ, GLD)
    2. Select date range and investment type
    3. Choose your strategy and rebalancing frequency
    4. Set transaction and management fees
    5. Click "Run Backtest" to see results!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💰 Money Freedom - Built with Streamlit | Data from Yahoo Finance</p>
    <p><em>Disclaimer: Past performance does not guarantee future results. This tool is for educational purposes only.</em></p>
</div>
""", unsafe_allow_html=True)
