import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Stock Price Dashboard",
    layout="wide"
)

st.title("📈 Stock Price Dashboard")
st.markdown(
    "Simple stock price visualization using **Streamlit**, **yfinance**, and **Plotly**"
)

# -------------------------------
# Sidebar - User Input
# -------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Example: AAPL, MSFT, TSLA, NVDA"
)

period = st.sidebar.selectbox(
    "Time Period",
    options=["1mo", "3mo", "6mo", "1y", "5y"],
    index=3
)

# -------------------------------
# Data Loading with Cache
# -------------------------------
@st.cache_data(show_spinner=True)
def load_stock_data(ticker_symbol, period_range):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period_range)
    return df

# -------------------------------
# Main App Logic
# -------------------------------
try:
    df = load_stock_data(ticker, period)

    if df.empty:
        st.warning("⚠️ No data found. Please check the ticker symbol.")
    else:
        st.subheader(f"📊 {ticker.upper()} Closing Price")

        fig = px.line(
            df,
            x=df.index,
            y="Close",
            labels={"Close": "Price (USD)", "index": "Date"},
            title=f"{ticker.upper()} Stock Price"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 View Raw Data"):
            st.dataframe(df.tail(20), use_container_width=True)

except Exception as e:
    st.error("❌ An unexpected error occurred while loading data.")
    st.exception(e)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Powered by Streamlit • Data from Yahoo Finance")
