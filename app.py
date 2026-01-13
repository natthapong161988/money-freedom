import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="แดชบอร์ดวิเคราะห์การลงทุน",
    layout="wide"
)

# =====================================================
# Font & Style (Thai-friendly)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

h1, h2, h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.title("📊 แดชบอร์ดวิเคราะห์การลงทุน")
st.markdown("วิเคราะห์หุ้นด้วยตัวเลขสำคัญ เพื่อการตัดสินใจลงทุนอย่างมีระบบ")
st.divider()

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🔧 ตั้งค่าการวิเคราะห์")

ticker = st.sidebar.text_input(
    "สัญลักษณ์หุ้น (Ticker)",
    value="AAPL"
)

period_label = st.sidebar.selectbox(
    "ช่วงเวลาข้อมูล",
    ["1 เดือน", "3 เดือน", "6 เดือน", "1 ปี", "5 ปี"],
    index=3
)

period_map = {
    "1 เดือน": "1mo",
    "3 เดือน": "3mo",
    "6 เดือน": "6mo",
    "1 ปี": "1y",
    "5 ปี": "5y"
}

# =====================================================
# Data Functions
# =====================================================
@st.cache_data
def load_price_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df


def calculate_basic_metrics(df):
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = (end_price - start_price) / start_price * 100
    return start_price, end_price, total_return


def calculate_cagr(df):
    days = (df.index[-1] - df.index[0]).days
    years = days / 365
    if years <= 0:
        return 0
    return ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / years) - 1) * 100


def calculate_volatility(df):
    daily_return = df["Close"].pct_change()
    return daily_return.std() * np.sqrt(252) * 100


def calculate_max_drawdown(df):
    cumulative_max = df["Close"].cummax()
    drawdown = (df["Close"] - cumulative_max) / cumulative_max
    return drawdown.min() * 100


def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    daily_return = df["Close"].pct_change().dropna()
    excess_return = daily_return.mean() * 252 - risk_free_rate
    volatility = daily_return.std() * np.sqrt(252)
    if volatility == 0:
        return 0
    return excess_return / volatility


def add_moving_averages(df):
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df

# =====================================================
# Main Logic
# =====================================================
try:
    df = load_price_data(ticker, period_map[period_label])

    if df.empty:
        st.warning("ไม่พบข้อมูล กรุณาตรวจสอบสัญลักษณ์หุ้น")
    else:
        df = add_moving_averages(df)

        start_price, end_price, total_return = calculate_basic_metrics(df)
        cagr = calculate_cagr(df)
        volatility = calculate_volatility(df)
        max_dd = calculate_max_drawdown(df)
        sharpe = calculate_sharpe_ratio(df)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("ราคาต้นงวด", f"${start_price:,.2f}")
        col2.metric("ราคาปัจจุบัน", f"${end_price:,.2f}")
        col3.metric("ผลตอบแทนรวม", f"{total_return:.2f}%")
        col4.metric("CAGR ต่อปี", f"{cagr:.2f}%")
        col5.metric("Volatility", f"{volatility:.2f}%")
        col6.metric("Max Drawdown", f"{max_dd:.2f}%")

        st.subheader("📈 กราฟราคาหุ้นและค่าเฉลี่ยเคลื่อนที่")

        fig = px.line(
            df,
            x=df.index,
            y=["Close", "MA50", "MA200"],
            labels={"value": "ราคา (USD)", "index": "วันที่", "variable": "เส้น"},
            title=f"ราคาหุ้น {ticker.upper()}"
        )

        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Risk-adjusted Performance")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with st.expander("📄 ตารางข้อมูลย้อนหลัง"):
            st.dataframe(df.tail(30), use_container_width=True)

except Exception as e:
    st.error("เกิดข้อผิดพลาดในการประมวลผลข้อมูล")
    st.exception(e)

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption("ข้อมูลจาก Yahoo Finance | ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")
