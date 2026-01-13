import streamlit as st

st.set_page_config(page_title="Test App", layout="centered")

st.title("✅ Streamlit Cloud ทำงานปกติ")
st.write("ถ้าคุณเห็นหน้านี้ แปลว่า environment ผ่าน")

# Test imports ทีละตัว
try:
    import pandas as pd
    st.success("pandas OK")
except Exception as e:
    st.error("pandas error")
    st.exception(e)

try:
    import numpy as np
    st.success("numpy OK")
except Exception as e:
    st.error("numpy error")
    st.exception(e)

try:
    import yfinance as yf
    st.success("yfinance OK")
except Exception as e:
    st.error("yfinance error")
    st.exception(e)

try:
    import plotly.express as px
    st.success("plotly OK")
except Exception as e:
    st.error("plotly error")
    st.exception(e)
