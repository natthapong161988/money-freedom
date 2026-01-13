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

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Money Freedom - ระบบทดสอบกลยุทธ์พอร์ตการลงทุน",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS แบบกำหนดเอง - ใช้ฟอนต์ที่อ่านง่าย
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00acc1;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
    
    h1, h2, h3 {
        font-family: 'Sarabun', sans-serif;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Sarabun', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# หัวข้อหลัก
st.markdown('<h1 class="main-header">💰 Money Freedom</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบทดสอบและวิเคราะห์กลยุทธ์พอร์ตการลงทุนแบบเชิงปริมาณ</p>', unsafe_allow_html=True)

# ฟังก์ชันช่วยเหลือ
@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    """ดึงข้อมูลราคาย้อนหลัง"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers[0]]
        data = data.ffill().bfill()
        return data
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}")
        return None

def calculate_returns(prices):
    """คำนวณผลตอบแทน"""
    return prices.pct_change().fillna(0)

def equal_weight(n_assets):
    """กลยุทธ์น้ำหนักเท่ากัน"""
    return np.ones(n_assets) / n_assets

def inverse_volatility_weight(returns, lookback):
    """กลยุทธ์น้ำหนักผกผันกับความผันผวน"""
    vol = returns.tail(lookback).std()
    inv_vol = 1 / (vol + 1e-8)
    return (inv_vol / inv_vol.sum()).values

def momentum_weight(prices, lookback):
    """กลยุทธ์โมเมนตัม"""
    momentum = prices.pct_change(lookback).iloc[-1]
    momentum = momentum.clip(lower=0)
    if momentum.sum() == 0:
        return equal_weight(len(momentum))
    return (momentum / momentum.sum()).values

def risk_parity_weight(returns, lookback):
    """กลยุทธ์ความเสี่ยงเท่ากัน (Risk Parity)"""
    cov_matrix = returns.tail(lookback).cov()
    inv_vol = 1 / (np.sqrt(np.diag(cov_matrix)) + 1e-8)
    weights = inv_vol / inv_vol.sum()
    return weights

def min_variance_weight(returns, lookback):
    """กลยุทธ์ความแปรปรวนต่ำสุด"""
    cov_matrix = returns.tail(lookback).cov().values
    n = len(cov_matrix)
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.ones(n) / n
    
    try:
        result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        return result.x if result.success else initial_weights
    except:
        return initial_weights

def max_sharpe_weight(returns, lookback, risk_free_rate=0.02):
    """กลยุทธ์อัตราส่วนชาร์ปสูงสุด"""
    mean_returns = returns.tail(lookback).mean() * 252
    cov_matrix = returns.tail(lookback).cov().values * 252
    n = len(mean_returns)
    
    def neg_sharpe(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol == 0:
            return 1e10
        return -(portfolio_return - risk_free_rate) / portfolio_vol
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.ones(n) / n
    
    try:
        result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        return result.x if result.success else initial_weights
    except:
        return initial_weights

def backtest_strategy(prices, returns, strategy, lookback, rebalance_freq, 
                     initial_capital, transaction_fee, management_fee, investment_type='lump_sum', dca_amount=0):
    """เครื่องมือทดสอบกลยุทธ์หลัก"""
    
    portfolio_value = [initial_capital]
    weights_history = []
    trade_log = []
    cash = initial_capital
    holdings = np.zeros(len(prices.columns))
    
    freq_map = {
        'รายวัน': 1,
        'รายสัปดาห์': 5,
        'รายเดือน': 21,
        'รายไตรมาส': 63,
        'รายปี': 252
    }
    rebalance_days = freq_map.get(rebalance_freq, 21)
    
    for i in range(lookback, len(prices)):
        current_prices = prices.iloc[i].values
        
        if investment_type == 'DCA' and i > lookback and i % 21 == 0:
            cash += dca_amount
        
        if i % rebalance_days == 0:
            if strategy == 'น้ำหนักเท่ากัน':
                weights = equal_weight(len(prices.columns))
            elif strategy == 'ผกผันกับความผันผวน':
                weights = inverse_volatility_weight(returns.iloc[:i], lookback)
            elif strategy == 'โมเมนตัม':
                weights = momentum_weight(prices.iloc[:i], lookback)
            elif strategy == 'ความเสี่ยงเท่ากัน':
                weights = risk_parity_weight(returns.iloc[:i], lookback)
            elif strategy == 'ความแปรปรวนต่ำสุด':
                weights = min_variance_weight(returns.iloc[:i], lookback)
            elif strategy == 'อัตราส่วนชาร์ปสูงสุด':
                weights = max_sharpe_weight(returns.iloc[:i], lookback)
            else:
                weights = equal_weight(len(prices.columns))
            
            weights_history.append({
                'date': prices.index[i],
                'weights': weights.copy()
            })
            
            total_value = cash + np.sum(holdings * current_prices)
            target_value = total_value * weights
            target_holdings = target_value / (current_prices + 1e-8)
            
            trades = target_holdings - holdings
            transaction_costs = np.sum(np.abs(trades * current_prices)) * transaction_fee
            
            holdings = target_holdings
            cash = total_value - np.sum(holdings * current_prices) - transaction_costs
            
            for j, ticker in enumerate(prices.columns):
                if abs(trades[j]) > 0.001:
                    trade_log.append({
                        'วันที่': prices.index[i],
                        'สินทรัพย์': ticker,
                        'การทำรายการ': 'ซื้อ' if trades[j] > 0 else 'ขาย',
                        'จำนวน': abs(trades[j]),
                        'ราคา': current_prices[j],
                        'มูลค่า': abs(trades[j] * current_prices[j])
                    })
        
        total_value = cash + np.sum(holdings * current_prices)
        
        if i > lookback:
            daily_mgmt_fee = management_fee / 252
            total_value *= (1 - daily_mgmt_fee)
        
        portfolio_value.append(total_value)
    
    return portfolio_value, weights_history, trade_log

def calculate_metrics(portfolio_values, initial_capital):
    """คำนวณตัวชี้วัดผลการดำเนินงาน"""
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    total_return = portfolio_values[-1] / initial_capital
    n_years = len(portfolio_values) / 252
    cagr = (total_return ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    
    volatility = returns.std() * np.sqrt(252) * 100
    
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'CAGR': cagr,
        'ความผันผวน': volatility,
        'อัตราส่วนชาร์ป': sharpe,
        'อัตราส่วนซอร์ทิโน': sortino,
        'การลดลงสูงสุด': max_drawdown,
        'มูลค่าสุดท้าย': portfolio_values[-1]
    }

def monte_carlo_simulation(returns, initial_value, n_simulations=1000, n_days=252):
    """การจำลองมอนติคาร์โล"""
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

# แถบด้านข้าง - พารามิเตอร์
st.sidebar.header("⚙️ การตั้งค่าพอร์ตการลงทุน")

st.sidebar.markdown("---")

# ส่วนที่ 1: สินทรัพย์
st.sidebar.subheader("📊 เลือกสินทรัพย์")
tickers_input = st.sidebar.text_input(
    "รหัสหุ้น/ETF (คั่นด้วยจุลภาค)",
    "SPY,QQQ,IWM,TLT,GLD",
    help="ใส่รหัสหุ้นหรือ ETF คั่นด้วยจุลภาค เช่น SPY,QQQ,GLD"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# ตรวจสอบว่ามีสินทรัพย์อย่างน้อย 2 รายการ
if len(tickers) < 2:
    st.sidebar.warning("⚠️ กรุณาเลือกสินทรัพย์อย่างน้อย 2 รายการ")

st.sidebar.markdown("---")

# ส่วนที่ 2: ช่วงเวลา
st.sidebar.subheader("📅 ช่วงเวลาทดสอบ")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("วันเริ่มต้น", datetime.now() - timedelta(days=365*5))
with col2:
    end_date = st.date_input("วันสิ้นสุด", datetime.now())

if start_date >= end_date:
    st.sidebar.error("⚠️ วันเริ่มต้นต้องน้อยกว่าวันสิ้นสุด")

st.sidebar.markdown("---")

# ส่วนที่ 3: การลงทุน
st.sidebar.subheader("💵 การตั้งค่าการลงทุน")
initial_capital = st.sidebar.number_input(
    "เงินทุนเริ่มต้น (บาท)", 
    min_value=10000, 
    value=100000, 
    step=10000,
    help="จำนวนเงินที่จะลงทุนในครั้งแรก"
)

investment_type = st.sidebar.radio(
    "ประเภทการลงทุน",
    ['ลงทุนครั้งเดียว', 'ลงทุนสม่ำเสมอ (DCA)'],
    help="ลงทุนครั้งเดียว = ลงทุนก้อนเดียว, DCA = ทยอยลงทุนทุกเดือน"
)

dca_amount = 0
if investment_type == 'ลงทุนสม่ำเสมอ (DCA)':
    dca_amount = st.sidebar.number_input(
        "จำนวนเงินลงทุนต่อเดือน (บาท)", 
        min_value=0, 
        value=5000, 
        step=1000,
        help="จำนวนเงินที่จะลงทุนเพิ่มทุกเดือน"
    )

st.sidebar.markdown("---")

# ส่วนที่ 4: กลยุทธ์
st.sidebar.subheader("🎯 การเลือกกลยุทธ์")
strategy = st.sidebar.selectbox(
    "กลยุทธ์การจัดสรรน้ำหนัก",
    ['น้ำหนักเท่ากัน', 'ผกผันกับความผันผวน', 'โมเมนตัม', 
     'ความเสี่ยงเท่ากัน', 'ความแปรปรวนต่ำสุด', 'อัตราส่วนชาร์ปสูงสุด'],
    help="เลือกวิธีการกระจายเงินลงทุนในแต่ละสินทรัพย์"
)

# คำอธิบายกลยุทธ์
strategy_descriptions = {
    'น้ำหนักเท่ากัน': '💡 กระจายเงินลงทุนเท่าๆ กันในทุกสินทรัพย์',
    'ผกผันกับความผันผวน': '💡 ลงทุนมากในสินทรัพย์ที่มีความผันผวนต่ำ',
    'โมเมนตัม': '💡 ลงทุนมากในสินทรัพย์ที่มีผลตอบแทนดีล่าสุด',
    'ความเสี่ยงเท่ากัน': '💡 ทำให้ทุกสินทรัพย์มีความเสี่ยงเท่ากัน',
    'ความแปรปรวนต่ำสุด': '💡 หาพอร์ตที่มีความผันผวนต่ำสุด',
    'อัตราส่วนชาร์ปสูงสุด': '💡 หาพอร์ตที่ให้ผลตอบแทนต่อความเสี่ยงสูงสุด'
}
st.sidebar.info(strategy_descriptions.get(strategy, ''))

lookback = st.sidebar.slider(
    "ระยะเวลามองย้อนหลัง (วัน)", 
    20, 252, 60,
    help="จำนวนวันที่ใช้คำนวณน้ำหนักการลงทุน"
)

rebalance_freq = st.sidebar.selectbox(
    "ความถี่ในการปรับสมดุล",
    ['รายวัน', 'รายสัปดาห์', 'รายเดือน', 'รายไตรมาส', 'รายปี'],
    index=2,
    help="ความถี่ในการปรับสัดส่วนการลงทุนใหม่"
)

st.sidebar.markdown("---")

# ส่วนที่ 5: ค่าธรรมเนียม
st.sidebar.subheader("💸 ค่าธรรมเนียม")
transaction_fee = st.sidebar.number_input(
    "ค่าธรรมเนียมการซื้อขาย (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.25, 
    step=0.05,
    help="ค่าธรรมเนียมที่เกิดขึ้นทุกครั้งที่ซื้อหรือขาย"
) / 100

management_fee = st.sidebar.number_input(
    "ค่าธรรมเนียมบริหาร (%/ปี)", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.5, 
    step=0.1,
    help="ค่าธรรมเนียมการบริหารจัดการต่อปี"
) / 100

st.sidebar.markdown("---")

# ปุ่มเริ่มทดสอบ
run_backtest = st.sidebar.button("🚀 เริ่มทดสอบกลยุทธ์", type="primary")

# เนื้อหาหลัก
if run_backtest:
    # ตรวจสอบข้อมูลก่อนเริ่ม
    if len(tickers) < 2:
        st.error("❌ กรุณาเลือกสินทรัพย์อย่างน้อย 2 รายการ")
    elif start_date >= end_date:
        st.error("❌ วันเริ่มต้นต้องน้อยกว่าวันสิ้นสุด")
    else:
        with st.spinner("⏳ กำลังดึงข้อมูลและทดสอบกลยุทธ์..."):
            prices = fetch_data(tickers, start_date, end_date)
            
            if prices is not None and not prices.empty:
                returns = calculate_returns(prices)
                
                # ทดสอบกลยุทธ์
                portfolio_values, weights_history, trade_log = backtest_strategy(
                    prices, returns, strategy, lookback, rebalance_freq,
                    initial_capital, transaction_fee, management_fee,
                    investment_type.replace('ลงทุนครั้งเดียว', 'lump_sum').replace('ลงทุนสม่ำเสมอ (DCA)', 'DCA'),
                    dca_amount
                )
                
                metrics = calculate_metrics(portfolio_values, initial_capital)
                
                # แสดงผลลัพธ์
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # ตัวชี้วัดหลัก
                st.markdown("## 📈 ตัวชี้วัดผลการดำเนินงาน")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "ผลตอบแทนต่อปี (CAGR)",
                        f"{metrics['CAGR']:.2f}%",
                        delta=f"{metrics['CAGR']:.2f}%" if metrics['CAGR'] > 0 else None
                    )
                
                with col2:
                    st.metric(
                        "ความผันผวน",
                        f"{metrics['ความผันผวน']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "อัตราส่วนชาร์ป",
                        f"{metrics['อัตราส่วนชาร์ป']:.2f}",
                        delta="ดี" if metrics['อัตราส่วนชาร์ป'] > 1 else "ปานกลาง" if metrics['อัตราส่วนชาร์ป'] > 0.5 else "ต่ำ"
                    )
                
                with col4:
                    st.metric(
                        "การลดลงสูงสุด",
                        f"{metrics['การลดลงสูงสุด']:.2f}%",
                        delta=f"{metrics['การลดลงสูงสุด']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col5:
                    st.metric(
                        "มูลค่าสุดท้าย",
                        f"฿{metrics['มูลค่าสุดท้าย']:,.0f}",
                        delta=f"฿{metrics['มูลค่าสุดท้าย'] - initial_capital:,.0f}"
                    )
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # กราฟหลัก
                tab1, tab2, tab3, tab4 = st.tabs(["📊 เส้นกราฟพอร์ต", "📉 การลดลง", "🎯 สัดส่วนการลงทุน", "🎲 การจำลอง"])
                
                with tab1:
                    st.markdown("### 📊 มูลค่าพอร์ตตลอดเวลา")
                    dates = prices.index[lookback:]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=portfolio_values[1:],
                        mode='lines',
                        name='มูลค่าพอร์ต',
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title='มูลค่าพอร์ตการลงทุนตลอดระยะเวลาทดสอบ',
                        xaxis_title='วันที่',
                        yaxis_title='มูลค่า (บาท)',
                        hovermode='x unified',
                        height=500,
                        font=dict(family="Sarabun", size=14),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # สรุปผลตอบแทน
                    total_return_pct = ((metrics['มูลค่าสุดท้าย'] - initial_capital) / initial_capital) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>📊 สรุปผลการลงทุน</h4>
                        <ul>
                            <li>เงินลงทุนเริ่มต้น: <b>฿{initial_capital:,.0f}</b></li>
                            <li>มูลค่าสุดท้าย: <b>฿{metrics['มูลค่าสุดท้าย']:,.0f}</b></li>
                            <li>กำไร/ขาดทุน: <b>฿{metrics['มูลค่าสุดท้าย'] - initial_capital:,.0f}</b></li>
                            <li>ผลตอบแทนรวม: <b>{total_return_pct:.2f}%</b></li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>📈 ประสิทธิภาพการลงทุน</h4>
                        <ul>
                            <li>ผลตอบแทนต่อปี (CAGR): <b>{metrics['CAGR']:.2f}%</b></li>
                            <li>ความผันผวนต่อปี: <b>{metrics['ความผันผวน']:.2f}%</b></li>
                            <li>อัตราส่วนชาร์ป: <b>{metrics['อัตราส่วนชาร์ป']:.2f}</b></li>
                            <li>อัตราส่วนซอร์ทิโน: <b>{metrics['อัตราส่วนซอร์ทิโน']:.2f}</b></li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### 📉 การวิเคราะห์การลดลง (Drawdown)")
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
                        name='การลดลง',
                        line=dict(color='#e74c3c', width=2),
                        fillcolor='rgba(231, 76, 60, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title='การลดลงของมูลค่าพอร์ตตลอดเวลา',
                        xaxis_title='วันที่',
                        yaxis_title='การลดลง (%)',
                        hovermode='x unified',
                        height=500,
                        font=dict(family="Sarabun", size=14),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>⚠️ คำเตือนเกี่ยวกับความเสี่ยง</h4>
                    <p>การลดลงสูงสุดของพอร์ตนี้อยู่ที่ <b>{metrics['การลดลงสูงสุด']:.2f}%</b></p>
                    <p>หมายความว่า ในช่วงที่แย่ที่สุด พอร์ตของคุณอาจลดลงจากจุดสูงสุดถึง {abs(metrics['การลดลงสูงสุด']):.2f}%</p>
                    <p>กรุณาพิจารณาความเสี่ยงนี้ก่อนตัดสินใจลงทุนจริง</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab3:
                    if weights_history:
                        st.markdown("### 🎯 สัดส่วนการลงทุนตลอดเวลา")
                        weights_df = pd.DataFrame([
                            {'วันที่': w['date'], **{tickers[i]: w['weights'][i] for i in range(len(tickers))}}
                            for w in weights_history
                        ]).set_index('วันที่')
                        
                        fig = go.Figure()
                        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
                        
                        for idx, ticker in enumerate(tickers):
                            fig.add_trace(go.Scatter(
                                x=weights_df.index,
                                y=weights_df[ticker] * 100,
                                mode='lines',
                                name=ticker,
                                stackgroup='one',
                                line=dict(width=0.5, color=colors[idx % len(colors)]),
                                fillcolor=colors[idx % len(colors)]
                            ))
                        
                        fig.update_layout(
                            title='สัดส่วนการลงทุนในแต่ละสินทรัพย์',
                            xaxis_title='วันที่',
                            yaxis_title='สัดส่วน (%)',
                            hovermode='x unified',
                            height=500,
                            font=dict(family="Sarabun", size=14),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # แสดงตารางน้ำหนักล่าสุด
                        st.markdown("#### 📋 สัดส่วนการลงทุนล่าสุด")
                        latest_weights = weights_history[-1]['weights']
                        weight_table = pd.DataFrame({
                            'สินทรัพย์': tickers,
                            'สัดส่วน (%)': [f"{w*100:.2f}%" for w in latest_weights],
                            'มูลค่าโดยประมาณ (บาท)': [f"฿{w*metrics['มูลค่าสุดท้าย']:,.0f}" for w in latest_weights]
                        })
                        st.dataframe(weight_table, use_container_width=True, hide_index=True)
                
                with tab4:
                    st.markdown("### 🎲 การจำลองมอนติคาร์โล (คาดการณ์ 1 ปีข้างหน้า)")
                    
                    with st.spinner("กำลังจำลองสถานการณ์ 1,000 รอบ..."):
                        mc_results = monte_carlo_simulation(returns_series, portfolio_values[-1], n_simulations=1000, n_days=252)
                        
                        fig = go.Figure()
                        
                        # แสดงเส้นตัวอย่าง 50 เส้น
                        for i in range(min(50, len(mc_results))):
                            fig.add_trace(go.Scatter(
                                y=mc_results[i],
                                mode='lines',
                                line=dict(color='rgba(102, 126, 234, 0.1)', width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # แสดงเปอร์เซนไทล์
                        percentiles = np.percentile(mc_results, [5, 50, 95], axis=0)
                        
                        fig.add_trace(go.Scatter(
                            y=percentiles[1],
                            mode='lines',
                            name='ค่ากลาง (50th)',
                            line=dict(color='#667eea', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            y=percentiles[2],
                            mode='lines',
                            name='กรณีดี (95th)',
                            line=dict(color='#43e97b', width=3, dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            y=percentiles[0],
                            mode='lines',
                            name='กรณีแย่ (5th)',
                            line=dict(color='#e74c3c', width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='การจำลองมอนติคาร์โล - คาดการณ์มูลค่าพอร์ต 1 ปีข้างหน้า',
                            xaxis_title='วันซื้อขาย',
                            yaxis_title='มูลค่าพอร์ต (บาท)',
                            hovermode='x unified',
                            height=500,
                            font=dict(family="Sarabun", size=14),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # สถิติการจำลอง
                        final_values = mc_results[:, -1]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "มูลค่าปัจจุบัน",
                                f"฿{portfolio_values[-1]:,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "คาดการณ์ (ค่ากลาง)",
                                f"฿{np.median(final_values):,.0f}",
                                delta=f"฿{np.median(final_values) - portfolio_values[-1]:,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "กรณีดี (95th)",
                                f"฿{np.percentile(final_values, 95):,.0f}",
                                delta=f"฿{np.percentile(final_values, 95) - portfolio_values[-1]:,.0f}"
                            )
                        
                        with col4:
                            st.metric(
                                "กรณีแย่ (5th)",
                                f"฿{np.percentile(final_values, 5):,.0f}",
                                delta=f"฿{np.percentile(final_values, 5) - portfolio_values[-1]:,.0f}",
                                delta_color="inverse"
                            )
                        
                        st.markdown("""
                        <div class="info-box">
                        <h4>💡 คำอธิบาย</h4>
                        <p>การจำลองมอนติคาร์โลเป็นการสร้างสถานการณ์ที่เป็นไปได้ 1,000 รูปแบบ โดยอิงจากผลตอบแทนและความผันผวนในอดีต</p>
                        <ul>
                            <li><b>ค่ากลาง (50th):</b> มูลค่าที่คาดว่าจะเกิดขึ้นในกรณีปกติ</li>
                            <li><b>กรณีดี (95th):</b> มูลค่าในกรณีที่ตลาดดีกว่าที่คาด (โอกาส 5%)</li>
                            <li><b>กรณีแย่ (5th):</b> มูลค่าในกรณีที่ตลาดแย่กว่าที่คาด (โอกาส 5%)</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # บันทึกการซื้อขาย
                if trade_log:
                    st.markdown("## 📋 บันทึกการซื้อขาย")
                    
                    trade_df = pd.DataFrame(trade_log)
                    
                    # สรุปการซื้อขาย
                    total_trades = len(trade_df)
                    total_buy = len(trade_df[trade_df['การทำรายการ'] == 'ซื้อ'])
                    total_sell = len(trade_df[trade_df['การทำรายการ'] == 'ขาย'])
                    total_volume = trade_df['มูลค่า'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("จำนวนรายการทั้งหมด", f"{total_trades:,}")
                    with col2:
                        st.metric("รายการซื้อ", f"{total_buy:,}")
                    with col3:
                        st.metric("รายการขาย", f"{total_sell:,}")
                    with col4:
                        st.metric("มูลค่ารวม", f"฿{total_volume:,.0f}")
                    
                    # แสดงตาราง
                    st.dataframe(
                        trade_df.style.format({
                            'จำนวน': '{:.4f}',
                            'ราคา': '฿{:.2f}',
                            'มูลค่า': '฿{:,.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # ปุ่มดาวน์โหลด
                    csv = trade_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 ดาวน์โหลดบันทึกการซื้อขาย (CSV)",
                        data=csv,
                        file_name=f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.error("❌ ไม่สามารถดึงข้อมูลได้ กรุณาตรวจสอบรหัสสินทรัพย์และลองใหม่อีกครั้ง")

else:
    # หน้าต้อนรับ
    st.markdown("""
    <div class="info-box">
    <h3>👋 ยินดีต้อนรับสู่ Money Freedom</h3>
    <p>ระบบทดสอบและวิเคราะห์กลยุทธ์พอร์ตการลงทุนแบบเชิงปริมาณ</p>
    <p><b>เริ่มต้นใช้งาน:</b> กรุณาตั้งค่าพารามิเตอร์ด้านซ้าย แล้วคลิกปุ่ม "เริ่มทดสอบกลยุทธ์"</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 กลยุทธ์การลงทุน
        
        **1. น้ำหนักเท่ากัน (Equal Weight)**
        - กระจายเงินลงทุนเท่าๆ กันในทุกสินทรัพย์
        - เหมาะสำหรับผู้ที่ต้องการความเรียบง่าย
        
        **2. ผกผันกับความผันผวน (Inverse Volatility)**
        - ลงทุนมากในสินทรัพย์ที่มีความผันผวนต่ำ
        - ช่วยลดความเสี่ยงของพอร์ต
        
        **3. โมเมนตัม (Momentum)**
        - ลงทุนมากในสินทรัพย์ที่มีผลตอบแทนดีล่าสุด
        - เหมาะสำหรับตลาดที่มีแนวโน้ม
        
        **4. ความเสี่ยงเท่ากัน (Risk Parity)**
        - ทำให้ทุกสินทรัพย์มีส่วนในความเสี่ยงเท่ากัน
        - สมดุลระหว่างผลตอบแทนและความเสี่ยง
        
        **5. ความแปรปรวนต่ำสุด (Minimum Variance)**
        - หาพอร์ตที่มีความผันผวนต่ำที่สุด
        - เหมาะสำหรับผู้ที่ไม่ชอบความเสี่ยง
        
        **6. อัตราส่วนชาร์ปสูงสุด (Maximum Sharpe)**
        - หาพอร์ตที่ให้ผลตอบแทนต่อความเสี่ยงสูงสุด
        - พยายามหาจุดสมดุลที่ดีที่สุด
        """)
    
    with col2:
        st.markdown("""
        ### 📊 เครื่องมือวิเคราะห์
        
        **การแสดงผลที่คุณจะได้รับ:**
        
        ✅ **ตัวชี้วัดผลการดำเนินงาน**
        - ผลตอบแทนต่อปี (CAGR)
        - ความผันผวน (Volatility)
        - อัตราส่วนชาร์ป (Sharpe Ratio)
        - การลดลงสูงสุด (Maximum Drawdown)
        
        ✅ **กราฟวิเคราะห์**
        - เส้นกราฟมูลค่าพอร์ต
        - กราฟการลดลง (Drawdown)
        - สัดส่วนการลงทุนตลอดเวลา
        - การจำลองมอนติคาร์โล
        
        ✅ **บันทึกการซื้อขาย**
        - รายละเอียดทุกรายการซื้อขาย
        - ดาวน์โหลดเป็นไฟล์ CSV
        
        ---
        
        ### 💡 คำแนะนำ
        
        1. **เริ่มต้นง่ายๆ** - ลองใช้กลยุทธ์น้ำหนักเท่ากันก่อน
        2. **ทดสอบหลายกลยุทธ์** - เปรียบเทียบผลลัพธ์
        3. **ดูที่ความเสี่ยง** - ไม่ใช่แค่ผลตอบแทน
        4. **พิจารณาค่าธรรมเนียม** - มีผลต่อผลตอบแทนระยะยาว
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ ข้อจำกัดความรับผิดชอบ</h4>
    <p><b>ข้อมูลสำคัญที่ควรทราบ:</b></p>
    <ul>
        <li>เครื่องมือนี้จัดทำขึ้นเพื่อการศึกษาและการวิจัยเท่านั้น</li>
        <li>ผลการทดสอบในอดีตไม่ได้การันตีผลตอบแทนในอนาคต</li>
        <li>การลงทุนมีความเสี่ยง ผู้ลงทุนอาจได้รับหรือสูญเสียเงินลงทุน</li>
        <li>กรุณาศึกษาข้อมูลและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุนจริง</li>
        <li>ผู้พัฒนาไม่รับผิดชอบต่อผลการลงทุนที่เกิดขึ้นจริง</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ส่วนท้าย
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <p style='font-size: 1.1rem; color: #666;'>💰 <b>Money Freedom</b> - สร้างด้วย Streamlit</p>
    <p style='font-size: 0.9rem; color: #999;'>ข้อมูลจาก Yahoo Finance | พัฒนาเพื่อการศึกษา</p>
    <p style='font-size: 0.85rem; color: #aaa; font-style: italic; margin-top: 1rem;'>
        ผลการดำเนินงานในอดีตมิได้เป็นสิ่งยืนยันถึงผลการดำเนินงานในอนาคต<br>
        การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจลงทุน
    </p>
</div>
""", unsafe_allow_html=True)
