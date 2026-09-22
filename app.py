import datetime
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NSE Live Movers & Tomorrow AI Predictor",
    page_icon="⚡",
    layout="wide"
)

# Dark theme card styling
st.markdown("""
    <style>
    .gain-box {
        background-color: #0d2818;
        border: 1px solid #238636;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    .loss-box {
        background-color: #2b1113;
        border: 1px solid #da3633;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Top 50 Liquid Nifty Large-Cap Stocks
BASKET_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "ASIANPAINT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "M&M.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "GRASIM.NS", "HCLTECH.NS",
    "WIPRO.NS", "ONGC.NS", "TECHM.NS", "NESTLEIND.NS", "INDUSINDBK.NS",
    "CIPLA.NS", "BPCL.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "HINDALCO.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "TATACONSUM.NS",
    "SHRIRAMFIN.NS", "BEL.NS", "TRENT.NS", "SBILIFE.NS", "HDFCLIFE.NS"
]

# --- Market Time & Status ---
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.datetime.now(ist)
is_weekday = now_ist.weekday() < 5
market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
is_live = is_weekday and (market_open <= now_ist <= market_close)

col_title, col_btn = st.columns([3, 1])
with col_title:
    st.title("⚡ NSE Market Intelligence & AI Predictor")
    if is_live:
        st.success(f"🟢 Market Status: LIVE (IST {now_ist.strftime('%I:%M:%S %p')})")
    else:
        st.info(f"⚪ Market Status: CLOSED / INACTIVE (Official EOD: {now_ist.strftime('%d-%b-%Y')})")

with col_btn:
    st.write("")
    if st.button("🔄 Refresh Market Data & Retrain", type="primary", use_container_width=True):
        st.cache_data.clear()

# --- Data Ingestion Engine ---
@st.cache_data(ttl=120, show_spinner=False)
def fetch_basket_market_data(tickers):
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365 * 2)  # 2 years of daily data
    
    df = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    return df

with st.spinner("Pulling real-time market data across universe..."):
    market_data = fetch_basket_market_data(BASKET_TICKERS)

if market_data.empty or "Close" not in market_data:
    st.error("Failed to connect to market feed. Verify network connection.")
    st.stop()

# --- Compute Today's Realized Gainers & Losers ---
close_all = market_data["Close"]
vol_all = market_data["Volume"]

today_records = []
for sym in BASKET_TICKERS:
    try:
        if sym not in close_all.columns:
            continue
        c = close_all[sym].dropna()
        if len(c) < 5:
            continue
        v = vol_all[sym].loc[c.index]
        
        ltp = float(c.iloc[-1])
        prev = float(c.iloc[-2])
        chg_rs = ltp - prev
        pct_chg = (chg_rs / prev) * 100.0
        
        # 14-day RSI
        diff = c.diff()
        gain = (diff.where(diff > 0, 0.0)).rolling(14).mean().iloc[-1]
        loss = (-diff.where(diff < 0, 0.0)).rolling(14).mean().iloc[-1]
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Volume Surge Ratio
        avg_v = float(v.rolling(20).mean().iloc[-1]) + 1e-5
        vol_mult = float(v.iloc[-1]) / avg_v

        today_records.append({
            "Symbol": sym.replace(".NS", ""),
            "Ticker": sym,
            "LTP (₹)": round(ltp, 2),
            "Change (₹)": round(chg_rs, 2),
            "Change (%)": round(pct_chg, 2),
            "RSI (14)": round(rsi, 1),
            "Vol Multiple": round(vol_mult, 2),
            "Raw_Change": pct_chg
        })
    except Exception:
        continue

summary_df = pd.DataFrame(today_records)
today_gainers = summary_df.sort_values(by="Raw_Change", ascending=False).head(10).reset_index(drop=True)
today_losers = summary_df.sort_values(by="Raw_Change", ascending=True).head(10).reset_index(drop=True)

# --- Train 1-Day Forward AI Prediction Engine ---
@st.cache_resource(show_spinner=False)
def train_tomorrow_predictor(raw_df, tickers):
    close_p = raw_df["Close"]
    
    # 1-day forward return target
    fwd_ret_1d = close_p.pct_change(1).shift(-1)
    universe_median = fwd_ret_1d.median(axis=1)
    excess_1d = fwd_ret_1d.sub(universe_median, axis=0)

    # Label: 1 if Top 25% outperformer tomorrow, 0 otherwise
    def label_top(row):
        valid = row.dropna()
        if len(valid) == 0:
            return row
        return (row >= valid.quantile(0.75)).astype(float)

    targets = excess_1d.apply(label_top, axis=1)
    target_flat = targets.stack().reset_index()
    target_flat.columns = ["Date", "Ticker", "target"]

    # Build features: 1D return, 5D return, RSI, Intraday Range, Volatility
    feat_rows = []
    for ticker in tickers:
        if ticker not in close_p.columns:
            continue
        c = close_p[ticker].dropna()
        h = raw_df["High"][ticker].loc[c.index]
        l = raw_df["Low"][ticker].loc[c.index]
        v = raw_df["Volume"][ticker].loc[c.index]

        r1 = c.pct_change(1)
        r5 = c.pct_change(5)
        diff = c.diff()
        gain = (diff.where(diff > 0, 0.0)).rolling(14).mean()
        loss = (-diff.where(diff < 0, 0.0)).rolling(14).mean()
        rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
        daily_range = (h - l) / (c + 1e-9)
        vol_ratio = v / (v.rolling(20).mean() + 1e-9)

        stk_df = pd.DataFrame({
            "Ticker": ticker,
            "r1": r1,
            "r5": r5,
            "rsi": rsi,
            "range": daily_range,
            "vol_ratio": vol_ratio
        }, index=c.index)
        feat_rows.append(stk_df)

    feat_matrix = pd.concat(feat_rows).reset_index().rename(columns={"index": "Date", "Date": "Date"})
    
    # Relative Ranks per Date
    for col in ["r1", "r5", "rsi", "range", "vol_ratio"]:
        feat_matrix[f"{col}_rank"] = feat_matrix.groupby("Date")[col].rank(pct=True)

    dataset = pd.merge(feat_matrix, target_flat, on=["Date", "Ticker"], how="inner").dropna()
    feature_cols = [c for c in dataset.columns if c not in ["Date", "Ticker", "target"]]

    X = dataset[feature_cols]
    y = dataset["target"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        verbosity=-1
    )
    model.fit(X, y)

    # Live Inference on the Latest Date
    latest_dt = feat_matrix["Date"].max()
    latest_features = feat_matrix[feat_matrix["Date"] == latest_dt].copy()
    probs = model.predict_proba(latest_features[feature_cols])[:, 1]
    latest_features["Gain_Probability"] = probs

    return latest_features[["Ticker", "Gain_Probability", "rsi", "r1"]], latest_dt

with st.spinner("Computing machine learning forward probabilities for tomorrow..."):
    ai_preds, pred_date = train_tomorrow_predictor(market_data, BASKET_TICKERS)

# Merge latest LTP
ai_preds["Symbol"] = ai_preds["Ticker"].apply(lambda x: x.replace(".NS", ""))
ai_preds = ai_preds.merge(summary_df[["Symbol", "LTP (₹)"]], on="Symbol", how="left")

# Top 10 Predicted Gainers & Losers for Tomorrow
tomorrow_gainers = ai_preds.sort_values(by="Gain_Probability", ascending=False).head(10).reset_index(drop=True)
tomorrow_losers = ai_preds.sort_values(by="Gain_Probability", ascending=True).head(10).reset_index(drop=True)

# ----------------- TABS & PRESENTATION -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Today's Top 10 Gainers",
    "🔻 Today's Top 10 Losers",
    "🔮 Tomorrow's AI Predictions (Gainers/Losers)",
    "📈 Interactive Stock Chart"
])

def style_positive(v):
    return "color: #3fb950; font-weight: bold;"

def style_negative(v):
    return "color: #f85149; font-weight: bold;"

with tab1:
    st.subheader(f"Top 10 Growth Performers Today ({summary_df['Symbol'].count()} Monitored)")
    top_5_g = today_gainers.head(5)
    cols = st.columns(5)
    for idx, c in enumerate(cols):
        item = top_5_g.iloc[idx]
        with c:
            st.markdown(f"""
            <div class="gain-box">
                <h4 style="margin:0; color:#3fb950;">#{idx+1} {item['Symbol']}</h4>
                <h3 style="margin:2px 0;">₹{item['LTP (₹)']}</h3>
                <p style="margin:0; font-size:13px; color:#3fb950; font-weight:bold;">+{item['Change (%)']}% (+₹{item['Change (₹)']})</p>
                <p style="margin:0; font-size:11px; color:#8b949e;">RSI: {item['RSI (14)']} | Vol: {item['Vol Multiple']}x</p>
            </div>
            """, unsafe_allow_html=True)

    st.dataframe(
        today_gainers[["Symbol", "LTP (₹)", "Change (₹)", "Change (%)", "RSI (14)", "Vol Multiple"]]
        .style.map(style_positive, subset=["Change (%)", "Change (₹)"]),
        use_container_width=True,
        height=380
    )

with tab2:
    st.subheader("Top 10 Declining Performers Today")
    top_5_l = today_losers.head(5)
    cols_l = st.columns(5)
    for idx, c in enumerate(cols_l):
        item = top_5_l.iloc[idx]
        with c:
            st.markdown(f"""
            <div class="loss-box">
                <h4 style="margin:0; color:#f85149;">#{idx+1} {item['Symbol']}</h4>
                <h3 style="margin:2px 0;">₹{item['LTP (₹)']}</h3>
                <p style="margin:0; font-size:13px; color:#f85149; font-weight:bold;">{item['Change (%)']}% (₹{item['Change (₹)']})</p>
                <p style="margin:0; font-size:11px; color:#8b949e;">RSI: {item['RSI (14)']} | Vol: {item['Vol Multiple']}x</p>
            </div>
            """, unsafe_allow_html=True)

    st.dataframe(
        today_losers[["Symbol", "LTP (₹)", "Change (₹)", "Change (%)", "RSI (14)", "Vol Multiple"]]
        .style.map(style_negative, subset=["Change (%)", "Change (₹)"]),
        use_container_width=True,
        height=380
    )

with tab3:
    st.subheader("🔮 Tomorrow's High-Probability Forecast")
    st.caption("Derived from cross-sectional volume buildup, range expansion, and price memory.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 🟢 Top 10 Predicted Gainers Tomorrow")
        disp_tg = tomorrow_gainers[["Symbol", "LTP (₹)", "Gain_Probability", "rsi"]].copy()
        disp_tg["AI Growth Conviction"] = disp_tg["Gain_Probability"].map(lambda x: f"{x * 100:.1f}%")
        disp_tg["RSI (14)"] = disp_tg["rsi"].round(1)
        st.dataframe(
            disp_tg[["Symbol", "LTP (₹)", "AI Growth Conviction", "RSI (14)"]],
            use_container_width=True,
            height=380
        )
    
    with col_p2:
        st.markdown("#### 🔴 Top 10 Predicted Losers / Weakness Tomorrow")
        disp_tl = tomorrow_losers[["Symbol", "LTP (₹)", "Gain_Probability", "rsi"]].copy()
        disp_tl["AI Underperform Risk"] = disp_tl["Gain_Probability"].map(lambda x: f"{(1 - x) * 100:.1f}%")
        disp_tl["RSI (14)"] = disp_tl["rsi"].round(1)
        st.dataframe(
            disp_tl[["Symbol", "LTP (₹)", "AI Underperform Risk", "RSI (14)"]],
            use_container_width=True,
            height=380
        )

with tab4:
    st.subheader("📊 Interactive Candlestick Chart & Trend Analysis")
    selected = st.selectbox("Select stock to inspect:", options=summary_df["Ticker"].tolist())
    
    # Download single stock data cleanly with flattened columns to guarantee Plotly loads
    stock_df = yf.download(selected, period="6mo", interval="1d", progress=False)
    
    if not stock_df.empty:
        # Handle MultiIndex column flattening if present
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)
            
        c_series = stock_df["Close"].squeeze()
        o_series = stock_df["Open"].squeeze()
        h_series = stock_df["High"].squeeze()
        l_series = stock_df["Low"].squeeze()
        dates = stock_df.index

        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=o_series,
            high=h_series,
            low=l_series,
            close=c_series,
            name=selected.replace(".NS", "")
        )])
        
        # 20 EMA and 50 EMA
        ema_20 = c_series.ewm(span=20, adjust=False).mean()
        ema_50 = c_series.ewm(span=50, adjust=False).mean()
        fig.add_trace(go.Scatter(x=dates, y=ema_20, mode='lines', line=dict(color='#ff9800', width=1.5), name='20 EMA'))
        fig.add_trace(go.Scatter(x=dates, y=ema_50, mode='lines', line=dict(color='#00e5ff', width=1.5), name='50 EMA'))

        fig.update_layout(
            title=f"{selected.replace('.NS', '')} - 6-Month Candlestick with 20 & 50 EMA",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=520,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price history available for the selected asset.")