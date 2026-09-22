import numpy as np
import pandas as pd
from src.stationarity import fractional_differentiation_ffd

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def build_feature_matrix(raw_df: pd.DataFrame, frac_d: float = 0.4) -> pd.DataFrame:
    feature_list = []
    close_df = raw_df["Close"]
    volume_df = raw_df["Volume"]
    
    tickers = close_df.columns if isinstance(close_df, pd.DataFrame) else [close_df.name]
    
    for ticker in tickers:
        c = close_df[ticker].dropna() if isinstance(close_df, pd.DataFrame) else close_df.dropna()
        v = volume_df[ticker].loc[c.index] if isinstance(volume_df, pd.DataFrame) else volume_df.loc[c.index]
        
        ret_5 = c.pct_change(5)
        ret_20 = c.pct_change(20)
        rsi_14 = compute_rsi(c, 14)
        macd_diff = compute_macd(c)
        vol_20 = c.pct_change().rolling(20).std()
        norm_volume = v / (v.rolling(20).mean() + 1e-9)
        frac_diff_price = fractional_differentiation_ffd(c, d=frac_d)
        
        stock_features = pd.DataFrame({
            "ticker": ticker,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "rsi_14": rsi_14,
            "macd_diff": macd_diff,
            "vol_20": vol_20,
            "norm_volume": norm_volume,
            "frac_diff_close": frac_diff_price
        }, index=c.index)
        feature_list.append(stock_features)
        
    full_df = pd.concat(feature_list)
    full_df = full_df.reset_index().rename(columns={"Date": "Date", "index": "Date"})

    # Cross-Sectional Ranking / Standardization per Day
    feature_cols = ["ret_5", "ret_20", "rsi_14", "macd_diff", "vol_20", "norm_volume", "frac_diff_close"]
    for col in feature_cols:
        # Rank features across all stocks on that day from 0.0 to 1.0
        full_df[f"{col}_rank"] = full_df.groupby("Date")[col].rank(pct=True)

    return full_df.sort_values(by=["Date", "ticker"])