from pathlib import Path
import pandas as pd
import yfinance as yf

def fetch_market_data(tickers: list, start_date: str, end_date: str, interval: str = "1d", raw_dir: str = "data/raw") -> pd.DataFrame:
    """Fetches multi-ticker OHLCV data from Yahoo Finance and caches it locally using pickle."""
    folder = Path(raw_dir)
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / "market_data.pkl"
    
    print(f"Downloading historical market data for {len(tickers)} assets...")
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    
    if df.empty:
        raise ValueError("Market data download returned an empty DataFrame. Check ticker symbols or date range.")

    df.to_pickle(cache_path)
    return df

def load_cached_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Loads cached raw market data."""
    cache_path = Path(raw_dir) / "market_data.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache found at {cache_path}")
    return pd.read_pickle(cache_path)