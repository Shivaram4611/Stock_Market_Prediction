import datetime
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from src.features import build_feature_matrix
from src.targets import compute_excess_returns

def generate_live_delivery_picks():
    # 1. Load Configurations
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg["data"]["tickers"]
    start_date = cfg["data"]["start_date"]
    
    # Always fetch up to today's date for live market analysis
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    print(f"Fetching clean historical and latest daily data up to {today_str}...")
    raw_df = yf.download(
        tickers=tickers,
        start=start_date,
        end=today_str,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if raw_df.empty:
        raise ValueError("Failed to retrieve market data. Check your network connection.")

    close_prices = raw_df["Close"]

    # 2. Build Targets on Historical Data
    print("Constructing excess return ranking targets...")
    horizon = cfg["model"]["horizon"]
    excess_ret = compute_excess_returns(close_prices, horizon=horizon)

    # Binary Delivery Target: 1 if in top 35% outperformers, 0 otherwise
    def get_binary_target(row):
        valid = row.dropna()
        if len(valid) == 0:
            return row
        threshold = valid.quantile(1.0 - cfg["model"]["long_threshold"] + 0.1)
        return (row >= threshold).astype(float)

    binary_targets = excess_ret.apply(get_binary_target, axis=1)
    target_flat = binary_targets.stack().reset_index()
    target_flat.columns = ["Date", "ticker", "target"]

    # 3. Extract Features
    print("Computing technical, trend, and rank features...")
    feature_matrix = build_feature_matrix(raw_df, frac_d=cfg["model"]["frac_d"])

    # Separate training data (dates with known forward returns) from latest live data
    dataset = pd.merge(feature_matrix, target_flat, on=["Date", "ticker"], how="inner")
    train_data = dataset.dropna(subset=["target"]).copy()

    feature_cols = [c for c in train_data.columns if c not in ["Date", "ticker", "target"]]
    X_train = train_data[feature_cols]
    y_train = train_data["target"].astype(int)

    # 4. Train Full-History Model
    print(f"Training LightGBM engine on {len(X_train)} historical records...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        force_col_wise=True
    )
    model.fit(X_train, y_train)

    # 5. Extract the Most Recent Trading Day for Live Inference
    latest_date = feature_matrix["Date"].max()
    latest_features = feature_matrix[feature_matrix["Date"] == latest_date].copy()

    X_live = latest_features[feature_cols]
    probabilities = model.predict_proba(X_live)[:, 1]  # Probability of being a top outperformer

    latest_features["Outperform_Probability"] = probabilities
    
    # 6. Format Final Recommendations Table
    results = latest_features[["ticker", "ret_20", "rsi_14", "Outperform_Probability"]].copy()
    results.rename(columns={
        "ticker": "Stock",
        "ret_20": "20D Return",
        "rsi_14": "RSI (14)",
        "Outperform_Probability": "Confidence Score"
    }, inplace=True)

    results["20D Return"] = results["20D Return"].map(lambda x: f"{x * 100:.2f}%")
    results["RSI (14)"] = results["RSI (14)"].map(lambda x: f"{x:.1f}")
    results = results.sort_values(by="Confidence Score", ascending=False).reset_index(drop=True)

    def delivery_action(score):
        if score >= 0.60:
            return "STRONG BUY (Delivery)"
        elif score >= 0.50:
            return "ACCUMULATE / WATCH"
        else:
            return "AVOID / HOLD"

    results["Recommendation"] = results["Confidence Score"].map(delivery_action)
    results["Confidence Score"] = results["Confidence Score"].map(lambda x: f"{x * 100:.2f}%")

    print(f"\n================ LIVE DELIVERY SCANNER ({latest_date.strftime('%Y-%m-%d')}) ================")
    print(results.to_string(index=False))
    print("=========================================================================\n")
    print("Key Delivery Rules:")
    print("• High-probability candidates (>60%) indicate relative strength for multi-day swing holding.")
    print("• Avoid entering candidates with RSI > 75 to minimize chasing overbought peaks.")

if __name__ == "__main__":
    generate_live_delivery_picks()