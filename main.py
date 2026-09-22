import yaml
import pandas as pd
import numpy as np
from src.data_loader import fetch_market_data
from src.targets import compute_excess_returns, generate_quintile_targets
from src.features import build_feature_matrix
from src.validation import PurgedGroupTimeSeriesSplit
from src.models.tree_models import train_lightgbm_ranker
from sklearn.metrics import accuracy_score

def run_pipeline():
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    raw_df = fetch_market_data(
        tickers=cfg["data"]["tickers"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"]
    )
    close_prices = raw_df["Close"]

    print("Computing cross-sectional excess targets...")
    excess_ret = compute_excess_returns(close_prices, horizon=cfg["model"]["horizon"])
    target_labels = generate_quintile_targets(
        excess_ret,
        top_q=cfg["model"]["long_threshold"],
        bottom_q=cfg["model"]["short_threshold"]
    )

    target_flat = target_labels.stack().reset_index()
    target_flat.columns = ["Date", "ticker", "target"]

    print("Extracting features with cross-sectional ranking...")
    feature_matrix = build_feature_matrix(raw_df, frac_d=cfg["model"]["frac_d"])

    dataset = pd.merge(feature_matrix, target_flat, on=["Date", "ticker"], how="inner")
    dataset = dataset.dropna().reset_index(drop=True)

    feature_cols = [c for c in dataset.columns if c not in ["Date", "ticker", "target"]]
    X = dataset[feature_cols]
    y = dataset["target"]

    print(f"Dataset shape: {X.shape[0]} samples across {X.shape[1]} features.")
    print(f"Beginning {cfg['model']['n_splits']}-Fold Purged & Embargoed Cross-Validation...")

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=cfg["model"]["n_splits"],
        horizon=cfg["model"]["horizon"],
        embargo_pct=cfg["model"]["embargo_pct"]
    )

    accuracies = []
    long_precisions = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset, date_col="Date"), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model, preds, long_prec = train_lightgbm_ranker(X_train, y_train, X_test, y_test)
        acc = accuracy_score(y_test, preds)
        
        accuracies.append(acc)
        long_precisions.append(long_prec)
        print(f"Fold {fold} | Out-of-Sample Accuracy: {acc * 100:.2f}% | Long Pick Precision: {long_prec * 100:.2f}%")

    print(f"\nMean Out-of-Sample Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Mean Long Selection Precision: {np.mean(long_precisions) * 100:.2f}%")

if __name__ == "__main__":
    run_pipeline()