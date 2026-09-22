# 📈 Quantitative Stock Market Intelligence & AI Predictor

An end-to-end quantitative stock market analysis and machine learning platform built with **Python**, **LightGBM**, and **Streamlit**. 

The system shifts away from noisy single-point price forecasting toward **cross-sectional excess return ranking** ($R_{i,t+k} - \text{Median}(R_{\text{universe}})$), identifying institutional momentum, stationary technical indicators, and real-time gainers and losers across liquid NSE large-cap equities.

---

## 🚀 Key Features

* **Live Market Top 10 Gainers & Losers:** Real-time intra-day percentage mover tracking during active market hours (9:15 AM – 3:30 PM IST), automatically locking onto official end-of-day (EOD) closing data post-market.
* **Next-Day AI Predictive Ranking:** Machine learning engine trained on multi-period excess returns, volume surge multiples, and intraday range dynamics to identify high-probability gainers and underperformers for the next trading session.
* **Leak-Free Machine Learning Pipeline:** Incorporates **Purged and Embargoed Walk-Forward Cross-Validation** to eliminate lookahead bias and serial correlation leakage.
* **Stationarity Preservation:** Employs **Fractional Differentiation (FFD)** ($d \approx 0.4$) to preserve long-term price memory patterns while removing non-stationary trend drifts.
* **Interactive Visualization:** Interactive Plotly candlestick charts with 20 EMA and 50 EMA overlays and multi-timeframe volume profiling.

---

## 📂 Project Architecture

```text
Stock_prediction_model/
│
├── config/
│   └── settings.yaml             # Universe tickers, prediction horizons, model parameters
│
├── data/
│   ├── raw/                      # Downloaded raw market feeds (.gitignore)
│   └── processed/                # Normalized feature matrices (.gitignore)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Automated Yahoo Finance data ingestion & serialization
│   ├── targets.py                # Cross-sectional excess return labeling
│   ├── features.py               # Technical indicators, momentum, and rank standardizers
│   ├── stationarity.py           # Fractional differentiation algorithms
│   ├── validation.py             # Purged & Embargoed Time-Series Cross-Validation
│   └── models/
│       ├── __init__.py
│       ├── tree_models.py        # LightGBM classifier & ranking models
│       └── ensemble.py           # Stacking meta-learner
│
├── app.py                       # Streamlit web dashboard
├── predict_today.py             # CLI live inference & delivery screening script
├── main.py                      # Purged cross-validation training pipeline
├── requirements.txt             # Project library dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation