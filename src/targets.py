import numpy as np
import pandas as pd

def compute_excess_returns(price_df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Calculates forward excess returns relative to the cross-sectional median."""
    forward_returns = price_df.pct_change(horizon).shift(-horizon)
    universe_median = forward_returns.median(axis=1)
    return forward_returns.sub(universe_median, axis=0)

def generate_quintile_targets(excess_returns: pd.DataFrame, top_q: float = 0.80, bottom_q: float = 0.20) -> pd.DataFrame:
    """Labels cross-sectional outperformance: 1.0 (Long), -1.0 (Short), 0.0 (Neutral)."""
    def label_row(row):
        valid = row.dropna()
        if len(valid) == 0:
            return row
        top = valid.quantile(top_q)
        bottom = valid.quantile(bottom_q)
        
        labels = pd.Series(0.0, index=row.index)
        labels[row >= top] = 1.0
        labels[row <= bottom] = -1.0
        labels[row.isna()] = np.nan
        return labels

    return excess_returns.apply(label_row, axis=1)