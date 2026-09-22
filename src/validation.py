import numpy as np
import pandas as pd

class PurgedGroupTimeSeriesSplit:
    """
    Purged and Embargoed Walk-Forward Cross-Validation.
    Prevents cross-contamination caused by overlapping prediction windows and autocorrelation.
    """
    def __init__(self, n_splits: int = 5, horizon: int = 10, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def split(self, df: pd.DataFrame, date_col: str = "Date"):
        unique_dates = np.sort(df[date_col].unique())
        n_dates = len(unique_dates)
        split_size = n_dates // (self.n_splits + 1)
        embargo_size = int(n_dates * self.embargo_pct)

        for i in range(1, self.n_splits + 1):
            train_end_idx = i * split_size
            test_start_idx = train_end_idx + self.horizon  # Purge forward overlapping window
            test_end_idx = test_start_idx + split_size

            if test_end_idx > n_dates:
                break

            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            train_idx = df[df[date_col].isin(train_dates)].index.values
            test_idx = df[df[date_col].isin(test_dates)].index.values

            yield train_idx, test_idx