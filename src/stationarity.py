import numpy as np
import pandas as pd

def get_weights_ffd(d: float, thres: float = 1e-4, max_lags: int = 100) -> np.ndarray:
    """Generates memory-preserving weights for Fixed-Width Window Fractional Differentiation."""
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def fractional_differentiation_ffd(series: pd.Series, d: float = 0.4, thres: float = 1e-4) -> pd.Series:
    """Applies Fractional Differentiation (FFD) to ensure stationarity while preserving memory."""
    weights = get_weights_ffd(d, thres=thres)
    width = len(weights) - 1
    
    res = {}
    values = series.values
    index = series.index
    
    for i in range(width, len(series)):
        window = values[i - width : i + 1]
        if np.isnan(window).any():
            continue
        res[index[i]] = np.dot(weights, window)
        
    return pd.Series(res)