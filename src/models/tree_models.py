import warnings
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd

# Suppress lightgbm deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def train_lightgbm_ranker(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Trains a LightGBM multi-class classifier on mapped labels (-1, 0, 1 -> 0, 1, 2)."""
    label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
    y_tr = y_train.map(label_map)
    y_va = y_val.map(label_map)

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        force_col_wise=True
    )

    clf.fit(
        X_train, y_tr,
        eval_set=[(X_val, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
    )

    preds = clf.predict(X_val)
    rev_map = {0: -1.0, 1: 0.0, 2: 1.0}
    preds_mapped = pd.Series(preds, index=y_val.index).map(rev_map)

    # Calculate Long precision (when model predicts Long, was it actually a top outperformer?)
    long_prec = precision_score(y_va, preds, labels=[2], average="micro", zero_division=0)

    return clf, preds_mapped, long_prec