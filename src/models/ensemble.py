import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class MetaLearnerStacker:
    """Combines out-of-fold probability outputs from base models using a Ridge/Logistic meta-learner."""
    def __init__(self):
        self.meta_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)

    def fit(self, base_probabilities: np.ndarray, y_true: np.ndarray):
        self.meta_model.fit(base_probabilities, y_true)

    def predict_proba(self, base_probabilities: np.ndarray) -> np.ndarray:
        return self.meta_model.predict_proba(base_probabilities)