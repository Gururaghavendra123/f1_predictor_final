"""
F1 predictor — ONE model.

A single XGBRegressor predicts each driver's expected finishing position from the
features in f1_features.py. Ranking, win and podium probabilities are all DERIVED
from that one score, so the outputs can never contradict each other (no "85% win
but predicted P6"). Drivers are ranked 1..N by ascending expected position, giving
unique, tie-free positions.

  expected_position  -> XGBRegressor output (continuous)
  predicted_position -> 1..N rank after sorting by expected_position
  win_probability    -> softmax over -expected_position (sums to ~1 across field)
  podium_probability -> logistic around an expected position of ~3.5

Training/validation orchestration lives in train.py. This module owns the model:
fit, rank a race, persist, load.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from f1_features import FEATURE_COLUMNS

MODEL_DIR = 'models'
WIN_TEMP = 2.0      # softmax temperature for win probability (higher = flatter)
PODIUM_K = 0.7      # logistic steepness for podium probability


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class F1Predictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = list(FEATURE_COLUMNS)
        self.snapshot = None
        self.trained = False

    # ---- training ----
    def fit(self, table, sample_weight=None):
        X = table[self.feature_columns].astype(float).values
        y = table['target_position'].astype(float).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.trained = True
        return self

    # ---- prediction ----
    def _predict_scores(self, feat_df):
        X = feat_df[self.feature_columns].astype(float).values
        return self.model.predict(self.scaler.transform(X))

    @staticmethod
    def _scores_to_probs(scores):
        s = np.asarray(scores, dtype=float)
        win = _softmax(-s / WIN_TEMP)
        podium = 1.0 / (1.0 + np.exp(-(3.5 - s) * PODIUM_K))
        return win, podium

    def rank_race(self, feature_rows):
        """Rank a field for one race.

        feature_rows: list of dicts, each with 'driver' + FEATURE_COLUMNS.
        Returns results sorted best-first with unique predicted_position 1..N.
        """
        df = pd.DataFrame(feature_rows)
        scores = self._predict_scores(df)
        win, podium = self._scores_to_probs(scores)

        order = np.argsort(scores)  # ascending expected position -> winner first
        results = []
        for rank, idx in enumerate(order, start=1):
            results.append({
                'driver': df.iloc[idx]['driver'],
                'predicted_position': rank,
                'expected_position': round(float(scores[idx]), 2),
                'win_probability': float(win[idx]),
                'podium_probability': float(podium[idx]),
                'confidence': float(podium[idx]),
            })
        return results

    # ---- persistence ----
    def save(self, snapshot=None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, os.path.join(MODEL_DIR, 'position_model.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
        if snapshot is not None:
            self.snapshot = snapshot
            with open(os.path.join(MODEL_DIR, 'driver_stats.json'), 'w') as f:
                json.dump(snapshot, f)
        print(f"Saved model + scaler + snapshot to {MODEL_DIR}/")

    def load(self):
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, 'position_model.pkl'))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            with open(os.path.join(MODEL_DIR, 'feature_columns.json')) as f:
                self.feature_columns = json.load(f)
            stats_path = os.path.join(MODEL_DIR, 'driver_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    self.snapshot = json.load(f)
            self.trained = True
            return True
        except Exception as e:
            print(f"Model load failed: {e}")
            return False
