import numpy as np
import pandas as pd
import logging
from typing import Tuple
from config.settings import Config

logger = logging.getLogger(__name__)


class KalmanFilterEngine:
    """
    Implements a Kalman Filter to dynamically estimate the hedge ratio
    between two cointegrated assets. Replaces static OLS regression.

    State vector: [hedge_ratio, intercept]
    Observation: price_y = hedge_ratio * price_x + intercept + noise
    """

    def __init__(
        self,
        delta: float = None,
        observation_noise: float = None,
    ):
        self.delta = delta or Config.KALMAN_DELTA
        self.Ve = observation_noise or Config.KALMAN_VE

        # State transition covariance (how fast hedge ratio can drift)
        self.Vw = self.delta / (1 - self.delta) * np.eye(2)

        # State estimate: [beta, alpha]
        self.theta = np.zeros(2)

        # State covariance matrix
        self.R = np.zeros((2, 2))

        self._initialized = False

    def update(self, price_x: float, price_y: float) -> Tuple[float, float, float]:
        """
        Process one observation and return updated hedge ratio, intercept, spread.

        Returns:
            hedge_ratio, intercept, spread (prediction error)
        """
        F = np.array([[price_x, 1.0]])  # observation matrix

        if not self._initialized:
            self.theta = np.array([1.0, 0.0])
            self.R = np.eye(2) * 1.0
            self._initialized = True

        # Predict step
        R_pred = self.R + self.Vw

        # Innovation (spread)
        y_hat = F @ self.theta
        e = price_y - y_hat[0]

        # Innovation covariance
        S = F @ R_pred @ F.T + self.Ve

        # Kalman gain
        K = R_pred @ F.T / S[0, 0]

        # Update state
        self.theta = self.theta + K.flatten() * e
        self.R = R_pred - K @ F @ R_pred

        hedge_ratio = float(self.theta[0])
        intercept = float(self.theta[1])
        spread = float(e)

        return hedge_ratio, intercept, spread

    def run_series(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
    ) -> pd.DataFrame:
        """
        Run Kalman filter over full historical series.
        Returns DataFrame with hedge_ratio, intercept, spread per row.
        """
        results = []
        self._initialized = False
        self.theta = np.zeros(2)
        self.R = np.zeros((2, 2))

        for px, py in zip(series_x.values, series_y.values):
            hr, intercept, spread = self.update(float(px), float(py))
            results.append({
                "hedge_ratio": hr,
                "intercept": intercept,
                "spread": spread,
            })

        df = pd.DataFrame(results, index=series_x.index)
        return df

    def reset(self):
        self._initialized = False
        self.theta = np.zeros(2)
        self.R = np.zeros((2, 2))
