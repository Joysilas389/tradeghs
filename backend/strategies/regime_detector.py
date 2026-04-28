import numpy as np
import pandas as pd
import logging
from typing import Tuple
from scipy import stats
from config.settings import Config

logger = logging.getLogger(__name__)


class RegimeDetector:
    MEAN_REVERTING = "MEAN_REVERTING"
    TRENDING = "TRENDING"

    def __init__(self, n_states: int = None):
        self.n_states = n_states or Config.HMM_STATES
        self._trained = False
        self._vol_threshold = None

    def _get_features(self, price_series: pd.Series) -> np.ndarray:
        returns = price_series.pct_change().dropna()
        rolling_vol = returns.rolling(5).std().fillna(returns.std())
        return returns.values, rolling_vol.values

    def fit(self, price_series: pd.Series) -> "RegimeDetector":
        try:
            if len(price_series) < 60:
                return self
            returns, rolling_vol = self._get_features(price_series)
            # Use rolling volatility median as threshold
            # Low vol = mean reverting, high vol = trending
            self._vol_threshold = float(np.median(rolling_vol))
            self._trained = True
            logger.info(f"RegimeDetector fitted. Vol threshold: {self._vol_threshold:.6f}")
        except Exception as exc:
            logger.error(f"RegimeDetector fit error: {exc}")
        return self

    def predict_regime(self, price_series: pd.Series) -> Tuple[str, np.ndarray]:
        try:
            returns, rolling_vol = self._get_features(price_series)
            if not self._trained or self._vol_threshold is None:
                threshold = float(np.median(rolling_vol))
            else:
                threshold = self._vol_threshold

            regimes = np.where(
                rolling_vol <= threshold,
                self.MEAN_REVERTING,
                self.TRENDING
            )
            current = str(regimes[-1]) if len(regimes) > 0 else self.MEAN_REVERTING
            return current, regimes
        except Exception as exc:
            logger.error(f"Regime prediction error: {exc}")
            return self.MEAN_REVERTING, np.array([self.MEAN_REVERTING])

    def get_current_regime(self, recent_prices: pd.Series) -> str:
        regime, _ = self.predict_regime(recent_prices)
        return regime

    def is_tradeable(self, recent_prices: pd.Series) -> bool:
        regime = self.get_current_regime(recent_prices)
        tradeable = regime == self.MEAN_REVERTING
        if not tradeable:
            logger.info("Regime: TRENDING — trade blocked")
        return tradeable
