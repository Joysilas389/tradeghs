import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from hmmlearn.hmm import GaussianHMM
from config.settings import Config

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Hidden Markov Model to detect market regimes:
        - State 0: Mean-reverting  → ALLOW trading
        - State 1: Trending        → BLOCK trading

    Features used: returns, volatility, spread z-score change.
    """

    MEAN_REVERTING = "MEAN_REVERTING"
    TRENDING = "TRENDING"

    def __init__(self, n_states: int = None):
        self.n_states = n_states or Config.HMM_STATES
        self._model: Optional[GaussianHMM] = None
        self._mean_reverting_state: Optional[int] = None
        self._trained = False

    def _build_features(self, price_series: pd.Series) -> np.ndarray:
        """
        Build feature matrix from price series.
        Features: [return, abs_return (volatility proxy), rolling_vol]
        """
        returns = price_series.pct_change().dropna()
        abs_returns = returns.abs()
        rolling_vol = returns.rolling(5).std().fillna(returns.std())

        features = np.column_stack([
            returns.values,
            abs_returns.values,
            rolling_vol.values,
        ])
        return features

    def fit(self, price_series: pd.Series) -> "RegimeDetector":
        """Train HMM on historical price series."""
        try:
            features = self._build_features(price_series)
            if len(features) < 100:
                logger.warning("Insufficient data for HMM training")
                return self

            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
            )
            self._model.fit(features)
            self._identify_states(features)
            self._trained = True
            logger.info(f"HMM trained. Mean-reverting state: {self._mean_reverting_state}")
        except Exception as exc:
            logger.error(f"HMM training failed: {exc}")
        return self

    def _identify_states(self, features: np.ndarray):
        """
        Identify which HMM state corresponds to mean-reverting vs trending.
        Mean-reverting state = lower volatility (abs_return feature).
        """
        states = self._model.predict(features)
        state_vols = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_vols[s] = float(np.mean(features[mask, 1]))  # mean abs_return

        # Lower volatility state = mean-reverting
        self._mean_reverting_state = min(state_vols, key=state_vols.get)
        logger.debug(f"State volatilities: {state_vols}")

    def predict_regime(self, price_series: pd.Series) -> Tuple[str, np.ndarray]:
        """
        Predict regime for each bar in the series.

        Returns:
            (current_regime_label, array_of_regime_labels)
        """
        if not self._trained or self._model is None:
            return self.MEAN_REVERTING, np.array([self.MEAN_REVERTING])

        try:
            features = self._build_features(price_series)
            states = self._model.predict(features)
            regimes = np.where(
                states == self._mean_reverting_state,
                self.MEAN_REVERTING,
                self.TRENDING,
            )
            current = str(regimes[-1]) if len(regimes) > 0 else self.MEAN_REVERTING
            return current, regimes
        except Exception as exc:
            logger.error(f"Regime prediction failed: {exc}")
            return self.MEAN_REVERTING, np.array([self.MEAN_REVERTING])

    def get_current_regime(self, recent_prices: pd.Series) -> str:
        """Get regime for the most recent observation."""
        regime, _ = self.predict_regime(recent_prices)
        return regime

    def is_tradeable(self, recent_prices: pd.Series) -> bool:
        """Returns True only when market is in mean-reverting regime."""
        regime = self.get_current_regime(recent_prices)
        tradeable = regime == self.MEAN_REVERTING
        if not tradeable:
            logger.info("HMM: TRENDING regime — trade blocked")
        return tradeable
