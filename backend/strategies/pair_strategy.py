import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime

from backend.strategies.kalman_filter import KalmanFilterEngine
from backend.strategies.cointegration import CointegrationEngine
from backend.strategies.regime_detector import RegimeDetector
from backend.strategies.signal_engine import ZScoreSignalEngine
from backend.risk.risk_engine import RiskEngine
from backend.data.data_handler import DataHandler
from config.settings import Config

logger = logging.getLogger(__name__)


class PairTradingStrategy:
    """
    Master orchestrator for the pair trading system.

    Coordinates:
        - Data fetching
        - Cointegration validation
        - Kalman filter (dynamic hedge ratio)
        - HMM regime detection
        - Z-score signal generation
        - Risk management
    """

    def __init__(
        self,
        symbol_x: str = None,
        symbol_y: str = None,
        capital_ghs: float = None,
    ):
        self.symbol_x = symbol_x or Config.PRIMARY_PAIR[0]
        self.symbol_y = symbol_y or Config.PRIMARY_PAIR[1]
        self.capital_ghs = capital_ghs or Config.INITIAL_CAPITAL_GHS

        self.data_handler = DataHandler()
        self.kalman = KalmanFilterEngine()
        self.cointegration = CointegrationEngine()
        self.regime_detector = RegimeDetector()
        self.signal_engine = ZScoreSignalEngine()
        self.risk_engine = RiskEngine(initial_capital=self.capital_ghs)

        self._series_x: Optional[pd.Series] = None
        self._series_y: Optional[pd.Series] = None
        self._kalman_results: Optional[pd.DataFrame] = None
        self._regime_labels: Optional[np.ndarray] = None
        self._pair_valid = False
        self._pair_metrics: dict = {}
        self._initialized = False

    def initialize(self, period: str = "5y") -> Dict:
        """
        Load data, validate pair, train HMM, run Kalman filter.
        Must be called before any signal generation.
        """
        logger.info(f"Initialising strategy for {self.symbol_x}/{self.symbol_y}")

        self._series_x, self._series_y = self.data_handler.fetch_pair_data(
            self.symbol_x, self.symbol_y, period=period
        )
        self.data_handler.get_ghs_rates()

        # Run Kalman filter
        self._kalman_results = self.kalman.run_series(self._series_x, self._series_y)
        spread_series = self._kalman_results["spread"]

        # Validate pair
        self._pair_valid, self._pair_metrics = self.cointegration.is_pair_valid(
            self._series_x, self._series_y, spread_series
        )

        if not self._pair_valid:
            logger.warning(f"Pair {self.symbol_x}/{self.symbol_y} failed validation")

        # Train HMM on primary series
        self.regime_detector.fit(self._series_x)

        # Predict regimes
        _, self._regime_labels = self.regime_detector.predict_regime(self._series_x)

        self._initialized = True
        logger.info(f"Strategy initialized. Pair valid: {self._pair_valid}")
        logger.info(f"Pair metrics: {self._pair_metrics}")

        return {
            "initialized": True,
            "pair_valid": self._pair_valid,
            "pair_metrics": self._pair_metrics,
            "data_points": len(self._series_x),
        }

    def get_latest_signal(self, current_position: str = "NONE") -> Dict:
        """
        Get the current trading signal based on latest data.
        """
        if not self._initialized:
            return {"error": "Strategy not initialized"}

        # Get current regime
        recent_window = min(200, len(self._series_x))
        recent_x = self._series_x.iloc[-recent_window:]
        regime = self.regime_detector.get_current_regime(recent_x)

        # Kalman filter latest values
        latest_kalman = self._kalman_results.iloc[-1]
        hedge_ratio = float(latest_kalman["hedge_ratio"])
        spread_series = self._kalman_results["spread"]

        # Z-score signal
        signal, zscore = self.signal_engine.get_current_signal(
            spread_series, current_position=current_position
        )

        # Block trading in trending regime
        if regime == RegimeDetector.TRENDING and signal in ("LONG", "SHORT"):
            logger.info(f"Signal {signal} blocked by HMM: TRENDING regime")
            signal = "NONE"

        can_trade, reason = self.risk_engine.can_open_trade(self.capital_ghs)
        if not can_trade and signal in ("LONG", "SHORT"):
            signal = "NONE"
            logger.warning(f"Trade blocked by risk engine: {reason}")

        spread_std = float(spread_series.rolling(Config.LOOKBACK_WINDOW).std().iloc[-1])

        return {
            "symbol_x": self.symbol_x,
            "symbol_y": self.symbol_y,
            "signal": signal,
            "zscore": round(zscore, 4) if not np.isnan(zscore) else None,
            "hedge_ratio": round(hedge_ratio, 6),
            "regime": regime,
            "spread": round(float(spread_series.iloc[-1]), 6),
            "spread_std": round(spread_std, 6),
            "pair_valid": self._pair_valid,
            "can_trade": can_trade,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_spread_series(self) -> pd.Series:
        if self._kalman_results is not None:
            return self._kalman_results["spread"]
        return pd.Series(dtype=float)

    def get_hedge_ratio_series(self) -> pd.Series:
        if self._kalman_results is not None:
            return self._kalman_results["hedge_ratio"]
        return pd.Series(dtype=float)

    def get_pair_metrics(self) -> dict:
        return self._pair_metrics

    def is_ready(self) -> bool:
        return self._initialized and self._pair_valid
