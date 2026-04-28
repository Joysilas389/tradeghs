import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config.settings import Config

logger = logging.getLogger(__name__)


class CointegrationEngine:
    """
    Tests for cointegration between two price series using
    Engle-Granger two-step method with rolling validation.
    """

    def __init__(self, pvalue_threshold: float = 0.05):
        self.pvalue_threshold = pvalue_threshold

    def test_cointegration(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
    ) -> Tuple[bool, float, float]:
        """
        Run Engle-Granger cointegration test.

        Returns:
            (is_cointegrated, pvalue, test_statistic)
        """
        try:
            if len(series_x) < 30 or len(series_y) < 30:
                return False, 1.0, 0.0

            score, pvalue, _ = coint(series_x.values, series_y.values)
            is_cointegrated = pvalue < self.pvalue_threshold
            logger.debug(f"Cointegration test: p={pvalue:.4f} cointegrated={is_cointegrated}")
            return is_cointegrated, float(pvalue), float(score)
        except Exception as exc:
            logger.error(f"Cointegration test error: {exc}")
            return False, 1.0, 0.0

    def test_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        ADF test for stationarity of the spread.

        Returns:
            (is_stationary, pvalue)
        """
        try:
            result = adfuller(series.dropna().values, autolag="AIC")
            pvalue = float(result[1])
            is_stationary = pvalue < self.pvalue_threshold
            return is_stationary, pvalue
        except Exception as exc:
            logger.error(f"ADF test error: {exc}")
            return False, 1.0

    def calculate_halflife(self, spread: pd.Series) -> float:
        """
        Estimate half-life of mean reversion using OLS on spread lag.
        Smaller = faster reversion = better for trading.
        """
        try:
            spread_lag = spread.shift(1).dropna()
            spread_delta = spread.diff().dropna()
            spread_lag, spread_delta = spread_lag.align(spread_delta, join="inner")

            model = OLS(spread_delta.values, add_constant(spread_lag.values)).fit()
            lambda_coef = model.params[1]

            if lambda_coef >= 0:
                return float("inf")

            halflife = -np.log(2) / lambda_coef
            return float(halflife)
        except Exception as exc:
            logger.error(f"Halflife calculation error: {exc}")
            return float("inf")

    def calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Hurst exponent: H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
        """
        try:
            lags = range(2, min(100, len(series) // 2))
            tau = [np.std(series.diff(lag).dropna()) for lag in lags]
            valid = [(lag, t) for lag, t in zip(lags, tau) if t > 0]
            if len(valid) < 5:
                return 0.5
            log_lags = np.log([v[0] for v in valid])
            log_tau = np.log([v[1] for v in valid])
            hurst = float(np.polyfit(log_lags, log_tau, 1)[0])
            return hurst
        except Exception as exc:
            logger.error(f"Hurst exponent error: {exc}")
            return 0.5

    def rolling_cointegration(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
        window: int = None,
    ) -> pd.Series:
        """
        Rolling cointegration test — detects structural breaks.
        Returns series of p-values over time.
        """
        window = window or Config.ROLLING_COINT_WINDOW
        pvalues = pd.Series(index=series_x.index, dtype=float)

        for i in range(window, len(series_x)):
            sx = series_x.iloc[i - window:i]
            sy = series_y.iloc[i - window:i]
            try:
                _, pv, _ = coint(sx.values, sy.values)
                pvalues.iloc[i] = pv
            except Exception:
                pvalues.iloc[i] = 1.0

        return pvalues

    def is_pair_valid(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
        spread: pd.Series,
    ) -> Tuple[bool, dict]:
        """
        Full validation of a trading pair.
        Returns (is_valid, metrics_dict)
        """
        is_coint, pvalue, score = self.test_cointegration(series_x, series_y)
        is_stationary, adf_pvalue = self.test_stationarity(spread)
        halflife = self.calculate_halflife(spread)
        hurst = self.calculate_hurst_exponent(spread)

        is_valid = (
            is_coint
            and is_stationary
            and halflife < Config.HALFLIFE_MAX_DAYS
            and hurst < 0.5
        )

        metrics = {
            "is_cointegrated": is_coint,
            "coint_pvalue": pvalue,
            "coint_score": score,
            "is_stationary": is_stationary,
            "adf_pvalue": adf_pvalue,
            "halflife_days": halflife,
            "hurst_exponent": hurst,
            "is_valid": is_valid,
        }

        logger.info(f"Pair validation: {metrics}")
        return is_valid, metrics
