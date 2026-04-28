import numpy as np
import pandas as pd
import logging
from typing import Tuple
from scipy import stats
from config.settings import Config

logger = logging.getLogger(__name__)


class CointegrationEngine:
    def __init__(self, pvalue_threshold: float = 0.05):
        self.pvalue_threshold = pvalue_threshold

    def _adf_test(self, series: np.ndarray) -> Tuple[float, float]:
        try:
            n = len(series)
            if n < 20:
                return 0.0, 1.0
            y = np.diff(series)
            x = series[:-1] - np.mean(series[:-1])
            if np.sum(x**2) < 1e-10:
                return 0.0, 1.0
            slope, intercept, _, _, _ = stats.linregress(x, y)
            residuals = y - (slope * x + intercept)
            s2 = np.sum(residuals**2) / max(n - 2, 1)
            se_slope = np.sqrt(s2 / np.sum(x**2))
            adf_stat = slope / se_slope if se_slope > 0 else 0.0
            pvalue = float(stats.norm.cdf(adf_stat))
            return float(adf_stat), pvalue
        except Exception as exc:
            logger.error(f"ADF error: {exc}")
            return 0.0, 1.0

    def test_cointegration(self, series_x: pd.Series, series_y: pd.Series) -> Tuple[bool, float, float]:
        try:
            if len(series_x) < 30:
                return False, 1.0, 0.0
            x = series_x.values.astype(float)
            y = series_y.values.astype(float)
            slope, intercept, _, _, _ = stats.linregress(x, y)
            residuals = y - (slope * x + intercept)
            adf_stat, pvalue = self._adf_test(residuals)
            return pvalue < self.pvalue_threshold, pvalue, float(adf_stat)
        except Exception as exc:
            logger.error(f"Cointegration error: {exc}")
            return False, 1.0, 0.0

    def test_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        try:
            adf_stat, pvalue = self._adf_test(series.dropna().values.astype(float))
            return pvalue < self.pvalue_threshold, pvalue
        except Exception as exc:
            logger.error(f"Stationarity error: {exc}")
            return False, 1.0

    def calculate_halflife(self, spread: pd.Series) -> float:
        try:
            s = spread.dropna().values.astype(float)
            if len(s) < 10:
                return float("inf")
            y = np.diff(s)
            x = s[:-1] - np.mean(s[:-1])
            if np.sum(x**2) < 1e-10:
                return float("inf")
            slope, _, _, _, _ = stats.linregress(x, y)
            if slope >= 0:
                return float("inf")
            return float(-np.log(2) / slope)
        except Exception as exc:
            logger.error(f"Halflife error: {exc}")
            return float("inf")

    def calculate_hurst_exponent(self, series: pd.Series) -> float:
        try:
            s = series.dropna().values.astype(float)
            lags = range(2, min(50, len(s) // 2))
            tau = [np.std(np.diff(s, lag)) for lag in lags]
            valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
            if len(valid) < 5:
                return 0.5
            log_lags = np.log([v[0] for v in valid])
            log_tau = np.log([v[1] for v in valid])
            return float(np.polyfit(log_lags, log_tau, 1)[0])
        except Exception as exc:
            logger.error(f"Hurst error: {exc}")
            return 0.5

    def rolling_cointegration(self, series_x: pd.Series, series_y: pd.Series, window: int = None) -> pd.Series:
        window = window or Config.ROLLING_COINT_WINDOW
        pvalues = pd.Series(index=series_x.index, dtype=float)
        for i in range(window, len(series_x)):
            sx = series_x.iloc[i - window:i]
            sy = series_y.iloc[i - window:i]
            _, pv, _ = self.test_cointegration(sx, sy)
            pvalues.iloc[i] = pv
        return pvalues

    def is_pair_valid(self, series_x: pd.Series, series_y: pd.Series, spread: pd.Series) -> Tuple[bool, dict]:
        is_coint, pvalue, score = self.test_cointegration(series_x, series_y)
        is_stationary, adf_pvalue = self.test_stationarity(spread)
        halflife = self.calculate_halflife(spread)
        hurst = self.calculate_hurst_exponent(spread)
        is_valid = is_coint and is_stationary and halflife < Config.HALFLIFE_MAX_DAYS and hurst < 0.5
        metrics = {
            "is_cointegrated": is_coint, "coint_pvalue": pvalue,
            "coint_score": score, "is_stationary": is_stationary,
            "adf_pvalue": adf_pvalue, "halflife_days": halflife,
            "hurst_exponent": hurst, "is_valid": is_valid,
        }
        logger.info(f"Pair validation: {metrics}")
        return is_valid, metrics
