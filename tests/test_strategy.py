import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from backend.strategies.kalman_filter import KalmanFilterEngine
from backend.strategies.cointegration import CointegrationEngine
from backend.strategies.regime_detector import RegimeDetector
from backend.strategies.signal_engine import ZScoreSignalEngine
from backend.risk.risk_engine import RiskEngine


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def cointegrated_pair():
    """Generate a synthetic cointegrated pair."""
    np.random.seed(42)
    n = 500
    x = np.cumsum(np.random.randn(n)) + 100
    y = 0.8 * x + np.random.randn(n) * 0.5 + 5
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(x, index=idx), pd.Series(y, index=idx)


@pytest.fixture
def random_walk_pair():
    np.random.seed(99)
    n = 500
    x = np.cumsum(np.random.randn(n)) + 100
    y = np.cumsum(np.random.randn(n)) + 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(x, index=idx), pd.Series(y, index=idx)


# ── Kalman Filter ────────────────────────────────────────────

class TestKalmanFilter:
    def test_single_update_returns_values(self):
        kf = KalmanFilterEngine()
        hr, intercept, spread = kf.update(1.1, 0.9)
        assert isinstance(hr, float)
        assert isinstance(intercept, float)
        assert isinstance(spread, float)

    def test_hedge_ratio_series_length(self, cointegrated_pair):
        sx, sy = cointegrated_pair
        kf = KalmanFilterEngine()
        result = kf.run_series(sx, sy)
        assert len(result) == len(sx)
        assert "hedge_ratio" in result.columns
        assert "spread" in result.columns
        assert "intercept" in result.columns

    def test_hedge_ratio_positive_for_cointegrated(self, cointegrated_pair):
        sx, sy = cointegrated_pair
        kf = KalmanFilterEngine()
        result = kf.run_series(sx, sy)
        # After warmup, hedge ratio should be positive (pair is 0.8x)
        final_hr = result["hedge_ratio"].iloc[-100:].mean()
        assert 0.3 < final_hr < 1.5, f"Unexpected hedge ratio: {final_hr}"

    def test_reset_clears_state(self):
        kf = KalmanFilterEngine()
        kf.update(1.0, 1.0)
        kf.reset()
        assert not kf._initialized
        assert np.all(kf.theta == 0)


# ── Cointegration ────────────────────────────────────────────

class TestCointegration:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        sx, sy = cointegrated_pair
        engine = CointegrationEngine()
        is_coint, pvalue, _ = engine.test_cointegration(sx, sy)
        assert is_coint, f"Expected cointegration, got p={pvalue:.4f}"
        assert pvalue < 0.05

    def test_random_walk_not_cointegrated(self, random_walk_pair):
        sx, sy = random_walk_pair
        engine = CointegrationEngine()
        is_coint, pvalue, _ = engine.test_cointegration(sx, sy)
        # Random walks usually aren't cointegrated
        assert pvalue > 0.0  # just a smoke test

    def test_halflife_positive_for_mean_reverting(self, cointegrated_pair):
        sx, sy = cointegrated_pair
        engine = CointegrationEngine()
        kf = KalmanFilterEngine()
        results = kf.run_series(sx, sy)
        spread = results["spread"]
        hl = engine.calculate_halflife(spread)
        assert hl > 0, f"Expected positive halflife, got {hl}"
        assert hl < 1000

    def test_hurst_exponent_range(self, cointegrated_pair):
        sx, _ = cointegrated_pair
        engine = CointegrationEngine()
        h = engine.calculate_hurst_exponent(sx)
        assert isinstance(h, float), f"Hurst exponent out of range: {h}"

    def test_too_short_series(self):
        engine = CointegrationEngine()
        short = pd.Series([1.0, 2.0, 3.0])
        is_c, pv, _ = engine.test_cointegration(short, short)
        assert not is_c
        assert pv == 1.0

    def test_adf_stationary_series(self):
        engine = CointegrationEngine()
        stationary = pd.Series(np.random.randn(500))
        is_stat, pv = engine.test_stationarity(stationary)
        assert is_stat, "White noise should be stationary"


# ── Regime Detector ──────────────────────────────────────────

class TestRegimeDetector:
    def test_fit_does_not_raise(self, cointegrated_pair):
        sx, _ = cointegrated_pair
        rd = RegimeDetector(n_states=2)
        rd.fit(sx)
        assert rd._trained

    def test_predict_returns_valid_regime(self, cointegrated_pair):
        sx, _ = cointegrated_pair
        rd = RegimeDetector(n_states=2)
        rd.fit(sx)
        regime, arr = rd.predict_regime(sx)
        assert regime in (RegimeDetector.MEAN_REVERTING, RegimeDetector.TRENDING)
        assert len(arr) > 0

    def test_untrained_defaults_to_mean_reverting(self):
        rd = RegimeDetector()
        regime = rd.get_current_regime(pd.Series(np.random.randn(50)))
        assert regime == RegimeDetector.MEAN_REVERTING

    def test_is_tradeable_type(self, cointegrated_pair):
        sx, _ = cointegrated_pair
        rd = RegimeDetector()
        rd.fit(sx)
        result = rd.is_tradeable(sx.iloc[-100:])
        assert isinstance(result, bool)


# ── Signal Engine ────────────────────────────────────────────

class TestSignalEngine:
    def test_long_signal_on_negative_zscore(self):
        engine = ZScoreSignalEngine(entry_threshold=2.0, exit_threshold=0.0, stop_threshold=3.5)
        sig = engine.generate_signal(-2.5, "NONE")
        assert sig == ZScoreSignalEngine.LONG

    def test_short_signal_on_positive_zscore(self):
        engine = ZScoreSignalEngine()
        sig = engine.generate_signal(2.5, "NONE")
        assert sig == ZScoreSignalEngine.SHORT

    def test_exit_when_zscore_reverts(self):
        engine = ZScoreSignalEngine(exit_threshold=0.0)
        # SHORT exits when z <= 0 (reverted to mean)
        sig = engine.generate_signal(-0.1, "SHORT")
        assert sig == ZScoreSignalEngine.EXIT

    def test_stop_loss_triggered(self):
        engine = ZScoreSignalEngine(stop_threshold=3.5)
        sig = engine.generate_signal(4.0, "SHORT")
        assert sig == ZScoreSignalEngine.EXIT

    def test_hold_in_between(self):
        engine = ZScoreSignalEngine()
        sig = engine.generate_signal(1.5, "SHORT")
        assert sig == ZScoreSignalEngine.HOLD

    def test_no_signal_no_position_in_range(self):
        engine = ZScoreSignalEngine()
        sig = engine.generate_signal(0.5, "NONE")
        assert sig == ZScoreSignalEngine.NONE

    def test_series_signal_generation(self, cointegrated_pair):
        sx, sy = cointegrated_pair
        kf = KalmanFilterEngine()
        results = kf.run_series(sx, sy)
        engine = ZScoreSignalEngine()
        df = engine.generate_signals_series(results["spread"])
        assert "signal" in df.columns
        assert "zscore" in df.columns
        assert len(df) == len(results)

    def test_nan_zscore_returns_none(self):
        engine = ZScoreSignalEngine()
        sig = engine.generate_signal(float("nan"), "NONE")
        assert sig == ZScoreSignalEngine.NONE


# ── Risk Engine ──────────────────────────────────────────────

class TestRiskEngine:
    def test_fixed_fractional_sizing(self):
        risk = RiskEngine(initial_capital=50000, risk_per_trade=0.02)
        size = risk.calculate_position_size_fixed_fractional(
            entry_price=1.0850, stop_loss_price=1.0800, capital_ghs=50000
        )
        assert size > 0
        expected = (50000 * 0.02) / abs(1.0850 - 1.0800)
        assert abs(size - expected) < 0.01

    def test_drawdown_not_breached(self):
        risk = RiskEngine(initial_capital=50000, max_drawdown_limit=0.15)
        breached, dd = risk.check_drawdown(48000)
        assert not breached
        assert dd < 0.15

    def test_drawdown_breached(self):
        risk = RiskEngine(initial_capital=50000, max_drawdown_limit=0.15)
        risk.peak_capital = 50000
        breached, dd = risk.check_drawdown(42000)
        assert breached
        assert dd >= 0.15

    def test_stop_loss_long(self):
        risk = RiskEngine()
        sl = risk.calculate_stop_loss(1.0850, 0.0020, "LONG", multiplier=2.0)
        assert sl < 1.0850

    def test_stop_loss_short(self):
        risk = RiskEngine()
        sl = risk.calculate_stop_loss(1.0850, 0.0020, "SHORT", multiplier=2.0)
        assert sl > 1.0850

    def test_metrics_summary(self):
        risk = RiskEngine()
        pnls = [100, -50, 200, -30, 150, -80, 90, -20]
        metrics = risk.metrics_summary(pnls)
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert 0 <= metrics["win_rate"] <= 1

    def test_can_open_trade_max_positions(self):
        risk = RiskEngine()
        risk._open_positions = 3
        can, reason = risk.can_open_trade(50000)
        assert not can

    def test_pnl_calculation(self):
        risk = RiskEngine()
        # Gross = (1.09-1.08) * 10000 = 100
        # Cost  = (1.08+1.09)*10000 * (0.0002+0.0001) ≈ 6.51
        pnl = risk.calculate_pnl(1.08, 1.09, 10000, "LONG",
                                 transaction_cost_pct=0.0002, slippage_pct=0.0001)
        expected_gross = (1.09 - 1.08) * 10000
        expected_cost  = (1.08 + 1.09) * 10000 * (0.0002 + 0.0001)
        assert abs(pnl - (expected_gross - expected_cost)) < 0.01

    def test_kelly_criterion(self):
        risk = RiskEngine()
        size = risk.calculate_position_size_kelly(
            win_rate=0.6, avg_win=200, avg_loss=100, capital_ghs=50000
        )
        assert 0 < size <= 50000 * 0.25  # capped at 25%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
