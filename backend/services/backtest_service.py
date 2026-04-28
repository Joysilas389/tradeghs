import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from backend.strategies.kalman_filter import KalmanFilterEngine
from backend.strategies.cointegration import CointegrationEngine
from backend.strategies.regime_detector import RegimeDetector
from backend.strategies.signal_engine import ZScoreSignalEngine
from backend.risk.risk_engine import RiskEngine
from backend.data.data_handler import DataHandler
from config.settings import Config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Full backtesting engine with:
        - Walk-forward validation
        - In-sample / out-of-sample split
        - Transaction costs + slippage
        - All performance metrics
    """

    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost_pct: float = None,
        slippage_pct: float = None,
    ):
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL_GHS
        self.tc = transaction_cost_pct or Config.TRANSACTION_COST_PCT
        self.slippage = slippage_pct or Config.SLIPPAGE_PCT

    def run(
        self,
        symbol_x: str,
        symbol_y: str,
        period: str = "5y",
        train_ratio: float = 0.7,
    ) -> Dict:
        """
        Full backtest with walk-forward validation.
        """
        logger.info(f"Starting backtest: {symbol_x}/{symbol_y} period={period}")

        handler = DataHandler()
        sx, sy = handler.fetch_pair_data(symbol_x, symbol_y, period=period)
        handler.get_ghs_rates()

        split_idx = int(len(sx) * train_ratio)
        sx_train, sy_train = sx.iloc[:split_idx], sy.iloc[:split_idx]
        sx_test, sy_test = sx.iloc[split_idx:], sy.iloc[split_idx:]

        logger.info(f"Train: {len(sx_train)} bars | Test: {len(sx_test)} bars")

        # Train on in-sample
        kalman_train = KalmanFilterEngine()
        kalman_train.run_series(sx_train, sy_train)  # warm up Kalman state

        regime_detector = RegimeDetector()
        regime_detector.fit(sx_train)

        # Run Kalman on full series (keeps state from training)
        kalman_full = KalmanFilterEngine()
        kalman_results = kalman_full.run_series(sx, sy)

        spread_full = kalman_results["spread"]
        hedge_ratio_full = kalman_results["hedge_ratio"]

        # Generate signals on test set only (no lookahead)
        signal_engine = ZScoreSignalEngine()
        zscore_full = signal_engine.calculate_zscore(spread_full)

        # Get test-period regimes
        _, regime_array = regime_detector.predict_regime(sx)
        regime_series = pd.Series(regime_array, index=sx.index[1:])  # -1 from pct_change

        equity_curve, trade_log = self._simulate_trades(
            sx=sx_test,
            sy=sy_test,
            spread=spread_full.iloc[split_idx:],
            zscore=zscore_full.iloc[split_idx:],
            hedge_ratio=hedge_ratio_full.iloc[split_idx:],
            regime_series=regime_series,
            handler=handler,
        )

        metrics = self._calculate_metrics(equity_curve, trade_log)
        metrics["initial_capital"] = self.initial_capital
        metrics["final_capital"] = equity_curve[-1] if equity_curve else self.initial_capital
        metrics["total_return_pct"] = round(
            (metrics["final_capital"] - self.initial_capital) / self.initial_capital * 100, 2
        )

        # Cointegration stats for info
        coint_engine = CointegrationEngine()
        _, coint_metrics = coint_engine.is_pair_valid(sx_train, sy_train, spread_full.iloc[:split_idx])
        metrics["coint_metrics"] = coint_metrics

        result = {
            "symbol_x": symbol_x,
            "symbol_y": symbol_y,
            "start_date": str(sx_test.index[0].date()),
            "end_date": str(sx_test.index[-1].date()),
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trade_log": trade_log,
        }

        logger.info(f"Backtest complete. Return: {metrics['total_return_pct']:.1f}% | Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        return result

    def _simulate_trades(
        self,
        sx: pd.Series,
        sy: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: pd.Series,
        regime_series: pd.Series,
        handler: DataHandler,
    ) -> Tuple[List[float], List[Dict]]:
        capital = self.initial_capital
        equity_curve = [capital]
        trade_log = []
        position = "NONE"
        entry_data = {}

        signal_engine = ZScoreSignalEngine()

        for i, (idx, z) in enumerate(zscore.items()):
            if pd.isna(z):
                equity_curve.append(capital)
                continue

            # Regime filter
            regime = regime_series.get(idx, RegimeDetector.MEAN_REVERTING)

            signal = signal_engine.generate_signal(z, position)

            # Block new entries in trending regime
            if regime == RegimeDetector.TRENDING and signal in ("LONG", "SHORT"):
                signal = "NONE"

            price_x = float(sx.get(idx, np.nan))
            price_y = float(sy.get(idx, np.nan))
            hr = float(hedge_ratio.get(idx, 1.0))

            if pd.isna(price_x) or pd.isna(price_y):
                equity_curve.append(capital)
                continue

            # Open trade
            if signal in ("LONG", "SHORT") and position == "NONE":
                spread_std = float(spread.rolling(Config.LOOKBACK_WINDOW).std().get(idx, 0.001))
                stop_price = price_y + (spread_std * Config.ZSCORE_STOP * (-1 if signal == "LONG" else 1))
                quantity = (capital * Config.RISK_PER_TRADE) / max(abs(price_y - stop_price), 1e-6)

                position = signal
                entry_data = {
                    "direction": signal,
                    "entry_price_x": price_x,
                    "entry_price_y": price_y,
                    "hedge_ratio": hr,
                    "quantity": quantity,
                    "entry_time": str(idx),
                    "entry_zscore": z,
                    "regime": regime,
                }
                # Apply slippage on entry
                capital -= abs(price_y * quantity) * (self.tc + self.slippage)

            # Close trade
            elif signal == "EXIT" and position != "NONE":
                if entry_data:
                    if position == "LONG":
                        gross = (price_y - entry_data["entry_price_y"]) * entry_data["quantity"]
                    else:
                        gross = (entry_data["entry_price_y"] - price_y) * entry_data["quantity"]

                    cost = abs(price_y * entry_data["quantity"]) * (self.tc + self.slippage)
                    pnl_usd = gross - cost
                    pnl_ghs = handler.convert_usd_to_ghs(pnl_usd)
                    capital += pnl_ghs

                    trade_log.append({
                        **entry_data,
                        "exit_price_y": price_y,
                        "exit_time": str(idx),
                        "exit_zscore": z,
                        "pnl_usd": round(pnl_usd, 4),
                        "pnl_ghs": round(pnl_ghs, 4),
                    })

                position = "NONE"
                entry_data = {}

            equity_curve.append(round(capital, 2))

        return equity_curve, trade_log

    def _calculate_metrics(self, equity_curve: List[float], trade_log: List[Dict]) -> Dict:
        if not equity_curve or len(equity_curve) < 2:
            return {}

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        # Sharpe ratio (annualised assuming daily data)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / (peak + 1e-8)
        max_dd = float(np.max(drawdowns))

        # Trade metrics
        pnls = [t["pnl_ghs"] for t in trade_log if "pnl_ghs" in t]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and sum(losses) != 0 else float("inf")
        )

        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 4),
            "total_trades": len(trade_log),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "total_pnl_ghs": round(sum(pnls), 2) if pnls else 0.0,
        }

    def walk_forward(
        self,
        symbol_x: str,
        symbol_y: str,
        period: str = "5y",
        n_windows: int = 5,
    ) -> List[Dict]:
        """
        Walk-forward optimization: splits data into N windows, trains on each,
        tests on next. Returns list of per-window results.
        """
        handler = DataHandler()
        sx, sy = handler.fetch_pair_data(symbol_x, symbol_y, period=period)

        window_size = len(sx) // (n_windows + 1)
        results = []

        for i in range(n_windows):
            train_end = (i + 1) * window_size
            test_end = min(train_end + window_size, len(sx))

            sx_train = sx.iloc[:train_end]
            sy_train = sy.iloc[:train_end]
            sx_test = sx.iloc[train_end:test_end]
            sy_test = sy.iloc[train_end:test_end]

            if len(sx_test) < 30:
                continue

            # Run mini-backtest for this window
            kalman = KalmanFilterEngine()
            kalman_train = kalman.run_series(sx_train, sy_train)
            kalman_test = kalman.run_series(sx_test, sy_test)

            regime_detector = RegimeDetector()
            regime_detector.fit(sx_train)
            _, regime_arr = regime_detector.predict_regime(sx_test)
            regime_series = pd.Series(regime_arr, index=sx_test.index[1:])

            signal_engine = ZScoreSignalEngine()
            zscore = signal_engine.calculate_zscore(kalman_test["spread"])

            equity_curve, trade_log = self._simulate_trades(
                sx=sx_test, sy=sy_test,
                spread=kalman_test["spread"],
                zscore=zscore,
                hedge_ratio=kalman_test["hedge_ratio"],
                regime_series=regime_series,
                handler=handler,
            )
            metrics = self._calculate_metrics(equity_curve, trade_log)
            metrics["window"] = i + 1
            metrics["train_bars"] = len(sx_train)
            metrics["test_bars"] = len(sx_test)
            metrics["start_date"] = str(sx_test.index[0].date())
            metrics["end_date"] = str(sx_test.index[-1].date())
            results.append(metrics)
            logger.info(f"Walk-forward window {i+1}: Sharpe={metrics.get('sharpe_ratio', 0):.2f}")

        return results
