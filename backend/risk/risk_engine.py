import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from config.settings import Config

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Manages position sizing, drawdown limits, stop-losses and exposure controls.

    Supports:
        - Fixed fractional sizing
        - Kelly criterion sizing
        - Max drawdown circuit breaker
        - Per-trade stop loss based on spread deviation
    """

    def __init__(
        self,
        initial_capital: float = None,
        risk_per_trade: float = None,
        max_drawdown_limit: float = None,
    ):
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL_GHS
        self.risk_per_trade = risk_per_trade or Config.RISK_PER_TRADE
        self.max_drawdown_limit = max_drawdown_limit or Config.MAX_DRAWDOWN_LIMIT
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self._open_positions = 0
        self._max_open_positions = 3

    def calculate_position_size_fixed_fractional(
        self,
        entry_price: float,
        stop_loss_price: float,
        capital_ghs: float,
    ) -> float:
        """
        Fixed fractional: risk a fixed % of capital per trade.
        Position size = (Capital × Risk%) / (Entry - StopLoss)
        """
        if entry_price <= 0 or abs(entry_price - stop_loss_price) < 1e-8:
            return 0.0

        risk_amount = capital_ghs * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        size = risk_amount / price_risk
        return float(size)

    def calculate_position_size_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital_ghs: float,
    ) -> float:
        """
        Kelly criterion: f* = (bp - q) / b
        b = avg_win/avg_loss, p = win_rate, q = 1 - p
        Capped at 25% to avoid over-leveraging.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return capital_ghs * self.risk_per_trade

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly_f = (b * p - q) / b
        kelly_f = max(0.0, min(kelly_f, 0.25))  # cap at 25%
        return float(capital_ghs * kelly_f)

    def calculate_stop_loss(
        self,
        entry_price: float,
        spread_std: float,
        direction: str,
        multiplier: float = 2.0,
    ) -> float:
        """
        Stop loss = entry ± (multiplier × spread_std).
        """
        offset = multiplier * spread_std
        if direction == "LONG":
            return float(entry_price - offset)
        else:
            return float(entry_price + offset)

    def check_drawdown(self, current_capital: float) -> Tuple[bool, float]:
        """
        Check if drawdown limit breached.

        Returns:
            (is_breached, current_drawdown_pct)
        """
        self.peak_capital = max(self.peak_capital, current_capital)
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        is_breached = drawdown >= self.max_drawdown_limit

        if is_breached:
            logger.warning(
                f"Max drawdown breached: {drawdown:.1%} >= {self.max_drawdown_limit:.1%}"
            )

        return is_breached, float(drawdown)

    def can_open_trade(self, current_capital: float) -> Tuple[bool, str]:
        """
        Pre-trade checks: drawdown, open position count.
        """
        breached, drawdown = self.check_drawdown(current_capital)
        if breached:
            return False, f"Drawdown limit reached ({drawdown:.1%})"

        if self._open_positions >= self._max_open_positions:
            return False, f"Max open positions ({self._max_open_positions}) reached"

        return True, "OK"

    def register_open(self):
        self._open_positions += 1

    def register_close(self):
        self._open_positions = max(0, self._open_positions - 1)

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        transaction_cost_pct: float = None,
        slippage_pct: float = None,
    ) -> float:
        """
        Calculate net P&L after costs and slippage.
        """
        tc = transaction_cost_pct or Config.TRANSACTION_COST_PCT
        sl = slippage_pct or Config.SLIPPAGE_PCT

        if direction == "LONG":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity

        cost = (entry_price + exit_price) * quantity * (tc + sl)
        return float(gross - cost)

    def metrics_summary(self, trade_pnls: list) -> dict:
        """
        Compute performance metrics from a list of trade P&Ls.
        """
        if not trade_pnls:
            return {}

        pnls = np.array(trade_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
        profit_factor = (
            float(np.sum(wins) / np.abs(np.sum(losses)))
            if np.sum(losses) != 0 else float("inf")
        )

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / (peak + 1e-8)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        else:
            sharpe = 0.0

        return {
            "total_trades": int(len(pnls)),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_pnl": round(float(np.sum(pnls)), 4),
        }
