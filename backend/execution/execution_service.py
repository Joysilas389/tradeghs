import logging
import json
from datetime import datetime
from typing import Optional, Dict, List
from backend.models.database import DatabaseManager, Trade, Signal
from backend.data.data_handler import DataHandler
from backend.risk.risk_engine import RiskEngine
from config.settings import Config

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Handles paper trade execution, order management, and trade logging.
    All trades are logged to SQLite. Designed to plug into MT5 later.
    """

    def __init__(self):
        self.db = DatabaseManager.get_instance()
        self.data_handler = DataHandler()
        self.risk_engine = RiskEngine()
        self._running = False
        self._current_position: Optional[Trade] = None

    def execute_signal(
        self,
        signal: str,
        symbol_x: str,
        symbol_y: str,
        price_x: float,
        price_y: float,
        hedge_ratio: float,
        zscore: float,
        spread: float,
        spread_std: float,
        regime: str,
    ) -> Optional[Dict]:
        """
        Execute a trade signal (paper mode). Logs to DB.
        """
        if signal in ("NONE", "HOLD"):
            return None

        session = self.db.get_session()
        try:
            rates = self.data_handler.get_ghs_rates()
            ghs_rate = rates.get("USD_GHS", 15.5)

            if signal in ("LONG", "SHORT"):
                return self._open_trade(
                    session, signal, symbol_x, symbol_y,
                    price_y, hedge_ratio, zscore, spread_std, regime, ghs_rate
                )
            elif signal == "EXIT" and self._current_position:
                return self._close_trade(session, price_y, zscore, ghs_rate)

        except Exception as exc:
            session.rollback()
            logger.error(f"Execution error: {exc}")
            return None
        finally:
            session.close()

    def _open_trade(
        self, session, direction, symbol_x, symbol_y,
        price_y, hedge_ratio, zscore, spread_std, regime, ghs_rate
    ) -> Dict:
        can_trade, reason = self.risk_engine.can_open_trade(self.risk_engine.current_capital)
        if not can_trade:
            logger.warning(f"Trade blocked: {reason}")
            return {"blocked": True, "reason": reason}

        stop_loss = self.risk_engine.calculate_stop_loss(price_y, spread_std, direction)
        quantity = self.risk_engine.calculate_position_size_fixed_fractional(
            entry_price=price_y,
            stop_loss_price=stop_loss,
            capital_ghs=self.risk_engine.current_capital,
        )

        trade = Trade(
            pair_symbol=f"{symbol_x}/{symbol_y}",
            direction=direction,
            entry_price=price_y,
            hedge_ratio=hedge_ratio,
            quantity=quantity,
            stop_loss=stop_loss,
            regime=regime,
            zscore_entry=zscore,
            status="OPEN",
        )
        session.add(trade)
        session.commit()
        session.refresh(trade)

        self._current_position = trade
        self.risk_engine.register_open()
        logger.info(f"Trade opened: {direction} {symbol_x}/{symbol_y} @ {price_y:.5f} qty={quantity:.2f}")
        return trade.to_dict()

    def _close_trade(self, session, price_y, zscore, ghs_rate) -> Dict:
        trade = session.get(Trade, self._current_position.id)
        if not trade:
            return {}

        if trade.direction == "LONG":
            pnl_usd = (price_y - trade.entry_price) * trade.quantity
        else:
            pnl_usd = (trade.entry_price - price_y) * trade.quantity

        cost = price_y * trade.quantity * (Config.TRANSACTION_COST_PCT + Config.SLIPPAGE_PCT)
        pnl_usd -= cost
        pnl_ghs = pnl_usd * ghs_rate

        trade.exit_price = price_y
        trade.exit_time = datetime.utcnow()
        trade.pnl_usd = round(pnl_usd, 6)
        trade.pnl_ghs = round(pnl_ghs, 2)
        trade.zscore_exit = zscore
        trade.status = "CLOSED"

        self.risk_engine.current_capital += pnl_ghs
        session.commit()
        session.refresh(trade)

        self.risk_engine.register_close()
        self._current_position = None
        logger.info(f"Trade closed: P&L = {pnl_ghs:.2f} GHS")
        return trade.to_dict()

    def save_signal(self, signal_data: Dict) -> None:
        session = self.db.get_session()
        try:
            sig = Signal(
                pair_symbol=f"{signal_data.get('symbol_x')}/{signal_data.get('symbol_y')}",
                signal_type=signal_data.get("signal", "NONE"),
                zscore=signal_data.get("zscore") or 0.0,
                spread=signal_data.get("spread") or 0.0,
                hedge_ratio=signal_data.get("hedge_ratio") or 1.0,
                regime=signal_data.get("regime"),
                halflife=signal_data.get("halflife"),
                is_cointegrated=signal_data.get("pair_valid", False),
            )
            session.add(sig)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Signal save error: {exc}")
        finally:
            session.close()

    def get_open_trades(self) -> List[Dict]:
        session = self.db.get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            return [t.to_dict() for t in trades]
        finally:
            session.close()

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        session = self.db.get_session()
        try:
            trades = (
                session.query(Trade)
                .order_by(Trade.created_at.desc())
                .limit(limit)
                .all()
            )
            return [t.to_dict() for t in trades]
        finally:
            session.close()
