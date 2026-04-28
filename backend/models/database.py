from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from config.settings import Config
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_symbol = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)   # LONG / SHORT
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    hedge_ratio = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    pnl_ghs = Column(Float, nullable=True)
    status = Column(String(20), default="OPEN")  # OPEN / CLOSED / STOPPED
    stop_loss = Column(Float, nullable=True)
    regime = Column(String(20), nullable=True)   # TRENDING / MEAN_REVERTING
    zscore_entry = Column(Float, nullable=True)
    zscore_exit = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "pair_symbol": self.pair_symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "hedge_ratio": self.hedge_ratio,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl_usd": self.pnl_usd,
            "pnl_ghs": self.pnl_ghs,
            "status": self.status,
            "regime": self.regime,
            "zscore_entry": self.zscore_entry,
        }


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_symbol = Column(String(50), nullable=False)
    signal_type = Column(String(20), nullable=False)  # LONG / SHORT / EXIT / NONE
    zscore = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    hedge_ratio = Column(Float, nullable=False)
    regime = Column(String(20), nullable=True)
    halflife = Column(Float, nullable=True)
    is_cointegrated = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    acted_on = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "pair_symbol": self.pair_symbol,
            "signal_type": self.signal_type,
            "zscore": self.zscore,
            "spread": self.spread,
            "hedge_ratio": self.hedge_ratio,
            "regime": self.regime,
            "halflife": self.halflife,
            "is_cointegrated": self.is_cointegrated,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class BacktestResult(Base):
    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_symbol = Column(String(50), nullable=False)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    transaction_cost_pct = Column(Float, nullable=True)
    slippage_pct = Column(Float, nullable=True)
    equity_curve = Column(Text, nullable=True)  # JSON string
    trade_log = Column(Text, nullable=True)      # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "pair_symbol": self.pair_symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "equity_curve": json.loads(self.equity_curve) if self.equity_curve else [],
            "trade_log": json.loads(self.trade_log) if self.trade_log else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=True)
    metric_label = Column(String(200), nullable=True)
    period = Column(String(50), nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_label": self.metric_label,
            "period": self.period,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


class DatabaseManager:
    _instance = None

    def __init__(self):
        self.engine = create_engine(
            Config.DATABASE_URL,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info("Database initialised")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_session(self):
        return self.SessionLocal()
