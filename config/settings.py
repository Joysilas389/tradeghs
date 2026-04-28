import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-prod")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pair_trading.db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    INITIAL_CAPITAL_GHS = float(os.getenv("INITIAL_CAPITAL_GHS", 50000))
    MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", 0.15))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
    EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY", "")

    # Trading pairs
    PRIMARY_PAIR = ("EURUSD=X", "USDCHF=X")
    FOREX_PAIRS = [
        "EURUSD=X", "USDCHF=X", "GBPUSD=X",
        "USDJPY=X", "AUDUSD=X", "NZDUSD=X"
    ]

    # Strategy params
    ZSCORE_ENTRY = 2.0
    ZSCORE_EXIT = 0.0
    ZSCORE_STOP = 3.5
    KALMAN_DELTA = 1e-4
    KALMAN_VE = 0.001
    HMM_STATES = 2
    HALFLIFE_MAX_DAYS = 30
    LOOKBACK_WINDOW = 60
    ROLLING_COINT_WINDOW = 120

    # Backtest defaults
    TRANSACTION_COST_PCT = 0.0002
    SLIPPAGE_PCT = 0.0001

    # MT5 (future)
    MT5_LOGIN = os.getenv("MT5_LOGIN")
    MT5_PASSWORD = os.getenv("MT5_PASSWORD")
    MT5_SERVER = os.getenv("MT5_SERVER")
