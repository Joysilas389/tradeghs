import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from config.settings import Config

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles all data fetching, cleaning, resampling and storage.
    Supports yfinance (free) and pluggable MT5 adapter.
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._ghs_rates: Dict[str, float] = {}

    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            logger.info(f"Fetching {symbol} period={period} interval={interval}")
            ticker = yf.Ticker(symbol)
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = self.clean_data(df)
            self._cache[cache_key] = df
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df.copy()

        except Exception as exc:
            logger.error(f"fetch_historical_data failed for {symbol}: {exc}")
            raise

    def fetch_pair_data(
        self,
        symbol_x: str,
        symbol_y: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> Tuple[pd.Series, pd.Series]:
        df_x = self.fetch_historical_data(symbol_x, period=period, interval=interval)
        df_y = self.fetch_historical_data(symbol_y, period=period, interval=interval)

        close_x = df_x["Close"]
        close_y = df_y["Close"]

        # Align on common dates
        combined = pd.concat([close_x, close_y], axis=1, join="inner")
        combined.columns = [symbol_x, symbol_y]
        combined.dropna(inplace=True)

        return combined[symbol_x], combined[symbol_y]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.dropna(subset=["Close"], inplace=True)
        # Remove outliers: price changes > 20% in single bar
        pct_change = df["Close"].pct_change().abs()
        df = df[pct_change < 0.20]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        return df

    def resample_data(self, df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
        return df["Close"].resample(rule).last().dropna().to_frame()

    def get_ghs_rates(self) -> Dict[str, float]:
        """Fetch USD/GHS and EUR/GHS from free API with fallback."""
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                rates = data.get("rates", {})
                ghs_per_usd = rates.get("GHS", 15.5)
                eur_per_usd = rates.get("EUR", 0.93)
                ghs_per_eur = ghs_per_usd / eur_per_usd
                self._ghs_rates = {
                    "USD_GHS": ghs_per_usd,
                    "EUR_GHS": ghs_per_eur,
                }
                logger.info(f"GHS rates updated: {self._ghs_rates}")
                return self._ghs_rates
        except Exception as exc:
            logger.warning(f"GHS rate fetch failed: {exc}, using fallback")

        # Fallback rates (update periodically)
        self._ghs_rates = {"USD_GHS": 15.5, "EUR_GHS": 16.8}
        return self._ghs_rates

    def convert_usd_to_ghs(self, amount_usd: float) -> float:
        if not self._ghs_rates:
            self.get_ghs_rates()
        return amount_usd * self._ghs_rates.get("USD_GHS", 15.5)

    def convert_eur_to_ghs(self, amount_eur: float) -> float:
        if not self._ghs_rates:
            self.get_ghs_rates()
        return amount_eur * self._ghs_rates.get("EUR_GHS", 16.8)


class SimulatedFeed:
    """
    Replays historical data candle-by-candle to simulate live feed.
    Used for paper trading and backtesting in real-time simulation mode.
    """

    def __init__(self, df: pd.DataFrame, speed_seconds: float = 0.1):
        self._data = df.reset_index()
        self._cursor = 0
        self._speed = speed_seconds
        self._running = False

    def start(self):
        self._running = True
        self._cursor = 0

    def stop(self):
        self._running = False

    def next_candle(self) -> Optional[dict]:
        if not self._running or self._cursor >= len(self._data):
            self._running = False
            return None
        row = self._data.iloc[self._cursor]
        self._cursor += 1
        return {
            "timestamp": str(row.get("Date", row.get("Datetime", self._cursor))),
            "open": float(row.get("Open", row.get("Close", 0))),
            "high": float(row.get("High", row.get("Close", 0))),
            "low": float(row.get("Low", row.get("Close", 0))),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        }

    def has_more(self) -> bool:
        return self._running and self._cursor < len(self._data)

    def progress_pct(self) -> float:
        if len(self._data) == 0:
            return 100.0
        return round(self._cursor / len(self._data) * 100, 1)


class MT5DataFeed:
    """
    Future MT5 pluggable adapter. Design stub — not active until MT5 credentials provided.
    """

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            ok = mt5.initialize(
                login=int(Config.MT5_LOGIN),
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER,
            )
            self._connected = ok
            return ok
        except Exception as exc:
            logger.warning(f"MT5 connect failed: {exc}")
            return False

    def fetch(self, symbol: str, timeframe, bars: int = 1000) -> pd.DataFrame:
        if not self._connected:
            raise RuntimeError("MT5 not connected")
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close",
                            "tick_volume": "Volume"}, inplace=True)
        return df
