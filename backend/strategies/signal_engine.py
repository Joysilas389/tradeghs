import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from config.settings import Config

logger = logging.getLogger(__name__)


class ZScoreSignalEngine:
    """
    Generates trading signals from normalized spread (z-score).

    Signal rules:
        z > +ENTRY  → SHORT spread (short Y, long X*hedge)
        z < -ENTRY  → LONG spread  (long Y, short X*hedge)
        |z| < EXIT  → CLOSE position
        |z| > STOP  → STOP LOSS
    """

    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"
    NONE = "NONE"

    def __init__(
        self,
        entry_threshold: float = None,
        exit_threshold: float = None,
        stop_threshold: float = None,
        lookback: int = None,
    ):
        self.entry = entry_threshold or Config.ZSCORE_ENTRY
        self.exit = exit_threshold or Config.ZSCORE_EXIT
        self.stop = stop_threshold or Config.ZSCORE_STOP
        self.lookback = lookback or Config.LOOKBACK_WINDOW

    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """Rolling z-score of the spread series."""
        w = window or self.lookback
        mean = spread.rolling(w).mean()
        std = spread.rolling(w).std()
        zscore = (spread - mean) / std.replace(0, np.nan)
        return zscore

    def generate_signal(self, zscore: float, current_position: str = NONE) -> str:
        """
        Generate signal for a single z-score value.

        Args:
            zscore: current z-score of spread
            current_position: LONG / SHORT / NONE (current open position)

        Returns:
            signal: LONG / SHORT / EXIT / HOLD / NONE
        """
        if np.isnan(zscore):
            return self.NONE

        # Stop loss check (overrides all)
        if abs(zscore) >= self.stop:
            if current_position in (self.LONG, self.SHORT):
                logger.warning(f"Stop loss triggered at z={zscore:.2f}")
                return self.EXIT

        # Exit signal — spread reverted to mean
        if current_position == self.LONG and zscore >= -self.exit:
            return self.EXIT
        if current_position == self.SHORT and zscore <= self.exit:
            return self.EXIT

        # Hold existing position
        if current_position == self.LONG and zscore < -self.exit:
            return self.HOLD
        if current_position == self.SHORT and zscore > self.exit:
            return self.HOLD

        # New entry signals (only when no open position)
        if current_position == self.NONE:
            if zscore > self.entry:
                return self.SHORT
            if zscore < -self.entry:
                return self.LONG

        return self.NONE

    def generate_signals_series(
        self,
        spread: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate signals over a full historical spread series.

        Returns DataFrame with: zscore, signal, position columns.
        """
        zscore = self.calculate_zscore(spread, window=window)
        signals = []
        position = self.NONE

        for idx, z in zscore.items():
            sig = self.generate_signal(z, position)
            if sig == self.LONG:
                position = self.LONG
            elif sig == self.SHORT:
                position = self.SHORT
            elif sig == self.EXIT:
                position = self.NONE
            signals.append({
                "timestamp": idx,
                "zscore": z,
                "signal": sig,
                "position": position,
                "spread": spread.get(idx, np.nan),
            })

        df = pd.DataFrame(signals).set_index("timestamp")
        return df

    def get_current_signal(
        self,
        spread_series: pd.Series,
        current_position: str = NONE,
        window: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Get the signal for the most recent bar.

        Returns:
            (signal, current_zscore)
        """
        zscore_series = self.calculate_zscore(spread_series, window=window)
        current_z = float(zscore_series.iloc[-1]) if not zscore_series.empty else np.nan
        signal = self.generate_signal(current_z, current_position)
        return signal, current_z
