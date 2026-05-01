"""
engine.py — Signal engine for paper trading bot
Indicators: AMA (Kaufman), EMA, RSI, Volume Average
Confluence scoring weighted by inverse covariance (signals with lower inter-correlation get higher weight)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class IndicatorValues:
    ama: float = np.nan
    ema: float = np.nan
    rsi: float = np.nan
    vol_avg: float = np.nan
    vol_ratio: float = np.nan  # current vol / vol_avg
    ama_slope: float = np.nan
    ema_slope: float = np.nan
    crossover: str = "NONE"    # BULLISH / BEARISH / NONE


@dataclass
class ConfluenceScore:
    raw_score: float = 0.0        # -1 to +1
    weighted_score: float = 0.0   # covariance-weighted
    components: dict = field(default_factory=dict)
    signal: Signal = Signal.HOLD
    strength: str = "WEAK"        # WEAK / MODERATE / STRONG


# ── Covariance-derived weights ──────────────────────────────────────────────
# From the matrix analysis:
#   AMA–EMA highly correlated (ρ≈0.90) → downweight both
#   RSI moderately correlated (ρ≈0.20) → normal weight
#   Vol Avg near-orthogonal (ρ<0.08) → upweight (most independent)
#
# Weights derived from inverse-variance of correlation off-diagonals:
# AMA  : 0.22 | EMA  : 0.18 | RSI  : 0.26 | VOL  : 0.34
# (AMA gets slight edge over EMA because it's adaptive, EMA is baseline)

COV_WEIGHTS = {
    "ama": 0.22,
    "ema": 0.18,
    "rsi": 0.26,
    "vol": 0.34,
}

# Signal thresholds
BUY_THRESHOLD = 0.35
SELL_THRESHOLD = -0.35
STRONG_THRESHOLD = 0.60


def compute_ama(close: np.ndarray, fast: int = 9, slow: int = 30, er_period: int = 10) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (AMA / KAMA).
    fast, slow: periods for smoothing constants.
    er_period: efficiency ratio lookback.
    """
    n = len(close)
    ama = np.full(n, np.nan)
    if n < er_period + 1:
        return ama

    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    # Initialise at first valid point
    ama[er_period] = close[er_period]

    for i in range(er_period + 1, n):
        # Direction = net price change over er_period bars
        direction = abs(close[i] - close[i - er_period])
        # Volatility = sum of absolute daily changes
        volatility = np.sum(np.abs(np.diff(close[i - er_period: i + 1])))

        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        ama[i] = ama[i - 1] + sc * (close[i] - ama[i - 1])

    return ama


def compute_ema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    ema = np.full(len(close), np.nan)
    start = period - 1
    if start >= len(close):
        return ema
    ema[start] = np.mean(close[:period])
    for i in range(start + 1, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi[i + 1] = 100 - (100 / (1 + rs))

    return rsi


def compute_volume_average(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple moving average of volume."""
    vol_avg = np.full(len(volume), np.nan)
    for i in range(period - 1, len(volume)):
        vol_avg[i] = np.mean(volume[i - period + 1: i + 1])
    return vol_avg


def detect_crossover(ama: np.ndarray, ema: np.ndarray, lookback: int = 3) -> str:
    """
    Detect AMA–EMA crossover within last `lookback` bars.
    Returns 'BULLISH', 'BEARISH', or 'NONE'.
    Skips NaN values to avoid False==False masking a real crossover.
    """
    n = len(ama)
    if n < lookback + 2:
        return "NONE"

    # Current bar must be valid
    if np.isnan(ama[-1]) or np.isnan(ema[-1]):
        return "NONE"

    curr_above = ama[-1] > ema[-1]

    # Scan backwards, skip NaN pairs
    for i in range(2, lookback + 2):
        if i >= n:
            break
        a_prev = ama[-i]
        e_prev = ema[-i]
        if np.isnan(a_prev) or np.isnan(e_prev):
            continue  # skip NaN bars, keep looking
        prev_above = a_prev > e_prev
        if curr_above != prev_above:
            return "BULLISH" if curr_above else "BEARISH"

    return "NONE"


def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute all indicators on OHLCV DataFrame.
    Returns DataFrame with indicator columns added.
    """
    close = df["close"].values
    volume = df["volume"].values

    df = df.copy()
    df["ama"] = compute_ama(
        close,
        fast=params.get("ama_fast", 9),
        slow=params.get("ama_slow", 30),
        er_period=params.get("ama_er", 10),
    )
    df["ema"] = compute_ema(close, period=params.get("ema_period", 20))
    df["rsi"] = compute_rsi(close, period=params.get("rsi_period", 14))
    df["vol_avg"] = compute_volume_average(volume, period=params.get("vol_period", 20))
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    # Slopes (normalised 3-bar)
    df["ama_slope"] = df["ama"].diff(3) / df["ama"].shift(3)
    df["ema_slope"] = df["ema"].diff(3) / df["ema"].shift(3)

    return df


def score_bar(row: pd.Series, prev_row: pd.Series,
              crossover: str, params: dict) -> ConfluenceScore:
    """
    Compute confluence score for a single bar.
    Each component scored in [-1, +1], then combined with COV_WEIGHTS.
    """
    components = {}

    # ── 1. AMA component ────────────────────────────────────────────────────
    # +1 if price above AMA and AMA sloping up; -1 if below and sloping down
    if not np.isnan(row["ama"]):
        price_vs_ama = 1.0 if row["close"] > row["ama"] else -1.0
        slope_sign = np.sign(row.get("ama_slope", 0))
        components["ama"] = 0.6 * price_vs_ama + 0.4 * slope_sign
    else:
        components["ama"] = 0.0

    # ── 2. EMA component ────────────────────────────────────────────────────
    if not np.isnan(row["ema"]):
        price_vs_ema = 1.0 if row["close"] > row["ema"] else -1.0
        slope_sign = np.sign(row.get("ema_slope", 0))
        # Crossover bonus: fresh crossover gets +0.3 boost to EMA component
        xo_bonus = 0.3 if crossover == "BULLISH" else (-0.3 if crossover == "BEARISH" else 0.0)
        components["ema"] = np.clip(0.5 * price_vs_ema + 0.3 * slope_sign + 0.2 * xo_bonus, -1, 1)
    else:
        components["ema"] = 0.0

    # ── 3. RSI component ────────────────────────────────────────────────────
    # Linear mapping: 30→-1, 50→0, 70→+1 (with extreme clipping)
    rsi_threshold_ob = params.get("rsi_ob", 65)
    rsi_threshold_os = params.get("rsi_os", 35)

    if not np.isnan(row["rsi"]):
        rsi = row["rsi"]
        if rsi >= rsi_threshold_ob:
            components["rsi"] = -0.5  # overbought = bearish pressure
        elif rsi <= rsi_threshold_os:
            components["rsi"] = 0.5   # oversold = bullish pressure
        else:
            # Neutral zone: linearly scaled, momentum direction matters
            mid = (rsi_threshold_ob + rsi_threshold_os) / 2
            rng = (rsi_threshold_ob - rsi_threshold_os) / 2
            components["rsi"] = (rsi - mid) / rng * 0.5
    else:
        components["rsi"] = 0.0

    # ── 4. Volume component ─────────────────────────────────────────────────
    # Vol ratio > vol_threshold confirms trend; low volume = uncertain
    vol_threshold = params.get("vol_threshold", 1.2)
    vol_strong = params.get("vol_strong", 2.0)

    if not np.isnan(row.get("vol_ratio", np.nan)):
        vr = row["vol_ratio"]
        if vr >= vol_strong:
            # High volume: confirms direction of AMA signal
            vol_dir = np.sign(components["ama"])
            components["vol"] = 1.0 * vol_dir
        elif vr >= vol_threshold:
            vol_dir = np.sign(components["ama"])
            components["vol"] = 0.5 * vol_dir
        else:
            components["vol"] = 0.0  # low volume = no confirmation
    else:
        components["vol"] = 0.0

    # ── Weighted confluence ──────────────────────────────────────────────────
    raw = sum(components[k] for k in components) / len(components)
    weighted = sum(COV_WEIGHTS[k] * components[k] for k in components)

    # Determine signal
    if weighted >= BUY_THRESHOLD:
        sig = Signal.BUY
    elif weighted <= SELL_THRESHOLD:
        sig = Signal.SELL
    else:
        sig = Signal.HOLD

    strength = "WEAK"  # default for any non-HOLD signal
    if sig != Signal.HOLD:
        if abs(weighted) >= STRONG_THRESHOLD:
            strength = "STRONG"
        elif abs(weighted) >= BUY_THRESHOLD + 0.10:
            strength = "MODERATE"
        else:
            strength = "WEAK"
    else:
        strength = "HOLD"

    return ConfluenceScore(
        raw_score=round(raw, 4),
        weighted_score=round(weighted, 4),
        components=components,
        signal=sig,
        strength=strength,
    )


def run_signals(df: pd.DataFrame, params: dict) -> list[dict]:
    """
    Run full signal scan on processed DataFrame.
    Returns list of signal dicts for each bar.
    """
    df = compute_indicators(df, params)
    ama_arr = df["ama"].values
    ema_arr = df["ema"].values

    results = []
    warmup = max(
        params.get("ama_slow", 30),
        params.get("ema_period", 20),
        params.get("rsi_period", 14),
        params.get("vol_period", 20),
    ) + 5  # +5 for slope calculation

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        xo = detect_crossover(ama_arr[:i+1], ema_arr[:i+1], lookback=params.get("xo_lookback", 3))
        score = score_bar(row, prev_row, xo, params)

        results.append({
            "date": df.index[i],
            "close": row["close"],
            "ama": row["ama"],
            "ema": row["ema"],
            "rsi": row["rsi"],
            "vol_ratio": row.get("vol_ratio", np.nan),
            "crossover": xo,
            "weighted_score": score.weighted_score,
            "raw_score": score.raw_score,
            "components": score.components,
            "signal": score.signal.value,
            "strength": score.strength,
        })

    return results
