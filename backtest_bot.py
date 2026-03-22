# -*- coding: utf-8 -*-
"""
SNIPER BOT v2 — Backtest with TradingView
For: دكتور محمد

Logic:
- Trend only
- Strong impulse first
- First pullback only
- Continuation confirmation candle
- ATR-based stop
- One trade per move
"""

import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# =========================
# CONFIG
# =========================

TV_SYMBOL = "SPX500"
TV_EXCHANGE = "FOREXCOM"
BARS = 3000

RR = 2.0
LOOKAHEAD_BARS = 12

ADX_MIN = 18
IMPULSE_ATR_MULT = 0.8
PULLBACK_MAX_BARS = 8
ATR_STOP_MULT = 0.9
ENTRY_BUFFER = 1.0

# =========================
# LOAD DATA
# =========================

tv = TvDatafeed()

def load_data():
    df = tv.get_hist(
        symbol=TV_SYMBOL,
        exchange=TV_EXCHANGE,
        interval=Interval.in_5_minute,
        n_bars=BARS
    )

    if df is None or df.empty:
        raise RuntimeError("TradingView returned no data")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df

# =========================
# INDICATORS
# =========================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema20"] = EMAIndicator(close=out["close"], window=20).ema_indicator()
    out["ema50"] = EMAIndicator(close=out["close"], window=50).ema_indicator()

    adx = ADXIndicator(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=14
    )
    out["adx"] = adx.adx()

    atr = AverageTrueRange(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=14
    )
    out["atr"] = atr.average_true_range()

    return out

# =========================
# HELPERS
# =========================

def trend_label(row) -> str:
    if row["ema20"] > row["ema50"] and row["adx"] >= ADX_MIN:
        return "UP"
    if row["ema20"] < row["ema50"] and row["adx"] >= ADX_MIN:
        return "DOWN"
    return "NONE"

def candle_body(row) -> float:
    return abs(float(row["close"]) - float(row["open"]))

def is_bull_candle(row) -> bool:
    return float(row["close"]) > float(row["open"])

def is_bear_candle(row) -> bool:
    return float(row["close"]) < float(row["open"])

def is_strong_impulse_up(df: pd.DataFrame, i: int) -> bool:
    if i < 3:
        return False

    row = df.iloc[i]
    atr = float(row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return False

    c0 = float(df["close"].iloc[i - 2])
    c1 = float(df["close"].iloc[i - 1])
    c2 = float(df["close"].iloc[i])

    body = candle_body(row)

    return (
        c0 < c1 < c2 and
        is_bull_candle(row) and
        body >= atr * IMPULSE_ATR_MULT and
        float(row["close"]) > float(row["ema20"])
    )

def is_strong_impulse_down(df: pd.DataFrame, i: int) -> bool:
    if i < 3:
        return False

    row = df.iloc[i]
    atr = float(row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return False

    c0 = float(df["close"].iloc[i - 2])
    c1 = float(df["close"].iloc[i - 1])
    c2 = float(df["close"].iloc[i])

    body = candle_body(row)

    return (
        c0 > c1 > c2 and
        is_bear_candle(row) and
        body >= atr * IMPULSE_ATR_MULT and
        float(row["close"]) < float(row["ema20"])
    )

def bullish_continuation_confirmation(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    return (
        is_bull_candle(row) and
        float(row["close"]) > float(prev["high"]) and
        float(row["close"]) > float(row["ema20"])
    )

def bearish_continuation_confirmation(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    return (
        is_bear_candle(row) and
        float(row["close"]) < float(prev["low"]) and
        float(row["close"]) < float(row["ema20"])
    )

# =========================
# FIND SETUPS
# =========================

def find_long_setup(df: pd.DataFrame, impulse_idx: int):
    impulse_row = df.iloc[impulse_idx]
    atr = float(impulse_row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    for j in range(impulse_idx + 1, min(impulse_idx + 1 + PULLBACK_MAX_BARS, len(df) - 1)):
        row = df.iloc[j]

        touched_pullback = (
            float(row["low"]) <= float(row["ema20"]) + ENTRY_BUFFER
        )

        invalid = float(row["low"]) < float(impulse_row["low"])

        if invalid:
            return None

        if touched_pullback and bullish_continuation_confirmation(df, j + 1):
            entry_row = df.iloc[j + 1]
            entry = float(entry_row["close"])
            stop = min(float(row["low"]), float(entry_row["low"])) - (atr * ATR_STOP_MULT)
            risk = entry - stop

            if risk <= 0:
                return None

            target = entry + (risk * RR)

            return {
                "direction": "BUY",
                "entry_idx": j + 1,
                "entry": entry,
                "stop": stop,
                "target": target,
                "atr": atr,
            }

    return None

def find_short_setup(df: pd.DataFrame, impulse_idx: int):
    impulse_row = df.iloc[impulse_idx]
    atr = float(impulse_row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    for j in range(impulse_idx + 1, min(impulse_idx + 1 + PULLBACK_MAX_BARS, len(df) - 1)):
        row = df.iloc[j]

        touched_pullback = (
            float(row["high"]) >= float(row["ema20"]) - ENTRY_BUFFER
        )

        invalid = float(row["high"]) > float(impulse_row["high"])

        if invalid:
            return None

        if touched_pullback and bearish_continuation_confirmation(df, j + 1):
            entry_row = df.iloc[j + 1]
            entry = float(entry_row["close"])
            stop = max(float(row["high"]), float(entry_row["high"])) + (atr * ATR_STOP_MULT)
            risk = stop - entry

            if risk <= 0:
                return None

            target = entry - (risk * RR)

            return {
                "direction": "SELL",
                "entry_idx": j + 1,
                "entry": entry,
                "stop": stop,
                "target": target,
                "atr": atr,
            }

    return None

# =========================
# TRADE SIMULATION
# =========================

def simulate_trade(df: pd.DataFrame, setup: dict):
    entry_idx = int(setup["entry_idx"])
    direction = setup["direction"]
    entry = float(setup["entry"])
    stop = float(setup["stop"])
    target = float(setup["target"])

    result = 0.0
    exit_idx = None

    for k in range(entry_idx + 1, min(entry_idx + 1 + LOOKAHEAD_BARS, len(df))):
        row = df.iloc[k]
        high = float(row["high"])
        low = float(row["low"])

        if direction == "BUY":
            if low <= stop:
                result = -1.0
                exit_idx = k
                break
            if high >= target:
                result = RR
                exit_idx = k
                break

        else:
            if high >= stop:
                result = -1.0
                exit_idx = k
                break
            if low <= target:
                result = RR
                exit_idx = k
                break

    if exit_idx is None:
        exit_idx = min(entry_idx + LOOKAHEAD_BARS, len(df) - 1)

    return result, exit_idx

# =========================
# BACKTEST ENGINE
# =========================

def run_backtest():
    df = load_data()
    df = add_indicators(df)
    df["trend"] = df.apply(trend_label, axis=1)

    trades = []
    i = 60

    while i < len(df) - LOOKAHEAD_BARS - 2:
        trend = df["trend"].iloc[i]

        setup = None

        if trend == "UP" and is_strong_impulse_up(df, i):
            setup = find_long_setup(df, i)

        elif trend == "DOWN" and is_strong_impulse_down(df, i):
            setup = find_short_setup(df, i)

        if setup is None:
            i += 1
            continue

        result, exit_idx = simulate_trade(df, setup)

        trades.append({
            "direction": setup["direction"],
            "entry": round(setup["entry"], 1),
            "stop": round(setup["stop"], 1),
            "target": round(setup["target"], 1),
            "result_r": result
        })

        i = max(exit_idx, setup["entry_idx"] + 1)

    total = len(trades)
    wins = sum(1 for t in trades if t["result_r"] > 0)
    losses = sum(1 for t in trades if t["result_r"] < 0)
    breakeven = sum(1 for t in trades if t["result_r"] == 0)
    net_r = sum(t["result_r"] for t in trades)
    winrate = (wins / total * 100.0) if total > 0 else 0.0

    print("\n===== SNIPER V2 BACKTEST (TradingView) =====")
    print(f"Total Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Breakeven: {breakeven}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Net R: {net_r:.2f}")

    print("\n===== SAMPLE TRADES =====")
    for t in trades[:10]:
        print(
            f"{t['direction']} | "
            f"Entry={t['entry']:.1f} | "
            f"Stop={t['stop']:.1f} | "
            f"Target={t['target']:.1f} | "
            f"Result={t['result_r']:+.1f}R"
        )

if __name__ == "__main__":
    run_backtest()
