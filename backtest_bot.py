# -*- coding: utf-8 -*-
"""
SNIPER BOT v1 — Backtest with TradingView
For: دكتور محمد
"""

import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from ta.trend import EMAIndicator

# =========================
# CONFIG
# =========================

TV_SYMBOL = "SPX500"
TV_EXCHANGE = "FOREXCOM"

BARS = 3000
RR = 2.0
RISK_PTS = 10.0
LOOKAHEAD_BARS = 10

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
    return out

# =========================
# TREND
# =========================

def get_trend(row) -> str:
    if row["ema20"] > row["ema50"]:
        return "UP"
    elif row["ema20"] < row["ema50"]:
        return "DOWN"
    return "NONE"

# =========================
# BACKTEST
# =========================

def run_backtest():
    df = load_data()
    df = add_indicators(df)
    df["trend"] = df.apply(get_trend, axis=1)

    trades = []

    for i in range(60, len(df) - LOOKAHEAD_BARS):
        row = df.iloc[i]

        trend = row["trend"]
        price = float(row["close"])
        ema20 = float(row["ema20"])

        # =========================
        # ENTRY LOGIC
        # =========================

        direction = None

        # BUY pullback in uptrend
        if trend == "UP" and price < ema20:
            direction = "BUY"

        # SELL pullback in downtrend
        elif trend == "DOWN" and price > ema20:
            direction = "SELL"

        if direction is None:
            continue

        entry = price

        if direction == "BUY":
            stop = entry - RISK_PTS
            target = entry + (RISK_PTS * RR)
        else:
            stop = entry + RISK_PTS
            target = entry - (RISK_PTS * RR)

        future = df.iloc[i + 1:i + 1 + LOOKAHEAD_BARS]

        result = 0.0

        for _, frow in future.iterrows():
            high = float(frow["high"])
            low = float(frow["low"])

            if direction == "BUY":
                if low <= stop:
                    result = -1.0
                    break
                if high >= target:
                    result = RR
                    break

            else:  # SELL
                if high >= stop:
                    result = -1.0
                    break
                if low <= target:
                    result = RR
                    break

        trades.append({
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "result_r": result
        })

    # =========================
    # RESULTS
    # =========================

    total = len(trades)
    wins = sum(1 for t in trades if t["result_r"] > 0)
    losses = sum(1 for t in trades if t["result_r"] < 0)
    breakeven = sum(1 for t in trades if t["result_r"] == 0)
    net_r = sum(t["result_r"] for t in trades)
    winrate = (wins / total * 100.0) if total > 0 else 0.0

    print("\n===== SNIPER BACKTEST (TradingView) =====")
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
