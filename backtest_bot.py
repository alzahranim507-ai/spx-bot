# -*- coding: utf-8 -*-
"""
SPX Hybrid Smart Bot - Backtest
For: دكتور محمد
"""

import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from ta.trend import EMAIndicator, ADXIndicator

# ================= LOAD DATA =================

tv = TvDatafeed()

def load_data():
    df = tv.get_hist(
        symbol="SPX500",
        exchange="FOREXCOM",
        interval=Interval.in_5_minute,
        n_bars=3000
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned from TradingView")

    df.columns = [c.lower() for c in df.columns]
    df = df.dropna().copy()
    return df

# ================= INDICATORS =================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()

    adx = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    df["adx"] = adx.adx()
    return df

# ================= MARKET MODE =================

def detect_mode(row) -> str:
    if row["adx"] > 25:
        return "TREND"
    elif row["adx"] < 20:
        return "RANGE"
    return "MIXED"

# ================= PIVOTS =================

def find_pivots(df: pd.DataFrame):
    highs = []
    lows = []

    for i in range(3, len(df) - 3):
        if (
            df["high"].iloc[i] > df["high"].iloc[i - 3:i].max()
            and df["high"].iloc[i] > df["high"].iloc[i + 1:i + 4].max()
        ):
            highs.append((i, float(df["high"].iloc[i])))

        if (
            df["low"].iloc[i] < df["low"].iloc[i - 3:i].min()
            and df["low"].iloc[i] < df["low"].iloc[i + 1:i + 4].min()
        ):
            lows.append((i, float(df["low"].iloc[i])))

    return highs, lows

# ================= LEVELS =================

def build_levels(pivots, threshold=5.0):
    levels = []

    for _, lvl in pivots:
        if not levels:
            levels.append(float(lvl))
        else:
            if all(abs(float(lvl) - x) > threshold for x in levels):
                levels.append(float(lvl))

    return sorted(levels)

# ================= ENTRY =================

def check_entry(df: pd.DataFrame, i: int, levels: list):
    price = float(df["close"].iloc[i])
    prev = float(df["close"].iloc[i - 1])
    mode = detect_mode(df.iloc[i])

    for lvl in levels:
        # TREND BREAKOUT
        if prev < lvl and price > lvl:
            return "BUY", float(lvl), mode

        if prev > lvl and price < lvl:
            return "SELL", float(lvl), mode

        # RANGE TOUCH
        if mode == "RANGE":
            if abs(price - lvl) < 4:
                if price < lvl:
                    return "BUY", float(lvl), mode
                else:
                    return "SELL", float(lvl), mode

    return None, None, mode

# ================= TRADE SIMULATION =================

def simulate_trades(df: pd.DataFrame, levels: list):
    trades = []

    i = 50  # نبدأ بعد توفر بيانات كفاية
    while i < len(df) - 10:
        direction, level, mode = check_entry(df, i, levels)

        if direction is None:
            i += 1
            continue

        entry = float(df["close"].iloc[i])

        # Stop
        if direction == "BUY":
            stop = float(level - 5)
        else:
            stop = float(level + 5)

        risk = abs(entry - stop)
        if risk <= 0:
            i += 1
            continue

        # Target = 2R
        if direction == "BUY":
            target = float(entry + (risk * 2))
        else:
            target = float(entry - (risk * 2))

        result = None

        # نمشي قدام 10 شمعات
        for j in range(i + 1, min(i + 11, len(df))):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])

            if direction == "BUY":
                if low <= stop:
                    result = -1.0
                    break
                if high >= target:
                    result = 2.0
                    break

            else:  # SELL
                if high >= stop:
                    result = -1.0
                    break
                if low <= target:
                    result = 2.0
                    break

        if result is not None:
            trades.append({
                "index": i,
                "mode": mode,
                "direction": direction,
                "entry": entry,
                "stop": stop,
                "target": target,
                "result_r": result
            })
            i += 10
        else:
            i += 1

    return trades

# ================= RESULTS =================

def analyze_results(trades: list):
    total = len(trades)
    wins = len([t for t in trades if t["result_r"] > 0])
    losses = len([t for t in trades if t["result_r"] < 0])
    net_r = sum(t["result_r"] for t in trades)

    trend_trades = len([t for t in trades if t["mode"] == "TREND"])
    range_trades = len([t for t in trades if t["mode"] == "RANGE"])
    mixed_trades = len([t for t in trades if t["mode"] == "MIXED"])

    winrate = (wins / total * 100) if total > 0 else 0.0

    return {
        "Total Trades": total,
        "Wins": wins,
        "Losses": losses,
        "Winrate": round(winrate, 2),
        "Net R": round(net_r, 2),
        "Trend Trades": trend_trades,
        "Range Trades": range_trades,
        "Mixed Trades": mixed_trades,
    }

# ================= RUN BACKTEST =================

def run_backtest():
    df = load_data()
    df = add_indicators(df)

    highs, lows = find_pivots(df)
    levels = build_levels(highs + lows)

    trades = simulate_trades(df, levels)
    results = analyze_results(trades)

    print("\n===== BACKTEST RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n===== SAMPLE TRADES =====")
    for t in trades[:10]:
        print(
            f"{t['mode']} | {t['direction']} | "
            f"Entry={t['entry']:.1f} | Stop={t['stop']:.1f} | "
            f"Target={t['target']:.1f} | Result={t['result_r']:+.1f}R"
        )

if __name__ == "__main__":
    run_backtest()
