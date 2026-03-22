# -*- coding: utf-8 -*-
"""
SNIPER BOT v1 — Backtest
For: دكتور محمد

Logic:
- Trade WITH trend only
- Enter on pullback (EMA zone)
- No reversal trading
- Clean + selective entries
"""

import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

SYMBOL = "ES=F"
INTERVAL = "5m"
LOOKBACK = "5d"

RR = 2.0
RISK = 10  # points

# =========================
# DATA
# =========================

df = yf.download(SYMBOL, interval=INTERVAL, period=LOOKBACK)

df.dropna(inplace=True)

# =========================
# INDICATORS
# =========================

df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()

# =========================
# MARKET TREND
# =========================

def get_trend(row):
    if row["EMA20"] > row["EMA50"]:
        return "UP"
    elif row["EMA20"] < row["EMA50"]:
        return "DOWN"
    else:
        return "NONE"

df["Trend"] = df.apply(get_trend, axis=1)

# =========================
# BACKTEST
# =========================

trades = []

for i in range(50, len(df) - 10):

    row = df.iloc[i]

    trend = row["Trend"]
    price = row["Close"]
    ema20 = row["EMA20"]

    # =========================
    # ENTRY LOGIC
    # =========================

    if trend == "UP":

        # pullback to EMA
        if price < ema20:

            entry = price
            stop = entry - RISK
            target = entry + (RISK * RR)

            future = df.iloc[i+1:i+10]

            hit_target = future["High"].max() >= target
            hit_stop = future["Low"].min() <= stop

            if hit_target and not hit_stop:
                result = +RR
            elif hit_stop:
                result = -1
            else:
                result = 0

            trades.append(("BUY", entry, stop, target, result))

    elif trend == "DOWN":

        if price > ema20:

            entry = price
            stop = entry + RISK
            target = entry - (RISK * RR)

            future = df.iloc[i+1:i+10]

            hit_target = future["Low"].min() <= target
            hit_stop = future["High"].max() >= stop

            if hit_target and not hit_stop:
                result = +RR
            elif hit_stop:
                result = -1
            else:
                result = 0

            trades.append(("SELL", entry, stop, target, result))


# =========================
# RESULTS
# =========================

wins = sum(1 for t in trades if t[4] > 0)
losses = sum(1 for t in trades if t[4] < 0)
total = len(trades)
net = sum(t[4] for t in trades)

winrate = (wins / total * 100) if total > 0 else 0

print("\n===== SNIPER BACKTEST =====")
print(f"Total Trades: {total}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Winrate: {winrate:.2f}%")
print(f"Net R: {net:.2f}")

print("\n===== SAMPLE TRADES =====")
for t in trades[:10]:
    print(t)
