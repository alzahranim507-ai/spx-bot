# -*- coding: utf-8 -*-
"""
SPX/ES Trading Bot (Balanced Scoring) — Yahoo Finance data
Rules (per Dr. Mohammed):
- Pre-Market & After-Hours: use ES=F (with -10 points adjustment)
- Regular Market (RTH): use ^GSPC (SPX Index)
- Hourly Update: every hour (NO entry/targets)
- Signals: Event-driven with Entry/SL/Targets/RR
- Mandatory: Structure + Key Levels
- Balanced Scoring threshold >= 4
- Telegram messages always include: "دكتور محمد"
"""

import os
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Python <3.9 fallback


# =========================
# Config
# =========================

@dataclass
class Config:
    # Symbols
    es_symbol: str = "ES=F"     # Futures (pre/after)
    spx_symbol: str = "^GSPC"   # Index (regular market)

    # ES adjustment to match your TradingView (TV) chart:
    # "TradingView ناقص 10 عن ياهو" => subtract 10 from Yahoo ES for alignment.
    es_points_adjust: float = 10.0

    # Structure / pivots
    pivot_left: int = 3
    pivot_right: int = 3

    # Levels
    level_touch_tolerance: float = 0.0012   # 0.12%
    level_cluster_tolerance: float = 0.0010
    max_levels: int = 6

    # Scoring
    score_threshold: int = 4
    signal_cooldown_minutes: int = 30

    # Loop
    loop_sleep_seconds: int = 30

    # Hourly update
    hourly_update: bool = True

    # Risk quality filter
    min_rr_to_t1: float = 1.2

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Identity
    user_title: str = "دكتور محمد"

    # Timezones
    tz_riyadh: str = "Asia/Riyadh"
    tz_ny: str = "America/New_York"


CFG = Config()


# =========================
# Time helpers
# =========================

def tzinfo(name: str):
    if ZoneInfo is None:
        # fallback: best-effort fixed offset (Riyadh +3, NY -5)
        if name == "Asia/Riyadh":
            return timezone(timedelta(hours=3))
        if name == "America/New_York":
            return timezone(timedelta(hours=-5))
        return timezone.utc
    return ZoneInfo(name)

TZ_RIYADH = tzinfo(CFG.tz_riyadh)
TZ_NY = tzinfo(CFG.tz_ny)

def now_riyadh() -> datetime:
    return datetime.now(TZ_RIYADH)

def now_ny() -> datetime:
    return datetime.now(TZ_NY)

def session_label() -> str:
    """
    Labels by NY time:
    - Pre-Market: 04:00–09:30
    - Market:     09:30–16:00
    - After-Hours: else
    """
    t = now_ny()
    hm = t.hour * 60 + t.minute
    pre_start = 4 * 60
    rth_start = 9 * 60 + 30
    rth_end = 16 * 60

    if pre_start <= hm < rth_start:
        return "Pre-Market"
    if rth_start <= hm < rth_end:
        return "Market"
    return "After-Hours"


# =========================
# Telegram
# =========================

def send_telegram(text: str):
    if not CFG.telegram_bot_token or not CFG.telegram_chat_id:
        print("[WARN] Telegram env vars not set. Printing message:\n", text)
        return

    url = f"https://api.telegram.org/bot{CFG.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": CFG.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            print("[WARN] Telegram send failed:", r.text)
    except Exception as e:
        print("[WARN] Telegram exception:", e)


# =========================
# Data fetching (Yahoo)
# =========================

def _yf_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True
    )

    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty data for {symbol} interval={interval} period={period}")

    # Normalize columns (yfinance can return multiindex with tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Keep standard OHLCV
    cols = ["open", "high", "low", "close"]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {symbol} data")

    # Ensure datetime index is tz-aware (assume UTC if naive)
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    return df

def apply_es_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract 10 points from ES OHLC so it matches your TradingView baseline."""
    out = df.copy()
    adj = CFG.es_points_adjust
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) - adj
    return out

def fetch_timeframes(symbol: str, is_es: bool):
    """
    Fetch:
    - 5m (7d), 15m (30d), 60m (60d)
    Then derive 4h from 60m resample.
    """
    df_5m = _yf_download(symbol, interval="5m", period="7d")
    df_15m = _yf_download(symbol, interval="15m", period="30d")
    df_1h = _yf_download(symbol, interval="60m", period="60d")

    if is_es:
        df_5m = apply_es_adjustment(df_5m)
        df_15m = apply_es_adjustment(df_15m)
        df_1h = apply_es_adjustment(df_1h)

    # Build 4H from 1H
    df_4h = (
        df_1h
        .resample("4H")
        .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"} if "volume" in df_1h.columns else
             {"open":"first","high":"max","low":"min","close":"last"})
        .dropna()
    )

    return df_4h, df_1h, df_15m, df_5m


def active_symbol_and_data():
    """
    Market session => ^GSPC
    Pre/After => ES=F (adjusted -10)
    """
    sess = session_label()
    if sess == "Market":
        sym = CFG.spx_symbol
        is_es = False
    else:
        sym = CFG.es_symbol
        is_es = True

    df_4h, df_1h, df_15m, df_5m = fetch_timeframes(sym, is_es=is_es)
    return sess, sym, is_es, df_4h, df_1h, df_15m, df_5m


# =========================
# Structure / pivots
# =========================

def find_pivots(series: pd.Series, left: int, right: int):
    arr = series.values
    piv_hi, piv_lo = [], []
    n = len(arr)

    for i in range(left, n - right):
        v = arr[i]
        wl = arr[i-left:i]
        wr = arr[i+1:i+1+right]
        if np.all(v > wl) and np.all(v >= wr):
            piv_hi.append(i)
        if np.all(v < wl) and np.all(v <= wr):
            piv_lo.append(i)
    return piv_hi, piv_lo

def last_two_swings(df: pd.DataFrame, left: int, right: int):
    hi_idx, _ = find_pivots(df["high"], left, right)
    _, lo_idx = find_pivots(df["low"], left, right)

    highs = [(df.index[i], float(df["high"].iloc[i])) for i in hi_idx][-4:]
    lows  = [(df.index[i], float(df["low"].iloc[i])) for i in lo_idx][-4:]
    return highs, lows

def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    def bias_from(df):
        highs, lows = last_two_swings(df, CFG.pivot_left, CFG.pivot_right)
        if len(highs) < 2 or len(lows) < 2:
            return "Weak"
        h1, h2 = highs[-2][1], highs[-1][1]
        l1, l2 = lows[-2][1], lows[-1][1]

        if h2 > h1 and l2 > l1:
            return "Bullish"
        if h2 < h1 and l2 < l1:
            return "Bearish"
        return "Weak"

    b1 = bias_from(df_1h)
    b4 = bias_from(df_4h)

    if b1 == "Bullish" and b4 == "Bullish":
        return "Bullish"
    if b1 == "Bearish" and b4 == "Bearish":
        return "Bearish"
    if b1 == "Weak" and b4 == "Weak":
        return "Range"
    return "Weak"


# =========================
# Levels
# =========================

def cluster_levels(levels: list[float], tol_frac: float, price_ref: float) -> list[float]:
    if not levels:
        return []
    levels_sorted = sorted(levels)
    clustered = [levels_sorted[0]]
    for lvl in levels_sorted[1:]:
        if abs(lvl - clustered[-1]) / price_ref <= tol_frac:
            clustered[-1] = (clustered[-1] + lvl) / 2.0
        else:
            clustered.append(lvl)
    return clustered

def extract_key_levels(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> list[float]:
    price = float(df_15m["close"].iloc[-1])

    # Swing points from 1H
    hi_idx, lo_idx = find_pivots(df_1h["high"], CFG.pivot_left, CFG.pivot_right)
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx[-10:]] if hi_idx else []
    swing_lows  = [float(df_1h["low"].iloc[i])  for i in lo_idx[-10:]] if lo_idx else []

    # Recent 15m range extremes
    recent = df_15m.tail(200)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    # Previous day high/low (Riyadh date split)
    prev = df_15m.tail(96*2).copy()  # ~2 days
    prev_local = prev.copy()
    prev_local.index = prev_local.index.tz_convert(TZ_RIYADH)
    prev_local["date"] = prev_local.index.date
    dates = sorted(prev_local["date"].unique())
    prev_day = dates[-2] if len(dates) >= 2 else dates[-1]
    prev_day_df = prev_local[prev_local["date"] == prev_day]
    prev_day_hi = float(prev_day_df["high"].max()) if len(prev_day_df) else np.nan
    prev_day_lo = float(prev_day_df["low"].min()) if len(prev_day_df) else np.nan

    candidates = []
    candidates += swing_highs + swing_lows
    candidates += [range_hi, range_lo]
    if np.isfinite(prev_day_hi): candidates.append(prev_day_hi)
    if np.isfinite(prev_day_lo): candidates.append(prev_day_lo)

    candidates = [float(x) for x in candidates if np.isfinite(x)]
    clustered = cluster_levels(candidates, CFG.level_cluster_tolerance, price)

    # Rank by proximity to current price, keep extremes too
    clustered = sorted(clustered, key=lambda x: abs(x - price))
    extremes = []
    if clustered:
        extremes = [min(clustered), max(clustered)]

    merged = cluster_levels(clustered + extremes, CFG.level_cluster_tolerance, price)
    merged = sorted(merged)

    if len(merged) > CFG.max_levels:
        around = sorted(merged, key=lambda x: abs(x - price))[:CFG.max_levels-2]
        merged = sorted(cluster_levels(around + [min(merged), max(merged)], CFG.level_cluster_tolerance, price))

    return merged

def near_level(price: float, level: float, tol_frac: float) -> bool:
    return abs(price - level) / price <= tol_frac

def fmt_levels(levels: list[float]) -> str:
    return ", ".join([f"{lvl:.1f}" for lvl in levels])


# =========================
# Indicators & Scoring
# =========================

def compute_indicators(df_5m: pd.DataFrame) -> pd.DataFrame:
    out = df_5m.copy()

    out["rsi"] = RSIIndicator(close=out["close"], window=14).rsi()

    st = StochRSIIndicator(close=out["close"], window=14, smooth1=3, smooth2=3)
    out["stochrsi_k"] = st.stochrsi_k()
    out["stochrsi_d"] = st.stochrsi_d()

    out["ema20"] = EMAIndicator(close=out["close"], window=20).ema_indicator()
    macd = MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist"] = macd.macd_diff()

    return out

def rejection_candle_score(df_5m: pd.DataFrame, direction: str) -> int:
    c = df_5m.iloc[-1]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    if direction == "Bullish":
        cond = (lower / rng >= 0.45) and (cl > o) and (upper / rng <= 0.25)
        return 2 if cond else 0
    if direction == "Bearish":
        cond = (upper / rng >= 0.45) and (cl < o) and (lower / rng <= 0.25)
        return 2 if cond else 0
    return 0

def break_retest_score(df_5m: pd.DataFrame, level: float, direction: str, tol_frac: float) -> int:
    price = float(df_5m["close"].iloc[-1])
    window = df_5m.tail(8)

    if direction == "Bullish":
        broke = (window["close"] > level).any()
        retest = near_level(price, level, tol_frac) or near_level(float(window["low"].min()), level, tol_frac)
        return 2 if (broke and retest and price >= level) else 0

    if direction == "Bearish":
        broke = (window["close"] < level).any()
        retest = near_level(price, level, tol_frac) or near_level(float(window["high"].max()), level, tol_frac)
        return 2 if (broke and retest and price <= level) else 0

    return 0

def rsi_turn_score(df_5m: pd.DataFrame, direction: str) -> int:
    r = df_5m["rsi"].tail(6).values
    if len(r) < 3 or np.isnan(r).any():
        return 0

    if direction == "Bullish":
        was_oversold = np.min(r) < 30
        turning_up = r[-1] > r[-2] > r[-3]
        return 1 if (was_oversold and turning_up) else 0

    if direction == "Bearish":
        was_overbought = np.max(r) > 70
        turning_down = r[-1] < r[-2] < r[-3]
        return 1 if (was_overbought and turning_down) else 0

    return 0

def stoch_cross_score(df_5m: pd.DataFrame, direction: str) -> int:
    k = df_5m["stochrsi_k"].tail(3).values
    d = df_5m["stochrsi_d"].tail(3).values
    if len(k) < 3 or np.isnan(k).any() or np.isnan(d).any():
        return 0

    prev = k[-2] - d[-2]
    curr = k[-1] - d[-1]

    if direction == "Bullish":
        return 1 if (prev < 0 and curr > 0) else 0
    if direction == "Bearish":
        return 1 if (prev > 0 and curr < 0) else 0
    return 0

def momentum_shift_score(df_5m: pd.DataFrame, direction: str) -> int:
    if len(df_5m) < 30:
        return 0

    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema20"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    hist_prev = float(df_5m["macd_hist"].iloc[-2])

    if direction == "Bullish":
        return 1 if (c > ema) or (hist_prev < 0 and hist > 0) else 0
    if direction == "Bearish":
        return 1 if (c < ema) or (hist_prev > 0 and hist < 0) else 0
    return 0


# =========================
# Trade plan (Entry/SL/Targets)
# =========================

def pick_targets(levels: list[float], entry: float, direction: str):
    if not levels:
        return None, None

    if direction == "Bullish":
        above = sorted([lvl for lvl in levels if lvl > entry])
        t1 = above[0] if len(above) >= 1 else None
        t2 = above[1] if len(above) >= 2 else None
        return t1, t2

    if direction == "Bearish":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        t1 = below[0] if len(below) >= 1 else None
        t2 = below[1] if len(below) >= 2 else None
        return t1, t2

    return None, None

def compute_trade_plan(df_5m: pd.DataFrame, levels: list[float], level_hit: float, direction: str, trigger: str):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    buffer = max(price * 0.0002, 0.5)

    if "Rejection" in trigger:
        if direction == "Bullish":
            entry = float(last["high"] + buffer)
            stop = float(min(last["low"], level_hit) - buffer)
        else:
            entry = float(last["low"] - buffer)
            stop = float(max(last["high"], level_hit) + buffer)
    elif "Break&Retest" in trigger:
        if direction == "Bullish":
            entry = float(max(price, level_hit + buffer))
            stop = float(level_hit - (price * 0.0015) - buffer)
        else:
            entry = float(min(price, level_hit - buffer))
            stop = float(level_hit + (price * 0.0015) + buffer)
    else:
        entry = price
        if direction == "Bullish":
            stop = float(level_hit - (price * 0.0015) - buffer)
        else:
            stop = float(level_hit + (price * 0.0015) + buffer)

    t1, t2 = pick_targets(levels, entry, direction)

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = (reward / risk) if risk > 0 else None

    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "rr": rr}


# =========================
# Bot state
# =========================

class BotState:
    def __init__(self):
        self.last_hour_sent = None
        self.last_signal_ts = {}  # key: f"{symbol}:{dir}:{round(level,1)}"

    def can_send_signal(self, symbol: str, direction: str, level: float) -> bool:
        key = f"{symbol}:{direction}:{round(level, 1)}"
        last = self.last_signal_ts.get(key)
        if last is None:
            return True
        return (datetime.utcnow() - last) >= timedelta(minutes=CFG.signal_cooldown_minutes)

    def mark_signal(self, symbol: str, direction: str, level: float):
        key = f"{symbol}:{direction}:{round(level, 1)}"
        self.last_signal_ts[key] = datetime.utcnow()

STATE = BotState()


# =========================
# Messaging
# =========================

def build_hourly_update(session: str, symbol: str, bias: str, levels: list[float], price: float, is_es: bool) -> str:
    note = ""
    if is_es:
        note = f"\n🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias: <b>{bias}</b>\n"
        f"💵 Price: {price:.1f}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}"
        f"{note}"
    )

def build_signal_message(session: str, symbol: str, bias: str, level_hit: float, score: int, reasons: list[str], plan: dict, is_es: bool) -> str:
    rr_txt = f"{plan['rr']:.2f}" if plan["rr"] is not None else "N/A"
    t1_txt = f"{plan['t1']:.1f}" if plan["t1"] is not None else "N/A"
    t2_txt = f"{plan['t2']:.1f}" if plan["t2"] is not None else "N/A"
    note = ""
    if is_es:
        note = f"\n🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Balanced)\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias: <b>{bias}</b>\n"
        f"🧱 Level: {level_hit:.1f}\n\n"
        f"✅ Entry: <b>{plan['entry']:.1f}</b>\n"
        f"🛑 Stop: <b>{plan['stop']:.1f}</b>\n"
        f"🎯 Target 1: <b>{t1_txt}</b>\n"
        f"🎯 Target 2: <b>{t2_txt}</b>\n"
        f"📐 RR (to T1): <b>{rr_txt}</b>\n\n"
        f"⭐ Score: <b>{score}/7</b>\n"
        f"🧠 Reason: {', '.join(reasons)}"
        f"{note}"
    )


# =========================
# Main evaluation
# =========================

def evaluate_once():
    session, symbol, is_es, df_4h, df_1h, df_15m, df_5m_raw = active_symbol_and_data()
    df_5m = compute_indicators(df_5m_raw)

    price = float(df_15m["close"].iloc[-1])

    # 1) Structure/Bias (mandatory)
    bias = structure_bias(df_1h, df_4h)

    # 2) Key levels (mandatory)
    levels = extract_key_levels(df_15m, df_1h)

    # Hourly update (no entry/targets)
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(build_hourly_update(session, symbol, bias, levels, price, is_es))
            STATE.last_hour_sent = current_hour

    # Gate: need clear bias + levels
    if bias in ("Weak", "Range") or not levels:
        return

    # Must be near a key level
    nearby = [lvl for lvl in levels if near_level(price, lvl, CFG.level_touch_tolerance)]
    if not nearby:
        return

    level_hit = min(nearby, key=lambda x: abs(x - price))

    # Scoring
    score = 0
    reasons = []

    rj = rejection_candle_score(df_5m, bias)
    if rj:
        score += rj
        reasons.append("Rejection")

    br = break_retest_score(df_5m, level_hit, bias, CFG.level_touch_tolerance)
    if br:
        score += br
        reasons.append("Break&Retest")

    rs = rsi_turn_score(df_5m, bias)
    if rs:
        score += rs
        reasons.append("RSI turn")

    sc = stoch_cross_score(df_5m, bias)
    if sc:
        score += sc
        reasons.append("Stoch cross")

    ms = momentum_shift_score(df_5m, bias)
    if ms:
        score += ms
        reasons.append("Momentum shift")

    if score < CFG.score_threshold:
        return

    if not STATE.can_send_signal(symbol, bias, level_hit):
        return

    trigger = "Rejection" if "Rejection" in reasons else ("Break&Retest" if "Break&Retest" in reasons else "Momentum")
    plan = compute_trade_plan(df_5m, levels, level_hit, bias, trigger)

    if plan["rr"] is not None and plan["rr"] < CFG.min_rr_to_t1:
        return

    send_telegram(build_signal_message(session, symbol, bias, level_hit, score, reasons, plan, is_es))
    STATE.mark_signal(symbol, bias, level_hit)


def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- Market: {CFG.spx_symbol}\n"
        f"- Pre/After: {CFG.es_symbol} (adjust -{CFG.es_points_adjust:.0f})\n"
        f"- Hourly Update: ON\n"
        f"- Signals: Balanced score ≥ {CFG.score_threshold}"
    )

    while True:
        try:
            evaluate_once()
        except Exception as e:
            # Keep errors local, avoid Telegram spam
            print("[ERROR]", e)
        time.sleep(CFG.loop_sleep_seconds)


if __name__ == "__main__":
    main()
