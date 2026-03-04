# -*- coding: utf-8 -*-
"""
ES Trading Bot (Hunter Smart + Trade Tracking + Daily Stats + Liquidity Sweep)
(per Dr. Mohammed)

Data: Yahoo Finance (yfinance)
Symbol: ES=F always (Pre/Market/After)
Adjustment: -10 points always (match TradingView baseline)

Timeframes:
- 1H + 4H: context/structure + level sourcing
- 15m: tighter level sourcing
- 5m: entries + confirmations + trade tracking updates

Messaging:
- Every message includes: "دكتور محمد"
- Hourly Update: every hour (no entry/targets)
- Signals: Hunter Smart (more entries, still filtered)
- Trade tracking: Entry triggered, 50% progress, T1, T2, Stop (smart confirm), Sweep notice
- Daily stats: trades/wins/losses/winrate/avgRR (sent when day rolls over Riyadh)

Railway env vars:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

requirements.txt:
yfinance
pandas
numpy
requests
ta
tzdata
"""

import os
import time
import math
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
    ZoneInfo = None


# =========================
# Config
# =========================

@dataclass
class Config:
    es_symbol: str = "ES=F"
    es_points_adjust: float = 10.0

    # Structure pivots (context)
    pivot_left: int = 3
    pivot_right: int = 3

    # Key levels
    level_touch_tolerance: float = 0.0013       # 0.13%
    level_cluster_tolerance: float = 0.0010
    max_levels: int = 6

    # Hunter Smart thresholds
    signal_score_threshold: int = 3             # Hunter Smart (more entries)
    min_rr_to_t1: float = 1.6                   # keep decent RR
    signal_cooldown_minutes: int = 20           # avoid spam

    # Trade tracking thresholds
    progress_notify_frac: float = 0.50          # 50% to T1 -> suggest BE
    tracking_poll_seconds: int = 30

    # Smart stop confirmation
    stop_confirm_seconds: int = 120             # wait 2 minutes to confirm stop if still beyond
    stop_confirm_5m_closes: int = 1             # require 1 full 5m close beyond stop (plus stoch confirm)
    stop_stoch_confirm: bool = True             # require Stoch RSI to agree when confirming stop

    # Liquidity sweep detection (stop hunt)
    sweep_buffer_points: float = 2.0            # how far beyond stop counts as sweep attempt
    sweep_recover_seconds: int = 180            # must recover back inside stop within 3 min

    # Hourly update
    hourly_update: bool = True

    # Loop
    loop_sleep_seconds: int = 30

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
# TZ helpers
# =========================

def tzinfo(name: str):
    if ZoneInfo is None:
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
        print("[WARN] Telegram env vars not set. Printing:\n", text)
        return

    url = f"https://api.telegram.org/bot{CFG.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": CFG.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.ok:
                return
            print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", e)
        time.sleep(1.5 * (attempt + 1))


# =========================
# Yahoo fetching (with retry)
# =========================

def _yf_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=True
            )
            if df is None or df.empty:
                raise RuntimeError(f"Yahoo empty data {symbol} interval={interval} period={period}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            for c in ["open", "high", "low", "close"]:
                if c not in df.columns:
                    raise RuntimeError(f"Missing column '{c}'")

            if df.index.tz is None:
                df.index = df.index.tz_localize(timezone.utc)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Yahoo download failed after retries: {last_err}")

def apply_es_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    adj = CFG.es_points_adjust
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) - adj
    return out

def fetch_timeframes():
    sym = CFG.es_symbol
    df_5m  = apply_es_adjustment(_yf_download(sym, "5m",  "7d"))
    df_15m = apply_es_adjustment(_yf_download(sym, "15m", "30d"))
    df_1h  = apply_es_adjustment(_yf_download(sym, "60m", "60d"))

    agg = {"open":"first","high":"max","low":"min","close":"last"}
    if "volume" in df_1h.columns:
        agg["volume"] = "sum"
    df_4h = df_1h.resample("4h").agg(agg).dropna()
    return sym, df_4h, df_1h, df_15m, df_5m


# =========================
# Structure / pivots (context only)
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
    highs = [float(df["high"].iloc[i]) for i in hi_idx][-3:]
    lows  = [float(df["low"].iloc[i])  for i in lo_idx][-3:]
    return highs, lows

def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    def bias_from(df):
        highs, lows = last_two_swings(df, CFG.pivot_left, CFG.pivot_right)
        if len(highs) < 2 or len(lows) < 2:
            return "Weak"
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]
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
# Key levels
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

    hi_idx, lo_idx = find_pivots(df_1h["high"], CFG.pivot_left, CFG.pivot_right)
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx[-10:]] if hi_idx else []
    swing_lows  = [float(df_1h["low"].iloc[i])  for i in lo_idx[-10:]] if lo_idx else []

    recent = df_15m.tail(200)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    prev = df_15m.tail(96*2).copy()
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

    clustered = sorted(clustered, key=lambda x: abs(x - price))
    extremes = [min(clustered), max(clustered)] if clustered else []
    merged = cluster_levels(clustered + extremes, CFG.level_cluster_tolerance, price)
    merged = sorted(merged)

    if len(merged) > CFG.max_levels:
        around = sorted(merged, key=lambda x: abs(x - price))[:CFG.max_levels-2]
        merged = sorted(cluster_levels(around + [min(merged), max(merged)], CFG.level_cluster_tolerance, price))
    return merged

def near_level(price: float, level: float, tol_frac: float) -> bool:
    return abs(price - level) / price <= tol_frac

def fmt_levels(levels: list[float]) -> str:
    return ", ".join([f"{x:.1f}" for x in levels])


# =========================
# Indicators (5m)
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

def stoch_cross(df_5m: pd.DataFrame, direction: str) -> bool:
    k = df_5m["stochrsi_k"].tail(3).values
    d = df_5m["stochrsi_d"].tail(3).values
    if len(k) < 3 or np.isnan(k).any() or np.isnan(d).any():
        return False
    prev = k[-2] - d[-2]
    curr = k[-1] - d[-1]
    if direction == "BUY":
        return prev < 0 and curr > 0 and np.min(k) < 0.25
    if direction == "SELL":
        return prev > 0 and curr < 0 and np.max(k) > 0.75
    return False

def rsi_turn(df_5m: pd.DataFrame, direction: str) -> bool:
    r = df_5m["rsi"].tail(6).values
    if len(r) < 3 or np.isnan(r).any():
        return False
    if direction == "BUY":
        return (np.min(r) < 35) and (r[-1] > r[-2] > r[-3])
    if direction == "SELL":
        return (np.max(r) > 65) and (r[-1] < r[-2] < r[-3])
    return False

def momentum_shift(df_5m: pd.DataFrame, direction: str) -> bool:
    if len(df_5m) < 30:
        return False
    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema20"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    hist_prev = float(df_5m["macd_hist"].iloc[-2])
    if direction == "BUY":
        return (c > ema) or (hist_prev < 0 and hist > 0)
    if direction == "SELL":
        return (c < ema) or (hist_prev > 0 and hist < 0)
    return False

def rejection_lite(df_5m: pd.DataFrame, direction: str) -> bool:
    c = df_5m.iloc[-1]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    # lite: allow 25% wick, not strict close
    if direction == "BUY":
        return (lower / rng >= 0.25) and (upper / rng <= 0.70)
    if direction == "SELL":
        return (upper / rng >= 0.25) and (lower / rng <= 0.70)
    return False

def break_retest(df_5m: pd.DataFrame, level: float, direction: str, tol_frac: float) -> bool:
    price = float(df_5m["close"].iloc[-1])
    window = df_5m.tail(8)
    if direction == "BUY":
        broke = (window["close"] > level).any()
        retest = near_level(price, level, tol_frac) or near_level(float(window["low"].min()), level, tol_frac)
        return bool(broke and retest and price >= level)
    if direction == "SELL":
        broke = (window["close"] < level).any()
        retest = near_level(price, level, tol_frac) or near_level(float(window["high"].max()), level, tol_frac)
        return bool(broke and retest and price <= level)
    return False

def liquidity_sweep(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    """
    Sweep idea (simple):
    - For BUY: wick below level but close back above level
    - For SELL: wick above level but close back below level
    """
    c = df_5m.iloc[-1]
    h, l, cl = float(c["high"]), float(c["low"]), float(c["close"])
    tol = max(level * CFG.level_touch_tolerance, 1.0)

    if direction == "BUY":
        return (l < level - tol) and (cl > level)
    if direction == "SELL":
        return (h > level + tol) and (cl < level)
    return False


# =========================
# Trade plan
# =========================

def pick_targets(levels: list[float], entry: float, direction: str):
    if not levels:
        return None, None
    if direction == "BUY":
        above = sorted([lvl for lvl in levels if lvl > entry])
        return (above[0] if len(above) >= 1 else None,
                above[1] if len(above) >= 2 else None)
    if direction == "SELL":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        return (below[0] if len(below) >= 1 else None,
                below[1] if len(below) >= 2 else None)
    return None, None

def compute_trade_plan(df_5m: pd.DataFrame, levels: list[float], level_hit: float, direction: str, trigger: str):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    buffer = max(price * 0.0002, 0.5)

    if trigger in ("Rejection", "Sweep"):
        if direction == "BUY":
            entry = float(last["high"] + buffer)
            stop  = float(min(last["low"], level_hit) - buffer)
        else:
            entry = float(last["low"] - buffer)
            stop  = float(max(last["high"], level_hit) + buffer)
    else:
        # Break&Retest
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer))
            stop  = float(level_hit - (price * 0.0015) - buffer)
        else:
            entry = float(min(price, level_hit - buffer))
            stop  = float(level_hit + (price * 0.0015) + buffer)

    t1, t2 = pick_targets(levels, entry, direction)

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = (reward / risk) if risk > 0 else None

    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "rr": rr}


# =========================
# State: cooldown, trade tracking, daily stats
# =========================

class BotState:
    def __init__(self):
        self.last_hour_sent = None
        self.last_signal_ts = {}     # per direction+level
        self.active_trade = None     # dict
        self.last_day = now_riyadh().date()

        # daily stats
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.daily_rr_sum = 0.0
        self.daily_rr_count = 0

    def can_signal(self, direction: str, level: float) -> bool:
        key = f"{direction}:{round(level, 1)}"
        last = self.last_signal_ts.get(key)
        if last is None:
            return True
        return (datetime.utcnow() - last) >= timedelta(minutes=CFG.signal_cooldown_minutes)

    def mark_signal(self, direction: str, level: float):
        key = f"{direction}:{round(level, 1)}"
        self.last_signal_ts[key] = datetime.utcnow()

    def roll_day_if_needed(self):
        today = now_riyadh().date()
        if today != self.last_day:
            # send report for previous day
            self.send_daily_report(self.last_day)
            # reset
            self.last_day = today
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self.daily_rr_sum = 0.0
            self.daily_rr_count = 0

    def send_daily_report(self, day_date):
        trades = self.daily_trades
        wins = self.daily_wins
        losses = self.daily_losses
        winrate = (wins / trades * 100.0) if trades > 0 else 0.0
        avg_rr = (self.daily_rr_sum / self.daily_rr_count) if self.daily_rr_count > 0 else 0.0

        msg = (
            f"📊 {CFG.user_title} — تقرير اليوم ({day_date})\n"
            f"Trades: {trades}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Winrate: {winrate:.0f}%\n"
            f"Avg RR (to T1): {avg_rr:.2f}"
        )
        send_telegram(msg)

STATE = BotState()


# =========================
# Messages
# =========================

def hourly_msg(session: str, symbol: str, bias: str, levels: list[float], price: float) -> str:
    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): <b>{bias}</b>\n"
        f"💵 Price: {price:.1f}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )

def signal_msg(session: str, symbol: str, bias: str, direction: str, level_hit: float, score: int, reasons: list[str], plan: dict) -> str:
    rr_txt = f"{plan['rr']:.2f}" if plan["rr"] is not None else "N/A"
    t1_txt = f"{plan['t1']:.1f}" if plan["t1"] is not None else "N/A"
    t2_txt = f"{plan['t2']:.1f}" if plan["t2"] is not None else "N/A"
    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"📍 Direction: <b>{direction}</b>\n"
        f"🧭 Context Bias (1H/4H): <b>{bias}</b>\n"
        f"🧱 Level: {level_hit:.1f}\n\n"
        f"✅ Entry: <b>{plan['entry']:.1f}</b>\n"
        f"🛑 Stop: <b>{plan['stop']:.1f}</b>\n"
        f"🎯 Target 1: <b>{t1_txt}</b>\n"
        f"🎯 Target 2: <b>{t2_txt}</b>\n"
        f"📐 RR (to T1): <b>{rr_txt}</b>\n\n"
        f"⭐ Score: <b>{score}/6</b>\n"
        f"🧠 Reason: {', '.join(reasons)}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )

def trade_update_msg(title: str, body: str) -> str:
    return f"{title}\n{body}\n🧾 {CFG.user_title}"

def _fmt_price(x):
    return "N/A" if x is None else f"{x:.1f}"


# =========================
# Hunter Smart scoring (trend is BONUS only)
# =========================

def score_setup(df_5m: pd.DataFrame, bias: str, level_hit: float, direction: str) -> tuple[int, list[str], str]:
    """
    Score out of 6 (Hunter Smart):
    - Level context is assumed (we only call when near a level)
    - Rejection-lite: +2
    - Break&Retest: +2
    - Liquidity sweep: +2
    - Stoch cross: +1
    - RSI turn: +1
    - Momentum shift: +1
    - Bias alignment: +1 (bonus, but cap total to 6)
    We then normalize/cap to 6 and decide by threshold>=3.
    """
    score = 0
    reasons = []
    trigger = None

    # Price action core (choose best trigger)
    sw = liquidity_sweep(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction)
    br = break_retest(df_5m, level_hit, direction, CFG.level_touch_tolerance)

    # prioritize sweep > br > rejection for trigger naming
    if sw:
        score += 2
        reasons.append("Liquidity Sweep")
        trigger = "Sweep"
    if br:
        score += 2
        reasons.append("Break&Retest")
        trigger = trigger or "Break&Retest"
    if rj:
        score += 2
        reasons.append("Rejection")
        trigger = trigger or "Rejection"

    # Confirmations
    if stoch_cross(df_5m, direction):
        score += 1
        reasons.append("Stoch RSI cross")
    if rsi_turn(df_5m, direction):
        score += 1
        reasons.append("RSI turn")
    if momentum_shift(df_5m, direction):
        score += 1
        reasons.append("Momentum shift")

    # Bias = bonus only
    if (bias == "Bullish" and direction == "BUY") or (bias == "Bearish" and direction == "SELL"):
        score += 1
        reasons.append("Trend bonus")

    # Cap score to 6 (so message stays clean)
    score = min(score, 6)

    # Must have at least ONE core price-action reason to trade
    if trigger is None:
        return 0, ["No price-action trigger"], "None"

    return score, reasons, trigger


# =========================
# Trade tracking (smart stop + sweep handling)
# =========================

def _trade_direction_from_bias(bias: str) -> str:
    # Hunter Smart: allow both, but choose default direction from bias if not clear later
    if bias == "Bullish":
        return "BUY"
    if bias == "Bearish":
        return "SELL"
    # Weak/Range: we decide by micro-move direction via last 8 candles
    return None

def _guess_direction_from_micro(df_5m: pd.DataFrame) -> str | None:
    if len(df_5m) < 9:
        return None
    w = df_5m.tail(9)
    start = float(w["close"].iloc[0])
    end = float(w["close"].iloc[-1])
    move = (end - start) / max(start, 1e-9)
    if move >= 0.002:   # +0.2%
        return "BUY"
    if move <= -0.002:  # -0.2%
        return "SELL"
    return None

def update_active_trade(df_5m: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if not tr:
        return

    direction = tr["direction"]
    entry = tr["entry"]
    stop = tr["stop"]
    t1 = tr["t1"]
    t2 = tr["t2"]

    # Entry trigger
    if tr["status"] == "pending":
        triggered = (last_price >= entry) if direction == "BUY" else (last_price <= entry)
        if triggered:
            tr["status"] = "live"
            tr["entry_time"] = datetime.utcnow()
            send_telegram(
                f"📍 {CFG.user_title} — الصفقة تفعلت\n"
                f"Direction: {direction}\n"
                f"Entry Triggered @ {last_price:.1f}\n"
                f"Stop: {_fmt_price(stop)}\n"
                f"Target1: {_fmt_price(t1)} | Target2: {_fmt_price(t2)}"
            )
        return

    # Live trade management
    if tr["status"] != "live":
        return

    # --- Smart Stop Logic ---
    breached_stop = (last_price <= stop) if direction == "BUY" else (last_price >= stop)
    if breached_stop and not tr.get("stop_pending"):
        tr["stop_pending"] = True
        tr["stop_first_breach_ts"] = datetime.utcnow()
        tr["stop_breach_price"] = last_price
        send_telegram(
            f"⚠️ {CFG.user_title} — اختبار وقف الخسارة\n"
            f"Price لمس الوقف: {last_price:.1f} | Stop: {stop:.1f}\n"
            f"جاري التأكد (Stop-hunt محتمل)"
        )

    if tr.get("stop_pending"):
        # If recovered back inside stop quickly => sweep
        inside = (last_price > stop) if direction == "BUY" else (last_price < stop)
        if inside:
            dt = (datetime.utcnow() - tr["stop_first_breach_ts"]).total_seconds()
            if dt <= CFG.sweep_recover_seconds and abs(tr["stop_breach_price"] - stop) <= (CFG.sweep_buffer_points + 10.0):
                # Sweep recovery
                send_telegram(
                    f"🔥 {CFG.user_title} — Liquidity Sweep\n"
                    f"السوق لمس الوقف ورجع داخل بسرعة ✅\n"
                    f"Trade ما زالت صالحة (يدويًا راقب)"
                )
            tr["stop_pending"] = False
            tr.pop("stop_first_breach_ts", None)
            tr.pop("stop_breach_price", None)

        else:
            # Still beyond stop, confirm with time + stoch/close logic
            dt = (datetime.utcnow() - tr["stop_first_breach_ts"]).total_seconds()

            # Require 1 full 5m close beyond stop
            last_close = float(df_5m["close"].iloc[-1])
            close_beyond = (last_close <= stop) if direction == "BUY" else (last_close >= stop)

            stoch_ok = True
            if CFG.stop_stoch_confirm:
                # If BUY trade (stop below) -> confirm bearish stoch cross (down); for SELL trade -> confirm bullish stoch cross (up)
                stoch_ok = stoch_cross(df_5m, "SELL" if direction == "BUY" else "BUY")

            if (dt >= CFG.stop_confirm_seconds and close_beyond and stoch_ok):
                # Confirm stop hit
                tr["status"] = "closed_stop"
                STATE.daily_trades += 1
                STATE.daily_losses += 1
                if tr.get("rr_to_t1") is not None:
                    STATE.daily_rr_sum += float(tr["rr_to_t1"])
                    STATE.daily_rr_count += 1

                send_telegram(
                    f"❌ {CFG.user_title} — وقف الخسارة تأكد\n"
                    f"Stop Hit ✅\n"
                    f"Direction: {direction}\n"
                    f"Stop: {stop:.1f} | Price: {last_price:.1f}\n"
                    f"ملاحظة: تأكدنا لتفادي الكذب (Stop-hunt)"
                )
                STATE.active_trade = None
                return

    # Target hits
    if t1 is not None and not tr.get("t1_hit"):
        hit_t1 = (last_price >= t1) if direction == "BUY" else (last_price <= t1)
        if hit_t1:
            tr["t1_hit"] = True
            send_telegram(
                f"🎯 {CFG.user_title} — Target 1 Hit\n"
                f"T1: {t1:.1f}\n"
                f"اقتراح: خذ جزء + حرّك الستوب إلى Entry"
            )

    if t2 is not None and not tr.get("t2_hit"):
        hit_t2 = (last_price >= t2) if direction == "BUY" else (last_price <= t2)
        if hit_t2:
            tr["t2_hit"] = True
            tr["status"] = "closed_t2"

            STATE.daily_trades += 1
            STATE.daily_wins += 1
            if tr.get("rr_to_t1") is not None:
                STATE.daily_rr_sum += float(tr["rr_to_t1"])
                STATE.daily_rr_count += 1

            send_telegram(
                f"🏆 {CFG.user_title} — Target 2 Hit\n"
                f"T2: {t2:.1f}\n"
                f"Trade completed ✅"
            )
            STATE.active_trade = None
            return

    # 50% progress notify toward T1 (suggest BE)
    if t1 is not None and not tr.get("progress_sent"):
        if direction == "BUY":
            dist_total = t1 - entry
            dist_now = last_price - entry
        else:
            dist_total = entry - t1
            dist_now = entry - last_price
        if dist_total > 0 and (dist_now / dist_total) >= CFG.progress_notify_frac:
            tr["progress_sent"] = True
            send_telegram(
                f"⚡ {CFG.user_title} — تحديث الصفقة\n"
                f"تحركنا ~{int(CFG.progress_notify_frac*100)}% نحو T1\n"
                f"اقتراح: نقل الستوب إلى Entry (BE) حسب إدارتك"
            )


# =========================
# Main evaluation
# =========================

def evaluate_once():
    STATE.roll_day_if_needed()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m_raw = fetch_timeframes()
    df_5m = compute_indicators(df_5m_raw)

    price = float(df_5m["close"].iloc[-1])  # use 5m last for tracking responsiveness
    bias = structure_bias(df_1h, df_4h)
    levels = extract_key_levels(df_15m, df_1h)

    # Hourly update
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_msg(session, symbol, bias, levels, price))
            STATE.last_hour_sent = current_hour

    # Trade tracking (always)
    update_active_trade(df_5m, price)

    # If there is an active trade, do not open another one
    if STATE.active_trade is not None:
        return

    # Must be near some level to even consider a trade
    if not levels:
        return
    nearby = [lvl for lvl in levels if near_level(price, lvl, CFG.level_touch_tolerance)]
    if not nearby:
        return
    level_hit = min(nearby, key=lambda x: abs(x - price))

    # Choose direction (Hunter Smart)
    direction = _trade_direction_from_bias(bias)
    if direction is None:
        direction = _guess_direction_from_micro(df_5m) or "SELL"  # default to sell if truly unclear

    # Score setup (trend is only bonus)
    score, reasons, trigger = score_setup(df_5m, bias, level_hit, direction)
    if score < CFG.signal_score_threshold:
        return

    # Cooldown per direction+level
    if not STATE.can_signal(direction, level_hit):
        return

    # Build plan
    plan = compute_trade_plan(df_5m, levels, level_hit, direction, trigger=trigger)
    if plan["rr"] is not None and plan["rr"] < CFG.min_rr_to_t1:
        return

    # Send signal
    send_telegram(signal_msg(session, symbol, bias, direction, level_hit, score, reasons, plan))
    STATE.mark_signal(direction, level_hit)

    # Create active trade to track (Hunter Smart)
    STATE.active_trade = {
        "direction": direction,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "t1": None if plan["t1"] is None else float(plan["t1"]),
        "t2": None if plan["t2"] is None else float(plan["t2"]),
        "rr_to_t1": None if plan["rr"] is None else float(plan["rr"]),
        "status": "pending",          # pending -> live -> closed
        "created_ts": datetime.utcnow()
    }


def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- All Sessions: {CFG.es_symbol} (adjust -{CFG.es_points_adjust:.0f})\n"
        f"- Hourly Update: ON\n"
        f"- Signals: Hunter Smart (Score ≥ {CFG.signal_score_threshold}, RR ≥ {CFG.min_rr_to_t1})\n"
        f"- Trade Tracking: ON (Entry/T1/T2/Stop smart confirm)\n"
        f"- Daily Stats: ON"
    )

    while True:
        try:
            evaluate_once()
        except Exception as e:
            # keep it quiet but informative
            send_telegram(f"❌ {CFG.user_title}: خطأ - {repr(e)}")
            print("[ERROR]", repr(e))
        time.sleep(CFG.loop_sleep_seconds)


if __name__ == "__main__":
    main()
