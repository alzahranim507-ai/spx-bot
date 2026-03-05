# -*- coding: utf-8 -*-
"""
ES Trading Bot — Yahoo Finance (yfinance)
(per Dr. Mohammed)

Core rules:
- Symbol ALWAYS: ES=F (Pre / Market / After)
- Always adjust ES by -10 pts (match TradingView baseline)
- Hourly Update: ON (no entry/targets inside hourly update)
- Signals: Hunter Smart + Smarter Liquidity Sweep
- Trade tracking: ON (entry trigger / stop confirm / targets / progress)
- Daily Reset: 00:00 Riyadh, close any active trade, then No-Signal window (10m)

New additions (as requested):
- Market State: Trending/Range/Weak + Direction + ADX(1H), updated every 15 minutes
- Liquidity state (Low/Normal/High/Extreme) using ES volume ratio
- Level Now (current price at signal time)
- Confidence % (score-based + small contextual nudges)
- Probability to hit T1 and T2 (heuristic, bounded; not a promise)
- Expected Move (1H) via ATR(1H)
- ETA to T1 via velocity (recent 5m movement) + liquidity adjustment

Railway env vars:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

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

    # pivots
    pivot_left: int = 3
    pivot_right: int = 3

    # key levels
    level_touch_tolerance: float = 0.0013
    level_cluster_tolerance: float = 0.0010
    max_levels: int = 6

    # signals / scoring
    signal_score_threshold: int = 3
    min_rr_to_t1: float = 1.6
    signal_cooldown_minutes: int = 20

    # liquidity sweep
    sweep_close_required: bool = True
    sweep_wick_min_points: float = 1.5
    sweep_stoch_confirm: bool = True
    sweep_priority: bool = True

    # trade tracking
    progress_notify_frac: float = 0.50
    stop_confirm_seconds: int = 120
    stop_stoch_confirm: bool = True

    # daily reset (Riyadh midnight)
    daily_reset_enabled: bool = True
    daily_reset_hour: int = 0
    daily_reset_minute: int = 0
    daily_reset_window_minutes: int = 5
    no_signal_after_reset_minutes: int = 10

    # market state
    market_state_update_minutes: int = 15
    adx_window: int = 14
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # liquidity calc
    liquidity_lookback_5m: int = 60  # last 60 candles = 5 hours
    liquidity_state_thresholds: tuple = (0.60, 1.25, 2.00)  # low, normal, high, extreme via vol_ratio

    # expected move (ATR)
    expected_move_atr_window: int = 14

    # ETA
    eta_velocity_lookback_5m: int = 24  # last 24 candles = 2 hours
    eta_min_velocity_pts_per_min: float = 0.15  # guardrail

    # hourly update
    hourly_update: bool = True

    # loop
    loop_sleep_seconds: int = 30

    # telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # identity
    user_title: str = "دكتور محمد"

    # timezones
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
        print("[WARN] Telegram env vars not set. Printing message:\n", text)
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
# Yahoo fetching (retry)
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
                raise RuntimeError(f"Yahoo empty data: {symbol} interval={interval} period={period}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            for c in ["open", "high", "low", "close"]:
                if c not in df.columns:
                    raise RuntimeError(f"Missing column '{c}' in Yahoo data")
            if "volume" not in df.columns:
                # some feeds may omit volume; we handle gracefully
                df["volume"] = np.nan

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
    df_1h  = apply_es_adjustment(_yf_download(sym, "60m", "90d"))

    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    df_4h = df_1h.resample("4h").agg(agg).dropna(subset=["open","high","low","close"])
    return sym, df_4h, df_1h, df_15m, df_5m


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

def last_swings_1h(df_1h: pd.DataFrame, max_points: int = 6):
    hi_idx, lo_idx = find_pivots(df_1h["high"], CFG.pivot_left, CFG.pivot_right)
    highs = [float(df_1h["high"].iloc[i]) for i in hi_idx][-max_points:] if hi_idx else []
    lows  = [float(df_1h["low"].iloc[i])  for i in lo_idx][-max_points:] if lo_idx else []
    return highs, lows

def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    def bias_from(df):
        hi_idx, lo_idx = find_pivots(df["high"], CFG.pivot_left, CFG.pivot_right)
        highs = [float(df["high"].iloc[i]) for i in hi_idx][-2:]
        lows  = [float(df["low"].iloc[i])  for i in lo_idx][-2:]
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
        if abs(lvl - clustered[-1]) / max(price_ref, 1e-9) <= tol_frac:
            clustered[-1] = (clustered[-1] + lvl) / 2.0
        else:
            clustered.append(lvl)
    return clustered

def extract_key_levels(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> list[float]:
    price = float(df_15m["close"].iloc[-1])

    swing_highs, swing_lows = last_swings_1h(df_1h, max_points=10)

    recent = df_15m.tail(200)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    candidates = []
    candidates += swing_highs + swing_lows
    candidates += [range_hi, range_lo]

    candidates = [float(x) for x in candidates if np.isfinite(x)]
    merged = cluster_levels(candidates, CFG.level_cluster_tolerance, price)

    merged = sorted(merged, key=lambda x: abs(x - price))
    merged = merged[: max(CFG.max_levels * 2, 12)]
    merged = sorted(cluster_levels(merged, CFG.level_cluster_tolerance, price))

    if len(merged) > CFG.max_levels:
        around = sorted(merged, key=lambda x: abs(x - price))[:CFG.max_levels-2]
        merged = sorted(cluster_levels(around + [min(merged), max(merged)], CFG.level_cluster_tolerance, price))
    return merged

def fmt_levels(levels: list[float]) -> str:
    return ", ".join([f"{x:.1f}" for x in levels])

def near_level(price: float, level: float, tol_frac: float) -> bool:
    return abs(price - level) / max(price, 1e-9) <= tol_frac


# =========================
# Indicators (5m)
# =========================

def compute_indicators_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
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
        return prev < 0 and curr > 0 and np.nanmin(k) < 0.25
    if direction == "SELL":
        return prev > 0 and curr < 0 and np.nanmax(k) > 0.75
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
    if direction == "BUY":
        return (lower / rng >= 0.25)
    if direction == "SELL":
        return (upper / rng >= 0.25)
    return False

def break_retest(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    price = float(df_5m["close"].iloc[-1])
    window = df_5m.tail(8)
    if direction == "BUY":
        broke = (window["close"] > level).any()
        retest = near_level(price, level, CFG.level_touch_tolerance) or near_level(float(window["low"].min()), level, CFG.level_touch_tolerance)
        return bool(broke and retest and price >= level)
    if direction == "SELL":
        broke = (window["close"] < level).any()
        retest = near_level(price, level, CFG.level_touch_tolerance) or near_level(float(window["high"].max()), level, CFG.level_touch_tolerance)
        return bool(broke and retest and price <= level)
    return False


# =========================
# Smarter Liquidity Sweep
# =========================

def compute_sweep_levels(df_1h: pd.DataFrame, df_15m: pd.DataFrame, key_levels: list[float]) -> list[float]:
    swings_hi, swings_lo = last_swings_1h(df_1h, max_points=6)
    last_24h = df_15m.tail(96)
    hi_24 = float(last_24h["high"].max())
    lo_24 = float(last_24h["low"].min())

    price = float(df_15m["close"].iloc[-1])
    candidates = []
    candidates += swings_hi + swings_lo
    candidates += [hi_24, lo_24]
    candidates += key_levels

    candidates = [float(x) for x in candidates if np.isfinite(x)]
    clustered = cluster_levels(sorted(candidates), CFG.level_cluster_tolerance, price)
    return clustered

def detect_sweep(df_5m: pd.DataFrame, level: float) -> tuple[bool, str]:
    c = df_5m.iloc[-1]
    h, l, cl = float(c["high"]), float(c["low"]), float(c["close"])
    min_wick = CFG.sweep_wick_min_points

    # Sweep Up -> SELL
    if h >= level + min_wick:
        if (not CFG.sweep_close_required) or (cl <= level):
            return True, "SELL"

    # Sweep Down -> BUY
    if l <= level - min_wick:
        if (not CFG.sweep_close_required) or (cl >= level):
            return True, "BUY"

    return False, ""


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
# Bot state
# =========================

class BotState:
    def __init__(self):
        self.last_hour_sent = None
        self.last_signal_ts = {}
        self.active_trade = None

        self.last_reset_date_riyadh = None
        self.no_signal_until_utc = None

        self.market_state_last_calc_utc = None
        self.market_state_label = "Unknown"
        self.market_state_dir = "Neutral"
        self.market_state_adx = None

    def can_signal(self, key: str) -> bool:
        if self.no_signal_until_utc is not None and datetime.utcnow() < self.no_signal_until_utc:
            return False
        last = self.last_signal_ts.get(key)
        if last is None:
            return True
        return (datetime.utcnow() - last) >= timedelta(minutes=CFG.signal_cooldown_minutes)

    def mark_signal(self, key: str):
        self.last_signal_ts[key] = datetime.utcnow()

STATE = BotState()


# =========================
# Daily reset (Riyadh midnight)
# =========================

def maybe_daily_reset():
    if not CFG.daily_reset_enabled:
        return

    t = now_riyadh()
    today = t.date()

    if STATE.last_reset_date_riyadh == today:
        return

    is_midnight_window = (
        t.hour == CFG.daily_reset_hour and
        CFG.daily_reset_minute <= t.minute < (CFG.daily_reset_minute + CFG.daily_reset_window_minutes)
    )
    if not is_midnight_window:
        return

    if STATE.active_trade is not None:
        send_telegram(
            f"🛎️ {CFG.user_title} — Daily Reset (Riyadh)\n"
            f"السبب: منتصف الليل بتوقيت السعودية.\n"
            f"تم إغلاق الصفقة لتفادي التعليق وبداية يوم جديد."
        )
    else:
        send_telegram(
            f"🛎️ {CFG.user_title} — Daily Reset (Riyadh)\n"
            f"السبب: منتصف الليل بتوقيت السعودية.\n"
            f"لا توجد صفقة فعّالة — تم بدء يوم جديد."
        )

    STATE.active_trade = None
    STATE.last_reset_date_riyadh = today
    STATE.no_signal_until_utc = datetime.utcnow() + timedelta(minutes=CFG.no_signal_after_reset_minutes)


# =========================
# Market State (fast + stable)
# =========================

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    if len(df_1h) < (CFG.adx_window + 5):
        return "Weak", "Neutral", None

    adx = ADXIndicator(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], window=CFG.adx_window).adx()
    adx_val = float(adx.iloc[-1]) if adx is not None and len(adx) else None

    bias = structure_bias(df_1h, df_4h)
    direction = "Neutral"
    if bias == "Bullish":
        direction = "Bullish"
    elif bias == "Bearish":
        direction = "Bearish"

    prev = STATE.market_state_label

    if adx_val is None or np.isnan(adx_val):
        return "Weak", direction, None

    # hysteresis
    if adx_val >= CFG.adx_trending_on:
        label = "Trending"
    elif adx_val <= CFG.adx_range_on:
        label = "Range"
    else:
        if prev == "Trending" and adx_val > CFG.adx_range_on:
            label = "Trending"
        elif prev == "Range" and adx_val < CFG.adx_trending_on:
            label = "Range"
        else:
            label = "Weak"

    return label, direction, adx_val

def maybe_update_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    nowu = datetime.utcnow()
    if STATE.market_state_last_calc_utc is None:
        should = True
    else:
        should = (nowu - STATE.market_state_last_calc_utc) >= timedelta(minutes=CFG.market_state_update_minutes)

    if not should:
        return

    label, direction, adx_val = compute_market_state(df_1h, df_4h)
    STATE.market_state_label = label
    STATE.market_state_dir = direction
    STATE.market_state_adx = adx_val
    STATE.market_state_last_calc_utc = nowu


# =========================
# Liquidity + Expected Move + ETA + Probability
# =========================

def liquidity_state(df_5m_raw: pd.DataFrame) -> tuple[str, float | None]:
    """
    Uses volume ratio = current_volume / avg_volume(lookback).
    Returns (state, ratio). If volume missing -> ("Normal", None).
    """
    if "volume" not in df_5m_raw.columns:
        return "Normal", None

    v = df_5m_raw["volume"].astype(float)
    if v.isna().all():
        return "Normal", None

    look = min(CFG.liquidity_lookback_5m, len(v))
    if look < 10:
        return "Normal", None

    avg = float(v.tail(look).mean())
    cur = float(v.iloc[-1])
    if not np.isfinite(avg) or avg <= 0 or not np.isfinite(cur):
        return "Normal", None

    ratio = cur / avg
    t_low, t_high, t_ext = CFG.liquidity_state_thresholds
    if ratio < t_low:
        return "Low", ratio
    if ratio < t_high:
        return "Normal", ratio
    if ratio < t_ext:
        return "High", ratio
    return "Extreme", ratio

def expected_move_1h(df_1h: pd.DataFrame) -> float | None:
    """
    Expected Move (1H) ≈ ATR(1H)
    """
    if len(df_1h) < (CFG.expected_move_atr_window + 5):
        return None
    atr = AverageTrueRange(
        high=df_1h["high"], low=df_1h["low"], close=df_1h["close"],
        window=CFG.expected_move_atr_window
    ).average_true_range()
    val = float(atr.iloc[-1])
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def eta_to_t1_minutes(df_5m: pd.DataFrame, price_now: float, t1: float, direction: str, liq_state: str) -> tuple[int, int] | None:
    """
    ETA based on velocity of recent 5m price changes.
    Return (min_eta, max_eta) in minutes.
    """
    if t1 is None or not np.isfinite(t1):
        return None

    look = min(CFG.eta_velocity_lookback_5m, len(df_5m))
    if look < 8:
        return None

    closes = df_5m["close"].tail(look).astype(float)
    diffs = closes.diff().abs().dropna()
    if diffs.empty:
        return None

    # velocity: points per minute (5m candles => each diff roughly per 5 min)
    avg_abs_move_per_5m = float(diffs.mean())
    vel = max(avg_abs_move_per_5m / 5.0, CFG.eta_min_velocity_pts_per_min)

    dist = (t1 - price_now) if direction == "BUY" else (price_now - t1)
    dist = abs(float(dist))
    if dist < 0.1:
        return (1, 5)

    base = dist / vel  # minutes

    # liquidity adjustment
    mult = 1.0
    if liq_state == "Low":
        mult = 1.35
    elif liq_state == "Normal":
        mult = 1.05
    elif liq_state == "High":
        mult = 0.90
    elif liq_state == "Extreme":
        mult = 0.80

    eta = base * mult

    # provide band (±25%)
    lo = int(max(5, round(eta * 0.75)))
    hi = int(max(lo + 5, round(eta * 1.25)))
    return lo, hi

def confidence_percent(score: int, direction: str, bias: str, session: str, market_state_label: str, liq_state: str) -> int:
    """
    Score-based confidence + small nudges. Bounded.
    """
    base = int(round((score / 6) * 100))
    adj = 0

    # with bias bonus
    if (bias == "Bullish" and direction == "BUY") or (bias == "Bearish" and direction == "SELL"):
        adj += 5

    # session / liquidity nudges
    if session in ("After-Hours", "Pre-Market"):
        adj -= 5

    if liq_state == "High":
        adj += 3
    elif liq_state == "Extreme":
        adj += 5
    elif liq_state == "Low":
        adj -= 5

    # market state nudges (soft)
    if market_state_label == "Trending" and ((bias == "Bullish" and direction == "BUY") or (bias == "Bearish" and direction == "SELL")):
        adj += 3
    if market_state_label == "Range" and bias == "Weak":
        adj += 2

    val = max(5, min(95, base + adj))
    return int(val)

def probability_t1_t2(conf: int, rr: float | None, dist_t1: float | None, dist_t2: float | None,
                     exp_move_1h: float | None, liq_state: str, market_state_label: str) -> tuple[int, int]:
    """
    Heuristic probability. Not a guarantee. Bounded 5..95.
    """
    t1 = float(conf)

    # RR penalty (bigger RR means farther target)
    if rr is not None and np.isfinite(rr):
        if rr > 2.5:
            t1 -= 8
        elif rr > 2.0:
            t1 -= 5
        elif rr < 1.7:
            t1 += 2

    # liquidity effect
    if liq_state == "Low":
        t1 -= 5
    elif liq_state == "High":
        t1 += 2
    elif liq_state == "Extreme":
        t1 += 3

    # market state effect (very light)
    if market_state_label == "Range":
        t1 += 2
    elif market_state_label == "Trending":
        t1 += 1

    # expected move sanity
    if exp_move_1h is not None and dist_t1 is not None and np.isfinite(dist_t1):
        if dist_t1 > exp_move_1h * 1.20:
            t1 -= 12
        elif dist_t1 > exp_move_1h * 1.00:
            t1 -= 6

    t1 = max(5, min(95, int(round(t1))))

    # T2 probability scales down
    t2 = t1
    if dist_t1 is not None and dist_t2 is not None and np.isfinite(dist_t1) and np.isfinite(dist_t2) and dist_t2 > 0:
        ratio = max(1.0, dist_t2 / max(dist_t1, 1e-9))
        # farther target => lower prob
        t2 = int(round(t1 * (0.70 / ratio)))
    else:
        t2 = int(round(t1 * 0.65))

    t2 = max(5, min(t1, min(90, t2)))
    return int(t1), int(t2)


# =========================
# Trade tracking (events)
# =========================

def _fmt_level(x):
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:.1f}"
    return str(x)

def update_active_trade(df_5m: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if not tr:
        return

    direction = tr["direction"]
    entry = tr["entry"]
    stop = tr["stop"]
    t1 = tr.get("t1")
    t2 = tr.get("t2")

    # Entry trigger
    if tr["status"] == "pending":
        triggered = (last_price >= entry) if direction == "BUY" else (last_price <= entry)
        if triggered:
            tr["status"] = "live"
            tr["live_since_ts"] = datetime.utcnow()
            send_telegram(
                f"📍 {CFG.user_title} — الصفقة تفعلت\n"
                f"Direction: {direction}\n"
                f"Entry Triggered @ {last_price:.1f}\n"
                f"Stop: {stop:.1f}\n"
                f"T1: {_fmt_level(t1)} | T2: {_fmt_level(t2)}"
            )
        return

    if tr["status"] != "live":
        return

    # Smart stop confirm
    breached = (last_price <= stop) if direction == "BUY" else (last_price >= stop)
    if breached and not tr.get("stop_pending"):
        tr["stop_pending"] = True
        tr["stop_first_breach_ts"] = datetime.utcnow()
        send_telegram(
            f"⚠️ {CFG.user_title} — اختبار وقف الخسارة\n"
            f"Price لمس الوقف: {last_price:.1f} | Stop: {stop:.1f}\n"
            f"جاري التأكد (Stop-hunt محتمل)"
        )

    if tr.get("stop_pending"):
        inside = (last_price > stop) if direction == "BUY" else (last_price < stop)
        if inside:
            tr["stop_pending"] = False
            tr.pop("stop_first_breach_ts", None)
        else:
            dt = (datetime.utcnow() - tr["stop_first_breach_ts"]).total_seconds()
            if dt >= CFG.stop_confirm_seconds:
                stoch_ok = True
                if CFG.stop_stoch_confirm:
                    stoch_ok = stoch_cross(df_5m, "SELL" if direction == "BUY" else "BUY")
                if stoch_ok:
                    send_telegram(
                        f"❌ {CFG.user_title} — وقف الخسارة تأكد\n"
                        f"Stop Hit ✅ | Direction: {direction}\n"
                        f"Stop: {stop:.1f} | Price: {last_price:.1f}"
                    )
                    STATE.active_trade = None
                    return

    # Targets
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
            send_telegram(
                f"🏆 {CFG.user_title} — Target 2 Hit\n"
                f"T2: {t2:.1f}\n"
                f"Trade completed ✅"
            )
            STATE.active_trade = None
            return

    # Progress to T1
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
# Messages
# =========================

def hourly_msg(session: str, symbol: str, bias: str, levels: list[float], price: float) -> str:
    adx_txt = "N/A" if STATE.market_state_adx is None else f"{STATE.market_state_adx:.1f}"
    mkt_txt = f"{STATE.market_state_label} | {STATE.market_state_dir} | ADX(1H): {adx_txt}"

    status = "None" if STATE.active_trade is None else STATE.active_trade.get("status", "Unknown")

    reset_block = ""
    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        left = int((STATE.no_signal_until_utc - datetime.utcnow()).total_seconds() // 60)
        reset_block = f"\n🧊 No-signal window: {left} min (after reset)"

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): <b>{bias}</b>\n"
        f"📈 Market State: <b>{mkt_txt}</b>{reset_block}\n"
        f"💵 Price: {price:.1f}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"📌 Active Trade: <b>{status}</b>\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )

def signal_msg_format(
    session: str,
    symbol: str,
    market_state_str: str,
    liquidity_str: str,
    level_now: float,
    direction: str,
    bias: str,
    level_hit: float,
    plan: dict,
    rr: float | None,
    score: int,
    confidence: int,
    p_t1: int,
    p_t2: int,
    exp_move: float | None,
    eta_band: tuple[int, int] | None,
    reasons: list[str],
) -> str:
    t1_txt = f"{plan['t1']:.1f}" if plan["t1"] is not None else "N/A"
    t2_txt = f"{plan['t2']:.1f}" if plan["t2"] is not None else "N/A"
    rr_txt = f"{rr:.2f}" if rr is not None and np.isfinite(rr) else "N/A"
    exp_txt = f"±{exp_move:.0f} pts" if exp_move is not None and np.isfinite(exp_move) else "N/A"
    eta_txt = "N/A" if eta_band is None else f"{eta_band[0]}–{eta_band[1]} min"

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n\n"
        f"📊 Market State: {market_state_str}\n"
        f"💧 Liquidity: {liquidity_str}\n"
        f"💰 Level Now: {level_now:.1f}\n\n"
        f"📍 Direction: {direction}\n"
        f"🧱 Level: {level_hit:.1f}\n\n"
        f"✅ Entry: {plan['entry']:.1f}\n"
        f"🛑 Stop: {plan['stop']:.1f}\n"
        f"🎯 Target 1: {t1_txt}\n"
        f"🎯 Target 2: {t2_txt}\n\n"
        f"📐 RR: {rr_txt}\n"
        f"⭐ Score: {score}/6\n"
        f"🔎 Confidence: {confidence}%\n\n"
        f"📊 Probability\n"
        f"T1 Hit: {p_t1}%\n"
        f"T2 Hit: {p_t2}%\n\n"
        f"📈 Expected Move (1H): {exp_txt}\n"
        f"⏱ ETA to T1: {eta_txt}\n\n"
        f"🧠 Reason: {', '.join(reasons)}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )


# =========================
# Scoring
# =========================

def score_setup(df_5m: pd.DataFrame, bias: str, level_hit: float, direction: str, is_sweep: bool) -> tuple[int, list[str], str]:
    score = 0
    reasons = []
    trigger = None

    if is_sweep:
        score += 2
        reasons.append("Liquidity Sweep")
        trigger = "Sweep"

    br = break_retest(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction)

    if br:
        score += 2
        reasons.append("Break&Retest")
        trigger = trigger or "Break&Retest"
    if rj:
        score += 2
        reasons.append("Rejection")
        trigger = trigger or "Rejection"

    if stoch_cross(df_5m, direction):
        score += 1
        reasons.append("Stoch RSI cross")
    if momentum_shift(df_5m, direction):
        score += 1
        reasons.append("Momentum shift")

    if (bias == "Bullish" and direction == "BUY") or (bias == "Bearish" and direction == "SELL"):
        score += 1
        reasons.append("Trend bonus")

    score = min(score, 6)

    if trigger is None:
        return 0, ["No trigger"], "None"
    return score, reasons, trigger


# =========================
# Main evaluation
# =========================

def evaluate_once():
    # 1) daily reset first
    maybe_daily_reset()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m_raw = fetch_timeframes()
    df_5m = compute_indicators_5m(df_5m_raw)

    price_now = float(df_5m["close"].iloc[-1])
    bias = structure_bias(df_1h, df_4h)
    key_levels = extract_key_levels(df_15m, df_1h)

    # 2) market state update
    maybe_update_market_state(df_1h, df_4h)
    adx_txt = "N/A" if STATE.market_state_adx is None else f"{STATE.market_state_adx:.1f}"
    market_state_str = f"{STATE.market_state_label} | {STATE.market_state_dir} | ADX(1H): {adx_txt}"

    # 3) liquidity
    liq_state, _liq_ratio = liquidity_state(df_5m_raw)

    # 4) trade tracking
    update_active_trade(df_5m, price_now)

    # 5) hourly update
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_msg(session, symbol, bias, key_levels, price_now))
            STATE.last_hour_sent = current_hour

    # 6) respect no-signal window after reset
    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        return

    # 7) one trade at a time
    if STATE.active_trade is not None:
        return

    # =========================
    # Sweep-first logic
    # =========================
    sweep_levels = compute_sweep_levels(df_1h, df_15m, key_levels)
    sweep_hit = None
    sweep_dir = ""
    for lvl in sorted(sweep_levels, key=lambda x: abs(x - price_now))[:12]:
        is_sw, d = detect_sweep(df_5m, float(lvl))
        if is_sw:
            sweep_hit = float(lvl)
            sweep_dir = d
            break

    if sweep_hit is not None and CFG.sweep_stoch_confirm:
        if not stoch_cross(df_5m, sweep_dir):
            sweep_hit = None
            sweep_dir = ""

    if sweep_hit is not None and CFG.sweep_priority:
        direction = sweep_dir
        level_hit = sweep_hit

        score, reasons, _trigger = score_setup(df_5m, bias, level_hit, direction, is_sweep=True)
        if score >= CFG.signal_score_threshold:
            plan = compute_trade_plan(df_5m, key_levels, level_hit, direction, trigger="Sweep")
            rr = plan["rr"]
            if rr is None or rr >= CFG.min_rr_to_t1:
                key = f"SWEEP:{direction}:{round(level_hit,1)}"
                if STATE.can_signal(key):
                    # ---- metrics ----
                    conf = confidence_percent(score, direction, bias, session, STATE.market_state_label, liq_state)
                    exp = expected_move_1h(df_1h)

                    dist_t1 = None
                    dist_t2 = None
                    if plan["t1"] is not None:
                        dist_t1 = abs((plan["t1"] - price_now) if direction == "BUY" else (price_now - plan["t1"]))
                    if plan["t2"] is not None:
                        dist_t2 = abs((plan["t2"] - price_now) if direction == "BUY" else (price_now - plan["t2"]))

                    p1, p2 = probability_t1_t2(conf, rr, dist_t1, dist_t2, exp, liq_state, STATE.market_state_label)
                    eta = None
                    if plan["t1"] is not None:
                        eta = eta_to_t1_minutes(df_5m, price_now, float(plan["t1"]), direction, liq_state)

                    msg = signal_msg_format(
                        session=session,
                        symbol=symbol,
                        market_state_str=market_state_str,
                        liquidity_str=liq_state,
                        level_now=price_now,
                        direction=direction,
                        bias=bias,
                        level_hit=level_hit,
                        plan=plan,
                        rr=rr,
                        score=score,
                        confidence=conf,
                        p_t1=p1,
                        p_t2=p2,
                        exp_move=exp,
                        eta_band=eta,
                        reasons=reasons,
                    )
                    send_telegram(msg)
                    STATE.mark_signal(key)
                    STATE.active_trade = {
                        "direction": direction,
                        "level": float(level_hit),
                        "entry": float(plan["entry"]),
                        "stop": float(plan["stop"]),
                        "t1": None if plan["t1"] is None else float(plan["t1"]),
                        "t2": None if plan["t2"] is None else float(plan["t2"]),
                        "status": "pending",
                        "created_ts": datetime.utcnow(),
                    }
        return

    # =========================
    # Normal logic: must be near a key level
    # =========================
    if not key_levels:
        return

    nearby = [lvl for lvl in key_levels if near_level(price_now, lvl, CFG.level_touch_tolerance)]
    if not nearby:
        return
    level_hit = float(min(nearby, key=lambda x: abs(x - price_now)))

    # direction preference
    if bias == "Bullish":
        direction = "BUY"
    elif bias == "Bearish":
        direction = "SELL"
    else:
        w = df_5m.tail(9)
        move = float(w["close"].iloc[-1] - w["close"].iloc[0])
        direction = "BUY" if move > 0 else "SELL"

    score, reasons, trigger = score_setup(df_5m, bias, level_hit, direction, is_sweep=False)
    if score < CFG.signal_score_threshold:
        return

    plan = compute_trade_plan(df_5m, key_levels, level_hit, direction, trigger=trigger)
    rr = plan["rr"]
    if rr is not None and rr < CFG.min_rr_to_t1:
        return

    key = f"NORM:{direction}:{round(level_hit,1)}"
    if not STATE.can_signal(key):
        return

    # ---- metrics ----
    conf = confidence_percent(score, direction, bias, session, STATE.market_state_label, liq_state)
    exp = expected_move_1h(df_1h)

    dist_t1 = None
    dist_t2 = None
    if plan["t1"] is not None:
        dist_t1 = abs((plan["t1"] - price_now) if direction == "BUY" else (price_now - plan["t1"]))
    if plan["t2"] is not None:
        dist_t2 = abs((plan["t2"] - price_now) if direction == "BUY" else (price_now - plan["t2"]))

    p1, p2 = probability_t1_t2(conf, rr, dist_t1, dist_t2, exp, liq_state, STATE.market_state_label)

    eta = None
    if plan["t1"] is not None:
        eta = eta_to_t1_minutes(df_5m, price_now, float(plan["t1"]), direction, liq_state)

    msg = signal_msg_format(
        session=session,
        symbol=symbol,
        market_state_str=market_state_str,
        liquidity_str=liq_state,
        level_now=price_now,
        direction=direction,
        bias=bias,
        level_hit=level_hit,
        plan=plan,
        rr=rr,
        score=score,
        confidence=conf,
        p_t1=p1,
        p_t2=p2,
        exp_move=exp,
        eta_band=eta,
        reasons=reasons,
    )
    send_telegram(msg)
    STATE.mark_signal(key)

    STATE.active_trade = {
        "direction": direction,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "t1": None if plan["t1"] is None else float(plan["t1"]),
        "t2": None if plan["t2"] is None else float(plan["t2"]),
        "status": "pending",
        "created_ts": datetime.utcnow(),
    }


def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- All Sessions: {CFG.es_symbol} (adjust -{CFG.es_points_adjust:.0f})\n"
        f"- Hourly Update: ON\n"
        f"- Market State: ADX(1H)+Structure, update {CFG.market_state_update_minutes}m\n"
        f"- Daily Reset: {CFG.daily_reset_hour:02d}:{CFG.daily_reset_minute:02d} Riyadh + No-signal {CFG.no_signal_after_reset_minutes}m\n"
        f"- Signals: Score ≥ {CFG.signal_score_threshold}, RR ≥ {CFG.min_rr_to_t1}\n"
        f"- Extras: Liquidity + Level Now + Confidence + Prob(T1/T2) + Expected Move + ETA"
    )

    while True:
        try:
            evaluate_once()
        except Exception as e:
            send_telegram(f"❌ {CFG.user_title}: خطأ - {repr(e)}")
            print("[ERROR]", repr(e))
        time.sleep(CFG.loop_sleep_seconds)


if __name__ == "__main__":
    main()
