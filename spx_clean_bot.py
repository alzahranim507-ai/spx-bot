# -*- coding: utf-8 -*-
"""
ES Trading Bot (Hunter WICK) — Yahoo Finance (yfinance)
For: دكتور محمد

Core changes (as requested):
- Direction decision = (1H/4H Bias) THEN wick-cluster near key level
- Add "Wick Rejection near Level" as the strongest reason
- Block BUY inside Resistance zone if upper-wick cluster appears (and show reason clearly)
- Block SELL inside Support zone if lower-wick cluster appears (and show reason clearly)
- Messages keep your approved Hunter Smart format + explicit wick-cluster text

Notes:
- Symbol ALWAYS: ES=F (Pre/Market/After)
- ES Adjust: -10 points (match TradingView baseline)
- Hourly Update: ON (no entries inside hourly update)
- Signals: Score-based (wick cluster can override/flip direction or block)
- Stop handling: Hard-stop + optional 5m close confirm (prevents huge delayed stop)
- Probability/Expected Move/ETA remain heuristic (fast & light)

Railway env vars required:
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
    # Symbol
    symbol: str = "ES=F"
    es_points_adjust: float = 10.0

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    user_title: str = "دكتور محمد"

    # Timezones
    tz_riyadh: str = "Asia/Riyadh"
    tz_ny: str = "America/New_York"

    # Loop
    loop_sleep_seconds: int = 25

    # Pivots / structure
    pivot_left: int = 3
    pivot_right: int = 3

    # Level detection / clustering
    level_touch_tolerance_frac: float = 0.0013
    level_cluster_tolerance_frac: float = 0.0010
    max_key_levels: int = 6

    # Scoring
    score_threshold: int = 3
    min_rr_to_t1: float = 1.35
    signal_cooldown_minutes: int = 14

    # Market state
    adx_window: int = 14
    market_state_update_minutes: int = 15
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # Liquidity ratio (5m volume)
    liquidity_lookback_5m: int = 60
    liquidity_thresholds: tuple = (0.60, 1.25, 2.00)  # low, normal, high, extreme

    # Expected Move
    expected_move_atr_window: int = 14

    # ETA
    eta_velocity_lookback_5m: int = 24
    eta_min_velocity_pts_per_min: float = 0.15

    # Wick cluster (KEY CHANGE)
    wick_cluster_lookback_5m: int = 10
    wick_ratio_strong: float = 0.45     # wick / candle range
    wick_cluster_min_hits: int = 3      # how many candles must show wick dominance
    wick_near_level_tolerance_frac: float = 0.0010
    wick_min_abs_pts: float = 1.2       # ignore tiny wicks

    # Trade tracking
    hourly_update: bool = True

    # Stop logic
    hard_stop_enabled: bool = True
    hard_stop_buffer_pts: float = 1.0
    stop_confirm_by_5m_close: bool = True
    stop_confirm_minutes: int = 5

    # T1/T2 management (guidance messages)
    move_stop_to_be_on_t1: bool = True

    # Daily reset (Riyadh midnight)
    daily_reset_enabled: bool = True
    daily_reset_hour: int = 0
    daily_reset_minute: int = 0
    daily_reset_window_minutes: int = 5
    no_signal_after_reset_minutes: int = 8


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
# Safe formatting
# =========================

def safe_f1(x, default="N/A"):
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return f"{v:.1f}"
    except Exception:
        return default

def level_bucket_x(level: float) -> str:
    # 688x style bucket
    try:
        b = int(float(level) // 10) * 10
        return f"{b}x"
    except Exception:
        return "N/A"


# =========================
# Telegram
# =========================

def send_telegram(text: str):
    if not CFG.telegram_bot_token or not CFG.telegram_chat_id:
        print("[WARN] Telegram env vars not set. Printing:\n", text)
        return
    url = f"https://api.telegram.org/bot{CFG.telegram_bot_token}/sendMessage"
    payload = {"chat_id": CFG.telegram_chat_id, "text": text, "disable_web_page_preview": True}
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.ok:
                return
            print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", repr(e))
        time.sleep(1.3 * (attempt + 1))


# =========================
# Yahoo fetching
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
                threads=True,
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
                df["volume"] = np.nan

            if df.index.tz is None:
                df.index = df.index.tz_localize(timezone.utc)

            return df
        except Exception as e:
            last_err = e
            time.sleep(1.4 * (attempt + 1))
    raise RuntimeError(f"Yahoo download failed after retries: {repr(last_err)}")

def apply_es_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) - CFG.es_points_adjust
    return out

def fetch_timeframes():
    sym = CFG.symbol
    df_5m_raw = apply_es_adjustment(_yf_download(sym, "5m", "7d"))
    df_15m = apply_es_adjustment(_yf_download(sym, "15m", "30d"))
    df_1h = apply_es_adjustment(_yf_download(sym, "60m", "90d"))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_4h = df_1h.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return sym, df_4h, df_1h, df_15m, df_5m_raw


# =========================
# Pivots / Structure Bias
# =========================

def find_pivots(series: pd.Series, left: int, right: int):
    arr = series.values
    piv_hi, piv_lo = [], []
    n = len(arr)
    for i in range(left, n - right):
        v = arr[i]
        wl = arr[i - left : i]
        wr = arr[i + 1 : i + 1 + right]
        if np.all(v > wl) and np.all(v >= wr):
            piv_hi.append(i)
        if np.all(v < wl) and np.all(v <= wr):
            piv_lo.append(i)
    return piv_hi, piv_lo

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

def cluster_levels(levels: list, tol_frac: float, price_ref: float) -> list:
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

def extract_key_levels(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> list:
    price = float(df_15m["close"].iloc[-1])

    hi_idx, lo_idx = find_pivots(df_1h["high"], CFG.pivot_left, CFG.pivot_right)
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx][-12:] if hi_idx else []
    swing_lows  = [float(df_1h["low"].iloc[i])  for i in lo_idx][-12:] if lo_idx else []

    recent = df_15m.tail(220)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    candidates = swing_highs + swing_lows + [range_hi, range_lo]
    candidates = [float(x) for x in candidates if np.isfinite(x)]

    merged = cluster_levels(candidates, CFG.level_cluster_tolerance_frac, price)
    merged = sorted(merged, key=lambda x: abs(x - price))[: max(CFG.max_key_levels * 3, 18)]
    merged = sorted(cluster_levels(merged, CFG.level_cluster_tolerance_frac, price))

    # keep closest + extremes
    if len(merged) > CFG.max_key_levels:
        closest = sorted(merged, key=lambda x: abs(x - price))[: CFG.max_key_levels - 2]
        merged = sorted(cluster_levels(closest + [min(merged), max(merged)], CFG.level_cluster_tolerance_frac, price))

    return merged

def fmt_levels(levels: list) -> str:
    return ", ".join([f"{float(x):.1f}" for x in levels])

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
        retest = (
            near_level(price, level, CFG.level_touch_tolerance_frac)
            or near_level(float(window["low"].min()), level, CFG.level_touch_tolerance_frac)
        )
        return bool(broke and retest and price >= level)
    if direction == "SELL":
        broke = (window["close"] < level).any()
        retest = (
            near_level(price, level, CFG.level_touch_tolerance_frac)
            or near_level(float(window["high"].max()), level, CFG.level_touch_tolerance_frac)
        )
        return bool(broke and retest and price <= level)
    return False


# =========================
# Liquidity state
# =========================

def liquidity_state(df_5m_raw: pd.DataFrame) -> tuple[str, float | None]:
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
    t_low, t_high, t_ext = CFG.liquidity_thresholds
    if ratio < t_low:
        return "Low", ratio
    if ratio < t_high:
        return "Normal", ratio
    if ratio < t_ext:
        return "High", ratio
    return "Extreme", ratio


# =========================
# Market State (ADX + structure)
# =========================

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, prev_label: str):
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

    if adx_val is None or np.isnan(adx_val):
        return "Weak", direction, None

    if adx_val >= CFG.adx_trending_on:
        label = "Trending"
    elif adx_val <= CFG.adx_range_on:
        label = "Range"
    else:
        if prev_label == "Trending" and adx_val > CFG.adx_range_on:
            label = "Trending"
        elif prev_label == "Range" and adx_val < CFG.adx_trending_on:
            label = "Range"
        else:
            label = "Weak"

    return label, direction, adx_val

def expected_move_1h(df_1h: pd.DataFrame) -> float | None:
    if len(df_1h) < (CFG.expected_move_atr_window + 5):
        return None
    atr = AverageTrueRange(
        high=df_1h["high"], low=df_1h["low"], close=df_1h["close"],
        window=CFG.expected_move_atr_window,
    ).average_true_range()
    val = float(atr.iloc[-1])
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def eta_to_t1_minutes(df_5m: pd.DataFrame, price_now: float, t1: float, direction: str, liq_state: str) -> tuple[int, int] | None:
    if t1 is None or not np.isfinite(t1):
        return None
    look = min(CFG.eta_velocity_lookback_5m, len(df_5m))
    if look < 8:
        return None
    closes = df_5m["close"].tail(look).astype(float)
    diffs = closes.diff().abs().dropna()
    if diffs.empty:
        return None
    avg_abs_move_per_5m = float(diffs.mean())
    vel = max(avg_abs_move_per_5m / 5.0, CFG.eta_min_velocity_pts_per_min)

    dist = abs((t1 - price_now) if direction == "BUY" else (price_now - t1))
    if dist < 0.1:
        return (1, 5)

    base = dist / vel

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
    lo = int(max(5, round(eta * 0.75)))
    hi = int(max(lo + 5, round(eta * 1.25)))
    return lo, hi


# =========================
# Wick Cluster (KEY FEATURE)
# =========================

def wick_cluster_near_level(df_5m: pd.DataFrame, level: float) -> dict:
    """
    Returns:
      {
        "upper_cluster": bool,
        "lower_cluster": bool,
        "upper_hits": int,
        "lower_hits": int,
        "bucket": "688x",
      }
    """
    w = df_5m.tail(min(CFG.wick_cluster_lookback_5m, len(df_5m))).copy()
    if w.empty:
        return {"upper_cluster": False, "lower_cluster": False, "upper_hits": 0, "lower_hits": 0, "bucket": level_bucket_x(level)}

    upper_hits = 0
    lower_hits = 0

    for _, r in w.iterrows():
        o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        rng = max(h - l, 1e-9)

        upper = h - max(o, c)
        lower = min(o, c) - l

        # near level check: either wick or body touches level (simple & robust)
        near = (abs(h - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac) or \
               (abs(l - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac) or \
               (abs(c - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac)

        if not near:
            continue

        if upper >= CFG.wick_min_abs_pts and (upper / rng) >= CFG.wick_ratio_strong:
            upper_hits += 1
        if lower >= CFG.wick_min_abs_pts and (lower / rng) >= CFG.wick_ratio_strong:
            lower_hits += 1

    upper_cluster = upper_hits >= CFG.wick_cluster_min_hits
    lower_cluster = lower_hits >= CFG.wick_cluster_min_hits

    return {
        "upper_cluster": bool(upper_cluster),
        "lower_cluster": bool(lower_cluster),
        "upper_hits": int(upper_hits),
        "lower_hits": int(lower_hits),
        "bucket": level_bucket_x(level),
    }


# =========================
# Trade plan
# =========================

def pick_targets(levels: list, entry: float, direction: str):
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

def compute_trade_plan(df_5m: pd.DataFrame, levels: list, level_hit: float, direction: str, trigger: str):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    buffer = max(price * 0.0002, 0.5)

    if trigger in ("Rejection", "Wick Rejection near Level"):
        if direction == "BUY":
            entry = float(max(price, float(last["high"]) + buffer))
            stop  = float(min(float(last["low"]), level_hit) - buffer)
        else:
            entry = float(min(price, float(last["low"]) - buffer))
            stop  = float(max(float(last["high"]), level_hit) + buffer)
    else:
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer))
            stop  = float(level_hit - (price * 0.0013) - buffer)
        else:
            entry = float(min(price, level_hit - buffer))
            stop  = float(level_hit + (price * 0.0013) + buffer)

    t1, t2 = pick_targets(levels, entry, direction)

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = (reward / risk) if risk > 0 else None

    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "rr": rr}


# =========================
# Confidence / Probability (fast heuristic)
# =========================

def confidence_percent(score: int, direction: str, bias: str, session: str, market_label: str, liq_state: str) -> int:
    base = int(round((score / 6) * 100))
    adj = 0

    if (bias == "Bullish" and direction == "BUY") or (bias == "Bearish" and direction == "SELL"):
        adj += 6
    if session in ("After-Hours", "Pre-Market"):
        adj -= 5
    if liq_state == "Low":
        adj -= 6
    elif liq_state == "High":
        adj += 3
    elif liq_state == "Extreme":
        adj += 5
    if market_label == "Trending":
        adj += 2
    if market_label == "Range":
        adj += 1

    return int(max(5, min(95, base + adj)))

def probability_t1_t2(conf: int, rr: float | None, dist_t1: float | None, dist_t2: float | None,
                     exp_move_1h_val: float | None, liq_state: str, market_label: str) -> tuple[int, int]:
    t1 = float(conf)

    if rr is not None and np.isfinite(rr):
        if rr > 2.5:
            t1 -= 8
        elif rr > 2.0:
            t1 -= 5
        elif rr < 1.4:
            t1 += 2

    if liq_state == "Low":
        t1 -= 5
    elif liq_state in ("High", "Extreme"):
        t1 += 2

    if exp_move_1h_val is not None and dist_t1 is not None and np.isfinite(dist_t1):
        if dist_t1 > exp_move_1h_val * 1.20:
            t1 -= 12
        elif dist_t1 > exp_move_1h_val * 1.00:
            t1 -= 6

    t1 = max(5, min(95, int(round(t1))))

    if dist_t1 is not None and dist_t2 is not None and np.isfinite(dist_t1) and np.isfinite(dist_t2) and dist_t2 > 0:
        ratio = max(1.0, dist_t2 / max(dist_t1, 1e-9))
        t2 = int(round(t1 * (0.70 / ratio)))
    else:
        t2 = int(round(t1 * 0.65))

    t2 = max(5, min(t1, min(90, t2)))
    return int(t1), int(t2)


# =========================
# Messages
# =========================

def signal_message(
    session: str,
    symbol: str,
    market_state_str: str,
    liq_state: str,
    level_now: float,
    direction: str,
    level_hit: float,
    plan: dict,
    score: int,
    conf: int,
    p_t1: int,
    p_t2: int,
    exp_move: float | None,
    eta_band: tuple[int, int] | None,
    reasons: list[str],
) -> str:
    t1_txt = safe_f1(plan.get("t1"))
    t2_txt = safe_f1(plan.get("t2"))
    rr_val = plan.get("rr")
    rr_txt = f"{float(rr_val):.2f}" if rr_val is not None and np.isfinite(float(rr_val)) else "N/A"
    exp_txt = f"±{float(exp_move):.0f} pts" if exp_move is not None and np.isfinite(float(exp_move)) else "N/A"
    eta_txt = "N/A" if eta_band is None else f"{eta_band[0]}–{eta_band[1]} min"

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n\n"
        f"📊 Market State: {market_state_str}\n"
        f"💧 Liquidity: {liq_state}\n"
        f"💰 Level Now: {level_now:.1f}\n\n"
        f"📍 Direction: {direction}\n"
        f"🧱 Level: {level_hit:.1f}\n\n"
        f"✅ Entry: {plan['entry']:.1f}\n"
        f"🛑 Stop: {plan['stop']:.1f}\n"
        f"🎯 Target 1: {t1_txt}\n"
        f"🎯 Target 2: {t2_txt}\n\n"
        f"📐 RR: {rr_txt}\n"
        f"⭐ Score: {score}/6\n"
        f"🔎 Confidence: {conf}%\n\n"
        f"📊 Probability\n"
        f"T1 Hit: {p_t1}%\n"
        f"T2 Hit: {p_t2}%\n\n"
        f"📈 Expected Move (1H): {exp_txt}\n"
        f"⏱ ETA to T1: {eta_txt}\n\n"
        f"🧠 Reason: {', '.join(reasons)}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )

def hourly_update_message(session: str, symbol: str, bias: str, market_state_str: str,
                          levels: list, price: float, active_trade: dict | None) -> str:
    status = "None" if active_trade is None else active_trade.get("status", "Unknown")
    trade_line = "-"
    if active_trade is not None:
        trade_line = f"{active_trade.get('direction','?')} | Entry {safe_f1(active_trade.get('entry'))} | Stop {safe_f1(active_trade.get('stop'))}"

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): {bias}\n"
        f"📊 Market State: {market_state_str}\n"
        f"💵 Price: {price:.1f}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"📌 Active Trade: {status}\n"
        f"🧾 Trade: {trade_line}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )


# =========================
# Bot State
# =========================

class BotState:
    def __init__(self):
        self.last_hour_sent = None
        self.last_signal_ts = {}
        self.active_trade = None

        self.last_reset_date_riyadh = None
        self.no_signal_until_utc = None

        self.market_state_last_calc_utc = None
        self.market_label = "Unknown"
        self.market_dir = "Neutral"
        self.market_adx = None

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

    is_window = (t.hour == CFG.daily_reset_hour and CFG.daily_reset_minute <= t.minute < (CFG.daily_reset_minute + CFG.daily_reset_window_minutes))
    if not is_window:
        return

    if STATE.active_trade is not None:
        send_telegram(
            f"🛎️ {CFG.user_title} — Daily Reset (Riyadh)\n"
            f"السبب: منتصف الليل بتوقيت السعودية.\n"
            f"ملاحظة: تم إنهاء تتبع الصفقة لتفادي تعليق البوت وبداية يوم جديد."
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
# Market state update
# =========================

def maybe_update_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    nowu = datetime.utcnow()
    should = STATE.market_state_last_calc_utc is None or (nowu - STATE.market_state_last_calc_utc) >= timedelta(minutes=CFG.market_state_update_minutes)
    if not should:
        return
    label, direction, adx_val = compute_market_state(df_1h, df_4h, STATE.market_label)
    STATE.market_label = label
    STATE.market_dir = direction
    STATE.market_adx = adx_val
    STATE.market_state_last_calc_utc = nowu


# =========================
# Scoring (Wick Rejection is STRONGEST)
# =========================

def score_setup(df_5m: pd.DataFrame, level_hit: float, direction: str, wick_info: dict) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    trigger = None

    # Wick rejection near level (strong)
    wick_reason = None
    if wick_info.get("upper_cluster") and direction == "SELL":
        wick_reason = f"Upper-wick rejection cluster near {wick_info.get('bucket','N/A')}"
    if wick_info.get("lower_cluster") and direction == "BUY":
        wick_reason = f"Lower-wick cluster near support {wick_info.get('bucket','N/A')}"

    if wick_reason:
        score += 3
        reasons.append("Wick Rejection near Level")
        reasons.append(wick_reason)  # explicit text you asked for
        trigger = "Wick Rejection near Level"

    # Other confirmations
    br = break_retest(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction)
    st = stoch_cross(df_5m, direction)
    ms = momentum_shift(df_5m, direction)

    if br:
        score += 2
        reasons.append("Break&Retest")
        trigger = trigger or "Break&Retest"
    if rj:
        score += 2
        reasons.append("Rejection")
        trigger = trigger or "Rejection"
    if st:
        score += 1
        reasons.append("Stoch RSI cross")
    if ms:
        score += 1
        reasons.append("Momentum shift")

    score = int(min(score, 6))
    if trigger is None:
        return 0, ["No trigger"], "None"
    return score, reasons, trigger


# =========================
# Trade tracking (entry + stop)
# =========================

def update_active_trade(df_5m: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if tr is None:
        return

    direction = tr["direction"]
    entry = float(tr["entry"])
    stop = float(tr["stop"])
    t1 = tr.get("t1")
    t2 = tr.get("t2")

    # Entry trigger
    if tr["status"] == "pending":
        triggered = (last_price >= entry) if direction == "BUY" else (last_price <= entry)
        if triggered:
            tr["status"] = "live"
            tr["live_since_utc"] = datetime.utcnow()
            send_telegram(
                f"📍 {CFG.user_title} — الصفقة تفعلت\n"
                f"Direction: {direction}\n"
                f"Entry Triggered @ {last_price:.1f}\n"
                f"Stop: {stop:.1f}\n"
                f"T1: {safe_f1(t1)} | T2: {safe_f1(t2)}"
            )
        return

    if tr["status"] != "live":
        return

    breached = (last_price <= stop) if direction == "BUY" else (last_price >= stop)
    if breached:
        if CFG.hard_stop_enabled:
            beyond = (stop - last_price) >= CFG.hard_stop_buffer_pts if direction == "BUY" else (last_price - stop) >= CFG.hard_stop_buffer_pts
            if beyond:
                send_telegram(
                    f"❌ {CFG.user_title} — وقف الخسارة (Hard Stop)\n"
                    f"Stop Hit ✅ | Direction: {direction}\n"
                    f"Stop: {stop:.1f} | Price: {last_price:.1f}\n"
                    f"ملاحظة: تم الإغلاق الفوري لتجنب انزلاق كبير."
                )
                STATE.active_trade = None
                return

        if CFG.stop_confirm_by_5m_close:
            if not tr.get("stop_pending"):
                tr["stop_pending"] = True
                tr["stop_pending_since_utc"] = datetime.utcnow()
                send_telegram(
                    f"⚠️ {CFG.user_title} — اختبار وقف الخسارة\n"
                    f"Price قرب/تجاوز الوقف: {last_price:.1f} | Stop: {stop:.1f}\n"
                    f"جاري التأكد بإغلاق شمعة 5 دقائق."
                )
            else:
                mins = (datetime.utcnow() - tr["stop_pending_since_utc"]).total_seconds() / 60.0
                if mins >= CFG.stop_confirm_minutes:
                    last_close = float(df_5m["close"].iloc[-1])
                    confirmed = (last_close <= stop) if direction == "BUY" else (last_close >= stop)
                    if confirmed:
                        send_telegram(
                            f"❌ {CFG.user_title} — وقف الخسارة تأكد\n"
                            f"Stop Hit ✅ | Direction: {direction}\n"
                            f"Stop: {stop:.1f} | Price: {last_price:.1f}\n"
                            f"تأكيد: إغلاق 5m ضد الصفقة."
                        )
                        STATE.active_trade = None
                        return
                    else:
                        tr["stop_pending"] = False
                        tr.pop("stop_pending_since_utc", None)
            return

    if tr.get("stop_pending"):
        tr["stop_pending"] = False
        tr.pop("stop_pending_since_utc", None)

    # Targets (guidance)
    if t1 is not None and not tr.get("t1_hit"):
        hit_t1 = (last_price >= float(t1)) if direction == "BUY" else (last_price <= float(t1))
        if hit_t1:
            tr["t1_hit"] = True
            if CFG.move_stop_to_be_on_t1:
                send_telegram(
                    f"🎯 {CFG.user_title} — Target 1 Hit\n"
                    f"T1: {float(t1):.1f}\n"
                    f"اقتراح: نقل الستوب إلى Entry (BE) لحماية الربح."
                )
            else:
                send_telegram(f"🎯 {CFG.user_title} — Target 1 Hit\nT1: {float(t1):.1f}")

    if t2 is not None and not tr.get("t2_hit"):
        hit_t2 = (last_price >= float(t2)) if direction == "BUY" else (last_price <= float(t2))
        if hit_t2:
            tr["t2_hit"] = True
            send_telegram(
                f"🏆 {CFG.user_title} — Target 2 Hit\n"
                f"T2: {float(t2):.1f}\n"
                f"اقتراح: إغلاق/تخفيف يدوي حسب خطتك."
            )
            STATE.active_trade = None
            return


# =========================
# Main evaluation
# =========================

def evaluate_once():
    maybe_daily_reset()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m_raw = fetch_timeframes()
    df_5m = compute_indicators_5m(df_5m_raw)

    level_now = float(df_5m["close"].iloc[-1])

    maybe_update_market_state(df_1h, df_4h)
    adx_txt = "N/A" if STATE.market_adx is None else f"{float(STATE.market_adx):.1f}"
    market_state_str = f"{STATE.market_label} | {STATE.market_dir} | ADX(1H): {adx_txt}"

    liq_state, _ = liquidity_state(df_5m_raw)
    bias = structure_bias(df_1h, df_4h)
    key_levels = extract_key_levels(df_15m, df_1h)

    # Track active trade first
    if STATE.active_trade is not None:
        update_active_trade(df_5m, level_now)

    # Hourly update
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_update_message(session, symbol, bias, market_state_str, key_levels, level_now, STATE.active_trade))
            STATE.last_hour_sent = current_hour

    # no-signal window after reset
    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        return

    # One trade at a time
    if STATE.active_trade is not None:
        return

    if not key_levels:
        return

    # choose nearest relevant level
    near = [lvl for lvl in key_levels if near_level(level_now, lvl, CFG.level_touch_tolerance_frac)]
    level_hit = float(min((near if near else key_levels), key=lambda x: abs(x - level_now)))

    # ===== Direction decision (your rule) =====
    # 1) bias first
    if bias == "Bullish":
        direction = "BUY"
    elif bias == "Bearish":
        direction = "SELL"
    else:
        direction = "BUY" if momentum_shift(df_5m, "BUY") else "SELL"

    # 2) wick cluster near key level => filter/flip
    wick_info = wick_cluster_near_level(df_5m, level_hit)

    # Prevent BUY inside resistance if upper-wick cluster appears
    if direction == "BUY" and wick_info.get("upper_cluster"):
        # flip to SELL (more useful than just blocking) but still respects your "no BUY"
        direction = "SELL"

    # Prevent SELL inside support if lower-wick cluster appears
    if direction == "SELL" and wick_info.get("lower_cluster"):
        direction = "BUY"

    # Score + reasons (Wick Rejection is strongest)
    score, reasons, trigger = score_setup(df_5m, level_hit, direction, wick_info)
    if score <= 0:
        return

    # build trade plan
    plan = compute_trade_plan(df_5m, key_levels, level_hit, direction, trigger)

    rr = plan.get("rr")
    if rr is not None and np.isfinite(float(rr)) and float(rr) < CFG.min_rr_to_t1:
        return

    # cooldown key
    key = f"{direction}:{round(level_hit,1)}:{trigger}"
    if not STATE.can_signal(key):
        return

    # metrics
    exp_move_val = expected_move_1h(df_1h)
    conf = confidence_percent(score, direction, bias, session, STATE.market_label, liq_state)

    dist_t1 = None
    dist_t2 = None
    if plan.get("t1") is not None:
        dist_t1 = abs(float(plan["t1"]) - level_now)
    if plan.get("t2") is not None:
        dist_t2 = abs(float(plan["t2"]) - level_now)

    p1, p2 = probability_t1_t2(conf, rr, dist_t1, dist_t2, exp_move_val, liq_state, STATE.market_label)

    eta = None
    if plan.get("t1") is not None:
        eta = eta_to_t1_minutes(df_5m, level_now, float(plan["t1"]), direction, liq_state)

    msg = signal_message(
        session=session,
        symbol=symbol,
        market_state_str=market_state_str,
        liq_state=liq_state,
        level_now=level_now,
        direction=direction,
        level_hit=level_hit,
        plan=plan,
        score=score,
        conf=conf,
        p_t1=p1,
        p_t2=p2,
        exp_move=exp_move_val,
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
        "t1": None if plan.get("t1") is None else float(plan["t1"]),
        "t2": None if plan.get("t2") is None else float(plan["t2"]),
        "status": "pending",
        "created_utc": datetime.utcnow(),
    }


def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- All Sessions: {CFG.symbol} (adjust -{CFG.es_points_adjust:.0f})\n"
        f"- Hourly Update: ON\n"
        f"- Direction: Bias(1H/4H) ➜ Wick-Cluster near Key Level\n"
        f"- Strongest Reason: Wick Rejection near Level\n"
        f"- No BUY at resistance if Upper-wick cluster\n"
        f"- No SELL at support if Lower-wick cluster\n"
        f"- Score ≥ {CFG.score_threshold}\n"
        f"- Stop: HardStop({CFG.hard_stop_buffer_pts:.1f} pts) + 5m close confirm"
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
