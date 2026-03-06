# -*- coding: utf-8 -*-
"""
ES Trading Bot (Hunter Smart v5 - Reaction Zones)
For: دكتور محمد

Main idea:
- Symbol ALWAYS: ES=F
- ES Adjust: -10 points
- Reaction Zones instead of rigid Support/Resistance naming
- Direction starts from 1H/4H Bias
- Then confirms by:
    * Wick cluster near zone
    * Sweep + reclaim
    * Stoch RSI cross
    * Momentum shift
    * Break & Retest
- Confidence / RR / Probability / Expected Move / ETA remain
- Mid-range trades are still allowed IF score/confidence support them
- Fixes:
    * break_retest function included
    * safer formatting
    * safer Telegram/network handling

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
    symbol: str = "ES=F"
    es_points_adjust: float = 10.0

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    user_title: str = "دكتور محمد"

    tz_riyadh: str = "Asia/Riyadh"
    tz_ny: str = "America/New_York"

    loop_sleep_seconds: int = 25

    # Structure
    pivot_left: int = 3
    pivot_right: int = 3

    # Reaction zones
    zone_lookback_5m: int = 150
    eq_tolerance_pts: float = 2.0
    zone_merge_tolerance_pts: float = 3.0
    min_zone_reactions: int = 2
    max_zone_distance_pts: float = 80.0
    max_zones_total: int = 6
    strong_move_min_pts: float = 10.0

    # Wick cluster
    wick_cluster_lookback_5m: int = 12
    wick_ratio_strong: float = 0.40
    wick_min_abs_pts: float = 1.0
    wick_cluster_min_hits: int = 3

    # Sweep
    sweep_min_break_pts: float = 1.0
    reclaim_required: bool = True

    # Scoring
    score_threshold: int = 3
    min_rr_to_t1: float = 1.35
    signal_cooldown_minutes: int = 15

    # Market state
    adx_window: int = 14
    market_state_update_minutes: int = 15
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # Liquidity
    liquidity_lookback_5m: int = 60
    liquidity_thresholds: tuple = (0.60, 1.25, 2.00)

    # ATR / expected move
    atr1h_window: int = 14

    # ETA
    eta_velocity_lookback_5m: int = 24
    eta_min_velocity_pts_per_min: float = 0.15

    # Trade management
    hourly_update: bool = True
    hard_stop_enabled: bool = True
    hard_stop_buffer_pts: float = 1.0
    stop_confirm_by_5m_close: bool = True
    stop_confirm_minutes: int = 5
    move_stop_to_be_on_t1: bool = True

    # T2 cap
    cap_t2_enabled: bool = True
    cap_factor_market: float = 1.25
    cap_factor_offhours: float = 1.00

    # Daily reset
    daily_reset_enabled: bool = True
    daily_reset_hour: int = 0
    daily_reset_minute: int = 0
    daily_reset_window_minutes: int = 5
    no_signal_after_reset_minutes: int = 8


CFG = Config()


# =========================
# Time helpers
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

def safe_f2(x, default="N/A"):
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return f"{v:.2f}"
    except Exception:
        return default


# =========================
# Telegram
# =========================

def send_telegram(text: str):
    if not CFG.telegram_bot_token or not CFG.telegram_chat_id:
        print("[WARN] Telegram env vars not set.\n", text)
        return

    url = f"https://api.telegram.org/bot{CFG.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": CFG.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.ok:
                return
            print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", repr(e))
        time.sleep(1.2 * (attempt + 1))


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
# Indicators
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
        return prev < 0 and curr > 0 and np.nanmin(k) < 0.30
    if direction == "SELL":
        return prev > 0 and curr < 0 and np.nanmax(k) > 0.70
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
        return (lower / rng) >= 0.25
    if direction == "SELL":
        return (upper / rng) >= 0.25
    return False

def break_retest(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    if len(df_5m) < 8:
        return False

    price = float(df_5m["close"].iloc[-1])
    window = df_5m.tail(8)
    tol = max(CFG.eq_tolerance_pts, 1.0)

    if direction == "BUY":
        broke = (window["close"] > level).any()
        retest = (
            abs(price - level) <= tol or
            abs(float(window["low"].min()) - level) <= tol
        )
        return bool(broke and retest and price >= level)

    if direction == "SELL":
        broke = (window["close"] < level).any()
        retest = (
            abs(price - level) <= tol or
            abs(float(window["high"].max()) - level) <= tol
        )
        return bool(broke and retest and price <= level)

    return False


# =========================
# Structure / Market State
# =========================

def find_pivots(series: pd.Series, left: int, right: int):
    arr = series.values
    piv_hi, piv_lo = [], []
    n = len(arr)

    for i in range(left, n - right):
        v = arr[i]
        wl = arr[i - left:i]
        wr = arr[i + 1:i + 1 + right]

        if np.all(v > wl) and np.all(v >= wr):
            piv_hi.append(i)
        if np.all(v < wl) and np.all(v <= wr):
            piv_lo.append(i)

    return piv_hi, piv_lo

def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    def bias_from(df: pd.DataFrame) -> str:
        hi_idx, lo_idx = find_pivots(df["high"], CFG.pivot_left, CFG.pivot_right)
        highs = [float(df["high"].iloc[i]) for i in hi_idx][-2:]
        lows  = [float(df["low"].iloc[i]) for i in lo_idx][-2:]

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

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, prev_label: str):
    if len(df_1h) < (CFG.adx_window + 5):
        return "Weak", "Neutral", None

    adx = ADXIndicator(
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=CFG.adx_window
    ).adx()

    adx_val = float(adx.iloc[-1]) if len(adx) else None
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


# =========================
# Liquidity / ATR / ETA
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
    low_t, high_t, ext_t = CFG.liquidity_thresholds

    if ratio < low_t:
        return "Low", ratio
    if ratio < high_t:
        return "Normal", ratio
    if ratio < ext_t:
        return "High", ratio
    return "Extreme", ratio

def atr_1h(df_1h: pd.DataFrame) -> float | None:
    if len(df_1h) < (CFG.atr1h_window + 5):
        return None

    atr = AverageTrueRange(
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=CFG.atr1h_window
    ).average_true_range()

    val = float(atr.iloc[-1])
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def eta_to_t1_minutes(df_5m: pd.DataFrame, price_now: float, t1: float, direction: str, liq_state: str):
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
# Reaction Zones Engine
# =========================

def merge_levels_to_zones(levels: list[float], merge_tolerance_pts: float) -> list[tuple[float, float]]:
    if not levels:
        return []

    levels = sorted(levels)
    zones = []
    start = levels[0]
    end = levels[0]

    for x in levels[1:]:
        if abs(x - end) <= merge_tolerance_pts:
            end = x
        else:
            zones.append((start, end))
            start = x
            end = x

    zones.append((start, end))
    return zones

def zone_mid(zone: tuple[float, float]) -> float:
    return (zone[0] + zone[1]) / 2.0

def fmt_zones(zones: list[tuple[float, float]]) -> str:
    if not zones:
        return "-"
    return ", ".join([f"{z[0]:.1f}-{z[1]:.1f}" for z in zones])

def nearest_zone(price: float, zones: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not zones:
        return None
    return min(zones, key=lambda z: abs(zone_mid(z) - price))

def price_inside_zone(price: float, zone: tuple[float, float] | None) -> bool:
    if zone is None:
        return False
    return zone[0] <= price <= zone[1]

def build_equal_high_low_candidates(df_5m: pd.DataFrame) -> tuple[list[float], list[float]]:
    w = df_5m.tail(min(CFG.zone_lookback_5m, len(df_5m))).copy()
    highs = w["high"].astype(float).values
    lows = w["low"].astype(float).values

    eq_highs = []
    eq_lows = []

    used_hi = np.zeros(len(highs), dtype=bool)
    for i in range(len(highs)):
        if used_hi[i]:
            continue
        group = [highs[i]]
        used_hi[i] = True
        for j in range(i + 1, len(highs)):
            if abs(highs[j] - highs[i]) <= CFG.eq_tolerance_pts:
                group.append(highs[j])
                used_hi[j] = True
        if len(group) >= CFG.min_zone_reactions:
            eq_highs.append(float(np.mean(group)))

    used_lo = np.zeros(len(lows), dtype=bool)
    for i in range(len(lows)):
        if used_lo[i]:
            continue
        group = [lows[i]]
        used_lo[i] = True
        for j in range(i + 1, len(lows)):
            if abs(lows[j] - lows[i]) <= CFG.eq_tolerance_pts:
                group.append(lows[j])
                used_lo[j] = True
        if len(group) >= CFG.min_zone_reactions:
            eq_lows.append(float(np.mean(group)))

    return eq_highs, eq_lows

def build_wick_candidates(df_5m: pd.DataFrame) -> tuple[list[float], list[float]]:
    w = df_5m.tail(min(CFG.zone_lookback_5m, len(df_5m))).copy()
    upper_levels = []
    lower_levels = []

    for _, r in w.iterrows():
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        rng = max(h - l, 1e-9)

        upper = h - max(o, c)
        lower = min(o, c) - l

        if upper >= CFG.wick_min_abs_pts and (upper / rng) >= CFG.wick_ratio_strong:
            upper_levels.append(h)
        if lower >= CFG.wick_min_abs_pts and (lower / rng) >= CFG.wick_ratio_strong:
            lower_levels.append(l)

    return upper_levels, lower_levels

def zone_has_displacement(df_5m: pd.DataFrame, zone: tuple[float, float], side: str) -> bool:
    w = df_5m.tail(min(CFG.zone_lookback_5m, len(df_5m))).copy()
    z_low, z_high = zone
    touched_indices = []

    for i in range(len(w)):
        h = float(w["high"].iloc[i])
        l = float(w["low"].iloc[i])
        c = float(w["close"].iloc[i])

        touched = (l <= z_high and h >= z_low) or (z_low <= c <= z_high)
        if touched:
            touched_indices.append(i)

    if not touched_indices:
        return False

    for idx in touched_indices:
        future = w.iloc[idx + 1: idx + 6]
        if future.empty:
            continue

        if side == "upper":
            move = float(w["close"].iloc[idx] - future["low"].min())
            if move >= CFG.strong_move_min_pts:
                return True
        else:
            move = float(future["high"].max() - w["close"].iloc[idx])
            if move >= CFG.strong_move_min_pts:
                return True

    return False

def detect_reaction_zones(df_5m: pd.DataFrame, price_now: float) -> list[tuple[float, float]]:
    eq_highs, eq_lows = build_equal_high_low_candidates(df_5m)
    wick_highs, wick_lows = build_wick_candidates(df_5m)

    raw_levels = eq_highs + eq_lows + wick_highs + wick_lows
    raw_levels = [x for x in raw_levels if abs(x - price_now) <= CFG.max_zone_distance_pts]

    zones = merge_levels_to_zones(raw_levels, CFG.zone_merge_tolerance_pts)

    filtered = []
    for z in zones:
        if zone_has_displacement(df_5m, z, "upper") or zone_has_displacement(df_5m, z, "lower"):
            filtered.append(z)

    filtered = sorted(filtered, key=lambda z: abs(zone_mid(z) - price_now))
    return filtered[:CFG.max_zones_total]


# =========================
# Zone Behavior
# =========================

def wick_cluster_on_zone(df_5m: pd.DataFrame, zone: tuple[float, float]) -> tuple[bool, bool, int, int]:
    if zone is None:
        return False, False, 0, 0

    w = df_5m.tail(min(CFG.wick_cluster_lookback_5m, len(df_5m))).copy()
    upper_hits = 0
    lower_hits = 0
    z_low, z_high = zone

    for _, r in w.iterrows():
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        rng = max(h - l, 1e-9)
        upper = h - max(o, c)
        lower = min(o, c) - l

        touched = (l <= z_high and h >= z_low) or (z_low <= c <= z_high)
        if not touched:
            continue

        if upper >= CFG.wick_min_abs_pts and (upper / rng) >= CFG.wick_ratio_strong:
            upper_hits += 1
        if lower >= CFG.wick_min_abs_pts and (lower / rng) >= CFG.wick_ratio_strong:
            lower_hits += 1

    return (
        upper_hits >= CFG.wick_cluster_min_hits,
        lower_hits >= CFG.wick_cluster_min_hits,
        upper_hits,
        lower_hits
    )

def sweep_reclaim_on_zone(df_5m: pd.DataFrame, zone: tuple[float, float], mode: str) -> bool:
    if zone is None or len(df_5m) < 2:
        return False

    r = df_5m.iloc[-1]
    h = float(r["high"])
    l = float(r["low"])
    c = float(r["close"])
    z_low, z_high = zone

    if mode == "upper":
        broke = h >= (z_high + CFG.sweep_min_break_pts)
        reclaim = c <= z_high if CFG.reclaim_required else True
        return bool(broke and reclaim)

    broke = l <= (z_low - CFG.sweep_min_break_pts)
    reclaim = c >= z_low if CFG.reclaim_required else True
    return bool(broke and reclaim)


# =========================
# Trade plan / targets
# =========================

def pick_targets_from_zones(entry: float, direction: str, zones: list[tuple[float, float]]):
    mids = sorted([zone_mid(z) for z in zones])

    if direction == "BUY":
        above = [x for x in mids if x > entry]
        t1 = above[0] if len(above) >= 1 else None
        t2 = above[1] if len(above) >= 2 else None
        return t1, t2

    below = sorted([x for x in mids if x < entry], reverse=True)
    t1 = below[0] if len(below) >= 1 else None
    t2 = below[1] if len(below) >= 2 else None
    return t1, t2

def compute_trade_plan(
    df_5m: pd.DataFrame,
    direction: str,
    acting_zone: tuple[float, float],
    trigger: str,
    zones: list[tuple[float, float]]
):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    buffer = max(price * 0.0002, 0.5)
    z_low, z_high = acting_zone
    z_mid = zone_mid(acting_zone)

    if trigger in ("Wick Rejection near Zone", "Sweep + Reclaim"):
        if direction == "BUY":
            entry = float(max(price, z_high + buffer))
            stop = float(min(float(last["low"]), z_low) - buffer)
        else:
            entry = float(min(price, z_low - buffer))
            stop = float(max(float(last["high"]), z_high) + buffer)
    else:
        if direction == "BUY":
            entry = float(max(price, z_mid + buffer))
            stop = float(z_low - buffer)
        else:
            entry = float(min(price, z_mid - buffer))
            stop = float(z_high + buffer)

    t1, t2 = pick_targets_from_zones(entry, direction, zones)

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = reward / risk if risk > 0 else None

    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "rr": rr}

def cap_t2_with_expected_move(entry: float, t2: float | None, direction: str, exp_move: float | None, session: str):
    if not CFG.cap_t2_enabled or t2 is None or exp_move is None or not np.isfinite(exp_move):
        return t2, False

    factor = CFG.cap_factor_market if session == "Market" else CFG.cap_factor_offhours
    cap_dist = exp_move * factor
    cap_level = entry + cap_dist if direction == "BUY" else entry - cap_dist

    capped = False
    if direction == "BUY" and t2 > cap_level:
        t2 = float(cap_level)
        capped = True
    if direction == "SELL" and t2 < cap_level:
        t2 = float(cap_level)
        capped = True

    return t2, capped


# =========================
# Confidence / Probability
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
                      exp_move_1h_val: float | None, liq_state: str, market_label: str):
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
    acting_zone: tuple[float, float],
    plan: dict,
    score: int,
    conf: int,
    p_t1: int,
    p_t2: int,
    exp_move: float | None,
    eta_band,
    reasons: list[str],
) -> str:
    exp_txt = f"±{float(exp_move):.0f} pts" if exp_move is not None and np.isfinite(float(exp_move)) else "N/A"
    eta_txt = "N/A" if eta_band is None else f"{eta_band[0]}–{eta_band[1]} min"
    zone_txt = f"{acting_zone[0]:.1f}-{acting_zone[1]:.1f}"

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n\n"
        f"📊 Market State: {market_state_str}\n"
        f"💧 Liquidity: {liq_state}\n"
        f"💰 Level Now: {safe_f1(level_now)}\n\n"
        f"📍 Direction: {direction}\n"
        f"🧱 Reaction Zone: {zone_txt}\n\n"
        f"✅ Entry: {safe_f1(plan.get('entry'))}\n"
        f"🛑 Stop: {safe_f1(plan.get('stop'))}\n"
        f"🎯 Target 1: {safe_f1(plan.get('t1'))}\n"
        f"🎯 Target 2: {safe_f1(plan.get('t2'))}\n\n"
        f"📐 RR: {safe_f2(plan.get('rr'))}\n"
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
                          zones: list[tuple[float, float]], price: float, active_trade):
    status = "None" if active_trade is None else active_trade.get("status", "Unknown")
    trade_line = "-"
    if active_trade is not None:
        trade_line = (
            f"{active_trade.get('direction','?')} | "
            f"Entry {safe_f1(active_trade.get('entry'))} | "
            f"Stop {safe_f1(active_trade.get('stop'))}"
        )

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): {bias}\n"
        f"📊 Market State: {market_state_str}\n"
        f"💵 Price: {safe_f1(price)}\n"
        f"🧱 Key Reaction Zones: {fmt_zones(zones)}\n"
        f"📌 Active Trade: {status}\n"
        f"🧾 Trade: {trade_line}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )


# =========================
# State
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
# Daily reset
# =========================

def maybe_daily_reset():
    if not CFG.daily_reset_enabled:
        return

    t = now_riyadh()
    today = t.date()

    if STATE.last_reset_date_riyadh == today:
        return

    is_window = (
        t.hour == CFG.daily_reset_hour and
        CFG.daily_reset_minute <= t.minute < (CFG.daily_reset_minute + CFG.daily_reset_window_minutes)
    )
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
    should = (
        STATE.market_state_last_calc_utc is None or
        (nowu - STATE.market_state_last_calc_utc) >= timedelta(minutes=CFG.market_state_update_minutes)
    )
    if not should:
        return

    label, direction, adx_val = compute_market_state(df_1h, df_4h, STATE.market_label)
    STATE.market_label = label
    STATE.market_dir = direction
    STATE.market_adx = adx_val
    STATE.market_state_last_calc_utc = nowu


# =========================
# Trade tracking
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

    if tr["status"] == "pending":
        triggered = (last_price >= entry) if direction == "BUY" else (last_price <= entry)
        if triggered:
            tr["status"] = "live"
            tr["live_since_utc"] = datetime.utcnow()
            send_telegram(
                f"📍 {CFG.user_title} — الصفقة تفعلت\n"
                f"Direction: {direction}\n"
                f"Entry Triggered @ {safe_f1(last_price)}\n"
                f"Stop: {safe_f1(stop)}\n"
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
                    f"Stop: {safe_f1(stop)} | Price: {safe_f1(last_price)}\n"
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
                    f"Price قرب/تجاوز الوقف: {safe_f1(last_price)} | Stop: {safe_f1(stop)}\n"
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
                            f"Stop: {safe_f1(stop)} | Price: {safe_f1(last_price)}\n"
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

    if t1 is not None and not tr.get("t1_hit"):
        hit_t1 = (last_price >= float(t1)) if direction == "BUY" else (last_price <= float(t1))
        if hit_t1:
            tr["t1_hit"] = True
            if CFG.move_stop_to_be_on_t1:
                send_telegram(
                    f"🎯 {CFG.user_title} — Target 1 Hit\n"
                    f"T1: {safe_f1(t1)}\n"
                    f"اقتراح: نقل الستوب إلى Entry (BE) لحماية الربح."
                )
            else:
                send_telegram(f"🎯 {CFG.user_title} — Target 1 Hit\nT1: {safe_f1(t1)}")

    if t2 is not None and not tr.get("t2_hit"):
        hit_t2 = (last_price >= float(t2)) if direction == "BUY" else (last_price <= float(t2))
        if hit_t2:
            tr["t2_hit"] = True
            send_telegram(
                f"🏆 {CFG.user_title} — Target 2 Hit\n"
                f"T2: {safe_f1(t2)}\n"
                f"اقتراح: تخفيف/إغلاق يدوي حسب خطتك."
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

    price_now = float(df_5m["close"].iloc[-1])

    maybe_update_market_state(df_1h, df_4h)
    adx_txt = "N/A" if STATE.market_adx is None else safe_f1(STATE.market_adx)
    market_state_str = f"{STATE.market_label} | {STATE.market_dir} | ADX(1H): {adx_txt}"

    liq_state, _ = liquidity_state(df_5m_raw)
    bias = structure_bias(df_1h, df_4h)

    zones = detect_reaction_zones(df_5m, price_now)

    if STATE.active_trade is not None:
        update_active_trade(df_5m, price_now)

    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(
                hourly_update_message(
                    session=session,
                    symbol=symbol,
                    bias=bias,
                    market_state_str=market_state_str,
                    zones=zones,
                    price=price_now,
                    active_trade=STATE.active_trade,
                )
            )
            STATE.last_hour_sent = current_hour

    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        return

    if STATE.active_trade is not None:
        return

    if not zones:
        return

    # base direction
    if bias == "Bullish":
        direction = "BUY"
    elif bias == "Bearish":
        direction = "SELL"
    else:
        direction = "BUY" if momentum_shift(df_5m, "BUY") else "SELL"

    acting_zone = nearest_zone(price_now, zones)
    if acting_zone is None:
        return

    inside = price_inside_zone(price_now, acting_zone)
    upper_cluster, lower_cluster, upper_hits, lower_hits = wick_cluster_on_zone(df_5m, acting_zone)
    sweep_upper = sweep_reclaim_on_zone(df_5m, acting_zone, "upper")
    sweep_lower = sweep_reclaim_on_zone(df_5m, acting_zone, "lower")

    reasons = []
    score = 0
    trigger = None

    # role by behavior, not by fixed naming
    if direction == "BUY" and inside and upper_cluster:
        direction = "SELL"
    if direction == "SELL" and inside and lower_cluster:
        direction = "BUY"

    # scoring
    if direction == "BUY":
        if lower_cluster:
            score += 3
            reasons.append(f"Lower-wick cluster near zone {safe_f1(zone_mid(acting_zone))}")
            trigger = "Wick Rejection near Zone"
        if sweep_lower:
            score += 2
            reasons.append("Sweep + Reclaim")
            trigger = trigger or "Sweep + Reclaim"
    else:
        if upper_cluster:
            score += 3
            reasons.append(f"Upper-wick rejection cluster near {safe_f1(zone_mid(acting_zone))}")
            trigger = "Wick Rejection near Zone"
        if sweep_upper:
            score += 2
            reasons.append("Sweep + Reclaim")
            trigger = trigger or "Sweep + Reclaim"

    br = break_retest(df_5m, zone_mid(acting_zone), direction)
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

    if score < CFG.score_threshold:
        return

    plan = compute_trade_plan(
        df_5m=df_5m,
        direction=direction,
        acting_zone=acting_zone,
        trigger=trigger or "Zone",
        zones=zones,
    )

    rr = plan.get("rr")
    if rr is not None and np.isfinite(float(rr)) and float(rr) < CFG.min_rr_to_t1:
        return

    exp_move_val = atr_1h(df_1h)

    if plan.get("t2") is not None:
        t2_new, capped = cap_t2_with_expected_move(plan["entry"], float(plan["t2"]), direction, exp_move_val, session)
        plan["t2"] = t2_new
        if capped:
            reasons.append("T2 capped")

    key = f"{direction}:{safe_f1(zone_mid(acting_zone))}:{trigger}"
    if not STATE.can_signal(key):
        return

    conf = confidence_percent(score, direction, bias, session, STATE.market_label, liq_state)

    dist_t1 = abs(float(plan["t1"]) - price_now) if plan.get("t1") is not None else None
    dist_t2 = abs(float(plan["t2"]) - price_now) if plan.get("t2") is not None else None
    p1, p2 = probability_t1_t2(conf, rr, dist_t1, dist_t2, exp_move_val, liq_state, STATE.market_label)

    eta = None
    if plan.get("t1") is not None:
        eta = eta_to_t1_minutes(df_5m, price_now, float(plan["t1"]), direction, liq_state)

    msg = signal_message(
        session=session,
        symbol=symbol,
        market_state_str=market_state_str,
        liq_state=liq_state,
        level_now=price_now,
        direction=direction,
        acting_zone=acting_zone,
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
        "zone": acting_zone,
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
        f"- Direction: Bias(1H/4H) + Reaction Zones\n"
        f"- Zones: Equal High/Low + Wick Clusters + Sweep + Displacement\n"
        f"- Mid-range trades still possible if score/confidence support them\n"
        f"- Stop: HardStop({CFG.hard_stop_buffer_pts:.1f} pts) + 5m close confirm\n"
        f"- Extras: Confidence + RR + Probability + Expected Move + ETA"
    )

    while True:
        try:
            evaluate_once()
        except Exception as e:
            err = repr(e)
            print("[ERROR]", err)

            transient_errors = (
                "RemoteDisconnected",
                "Connection aborted",
                "ConnectionResetError",
                "ReadTimeout",
                "ConnectTimeout",
            )

            if any(x in err for x in transient_errors):
                time.sleep(CFG.loop_sleep_seconds)
                continue

            send_telegram(f"❌ {CFG.user_title}: خطأ - {err}")
            time.sleep(CFG.loop_sleep_seconds)


if __name__ == "__main__":
    main()
