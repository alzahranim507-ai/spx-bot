# -*- coding: utf-8 -*-
"""
backtest_bot.py
Backtest for Hunter WICK PRO logic on TradingView FOREXCOM:SPX500

What it does:
- Uses the SAME core logic as the live bot:
  * structure bias
  * key levels
  * wick rejection near level
  * break & retest
  * rejection
  * momentum shift
  * stoch cross
  * confidence / trade type
- Simulates one trade at a time over historical 5m candles
- Prints summary:
  * total signals
  * triggered trades
  * stop hits
  * T1/T2/T3 hits
  * no-trigger trades
  * win rate

Notes:
- Conservative candle handling:
  if stop and target are both touched in the same candle after entry,
  stop is assumed first.
- No Telegram here
- No infinite loop
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tvDatafeed import TvDatafeed, Interval

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
    tv_symbol: str = "SPX500"
    tv_exchange: str = "FOREXCOM"
    tv_username: str = os.getenv("TV_USERNAME", "")
    tv_password: str = os.getenv("TV_PASSWORD", "")

    user_title: str = "دكتور محمد"

    tz_riyadh: str = "Asia/Riyadh"
    tz_ny: str = "America/New_York"

    # Structure / pivots
    pivot_left: int = 3
    pivot_right: int = 3

    # Level detection / clustering
    level_touch_tolerance_frac: float = 0.0013
    level_cluster_tolerance_frac: float = 0.0010
    max_key_levels: int = 6

    # Scoring / quality
    score_threshold: int = 3
    min_rr_to_t1: float = 1.35

    # Market state
    adx_window: int = 14
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # Liquidity
    liquidity_lookback_5m: int = 60
    liquidity_thresholds: tuple = (0.60, 1.25, 2.00)

    # Expected move
    expected_move_atr_window: int = 14

    # Wick cluster
    wick_cluster_lookback_5m: int = 10
    wick_ratio_strong: float = 0.45
    wick_cluster_min_hits: int = 3
    wick_near_level_tolerance_frac: float = 0.0010
    wick_min_abs_pts: float = 1.2

    # Trade strength
    strong_score_threshold: int = 5
    strong_conf_threshold: int = 62
    standard_conf_threshold: int = 50

    # Dynamic targets
    enable_t3: bool = True
    atr_fallback_t1_mult_weak: float = 0.28
    atr_fallback_t2_mult_weak: float = 0.45
    atr_fallback_t1_mult_standard: float = 0.35
    atr_fallback_t2_mult_standard: float = 0.65
    atr_fallback_t3_mult_strong: float = 1.00

    # Entry logic
    aggressive_entry_for_strong_wick: bool = True

    # Session filter
    offhours_min_score: int = 4
    offhours_block_weak: bool = True

    # Bars to fetch
    bars_5m: int = 4500
    bars_15m: int = 1800
    bars_1h: int = 1800

    # Backtest behavior
    warmup_5m_bars: int = 500
    max_future_bars: int = 72   # 72 x 5m = 6 hours to resolve a trade


CFG = Config()


# =========================
# TradingView client
# =========================

def make_tv_client() -> TvDatafeed:
    if CFG.tv_username and CFG.tv_password:
        return TvDatafeed(CFG.tv_username, CFG.tv_password)
    return TvDatafeed()

TV = make_tv_client()


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

def now_ny_from_ts(ts: pd.Timestamp) -> datetime:
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_NY)

def session_label_from_ts(ts: pd.Timestamp) -> str:
    t = now_ny_from_ts(ts)
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
    try:
        b = int(float(level) // 10) * 10
        return f"{b}x"
    except Exception:
        return "N/A"


# =========================
# TradingView fetching
# =========================

def _tv_get_hist(symbol: str, exchange: str, interval: Interval, n_bars: int) -> pd.DataFrame:
    df = TV.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        n_bars=n_bars,
    )

    if df is None or df.empty:
        raise RuntimeError(f"TradingView empty data: {exchange}:{symbol} interval={interval} n_bars={n_bars}")

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    for c in ["open", "high", "low", "close"]:
        if c not in out.columns:
            raise RuntimeError(f"Missing column '{c}' in TradingView data")

    if "volume" not in out.columns:
        out["volume"] = np.nan

    out = out.sort_index()

    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out

def fetch_timeframes():
    symbol = CFG.tv_symbol
    exchange = CFG.tv_exchange

    df_5m = _tv_get_hist(symbol, exchange, Interval.in_5_minute, CFG.bars_5m)
    df_15m = _tv_get_hist(symbol, exchange, Interval.in_15_minute, CFG.bars_15m)
    df_1h = _tv_get_hist(symbol, exchange, Interval.in_1_hour, CFG.bars_1h)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_4h = df_1h.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])

    return f"{exchange}:{symbol}", df_4h, df_1h, df_15m, df_5m


# =========================
# Core analysis functions
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
    def bias_from(df):
        hi_idx, lo_idx = find_pivots(df["high"], CFG.pivot_left, CFG.pivot_right)
        highs = [float(df["high"].iloc[i]) for i in hi_idx][-2:]
        lows = [float(df["low"].iloc[i]) for i in lo_idx][-2:]

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
    swing_lows = [float(df_1h["low"].iloc[i]) for i in lo_idx][-12:] if lo_idx else []

    recent = df_15m.tail(220)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    candidates = swing_highs + swing_lows + [range_hi, range_lo]
    candidates = [float(x) for x in candidates if np.isfinite(x)]

    merged = cluster_levels(candidates, CFG.level_cluster_tolerance_frac, price)
    merged = sorted(merged, key=lambda x: abs(x - price))[:max(CFG.max_key_levels * 3, 18)]
    merged = sorted(cluster_levels(merged, CFG.level_cluster_tolerance_frac, price))

    if len(merged) > CFG.max_key_levels:
        closest = sorted(merged, key=lambda x: abs(x - price))[:CFG.max_key_levels - 2]
        merged = sorted(cluster_levels(closest + [min(merged), max(merged)], CFG.level_cluster_tolerance_frac, price))

    return merged

def near_level(price: float, level: float, tol_frac: float) -> bool:
    return abs(price - level) / max(price, 1e-9) <= tol_frac

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
        return lower / rng >= 0.25
    if direction == "SELL":
        return upper / rng >= 0.25
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

def liquidity_state(df_5m: pd.DataFrame) -> tuple[str, float | None]:
    if "volume" not in df_5m.columns:
        return "Normal", None

    v = df_5m["volume"].astype(float)
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

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, prev_label: str = "Range"):
    if len(df_1h) < (CFG.adx_window + 5):
        return "Weak", "Neutral", None

    adx = ADXIndicator(
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=CFG.adx_window
    ).adx()

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
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=CFG.expected_move_atr_window,
    ).average_true_range()

    val = float(atr.iloc[-1])
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def wick_cluster_near_level(df_5m: pd.DataFrame, level: float) -> dict:
    w = df_5m.tail(min(CFG.wick_cluster_lookback_5m, len(df_5m))).copy()
    if w.empty:
        return {
            "upper_cluster": False,
            "lower_cluster": False,
            "upper_hits": 0,
            "lower_hits": 0,
            "bucket": level_bucket_x(level)
        }

    upper_hits = 0
    lower_hits = 0

    for _, r in w.iterrows():
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        rng = max(h - l, 1e-9)

        upper = h - max(o, c)
        lower = min(o, c) - l

        near = (
            abs(h - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac
            or abs(l - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac
            or abs(c - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac
        )

        if not near:
            continue

        if upper >= CFG.wick_min_abs_pts and (upper / rng) >= CFG.wick_ratio_strong:
            upper_hits += 1
        if lower >= CFG.wick_min_abs_pts and (lower / rng) >= CFG.wick_ratio_strong:
            lower_hits += 1

    return {
        "upper_cluster": upper_hits >= CFG.wick_cluster_min_hits,
        "lower_cluster": lower_hits >= CFG.wick_cluster_min_hits,
        "upper_hits": upper_hits,
        "lower_hits": lower_hits,
        "bucket": level_bucket_x(level),
    }

def score_level_importance(df_5m: pd.DataFrame, level_now: float, level: float) -> float:
    wick_info = wick_cluster_near_level(df_5m, level)
    distance_pts = abs(level_now - level)

    score = 0.0
    score += max(0.0, 20.0 - distance_pts) * 0.15
    score += float(wick_info["upper_hits"] + wick_info["lower_hits"]) * 1.4

    if near_level(level_now, level, CFG.level_touch_tolerance_frac):
        score += 2.0

    recent = df_5m.tail(30)
    touches = 0
    for _, r in recent.iterrows():
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        if abs(h - level) <= 2.0 or abs(l - level) <= 2.0 or abs(c - level) <= 2.0:
            touches += 1
    score += touches * 0.4

    return float(score)

def choose_best_level(df_5m: pd.DataFrame, level_now: float, key_levels: list) -> tuple[float, dict]:
    ranked = []
    for lvl in key_levels:
        wick_info = wick_cluster_near_level(df_5m, float(lvl))
        imp = score_level_importance(df_5m, level_now, float(lvl))
        ranked.append((imp, float(lvl), wick_info))

    ranked.sort(key=lambda x: x[0], reverse=True)
    near_candidates = [x for x in ranked if abs(x[1] - level_now) <= 18.0]
    chosen = near_candidates[0] if near_candidates else ranked[0]
    return chosen[1], chosen[2]

def classify_trade_strength(score: int, conf: int) -> str:
    if score >= CFG.strong_score_threshold or conf >= CFG.strong_conf_threshold:
        return "Strong"
    if conf >= CFG.standard_conf_threshold or score >= 4:
        return "Standard"
    return "Weak"

def atr_fallback_targets(entry: float, direction: str, exp_move_val: float | None, trade_type: str):
    if exp_move_val is None or not np.isfinite(exp_move_val):
        return None, None, None

    if trade_type == "Weak":
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_weak
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_weak
        d3 = None
    elif trade_type == "Standard":
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_standard
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_standard
        d3 = None
    else:
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_standard
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_standard
        d3 = exp_move_val * CFG.atr_fallback_t3_mult_strong if CFG.enable_t3 else None

    if direction == "BUY":
        return (
            entry + d1 if d1 is not None else None,
            entry + d2 if d2 is not None else None,
            entry + d3 if d3 is not None else None,
        )
    else:
        return (
            entry - d1 if d1 is not None else None,
            entry - d2 if d2 is not None else None,
            entry - d3 if d3 is not None else None,
        )

def pick_targets(levels: list, entry: float, direction: str):
    if not levels:
        return None, None
    if direction == "BUY":
        above = sorted([lvl for lvl in levels if lvl > entry])
        return (
            above[0] if len(above) >= 1 else None,
            above[1] if len(above) >= 2 else None
        )
    if direction == "SELL":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        return (
            below[0] if len(below) >= 1 else None,
            below[1] if len(below) >= 2 else None
        )
    return None, None

def compute_trade_plan(
    df_5m: pd.DataFrame,
    levels: list,
    level_hit: float,
    direction: str,
    trigger: str,
    trade_type: str,
    exp_move_val: float | None,
):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    buffer = max(price * 0.0002, 0.5)

    if trigger == "Wick Rejection near Level":
        if direction == "BUY":
            if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                entry = float(max(price, level_hit + buffer * 0.4))
            else:
                entry = float(max(price, (last_high + level_hit) / 2.0))
            stop = float(min(last_low, level_hit) - buffer)
        else:
            if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                entry = float(min(price, level_hit - buffer * 0.4))
            else:
                entry = float(min(price, (last_low + level_hit) / 2.0))
            stop = float(max(last_high, level_hit) + buffer)

    elif trigger == "Rejection":
        if direction == "BUY":
            entry = float(max(price, last_high - buffer * 0.4))
            stop = float(min(last_low, level_hit) - buffer)
        else:
            entry = float(min(price, last_low + buffer * 0.4))
            stop = float(max(last_high, level_hit) + buffer)
    else:
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer))
            stop = float(level_hit - (price * 0.0013) - buffer)
        else:
            entry = float(min(price, level_hit - buffer))
            stop = float(level_hit + (price * 0.0013) + buffer)

    t1, t2 = pick_targets(levels, entry, direction)
    atr_t1, atr_t2, atr_t3 = atr_fallback_targets(entry, direction, exp_move_val, trade_type)

    if t1 is None:
        t1 = atr_t1
    if t2 is None:
        t2 = atr_t2

    t3 = atr_t3 if trade_type == "Strong" else None

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = reward / risk if risk > 0 else None

    return {
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "rr": rr,
        "trade_type": trade_type,
    }

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

def score_setup(df_5m: pd.DataFrame, level_hit: float, direction: str, wick_info: dict) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    trigger = None

    wick_reason = None
    if wick_info.get("upper_cluster") and direction == "SELL":
        wick_reason = f"Upper-wick rejection cluster near {wick_info.get('bucket','N/A')}"
    if wick_info.get("lower_cluster") and direction == "BUY":
        wick_reason = f"Lower-wick cluster near support {wick_info.get('bucket','N/A')}"

    if wick_reason:
        score += 3
        reasons.append("Wick Rejection near Level")
        reasons.append(wick_reason)
        trigger = "Wick Rejection near Level"

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
# Signal generator from real bot logic
# =========================

def generate_signal_from_data(df_4h, df_1h, df_15m, df_5m):
    if len(df_5m) < 60 or len(df_15m) < 100 or len(df_1h) < 100 or len(df_4h) < 30:
        return None

    df_5m = compute_indicators_5m(df_5m)
    current_ts = df_5m.index[-1]
    session = session_label_from_ts(current_ts)
    level_now = float(df_5m["close"].iloc[-1])

    market_label, _, _ = compute_market_state(df_1h, df_4h, "Range")
    liq_state, _ = liquidity_state(df_5m)
    bias = structure_bias(df_1h, df_4h)
    key_levels = extract_key_levels(df_15m, df_1h)
    exp_move_val = expected_move_1h(df_1h)

    if not key_levels:
        return None

    level_hit, wick_info = choose_best_level(df_5m, level_now, key_levels)

    if bias == "Bullish":
        direction = "BUY"
    elif bias == "Bearish":
        direction = "SELL"
    else:
        direction = "BUY" if momentum_shift(df_5m, "BUY") else "SELL"

    sell_confirm = (
        momentum_shift(df_5m, "SELL")
        or stoch_cross(df_5m, "SELL")
        or break_retest(df_5m, level_hit, "SELL")
    )
    buy_confirm = (
        momentum_shift(df_5m, "BUY")
        or stoch_cross(df_5m, "BUY")
        or break_retest(df_5m, level_hit, "BUY")
    )

    blocked = False

    if direction == "BUY" and wick_info.get("upper_cluster"):
        if sell_confirm:
            direction = "SELL"
        else:
            blocked = True

    if direction == "SELL" and wick_info.get("lower_cluster"):
        if buy_confirm:
            direction = "BUY"
        else:
            blocked = True

    if blocked:
        return None

    score, reasons, trigger = score_setup(df_5m, level_hit, direction, wick_info)
    if score <= 0 or score < CFG.score_threshold:
        return None

    conf = confidence_percent(score, direction, bias, session, market_label, liq_state)
    trade_type = classify_trade_strength(score, conf)

    if session in ("After-Hours", "Pre-Market"):
        if score < CFG.offhours_min_score:
            return None
        if CFG.offhours_block_weak and trade_type == "Weak":
            return None

    plan = compute_trade_plan(
        df_5m=df_5m,
        levels=key_levels,
        level_hit=level_hit,
        direction=direction,
        trigger=trigger,
        trade_type=trade_type,
        exp_move_val=exp_move_val,
    )

    rr = plan.get("rr")
    if rr is not None and np.isfinite(float(rr)) and float(rr) < CFG.min_rr_to_t1:
        return None

    return {
        "time": current_ts,
        "direction": direction,
        "level_now": level_now,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "t1": None if plan.get("t1") is None else float(plan["t1"]),
        "t2": None if plan.get("t2") is None else float(plan["t2"]),
        "t3": None if plan.get("t3") is None else float(plan["t3"]),
        "rr": None if plan.get("rr") is None else float(plan["rr"]),
        "score": int(score),
        "confidence": int(conf),
        "trade_type": trade_type,
        "trigger": trigger,
        "reasons": reasons,
        "session": session,
        "market_label": market_label,
        "liq_state": liq_state,
    }


# =========================
# Trade simulator
# =========================

def simulate_trade_outcome(trade: dict, future_5m: pd.DataFrame, max_bars: int = 72):
    """
    Conservative handling:
    - pending until entry touched
    - once live, if stop and target touched in same candle => stop first
    """
    direction = trade["direction"]
    entry = trade["entry"]
    stop = trade["stop"]
    t1 = trade.get("t1")
    t2 = trade.get("t2")
    t3 = trade.get("t3")

    triggered = False
    t1_hit = False
    t2_hit = False
    t3_hit = False

    bars_to_check = future_5m.head(max_bars)

    for idx, (_, row) in enumerate(bars_to_check.iterrows(), start=1):
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        if not triggered:
            if direction == "BUY":
                if h >= entry:
                    triggered = True
                else:
                    continue
            else:
                if l <= entry:
                    triggered = True
                else:
                    continue

        # Once trade is live, conservative intrabar order:
        # assume stop first if both touched in same candle

        if direction == "BUY":
            if l <= stop:
                return {
                    "status": "STOP",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": stop,
                    "r_result": -1.0 if not t1_hit else 0.0,
                }

            if t1 is not None and not t1_hit and h >= t1:
                t1_hit = True
            if t2 is not None and not t2_hit and h >= t2:
                t2_hit = True
            if t3 is not None and not t3_hit and h >= t3:
                t3_hit = True
                return {
                    "status": "T3",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": t3,
                    "r_result": 3.0,
                }

            if t2_hit:
                return {
                    "status": "T2",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": t2,
                    "r_result": 2.0,
                }

        else:  # SELL
            if h >= stop:
                return {
                    "status": "STOP",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": stop,
                    "r_result": -1.0 if not t1_hit else 0.0,
                }

            if t1 is not None and not t1_hit and l <= t1:
                t1_hit = True
            if t2 is not None and not t2_hit and l <= t2:
                t2_hit = True
            if t3 is not None and not t3_hit and l <= t3:
                t3_hit = True
                return {
                    "status": "T3",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": t3,
                    "r_result": 3.0,
                }

            if t2_hit:
                return {
                    "status": "T2",
                    "triggered": True,
                    "t1_hit": t1_hit,
                    "t2_hit": t2_hit,
                    "t3_hit": t3_hit,
                    "exit_bar": idx,
                    "exit_price": t2,
                    "r_result": 2.0,
                }

    if not triggered:
        return {
            "status": "NO_TRIGGER",
            "triggered": False,
            "t1_hit": False,
            "t2_hit": False,
            "t3_hit": False,
            "exit_bar": max_bars,
            "exit_price": None,
            "r_result": 0.0,
        }

    if t1_hit and not t2_hit:
        return {
            "status": "T1_ONLY_TIMEOUT",
            "triggered": True,
            "t1_hit": True,
            "t2_hit": False,
            "t3_hit": False,
            "exit_bar": max_bars,
            "exit_price": t1,
            "r_result": 1.0,
        }

    return {
        "status": "TIMEOUT",
        "triggered": True,
        "t1_hit": t1_hit,
        "t2_hit": t2_hit,
        "t3_hit": t3_hit,
        "exit_bar": max_bars,
        "exit_price": c if "c" in locals() else None,
        "r_result": 0.0,
    }


# =========================
# Backtest runner
# =========================

def run_backtest():
    print("starting backtest...")

    symbol, df_4h_all, df_1h_all, df_15m_all, df_5m_all = fetch_timeframes()

    print("symbol:", symbol)
    print("5m bars:", len(df_5m_all))
    print("15m bars:", len(df_15m_all))
    print("1h bars:", len(df_1h_all))
    print("4h bars:", len(df_4h_all))

    results = []
    i = CFG.warmup_5m_bars

    while i < len(df_5m_all) - CFG.max_future_bars - 2:
        current_ts = df_5m_all.index[i]

        df_5m = df_5m_all.iloc[:i + 1].copy()
        df_15m = df_15m_all[df_15m_all.index <= current_ts].copy()
        df_1h = df_1h_all[df_1h_all.index <= current_ts].copy()
        df_4h = df_4h_all[df_4h_all.index <= current_ts].copy()

        signal = generate_signal_from_data(df_4h, df_1h, df_15m, df_5m)

        if signal is None:
            i += 1
            continue

        future_5m = df_5m_all.iloc[i + 1:i + 1 + CFG.max_future_bars].copy()
        outcome = simulate_trade_outcome(signal, future_5m, CFG.max_future_bars)

        results.append({
            "time": signal["time"],
            "session": signal["session"],
            "direction": signal["direction"],
            "trade_type": signal["trade_type"],
            "score": signal["score"],
            "confidence": signal["confidence"],
            "entry": signal["entry"],
            "stop": signal["stop"],
            "t1": signal["t1"],
            "t2": signal["t2"],
            "t3": signal["t3"],
            "status": outcome["status"],
            "triggered": outcome["triggered"],
            "t1_hit": outcome["t1_hit"],
            "t2_hit": outcome["t2_hit"],
            "t3_hit": outcome["t3_hit"],
            "r_result": outcome["r_result"],
            "bars_in_trade": outcome["exit_bar"],
        })

        # one trade at a time: jump forward until this trade resolves
        i += max(1, outcome["exit_bar"])

    res = pd.DataFrame(results)

    print("\n------ BACKTEST RESULTS ------")
    print("Signals:", len(res))

    if res.empty:
        print("No signals found.")
        return

    triggered = int(res["triggered"].sum())
    no_trigger = int((res["status"] == "NO_TRIGGER").sum())
    stop_hits = int((res["status"] == "STOP").sum())
    t1_only = int((res["status"] == "T1_ONLY_TIMEOUT").sum())
    t2_hits = int((res["status"] == "T2").sum())
    t3_hits = int((res["status"] == "T3").sum())
    timeouts = int((res["status"] == "TIMEOUT").sum())

    wins_t1_plus = int(((res["t1_hit"] == True) | (res["status"].isin(["T2", "T3", "T1_ONLY_TIMEOUT"]))).sum())
    net_r = round(float(res["r_result"].sum()), 2)
    avg_conf = round(float(res["confidence"].mean()), 2)
    avg_score = round(float(res["score"].mean()), 2)

    print("Triggered trades:", triggered)
    print("No trigger:", no_trigger)
    print("Stop hits:", stop_hits)
    print("T1 only:", t1_only)
    print("T2 hits:", t2_hits)
    print("T3 hits:", t3_hits)
    print("Timeouts:", timeouts)
    print("Wins (T1+):", wins_t1_plus)

    if triggered > 0:
        winrate = round((wins_t1_plus / triggered) * 100, 2)
        stoprate = round((stop_hits / triggered) * 100, 2)
        print("Winrate (T1+ / triggered):", winrate, "%")
        print("Stop rate:", stoprate, "%")

    print("Average confidence:", avg_conf)
    print("Average score:", avg_score)
    print("Net R result:", net_r)

    print("\nBy trade type:")
    for tt in ["Weak", "Standard", "Strong"]:
        sub = res[res["trade_type"] == tt]
        if len(sub) == 0:
            continue
        sub_triggered = int(sub["triggered"].sum())
        sub_wins = int(((sub["t1_hit"] == True) | (sub["status"].isin(["T2", "T3", "T1_ONLY_TIMEOUT"]))).sum())
        sub_net_r = round(float(sub["r_result"].sum()), 2)
        wr = round((sub_wins / sub_triggered) * 100, 2) if sub_triggered > 0 else 0.0
        print(f"{tt}: signals={len(sub)}, triggered={sub_triggered}, winrate={wr}%, netR={sub_net_r}")

    # optional: save results
    out_path = "backtest_results.csv"
    res.to_csv(out_path, index=False)
    print("\nSaved detailed results to:", out_path)


if __name__ == "__main__":
    run_backtest()
    print("backtest finished successfully")
