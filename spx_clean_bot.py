# -*- coding: utf-8 -*-
"""
SPX500 Trading Bot (Hunter Smart - 3-State Regime + Hybrid + Momentum)
For: دكتور محمد

PART 1/4
- Imports
- Config
- TradingView / Telegram / Helpers
- Fetching
- Structure / Levels
- Indicators
- Regime classification
- Bot state
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from tvDatafeed import TvDatafeed, Interval
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    # TradingView
    tv_symbol: str = "SPX500"
    tv_exchange: str = "FOREXCOM"
    tv_username: str = os.getenv("TV_USERNAME", "")
    tv_password: str = os.getenv("TV_PASSWORD", "")

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    user_title: str = "دكتور محمد"

    # Timezones
    tz_riyadh: str = "Asia/Riyadh"
    tz_ny: str = "America/New_York"

    # Loop
    loop_sleep_seconds: int = 20

    # Fetch bars
    bars_5m: int = 1200
    bars_15m: int = 900
    bars_1h: int = 500

    # TradingView resilience
    tv_retry_attempts: int = 3
    tv_retry_sleep_base: float = 2.0
    tv_fallback_bars: tuple = (1000, 800, 700, 600, 500, 400, 300)

    # Error reporting
    error_notify_cooldown_minutes: int = 15

    # Structure / pivots
    pivot_left: int = 3
    pivot_right: int = 3

    # Level detection
    level_cluster_tolerance_frac: float = 0.0010
    max_key_levels: int = 8
    level_zone_near_pts: float = 18.0
    level_touch_tolerance_frac: float = 0.0013
    level_min_touch_count: int = 2
    level_strong_touch_count: int = 3

    # Signal cooldown
    signal_cooldown_minutes: int = 14

    # Hourly update
    hourly_update: bool = True

    # Daily reset
    daily_reset_enabled: bool = True
    daily_reset_hour: int = 0
    daily_reset_minute: int = 0
    daily_reset_window_minutes: int = 5
    no_signal_after_reset_minutes: int = 8

    # Indicator windows
    adx_window: int = 14
    expected_move_atr_window: int = 14
    atr15_window: int = 14

    # Liquidity
    liquidity_lookback_5m: int = 60
    liquidity_thresholds: tuple = (0.60, 1.25, 2.00)

    # Range / Regime
    enable_regime_filters: bool = True
    range_lookback_15m: int = 96
    range_min_width_pts: float = 16.0

    # ADX hysteresis
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # Clean range quality
    range_touch_tolerance_atr_mult: float = 0.25
    range_min_touches_each_side: int = 2
    range_max_sweeps_total: int = 6
    range_max_width_atr: float = 6.0
    range_max_slope_atr: float = 0.55

    # Mid-zone
    enable_mid_range_skip: bool = True
    mid_no_trade_pct: float = 0.18
    mid_atr_floor_mult: float = 0.40

    # Edge zones for clean range
    enable_edge_only_range: bool = True
    edge_zone_pct: float = 0.32
    edge_atr_floor_mult: float = 0.55

    # Fake breakout / trap filter
    enable_fake_break_filter: bool = True
    fake_break_buffer_atr_mult: float = 0.10
    fake_break_wick_body_ratio: float = 1.50
    fake_break_cooldown_bars_5m: int = 3

    # Closed-bar confirmation
    require_closed_bar_confirmation: bool = True

    # Trend protection
    enable_trend_protection: bool = True

    # Anti chase
    max_chase_distance_pts_clean_range: float = 16.0
    max_chase_distance_pts_trend: float = 12.0
    max_chase_distance_pts_messy: float = 8.0

    # Score / RR
    score_threshold: int = 3
    strong_score_threshold: int = 5
    strong_conf_threshold: int = 62
    standard_conf_threshold: int = 50
    min_rr_to_t1: float = 1.20

    # Off-hours control
    offhours_min_score: int = 4
    offhours_block_weak: bool = True

    # Momentum confirmation
    require_momentum_confirmation: bool = True

    # Targets
    enable_t3: bool = True
    enable_dynamic_t4: bool = True
    atr_fallback_t1_mult_weak: float = 0.28
    atr_fallback_t2_mult_weak: float = 0.45
    atr_fallback_t1_mult_standard: float = 0.35
    atr_fallback_t2_mult_standard: float = 0.65
    atr_fallback_t3_mult_strong: float = 1.00
    dynamic_t3_atr_mult: float = 1.15
    dynamic_t4_atr_mult: float = 1.45

    # Stop logic
    hard_stop_enabled: bool = True
    hard_stop_buffer_pts: float = 1.0
    stop_confirm_by_5m_close: bool = True
    stop_confirm_minutes: int = 5
    move_stop_to_be_on_t1: bool = True
    move_stop_to_t1_on_t2: bool = True
    move_stop_to_t2_on_t3: bool = True

    # Entry tuning
    aggressive_entry_for_strong_wick: bool = True

    # Hybrid
    enable_hybrid_entries: bool = True
    enable_early_entries_in_clean_range: bool = True
    enable_early_entries_in_messy: bool = False

    # Early entry detection
    early_two_candle_reversal: bool = True
    early_need_rejection_or_2bar: bool = True

    # Position sizing hint (for Telegram text only)
    strong_risk_scale: float = 1.00
    early_risk_scale: float = 0.50

    # Messy mode
    messy_allows_only_strong_break_retest: bool = True

    # Market state refresh
    market_state_update_minutes: int = 10


CFG = Config()


# =========================================================
# TradingView client
# =========================================================

def make_tv_client() -> TvDatafeed:
    if CFG.tv_username and CFG.tv_password:
        return TvDatafeed(CFG.tv_username, CFG.tv_password)
    return TvDatafeed()


TV = make_tv_client()
LAST_GOOD_DATA = {}


# =========================================================
# TZ helpers
# =========================================================

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


# =========================================================
# Safe helpers
# =========================================================

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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def level_bucket_x(level: float) -> str:
    try:
        b = int(float(level) // 10) * 10
        return f"{b}x"
    except Exception:
        return "N/A"


# =========================================================
# Telegram
# =========================================================

def send_telegram(text: str):
    if not CFG.telegram_bot_token or not CFG.telegram_chat_id:
        print("[WARN] Telegram env vars not set. Printing:\n", text)
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


# =========================================================
# TradingView fetching
# =========================================================

def _normalize_tv_df(df: pd.DataFrame) -> pd.DataFrame:
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
    if out.empty:
        raise RuntimeError("Normalized TradingView dataframe is empty")

    return out


def reinit_tv_client():
    global TV
    TV = make_tv_client()
    print("[INFO] Reinitialized TradingView client")


def _tv_get_hist(symbol: str, exchange: str, interval: Interval, n_bars: int) -> pd.DataFrame:
    global TV, LAST_GOOD_DATA

    cache_key = f"{exchange}:{symbol}:{interval}"
    bar_options = [n_bars]
    for x in CFG.tv_fallback_bars:
        if x not in bar_options:
            bar_options.append(x)

    last_err = None

    for attempt in range(CFG.tv_retry_attempts):
        for bars in bar_options:
            try:
                df = TV.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    n_bars=bars,
                )

                if df is None or df.empty:
                    last_err = RuntimeError(
                        f"TradingView empty data: {exchange}:{symbol} interval={interval} n_bars={bars}"
                    )
                    continue

                out = _normalize_tv_df(df)
                min_needed = 120 if interval == Interval.in_1_hour else 150
                if len(out) < min_needed:
                    last_err = RuntimeError(
                        f"TradingView too few bars: {exchange}:{symbol} interval={interval} got={len(out)}"
                    )
                    continue

                LAST_GOOD_DATA[cache_key] = out.copy()
                return out

            except Exception as e:
                last_err = e

        try:
            reinit_tv_client()
        except Exception as reinit_err:
            last_err = reinit_err

        time.sleep(CFG.tv_retry_sleep_base * (attempt + 1))

    cached = LAST_GOOD_DATA.get(cache_key)
    if cached is not None and not cached.empty:
        print(f"[WARN] Using cached data for {cache_key}: {repr(last_err)}")
        return cached.copy()

    raise RuntimeError(
        f"TradingView fetch failed repeatedly: {exchange}:{symbol} interval={interval} | last_error={repr(last_err)}"
    )


def fetch_timeframes():
    symbol = CFG.tv_symbol
    exchange = CFG.tv_exchange

    df_5m = _tv_get_hist(symbol, exchange, Interval.in_5_minute, CFG.bars_5m)
    df_15m = _tv_get_hist(symbol, exchange, Interval.in_15_minute, CFG.bars_15m)
    df_1h = _tv_get_hist(symbol, exchange, Interval.in_1_hour, CFG.bars_1h)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_4h = df_1h.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])

    return f"{exchange}:{symbol}", df_4h, df_1h, df_15m, df_5m


# =========================================================
# Structure / Levels
# =========================================================

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
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx][-14:] if hi_idx else []
    swing_lows = [float(df_1h["low"].iloc[i]) for i in lo_idx][-14:] if lo_idx else []

    recent = df_15m.tail(220)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    candidates = swing_highs + swing_lows + [range_hi, range_lo]
    candidates = [float(x) for x in candidates if np.isfinite(x)]

    merged = cluster_levels(candidates, CFG.level_cluster_tolerance_frac, price)
    merged = sorted(merged, key=lambda x: abs(x - price))[: max(CFG.max_key_levels * 4, 20)]
    merged = sorted(cluster_levels(merged, CFG.level_cluster_tolerance_frac, price))

    if len(merged) > CFG.max_key_levels:
        closest = sorted(merged, key=lambda x: abs(x - price))[: CFG.max_key_levels - 2]
        merged = sorted(cluster_levels(
            closest + [min(merged), max(merged)],
            CFG.level_cluster_tolerance_frac,
            price
        ))

    return merged


def fmt_levels(levels: list) -> str:
    return ", ".join([f"{float(x):.1f}" for x in levels])


def near_level(price: float, level: float, tol_frac: float) -> bool:
    return abs(price - level) / max(price, 1e-9) <= tol_frac


# =========================================================
# Indicators
# =========================================================

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


def latest_atr_15m(df_15m: pd.DataFrame, window: int = None) -> float | None:
    if window is None:
        window = CFG.atr15_window

    if len(df_15m) < window + 3:
        return None

    atr = AverageTrueRange(
        high=df_15m["high"],
        low=df_15m["low"],
        close=df_15m["close"],
        window=window
    ).average_true_range()

    val = float(atr.iloc[-1])
    if not np.isfinite(val) or val <= 0:
        return None

    return val


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


def liquidity_state(df_5m: pd.DataFrame):
    if len(df_5m) < CFG.liquidity_lookback_5m + 5:
        return "Normal", None

    recent = df_5m.tail(CFG.liquidity_lookback_5m)

    ranges = (recent["high"] - recent["low"]).astype(float)
    avg_range = float(ranges.mean())

    current_range = float(df_5m["high"].iloc[-1] - df_5m["low"].iloc[-1])

    if avg_range <= 0:
        return "Normal", None

    ratio = current_range / avg_range

    low, normal, high = CFG.liquidity_thresholds

    if ratio < low:
        return "Low", ratio
    elif ratio < normal:
        return "Normal", ratio
    elif ratio < high:
        return "High", ratio
    else:
        return "Extreme", ratio


# =========================================================
# Regime classification helpers
# =========================================================

def compute_range_info(df_15m: pd.DataFrame) -> dict:
    look = min(CFG.range_lookback_15m, len(df_15m))
    recent = df_15m.tail(look).copy()

    r_high = float(recent["high"].max())
    r_low = float(recent["low"].min())
    r_mid = (r_high + r_low) / 2.0
    r_width = r_high - r_low

    atr15 = latest_atr_15m(df_15m)
    if atr15 is None:
        atr15 = max(r_width * 0.08, 1.0)

    width_atr = r_width / max(atr15, 1e-9)

    mid_dist = max(r_width * (CFG.mid_no_trade_pct / 2.0), atr15 * CFG.mid_atr_floor_mult)
    edge_dist = max(r_width * CFG.edge_zone_pct, atr15 * CFG.edge_atr_floor_mult)

    touch_tol = atr15 * CFG.range_touch_tolerance_atr_mult
    highs = recent["high"].astype(float).values
    lows = recent["low"].astype(float).values
    closes = recent["close"].astype(float).values

    touches_high = int(np.sum(np.abs(highs - r_high) <= touch_tol))
    touches_low = int(np.sum(np.abs(lows - r_low) <= touch_tol))

    buf = atr15 * CFG.fake_break_buffer_atr_mult
    sweep_up = int(np.sum((highs > (r_high + buf)) & (closes < r_high)))
    sweep_dn = int(np.sum((lows < (r_low - buf)) & (closes > r_low)))
    sweeps_total = sweep_up + sweep_dn

    ema20 = recent["close"].ewm(span=20, adjust=False).mean()
    slope_ref = 10 if len(ema20) > 10 else max(1, len(ema20) - 1)
    slope_atr = (
        abs(float(ema20.iloc[-1]) - float(ema20.iloc[-1 - slope_ref])) / max(atr15, 1e-9)
        if len(ema20) > slope_ref else 0.0
    )

    valid = np.isfinite(r_high) and np.isfinite(r_low) and (r_width >= CFG.range_min_width_pts)

    clean_range = (
        valid
        and width_atr <= CFG.range_max_width_atr
        and touches_high >= CFG.range_min_touches_each_side
        and touches_low >= CFG.range_min_touches_each_side
        and sweeps_total <= CFG.range_max_sweeps_total
        and slope_atr <= CFG.range_max_slope_atr
    )

    return {
        "valid": bool(valid),
        "high": float(r_high),
        "low": float(r_low),
        "mid": float(r_mid),
        "width": float(r_width),
        "atr15": float(atr15),
        "width_atr": float(width_atr),
        "mid_dist": float(mid_dist),
        "edge_dist": float(edge_dist),
        "touch_tol": float(touch_tol),
        "touches_high": int(touches_high),
        "touches_low": int(touches_low),
        "sweep_up": int(sweep_up),
        "sweep_dn": int(sweep_dn),
        "sweeps_total": int(sweeps_total),
        "slope_atr": float(slope_atr),
        "clean_range": bool(clean_range),
    }


def price_in_mid_range(price: float, range_info: dict) -> bool:
    if not range_info.get("valid", False):
        return False
    return abs(price - range_info["mid"]) < range_info["mid_dist"]


def price_near_range_high(price: float, range_info: dict) -> bool:
    if not range_info.get("valid", False):
        return False
    return price >= (range_info["high"] - range_info["edge_dist"])


def price_near_range_low(price: float, range_info: dict) -> bool:
    if not range_info.get("valid", False):
        return False
    return price <= (range_info["low"] + range_info["edge_dist"])


def range_position_label(price: float, range_info: dict) -> str:
    if not range_info.get("valid", False):
        return "Unknown"
    if price_near_range_high(price, range_info):
        return "Upper Edge"
    if price_near_range_low(price, range_info):
        return "Lower Edge"
    if price_in_mid_range(price, range_info):
        return "Mid"
    return "Inner"


def compute_adx_and_market_direction(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    if len(df_1h) < (CFG.adx_window + 5):
        return None, "Neutral", "Weak"

    adx_ind = ADXIndicator(
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=CFG.adx_window,
    )
    adx_val = float(adx_ind.adx().iloc[-1])

    bias = structure_bias(df_1h, df_4h)
    direction = "Neutral"
    if bias == "Bullish":
        direction = "Bullish"
    elif bias == "Bearish":
        direction = "Bearish"

    return adx_val, direction, bias


def trend_direction_from_bias_and_state(bias: str, market_dir: str) -> str:
    if market_dir in ("Bullish", "Bearish"):
        return market_dir
    if bias in ("Bullish", "Bearish"):
        return bias
    return "Neutral"


def classify_market_regime(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_15m: pd.DataFrame, prev_label: str):
    adx_val, market_dir, bias = compute_adx_and_market_direction(df_1h, df_4h)
    range_info = compute_range_info(df_15m)

    if adx_val is None or not np.isfinite(adx_val):
        return {
            "label": "Weak",
            "dir": market_dir,
            "bias": bias,
            "adx": None,
            "range_info": range_info,
        }

    if adx_val >= CFG.adx_trending_on:
        label = "Trending"
    elif adx_val <= CFG.adx_range_on:
        label = "Clean Range" if range_info.get("clean_range", False) else "Messy"
    else:
        if prev_label == "Trending":
            label = "Trending"
        else:
            label = "Clean Range" if range_info.get("clean_range", False) else "Messy"

    return {
        "label": label,
        "dir": market_dir,
        "bias": bias,
        "adx": float(adx_val),
        "range_info": range_info,
    }


# =========================================================
# Bot State
# =========================================================

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
        self.market_bias = "Weak"
        self.market_adx = None
        self.range_info = {}

        self.last_error_notify_utc = None

        self.block_long_until_bar_5m = 0
        self.block_short_until_bar_5m = 0
        self.bar_counter_5m = 0

    def can_signal(self, key: str) -> bool:
        if self.no_signal_until_utc is not None and datetime.utcnow() < self.no_signal_until_utc:
            return False

        last = self.last_signal_ts.get(key)
        if last is None:
            return True

        return (datetime.utcnow() - last) >= timedelta(minutes=CFG.signal_cooldown_minutes)

    def mark_signal(self, key: str):
        self.last_signal_ts[key] = datetime.utcnow()

    def should_notify_error(self) -> bool:
        nowu = datetime.utcnow()
        if self.last_error_notify_utc is None:
            self.last_error_notify_utc = nowu
            return True

        if (nowu - self.last_error_notify_utc) >= timedelta(minutes=CFG.error_notify_cooldown_minutes):
            self.last_error_notify_utc = nowu
            return True

        return False

    def long_blocked(self) -> bool:
        return self.bar_counter_5m < self.block_long_until_bar_5m

    def short_blocked(self) -> bool:
        return self.bar_counter_5m < self.block_short_until_bar_5m

    def block_long(self, bars: int):
        self.block_long_until_bar_5m = max(self.block_long_until_bar_5m, self.bar_counter_5m + int(bars))

    def block_short(self, bars: int):
        self.block_short_until_bar_5m = max(self.block_short_until_bar_5m, self.bar_counter_5m + int(bars))


STATE = BotState()

# =========================================================
# Price action helpers
# =========================================================

def candle_parts(row: pd.Series) -> tuple[float, float, float, float, float, float, float]:
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = max(h - l, 1e-9)
    return o, h, l, c, body, upper, lower


def rejection_lite(df_5m: pd.DataFrame, direction: str) -> bool:
    c = df_5m.iloc[-1]
    _, h, l, _, _, upper, lower = candle_parts(c)
    rng = max(h - l, 1e-9)

    if direction == "BUY":
        return (lower / rng) >= 0.25
    if direction == "SELL":
        return (upper / rng) >= 0.25
    return False


def strong_rejection(df_5m: pd.DataFrame, direction: str) -> bool:
    c = df_5m.iloc[-1]
    _, h, l, _, body, upper, lower = candle_parts(c)
    rng = max(h - l, 1e-9)

    if direction == "BUY":
        return (lower / rng) >= 0.38 and lower > max(body, 0.8)
    if direction == "SELL":
        return (upper / rng) >= 0.38 and upper > max(body, 0.8)
    return False


def two_bar_reversal(df_5m: pd.DataFrame, direction: str) -> bool:
    if len(df_5m) < 3:
        return False

    c1 = df_5m.iloc[-1]
    c2 = df_5m.iloc[-2]

    if direction == "BUY":
        return float(c1["close"]) > float(c1["open"]) and float(c2["close"]) > float(c2["open"])
    if direction == "SELL":
        return float(c1["close"]) < float(c1["open"]) and float(c2["close"]) < float(c2["open"])
    return False


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


def strong_buy_confirmation(df_5m: pd.DataFrame) -> bool:
    if len(df_5m) < 5:
        return False

    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema20"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])
    k = float(df_5m["stochrsi_k"].iloc[-1])
    d = float(df_5m["stochrsi_d"].iloc[-1])

    return (c > ema) and (macd_hist > 0) and (k >= d)


def strong_sell_confirmation(df_5m: pd.DataFrame) -> bool:
    if len(df_5m) < 5:
        return False

    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema20"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])
    k = float(df_5m["stochrsi_k"].iloc[-1])
    d = float(df_5m["stochrsi_d"].iloc[-1])

    return (c < ema) and (macd_hist < 0) and (k <= d)


def momentum_breakout(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    if len(df_5m) < 8:
        return False

    closes = df_5m["close"].astype(float).tail(3).values
    ema = float(df_5m["ema20"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])

    prev_high = float(df_5m["high"].iloc[-6:-1].max())
    prev_low = float(df_5m["low"].iloc[-6:-1].min())

    if direction == "BUY":
        breakout_ref = max(float(level), prev_high)
        return (
            closes[0] < closes[1] < closes[2]
            and closes[2] > breakout_ref
            and closes[2] > ema
            and macd_hist > 0
        )

    if direction == "SELL":
        breakout_ref = min(float(level), prev_low)
        return (
            closes[0] > closes[1] > closes[2]
            and closes[2] < breakout_ref
            and closes[2] < ema
            and macd_hist < 0
        )

    return False


def break_retest(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    window = df_5m.tail(10)
    if len(window) < 4:
        return False

    last_close = float(window["close"].iloc[-1])
    prev_close = float(window["close"].iloc[-2])
    last_low = float(window["low"].iloc[-1])
    last_high = float(window["high"].iloc[-1])

    if direction == "BUY":
        broke = (window["close"] > level).sum() >= 2
        retest = abs(last_low - level) <= 2.0 or abs(prev_close - level) <= 2.0
        hold = last_close > level
        return bool(broke and retest and hold)

    if direction == "SELL":
        broke = (window["close"] < level).sum() >= 2
        retest = abs(last_high - level) <= 2.0 or abs(prev_close - level) <= 2.0
        hold = last_close < level
        return bool(broke and retest and hold)

    return False


# =========================================================
# Fake breakout / trap detection
# =========================================================

def detect_fake_breaks(df_15m: pd.DataFrame, range_info: dict, market_label: str) -> dict:
    out = {
        "bull_trap": False,
        "bear_trap": False,
        "reason": None,
    }

    if not CFG.enable_fake_break_filter:
        return out
    if market_label != "Clean Range":
        return out
    if not range_info.get("valid", False):
        return out
    if len(df_15m) < 3:
        return out

    last = df_15m.iloc[-1]
    _, h, l, c, body, upper, lower = candle_parts(last)

    wick_body_ratio_up = upper / max(body, 0.1)
    wick_body_ratio_dn = lower / max(body, 0.1)
    buf = range_info["atr15"] * CFG.fake_break_buffer_atr_mult

    bull_trap = (
        h > (range_info["high"] + buf)
        and c < range_info["high"]
        and wick_body_ratio_up >= CFG.fake_break_wick_body_ratio
    )

    bear_trap = (
        l < (range_info["low"] - buf)
        and c > range_info["low"]
        and wick_body_ratio_dn >= CFG.fake_break_wick_body_ratio
    )

    if bull_trap:
        out["bull_trap"] = True
        out["reason"] = "Bull Trap above range high"
    elif bear_trap:
        out["bear_trap"] = True
        out["reason"] = "Bear Trap below range low"

    return out


# =========================================================
# Wick / Level quality
# =========================================================

def wick_cluster_near_level(df_5m: pd.DataFrame, level: float) -> dict:
    w = df_5m.tail(10).copy()
    if w.empty:
        return {
            "upper_cluster": False,
            "lower_cluster": False,
            "upper_hits": 0,
            "lower_hits": 0,
            "bucket": level_bucket_x(level),
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
            (abs(h - level) / max(level, 1e-9) <= CFG.level_touch_tolerance_frac)
            or (abs(l - level) / max(level, 1e-9) <= CFG.level_touch_tolerance_frac)
            or (abs(c - level) / max(level, 1e-9) <= CFG.level_touch_tolerance_frac)
        )

        if not near:
            continue

        if upper >= 1.2 and (upper / rng) >= 0.45:
            upper_hits += 1
        if lower >= 1.2 and (lower / rng) >= 0.45:
            lower_hits += 1

    return {
        "upper_cluster": upper_hits >= 3,
        "lower_cluster": lower_hits >= 3,
        "upper_hits": int(upper_hits),
        "lower_hits": int(lower_hits),
        "bucket": level_bucket_x(level),
    }


def count_level_touches(df_5m: pd.DataFrame, level: float, lookback: int = 40, abs_tol: float = 2.0) -> int:
    recent = df_5m.tail(min(lookback, len(df_5m)))
    touches = 0

    for _, r in recent.iterrows():
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        if abs(h - level) <= abs_tol or abs(l - level) <= abs_tol or abs(c - level) <= abs_tol:
            touches += 1

    return touches


def level_quality_info(df_5m: pd.DataFrame, level_now: float, level: float) -> dict:
    wick_info = wick_cluster_near_level(df_5m, level)
    touches = count_level_touches(df_5m, level)
    distance_pts = abs(level_now - level)

    quality_score = 0.0
    quality_score += max(0.0, 22.0 - distance_pts) * 0.14
    quality_score += float(wick_info["upper_hits"] + wick_info["lower_hits"]) * 1.5
    quality_score += float(touches) * 0.45

    if near_level(level_now, level, CFG.level_touch_tolerance_frac):
        quality_score += 2.0

    strong = (
        touches >= CFG.level_strong_touch_count
        or wick_info["upper_hits"] >= 2
        or wick_info["lower_hits"] >= 2
        or wick_info["upper_cluster"]
        or wick_info["lower_cluster"]
    )

    tradable = (
        touches >= CFG.level_min_touch_count
        or wick_info["upper_hits"] >= 1
        or wick_info["lower_hits"] >= 1
    )

    return {
        "touches": touches,
        "distance_pts": distance_pts,
        "quality_score": float(quality_score),
        "strong": bool(strong),
        "tradable": bool(tradable),
        "wick_info": wick_info,
    }


def choose_best_level(df_5m: pd.DataFrame, level_now: float, key_levels: list) -> tuple[float, dict]:
    ranked = []
    for lvl in key_levels:
        info = level_quality_info(df_5m, level_now, float(lvl))
        ranked.append((info["quality_score"], float(lvl), info))

    ranked.sort(key=lambda x: x[0], reverse=True)
    near_candidates = [x for x in ranked if abs(x[1] - level_now) <= CFG.level_zone_near_pts]
    chosen = near_candidates[0] if near_candidates else ranked[0]
    return chosen[1], chosen[2]


# =========================================================
# Direction / confirmations
# =========================================================

def buy_confirmations(df_5m: pd.DataFrame, level_hit: float) -> bool:
    return (
        momentum_shift(df_5m, "BUY")
        or stoch_cross(df_5m, "BUY")
        or break_retest(df_5m, level_hit, "BUY")
        or rejection_lite(df_5m, "BUY")
    )


def sell_confirmations(df_5m: pd.DataFrame, level_hit: float) -> bool:
    return (
        momentum_shift(df_5m, "SELL")
        or stoch_cross(df_5m, "SELL")
        or break_retest(df_5m, level_hit, "SELL")
        or rejection_lite(df_5m, "SELL")
    )


def strong_setup_ready(df_5m: pd.DataFrame, level_hit: float, direction: str, trap_info: dict) -> bool:
    if direction == "BUY":
        return (
            trap_info.get("bear_trap", False)
            or strong_rejection(df_5m, "BUY")
            or break_retest(df_5m, level_hit, "BUY")
            or strong_buy_confirmation(df_5m)
        )

    if direction == "SELL":
        return (
            trap_info.get("bull_trap", False)
            or strong_rejection(df_5m, "SELL")
            or break_retest(df_5m, level_hit, "SELL")
            or strong_sell_confirmation(df_5m)
        )

    return False


def early_setup_ready(df_5m: pd.DataFrame, direction: str) -> bool:
    if direction == "BUY":
        if CFG.early_need_rejection_or_2bar:
            return rejection_lite(df_5m, "BUY") or two_bar_reversal(df_5m, "BUY")
        return two_bar_reversal(df_5m, "BUY")

    if direction == "SELL":
        if CFG.early_need_rejection_or_2bar:
            return rejection_lite(df_5m, "SELL") or two_bar_reversal(df_5m, "SELL")
        return two_bar_reversal(df_5m, "SELL")

    return False


def choose_direction_by_regime(
    df_5m: pd.DataFrame,
    level_hit: float,
    level_now: float,
    bias: str,
    range_info: dict,
    trap_info: dict,
) -> tuple[str | None, str | None, list[str]]:
    """
    returns: direction, entry_mode, notes
    entry_mode: Strong / Early / None
    """
    notes: list[str] = []

    buy_ok = buy_confirmations(df_5m, level_hit)
    sell_ok = sell_confirmations(df_5m, level_hit)

    # =====================================================
    # 1) TRENDING
    # =====================================================
    if STATE.market_label == "Trending":
        notes.append("Regime = Trending")

        if STATE.market_dir == "Bullish":
            if strong_setup_ready(df_5m, level_hit, "BUY", trap_info) and buy_ok:
                notes.append("Trend bullish → BUY only")
                return "BUY", "Strong", notes
            notes.append("Trend bullish but no strong BUY setup")
            return None, None, notes

        if STATE.market_dir == "Bearish":
            if strong_setup_ready(df_5m, level_hit, "SELL", trap_info) and sell_ok:
                notes.append("Trend bearish → SELL only")
                return "SELL", "Strong", notes
            notes.append("Trend bearish but no strong SELL setup")
            return None, None, notes

        notes.append("Trend neutral direction → skip")
        return None, None, notes

    # =====================================================
    # 2) CLEAN RANGE
    # =====================================================
    if STATE.market_label == "Clean Range" and range_info.get("valid", False):
        pos = range_position_label(level_now, range_info)
        notes.append(f"Regime = Clean Range | Pos = {pos}")

        if price_near_range_high(level_now, range_info):
            if strong_setup_ready(df_5m, level_hit, "SELL", trap_info) and sell_ok:
                notes.append("Upper edge + strong sell setup")
                return "SELL", "Strong", notes

            if CFG.enable_hybrid_entries and CFG.enable_early_entries_in_clean_range:
                if early_setup_ready(df_5m, "SELL"):
                    notes.append("Upper edge + early sell weakness")
                    return "SELL", "Early", notes

            notes.append("Upper edge but no valid SELL setup")
            return None, None, notes

        if price_near_range_low(level_now, range_info):
            if strong_setup_ready(df_5m, level_hit, "BUY", trap_info) and buy_ok:
                notes.append("Lower edge + strong buy setup")
                return "BUY", "Strong", notes

            if CFG.enable_hybrid_entries and CFG.enable_early_entries_in_clean_range:
                if early_setup_ready(df_5m, "BUY"):
                    notes.append("Lower edge + early buy strength")
                    return "BUY", "Early", notes

            notes.append("Lower edge but no valid BUY setup")
            return None, None, notes

        notes.append("Inside clean range away from edges → skip")
        return None, None, notes

    # =====================================================
    # 3) MESSY
    # =====================================================
    if STATE.market_label == "Messy":
        notes.append("Regime = Messy")

        # Momentum Mode first
        if bias == "Bullish":
            if momentum_breakout(df_5m, level_hit, "BUY"):
                notes.append("Messy → Momentum breakout BUY")
                return "BUY", "Strong", notes

        if bias == "Bearish":
            if momentum_breakout(df_5m, level_hit, "SELL"):
                notes.append("Messy → Momentum breakout SELL")
                return "SELL", "Strong", notes

        if CFG.messy_allows_only_strong_break_retest:
            if bias == "Bullish":
                if break_retest(df_5m, level_hit, "BUY") and strong_buy_confirmation(df_5m):
                    notes.append("Messy → only strong BUY break-retest")
                    return "BUY", "Strong", notes

            if bias == "Bearish":
                if break_retest(df_5m, level_hit, "SELL") and strong_sell_confirmation(df_5m):
                    notes.append("Messy → only strong SELL break-retest")
                    return "SELL", "Strong", notes

            notes.append("Messy regime without momentum/break-retest")
            return None, None, notes

        if bias == "Bullish" and strong_buy_confirmation(df_5m):
            notes.append("Messy fallback → strong BUY only")
            return "BUY", "Strong", notes

        if bias == "Bearish" and strong_sell_confirmation(df_5m):
            notes.append("Messy fallback → strong SELL only")
            return "SELL", "Strong", notes

        return None, None, notes

    # =====================================================
    # 4) WEAK / FALLBACK
    # =====================================================
    notes.append("Regime = Weak fallback")

    if bias == "Bullish" and buy_ok:
        return "BUY", "Strong", notes
    if bias == "Bearish" and sell_ok:
        return "SELL", "Strong", notes

    return None, None, notes

# =========================================================
# Trap cooldown
# =========================================================

def update_trap_cooldown(trap_info: dict):
    if trap_info.get("bull_trap"):
        STATE.block_long(CFG.fake_break_cooldown_bars_5m)
    if trap_info.get("bear_trap"):
        STATE.block_short(CFG.fake_break_cooldown_bars_5m)


# =========================================================
# Blocking / filters
# =========================================================

def direction_allowed_by_trend(direction: str, trend_dir: str) -> bool:
    if not CFG.enable_trend_protection:
        return True
    if STATE.market_label != "Trending":
        return True

    if trend_dir == "Bullish" and direction == "SELL":
        return False
    if trend_dir == "Bearish" and direction == "BUY":
        return False
    return True


def direction_allowed_by_range_location(direction: str, level_now: float, range_info: dict) -> bool:
    if STATE.market_label != "Clean Range":
        return True
    if not CFG.enable_edge_only_range:
        return True
    if not range_info.get("valid", False):
        return False

    if CFG.enable_mid_range_skip and price_in_mid_range(level_now, range_info):
        return False

    if direction == "SELL":
        return price_near_range_high(level_now, range_info)

    if direction == "BUY":
        return price_near_range_low(level_now, range_info)

    return False


def distance_from_best_location(direction: str, level_now: float, range_info: dict) -> float | None:
    if not range_info.get("valid", False):
        return None

    if STATE.market_label == "Clean Range":
        if direction == "SELL":
            return abs(range_info["high"] - level_now)
        if direction == "BUY":
            return abs(level_now - range_info["low"])

    if STATE.market_label == "Trending":
        return None

    if STATE.market_label == "Messy":
        return None

    return None


def is_chasing_too_far(direction: str, level_now: float, range_info: dict) -> bool:
    dist = distance_from_best_location(direction, level_now, range_info)

    if dist is None:
        return False

    if STATE.market_label == "Clean Range":
        return dist > CFG.max_chase_distance_pts_clean_range

    if STATE.market_label == "Trending":
        return dist > CFG.max_chase_distance_pts_trend

    if STATE.market_label == "Messy":
        return dist > CFG.max_chase_distance_pts_messy

    return False


def block_reason_for_direction(
    direction: str,
    level_now: float,
    range_info: dict,
    trend_dir: str,
    trap_info: dict,
    entry_mode: str | None,
) -> str | None:
    if not direction_allowed_by_trend(direction, trend_dir):
        return f"Blocked by trend protection ({trend_dir})"

    if not direction_allowed_by_range_location(direction, level_now, range_info):
        pos = range_position_label(level_now, range_info)
        return f"Blocked by range-location filter ({pos})"

    if direction == "BUY" and STATE.long_blocked():
        return "Blocked by recent bull-trap cooldown"
    if direction == "SELL" and STATE.short_blocked():
        return "Blocked by recent bear-trap cooldown"

    if trap_info.get("bull_trap") and direction == "BUY":
        return "Blocked by active bull trap"
    if trap_info.get("bear_trap") and direction == "SELL":
        return "Blocked by active bear trap"

    if is_chasing_too_far(direction, level_now, range_info):
        return "Blocked by anti-chase filter"

    if STATE.market_label == "Messy":
        if entry_mode == "Early" and not CFG.enable_early_entries_in_messy:
            return "Blocked: Early entries disabled in Messy"

    return None


# =========================================================
# Score / confidence / probabilities
# =========================================================

def score_setup(
    df_5m: pd.DataFrame,
    level_hit: float,
    direction: str,
    wick_info: dict,
    entry_mode: str,
) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    trigger = None

    wick_reason = None
    if wick_info.get("upper_cluster") and direction == "SELL":
        wick_reason = f"Upper-wick rejection cluster near {wick_info.get('bucket', 'N/A')}"
    if wick_info.get("lower_cluster") and direction == "BUY":
        wick_reason = f"Lower-wick cluster near support {wick_info.get('bucket', 'N/A')}"

    if wick_reason:
        score += 2 if entry_mode == "Early" else 3
        reasons.append("Wick Rejection near Level")
        reasons.append(wick_reason)
        trigger = "Wick Rejection near Level"

    br = break_retest(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction)
    sr = strong_rejection(df_5m, direction)
    st = stoch_cross(df_5m, direction)
    ms = momentum_shift(df_5m, direction)
    tb = two_bar_reversal(df_5m, direction)

    if br:
        score += 2
        reasons.append("Break&Retest")
        trigger = trigger or "Break&Retest"

    if sr:
        score += 2
        reasons.append("Strong Rejection")
        trigger = trigger or "Strong Rejection"
    elif rj:
        score += 1 if entry_mode == "Early" else 2
        reasons.append("Rejection")
        trigger = trigger or "Rejection"

    if tb and entry_mode == "Early":
        score += 1
        reasons.append("Two-bar reversal")

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


def classify_trade_strength(score: int, conf: int, entry_mode: str) -> str:
    if entry_mode == "Early":
        return "Early"

    if score >= CFG.strong_score_threshold or conf >= CFG.strong_conf_threshold:
        return "Strong"
    if conf >= CFG.standard_conf_threshold or score >= 4:
        return "Standard"
    return "Weak"


def confidence_percent(
    score: int,
    direction: str,
    bias: str,
    session: str,
    market_label: str,
    liq_state: str,
    entry_mode: str,
) -> int:
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
        adj += 4
    elif market_label == "Clean Range":
        adj += 2
    elif market_label == "Messy":
        adj -= 7

    if entry_mode == "Early":
        adj -= 10

    return int(max(5, min(95, base + adj)))


def probability_t1_t2(
    conf: int,
    rr: float | None,
    dist_t1: float | None,
    dist_t2: float | None,
    exp_move_1h_val: float | None,
    liq_state: str,
    market_label: str,
    entry_mode: str,
) -> tuple[int, int]:
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

    if market_label == "Messy":
        t1 -= 8
    elif market_label == "Trending":
        t1 += 2

    if entry_mode == "Early":
        t1 -= 8

    if exp_move_1h_val is not None and dist_t1 is not None and np.isfinite(dist_t1):
        if dist_t1 > exp_move_1h_val * 1.20:
            t1 -= 12
        elif dist_t1 > exp_move_1h_val * 1.00:
            t1 -= 6

    t1 = max(5, min(95, int(round(t1))))

    if (
        dist_t1 is not None and dist_t2 is not None
        and np.isfinite(dist_t1) and np.isfinite(dist_t2)
        and dist_t2 > 0
    ):
        ratio = max(1.0, dist_t2 / max(dist_t1, 1e-9))
        t2 = int(round(t1 * (0.70 / ratio)))
    else:
        t2 = int(round(t1 * 0.65))

    if entry_mode == "Early":
        t2 = int(round(t2 * 0.82))

    t2 = max(5, min(t1, min(90, t2)))
    return int(t1), int(t2)


# =========================================================
# ETA helper
# =========================================================

def eta_to_t1_minutes(
    df_5m: pd.DataFrame,
    price_now: float,
    t1: float,
    direction: str,
    liq_state: str,
) -> tuple[int, int] | None:
    if t1 is None or not np.isfinite(t1):
        return None

    look = min(24, len(df_5m))
    if look < 8:
        return None

    closes = df_5m["close"].tail(look).astype(float)
    diffs = closes.diff().abs().dropna()
    if diffs.empty:
        return None

    avg_abs_move_per_5m = float(diffs.mean())
    vel = max(avg_abs_move_per_5m / 5.0, 0.15)

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


# =========================================================
# Messages
# =========================================================

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
    range_info: dict | None = None,
) -> str:
    t1_txt = safe_f1(plan.get("t1"))
    t2_txt = safe_f1(plan.get("t2"))
    t3_txt = safe_f1(plan.get("t3"))
    rr_txt = safe_f2(plan.get("rr"))
    exp_txt = f"±{float(exp_move):.0f} pts" if exp_move is not None and np.isfinite(float(exp_move)) else "N/A"
    eta_txt = "N/A" if eta_band is None else f"{eta_band[0]}–{eta_band[1]} min"

    trade_type = plan.get("trade_type", "N/A")
    entry_mode = plan.get("entry_mode", "N/A")
    risk_scale = plan.get("risk_scale", 1.0)

    range_pos = "N/A"
    range_desc = "Range: N/A"
    if range_info is not None and range_info.get("valid", False):
        range_pos = range_position_label(level_now, range_info)
        range_desc = (
            f"{safe_f1(range_info['low'])} - {safe_f1(range_info['high'])} "
            f"| Mid {safe_f1(range_info['mid'])}"
        )

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: TradingView | Symbol: {symbol}\n\n"
        f"📊 Market State: {market_state_str}\n"
        f"💧 Liquidity: {liq_state}\n"
        f"💰 Level Now: {safe_f1(level_now)}\n"
        f"🧭 Range Position: {range_pos}\n"
        f"📦 Range Box: {range_desc}\n\n"
        f"📍 Direction: {direction}\n"
        f"🧱 Level: {safe_f1(level_hit)}\n"
        f"🧬 Trade Type: {trade_type}\n"
        f"⚙️ Entry Mode: {entry_mode}\n"
        f"📏 Risk Scale: x{safe_f2(risk_scale)}\n\n"
        f"✅ Entry: {safe_f1(plan.get('entry'))}\n"
        f"🛑 Stop: {safe_f1(plan.get('stop'))}\n"
        f"🎯 Target 1: {t1_txt}\n"
        f"🎯 Target 2: {t2_txt}\n"
        f"🎯 Target 3: {t3_txt}\n\n"
        f"📐 RR: {rr_txt}\n"
        f"⭐ Score: {score}/6\n"
        f"🔎 Confidence: {conf}%\n\n"
        f"📊 Probability\n"
        f"T1 Hit: {p_t1}%\n"
        f"T2 Hit: {p_t2}%\n\n"
        f"📈 Expected Move (1H): {exp_txt}\n"
        f"⏱ ETA to T1: {eta_txt}\n\n"
        f"🧠 Reason: {', '.join(reasons)}"
    )


def hourly_update_message(
    session: str,
    symbol: str,
    bias: str,
    market_state_str: str,
    levels: list,
    price: float,
    active_trade: dict | None,
    range_info: dict | None = None,
) -> str:
    status = "None" if active_trade is None else active_trade.get("status", "Unknown")
    trade_line = "-"

    if active_trade is not None:
        trade_line = (
            f"{active_trade.get('direction', '?')} | "
            f"Entry {safe_f1(active_trade.get('entry'))} | "
            f"Stop {safe_f1(active_trade.get('stop'))} | "
            f"T1 {safe_f1(active_trade.get('t1'))} | "
            f"T2 {safe_f1(active_trade.get('t2'))} | "
            f"T3 {safe_f1(active_trade.get('t3'))} | "
            f"Mode {active_trade.get('entry_mode', 'N/A')}"
        )

    range_line = "Range: N/A"
    quality_line = "Range Quality: N/A"

    if range_info is not None and range_info.get("valid", False):
        range_line = (
            f"Range: {safe_f1(range_info['low'])} - {safe_f1(range_info['high'])} "
            f"| Mid {safe_f1(range_info['mid'])}"
        )
        quality_line = (
            f"Clean={range_info.get('clean_range', False)} | "
            f"WidthATR={safe_f2(range_info.get('width_atr'))} | "
            f"Touches(H/L)={range_info.get('touches_high')}/{range_info.get('touches_low')} | "
            f"Sweeps={range_info.get('sweeps_total')}"
        )

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: TradingView | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): {bias}\n"
        f"📊 Market State: {market_state_str}\n"
        f"💵 Price: {safe_f1(price)}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"🧭 {range_line}\n"
        f"🧪 {quality_line}\n"
        f"📌 Active Trade: {status}\n"
        f"🧾 Trade: {trade_line}"
    )


# =========================================================
# Reset / Market-state cache
# =========================================================

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


def maybe_update_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_15m: pd.DataFrame):
    nowu = datetime.utcnow()
    should = STATE.market_state_last_calc_utc is None or (
        nowu - STATE.market_state_last_calc_utc
    ) >= timedelta(minutes=CFG.market_state_update_minutes)

    if not should:
        return

    regime = classify_market_regime(df_1h, df_4h, df_15m, STATE.market_label)
    STATE.market_label = regime["label"]
    STATE.market_dir = regime["dir"]
    STATE.market_bias = regime["bias"]
    STATE.market_adx = regime["adx"]
    STATE.range_info = regime["range_info"]
    STATE.market_state_last_calc_utc = nowu

# =========================================================
# Targets / planning
# =========================================================

def atr_fallback_targets(entry: float, direction: str, exp_move_val: float | None, trade_type: str):
    if exp_move_val is None or not np.isfinite(exp_move_val):
        return None, None, None

    if trade_type == "Weak":
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_weak
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_weak
        d3 = None
    elif trade_type in ("Standard", "Early"):
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_standard
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_standard
        d3 = None
    else:
        d1 = exp_move_val * CFG.atr_fallback_t1_mult_standard
        d2 = exp_move_val * CFG.atr_fallback_t2_mult_standard
        d3 = exp_move_val * CFG.atr_fallback_t3_mult_strong if CFG.enable_t3 else None

    if direction == "BUY":
        t1 = entry + d1 if d1 is not None else None
        t2 = entry + d2 if d2 is not None else None
        t3 = entry + d3 if d3 is not None else None
    else:
        t1 = entry - d1 if d1 is not None else None
        t2 = entry - d2 if d2 is not None else None
        t3 = entry - d3 if d3 is not None else None

    return t1, t2, t3


def pick_targets(levels: list, entry: float, direction: str):
    if not levels:
        return None, None

    if direction == "BUY":
        above = sorted([lvl for lvl in levels if lvl > entry])
        return (
            above[0] if len(above) >= 1 else None,
            above[1] if len(above) >= 2 else None,
        )

    if direction == "SELL":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        return (
            below[0] if len(below) >= 1 else None,
            below[1] if len(below) >= 2 else None,
        )

    return None, None


def risk_scale_from_entry_mode(entry_mode: str) -> float:
    if entry_mode == "Early":
        return float(CFG.early_risk_scale)
    return float(CFG.strong_risk_scale)


def trade_type_from_entry_mode(entry_mode: str, classified_strength: str) -> str:
    if entry_mode == "Early":
        return "Early"
    return classified_strength


# =========================================================
# Entry / Stop planner
# =========================================================

def compute_trade_plan(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    levels: list,
    level_hit: float,
    direction: str,
    trigger: str,
    trade_type: str,
    entry_mode: str,
    exp_move_val: float | None,
    range_info: dict | None = None,
):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])

    atr15 = latest_atr_15m(df_15m)
    if atr15 is None:
        atr15 = max(abs(last_high - last_low), 4.0)

    buffer = max(price * 0.0002, 0.5)

    # =====================================================
    # CLEAN RANGE MODE
    # =====================================================
    if STATE.market_label == "Clean Range" and range_info is not None and range_info.get("valid", False):
        if direction == "BUY":
            anchor = min(level_hit, range_info["low"] + range_info["edge_dist"])

            if entry_mode == "Early":
                entry = float(max(price, anchor + max(0.3, atr15 * 0.06)))
                stop = float(min(last_low, range_info["low"], level_hit) - max(buffer, atr15 * 0.18))
            else:
                if trigger == "Wick Rejection near Level":
                    entry = float(max(price, anchor + max(0.4, atr15 * 0.08)))
                elif trigger == "Break&Retest":
                    entry = float(max(price, anchor + max(0.8, atr15 * 0.12)))
                else:
                    entry = float(max(price, anchor + max(0.5, atr15 * 0.10)))

                stop = float(min(last_low, range_info["low"], level_hit) - max(buffer, atr15 * 0.22))

        else:  # SELL
            anchor = max(level_hit, range_info["high"] - range_info["edge_dist"])

            if entry_mode == "Early":
                entry = float(min(price, anchor - max(0.3, atr15 * 0.06)))
                stop = float(max(last_high, range_info["high"], level_hit) + max(buffer, atr15 * 0.18))
            else:
                if trigger == "Wick Rejection near Level":
                    entry = float(min(price, anchor - max(0.4, atr15 * 0.08)))
                elif trigger == "Break&Retest":
                    entry = float(min(price, anchor - max(0.8, atr15 * 0.12)))
                else:
                    entry = float(min(price, anchor - max(0.5, atr15 * 0.10)))

                stop = float(max(last_high, range_info["high"], level_hit) + max(buffer, atr15 * 0.22))

    # =====================================================
    # TREND / MESSY / FALLBACK
    # =====================================================
    else:
        if entry_mode == "Early":
            if direction == "BUY":
                entry = float(max(price, level_hit + max(buffer * 0.2, atr15 * 0.05)))
                stop = float(min(last_low, level_hit) - max(buffer, atr15 * 0.16))
            else:
                entry = float(min(price, level_hit - max(buffer * 0.2, atr15 * 0.05)))
                stop = float(max(last_high, level_hit) + max(buffer, atr15 * 0.16))
        else:
            if trigger == "Wick Rejection near Level":
                if direction == "BUY":
                    if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                        entry = float(max(price, level_hit + buffer * 0.4))
                    else:
                        entry = float(max(price, (last_high + level_hit) / 2.0))
                    stop = float(min(last_low, level_hit) - max(buffer, atr15 * 0.18))
                else:
                    if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                        entry = float(min(price, level_hit - buffer * 0.4))
                    else:
                        entry = float(min(price, (last_low + level_hit) / 2.0))
                    stop = float(max(last_high, level_hit) + max(buffer, atr15 * 0.18))

            elif trigger == "Rejection" or trigger == "Strong Rejection":
                if direction == "BUY":
                    entry = float(max(price, last_high - buffer * 0.4))
                    stop = float(min(last_low, level_hit) - max(buffer, atr15 * 0.18))
                else:
                    entry = float(min(price, last_low + buffer * 0.4))
                    stop = float(max(last_high, level_hit) + max(buffer, atr15 * 0.18))
            else:
                if direction == "BUY":
                    entry = float(max(price, level_hit + max(buffer, atr15 * 0.08)))
                    stop = float(level_hit - max(price * 0.0011, atr15 * 0.22) - buffer)
                else:
                    entry = float(min(price, level_hit - max(buffer, atr15 * 0.08)))
                    stop = float(level_hit + max(price * 0.0011, atr15 * 0.22) + buffer)

    # =====================================================
    # Targets
    # =====================================================
    t1, t2 = pick_targets(levels, entry, direction)
    atr_t1, atr_t2, atr_t3 = atr_fallback_targets(entry, direction, exp_move_val, trade_type)

    if t1 is None:
        t1 = atr_t1
    if t2 is None:
        t2 = atr_t2

    t3 = None
    if trade_type == "Strong" and atr_t3 is not None:
        if direction == "BUY":
            if t2 is None or atr_t3 > t2:
                t3 = atr_t3
        else:
            if t2 is None or atr_t3 < t2:
                t3 = atr_t3

    if entry_mode == "Early":
        t3 = None

    targets = [t for t in [t1, t2, t3] if t is not None and np.isfinite(t)]
    if direction == "BUY":
        targets = sorted(set(targets))
    else:
        targets = sorted(set(targets), reverse=True)

    t1 = targets[0] if len(targets) > 0 else None
    t2 = targets[1] if len(targets) > 1 else None
    t3 = targets[2] if len(targets) > 2 else None

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = (reward / risk) if risk > 0 else None

    return {
        "entry": entry,
        "stop": stop,
        "initial_stop": stop,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": None,
        "rr": rr,
        "trade_type": trade_type,
        "entry_mode": entry_mode,
        "risk_scale": risk_scale_from_entry_mode(entry_mode),
    }


# =========================================================
# Dynamic T3 / T4 expansion
# =========================================================

def next_levels_beyond(levels: list, price_ref: float, direction: str) -> list:
    if direction == "BUY":
        return sorted([lvl for lvl in levels if lvl > price_ref])
    return sorted([lvl for lvl in levels if lvl < price_ref], reverse=True)


def dynamic_extension_allowed(df_5m: pd.DataFrame, direction: str) -> bool:
    if direction == "BUY":
        return strong_buy_confirmation(df_5m)
    return strong_sell_confirmation(df_5m)


def dynamic_target_from_market_or_atr(
    levels: list,
    current_price: float,
    direction: str,
    exp_move_val: float | None,
    used_targets: list[float],
    stage: str,
) -> float | None:
    candidates = next_levels_beyond(levels, current_price, direction)

    for lvl in candidates:
        if all(abs(lvl - x) > 1.0 for x in used_targets):
            return float(lvl)

    if exp_move_val is None or not np.isfinite(exp_move_val):
        return None

    dist = exp_move_val * (CFG.dynamic_t3_atr_mult if stage == "T3" else CFG.dynamic_t4_atr_mult)

    if direction == "BUY":
        candidate = current_price + dist
    else:
        candidate = current_price - dist

    if all(abs(candidate - x) > 1.0 for x in used_targets):
        return float(candidate)

    return None


def maybe_expand_targets(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if tr is None or tr.get("status") != "live":
        return

    if tr.get("entry_mode") == "Early":
        return

    direction = tr["direction"]
    used_targets = [x for x in [tr.get("t1"), tr.get("t2"), tr.get("t3"), tr.get("t4")] if x is not None]

    if not dynamic_extension_allowed(df_5m, direction):
        return

    exp_move_val = expected_move_1h(df_1h)
    levels = extract_key_levels(df_15m, df_1h)

    if tr.get("t2_hit") and not tr.get("t3_defined") and CFG.enable_t3:
        new_t3 = dynamic_target_from_market_or_atr(
            levels=levels,
            current_price=last_price,
            direction=direction,
            exp_move_val=exp_move_val,
            used_targets=used_targets,
            stage="T3",
        )
        if new_t3 is not None:
            tr["t3"] = float(new_t3)
            tr["t3_defined"] = True
            if CFG.move_stop_to_t1_on_t2 and tr.get("t1") is not None:
                tr["stop"] = float(tr["t1"])
            send_telegram(
                f"🧠 {CFG.user_title} — Dynamic Target Expansion\n"
                f"After T2, momentum still strong ✅\n"
                f"New T3: {safe_f1(tr['t3'])}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )

    if tr.get("t3_hit") and not tr.get("t4_defined") and CFG.enable_dynamic_t4:
        used_targets = [x for x in [tr.get("t1"), tr.get("t2"), tr.get("t3"), tr.get("t4")] if x is not None]
        new_t4 = dynamic_target_from_market_or_atr(
            levels=levels,
            current_price=last_price,
            direction=direction,
            exp_move_val=exp_move_val,
            used_targets=used_targets,
            stage="T4",
        )
        if new_t4 is not None:
            tr["t4"] = float(new_t4)
            tr["t4_defined"] = True
            if CFG.move_stop_to_t2_on_t3 and tr.get("t2") is not None:
                tr["stop"] = float(tr["t2"])
            send_telegram(
                f"🚀 {CFG.user_title} — Dynamic Runner Extended\n"
                f"After T3, trend still alive ✅\n"
                f"New T4: {safe_f1(tr['t4'])}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )


# =========================================================
# Active trade management
# =========================================================

def update_active_trade(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if tr is None:
        return

    direction = tr["direction"]
    entry = float(tr["entry"])
    stop = float(tr["stop"])
    t1 = tr.get("t1")
    t2 = tr.get("t2")
    t3 = tr.get("t3")
    t4 = tr.get("t4")

    # =====================================================
    # Pending -> live
    # =====================================================
    if tr["status"] == "pending":
        triggered = (last_price >= entry) if direction == "BUY" else (last_price <= entry)
        if triggered:
            tr["status"] = "live"
            tr["live_since_utc"] = datetime.utcnow()
            send_telegram(
                f"📍 {CFG.user_title} — الصفقة تفعلت\n"
                f"Direction: {direction}\n"
                f"Entry Mode: {tr.get('entry_mode', 'N/A')}\n"
                f"Risk Scale: x{safe_f2(tr.get('risk_scale', 1.0))}\n"
                f"Entry Triggered @ {safe_f1(last_price)}\n"
                f"Stop: {safe_f1(stop)}\n"
                f"T1: {safe_f1(t1)} | T2: {safe_f1(t2)} | T3: {safe_f1(t3)} | T4: {safe_f1(t4)}"
            )
        return

    if tr["status"] != "live":
        return

    # =====================================================
    # Stop logic
    # =====================================================
    breached = (last_price <= stop) if direction == "BUY" else (last_price >= stop)

    if breached:
        if CFG.hard_stop_enabled:
            beyond = (
                (stop - last_price) >= CFG.hard_stop_buffer_pts
                if direction == "BUY"
                else (last_price - stop) >= CFG.hard_stop_buffer_pts
            )
            if beyond:
                send_telegram(
                    f"❌ {CFG.user_title} — وقف الخسارة (Hard Stop)\n"
                    f"Stop Hit ✅ | Direction: {direction}\n"
                    f"Mode: {tr.get('entry_mode', 'N/A')}\n"
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
                            f"Mode: {tr.get('entry_mode', 'N/A')}\n"
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

    # =====================================================
    # T1
    # =====================================================
    if t1 is not None and not tr.get("t1_hit"):
        hit_t1 = (last_price >= float(t1)) if direction == "BUY" else (last_price <= float(t1))
        if hit_t1:
            tr["t1_hit"] = True

            if CFG.move_stop_to_be_on_t1:
                tr["stop"] = float(entry)

            send_telegram(
                f"🎯 {CFG.user_title} — Target 1 Hit\n"
                f"Mode: {tr.get('entry_mode', 'N/A')}\n"
                f"T1: {safe_f1(t1)}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )

            if tr.get("entry_mode") == "Early" and tr.get("t2") is None:
                send_telegram(
                    f"✅ {CFG.user_title} — Early trade completed\n"
                    f"T1 was the main target.\n"
                    f"اقتراح: إغلاق أو تتبع يدوي."
                )
                STATE.active_trade = None
                return

    # =====================================================
    # T2
    # =====================================================
    if t2 is not None and not tr.get("t2_hit"):
        hit_t2 = (last_price >= float(t2)) if direction == "BUY" else (last_price <= float(t2))
        if hit_t2:
            tr["t2_hit"] = True
            if CFG.move_stop_to_t1_on_t2 and tr.get("t1") is not None:
                tr["stop"] = float(tr["t1"])

            send_telegram(
                f"🏆 {CFG.user_title} — Target 2 Hit\n"
                f"Mode: {tr.get('entry_mode', 'N/A')}\n"
                f"T2: {safe_f1(t2)}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )

            if tr.get("entry_mode") == "Early":
                send_telegram(
                    f"✅ {CFG.user_title} — Early trade completed\n"
                    f"T2 reached.\n"
                    f"اقتراح: إغلاق كامل."
                )
                STATE.active_trade = None
                return

    maybe_expand_targets(df_5m, df_15m, df_1h, last_price)

    # =====================================================
    # T3
    # =====================================================
    t3 = tr.get("t3")
    if t3 is not None and not tr.get("t3_hit"):
        hit_t3 = (last_price >= float(t3)) if direction == "BUY" else (last_price <= float(t3))
        if hit_t3:
            tr["t3_hit"] = True
            if CFG.move_stop_to_t2_on_t3 and tr.get("t2") is not None:
                tr["stop"] = float(tr["t2"])

            send_telegram(
                f"🚀 {CFG.user_title} — Target 3 Hit\n"
                f"Mode: {tr.get('entry_mode', 'N/A')}\n"
                f"T3: {safe_f1(t3)}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )

    maybe_expand_targets(df_5m, df_15m, df_1h, last_price)

    # =====================================================
    # T4
    # =====================================================
    t4 = tr.get("t4")
    if t4 is not None and not tr.get("t4_hit"):
        hit_t4 = (last_price >= float(t4)) if direction == "BUY" else (last_price <= float(t4))
        if hit_t4:
            tr["t4_hit"] = True
            send_telegram(
                f"🔥 {CFG.user_title} — Target 4 Hit\n"
                f"T4: {safe_f1(t4)}\n"
                f"اقتراح: إغلاق كامل / Final runner exit"
            )
            STATE.active_trade = None
            return


# =========================================================
# Main evaluation
# =========================================================

def evaluate_once():
    maybe_daily_reset()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m = fetch_timeframes()

    # advance 5m bar counter
    STATE.bar_counter_5m += 1

    # indicators
    df_5m = compute_indicators_5m(df_5m)
    level_now = float(df_5m["close"].iloc[-1])

    # market state
    maybe_update_market_state(df_1h, df_4h, df_15m)
    adx_txt = "N/A" if STATE.market_adx is None else safe_f1(STATE.market_adx)
    market_state_str = f"{STATE.market_label} | {STATE.market_dir} | ADX(1H): {adx_txt}"

    # context
    liq_state, _ = liquidity_state(df_5m)
    bias = STATE.market_bias if STATE.market_bias else structure_bias(df_1h, df_4h)
    trend_dir = trend_direction_from_bias_and_state(bias, STATE.market_dir)

    key_levels = extract_key_levels(df_15m, df_1h)
    exp_move_val = expected_move_1h(df_1h)
    range_info = STATE.range_info if STATE.range_info else compute_range_info(df_15m)

    # fake breakout / traps
    trap_info = detect_fake_breaks(df_15m, range_info, STATE.market_label)
    update_trap_cooldown(trap_info)

    # update active trade first
    if STATE.active_trade is not None:
        update_active_trade(df_5m, df_15m, df_1h, level_now)

    # hourly update
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_update_message(
                session=session,
                symbol=symbol,
                bias=bias,
                market_state_str=market_state_str,
                levels=key_levels,
                price=level_now,
                active_trade=STATE.active_trade,
                range_info=range_info,
            ))
            STATE.last_hour_sent = current_hour

    # block after reset
    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        return

    # one trade at a time
    if STATE.active_trade is not None:
        return

    if not key_levels:
        return

    # choose best level
    level_hit, level_info = choose_best_level(df_5m, level_now, key_levels)
    wick_info = level_info["wick_info"]

    # choose direction + entry mode
    direction, entry_mode, direction_notes = choose_direction_by_regime(
        df_5m=df_5m,
        level_hit=level_hit,
        level_now=level_now,
        bias=bias,
        range_info=range_info,
        trap_info=trap_info,
    )

    if direction is None or entry_mode is None:
        return

    # block checks
    block_reason = block_reason_for_direction(
        direction=direction,
        level_now=level_now,
        range_info=range_info,
        trend_dir=trend_dir,
        trap_info=trap_info,
        entry_mode=entry_mode,
    )
    if block_reason is not None:
        return

    # closed-bar confirmation
    if CFG.require_closed_bar_confirmation:
        # tvdatafeed already gives closed bars in this workflow
        pass

    # extra 15m confirmation in clean range reversals
    if STATE.market_label == "Clean Range" and range_info.get("valid", False):
        last15 = df_15m.iloc[-1]
        c15 = float(last15["close"])
        h15 = float(last15["high"])
        l15 = float(last15["low"])

        if direction == "SELL" and price_near_range_high(level_now, range_info):
            if h15 > range_info["high"] and c15 > range_info["high"]:
                return

        if direction == "BUY" and price_near_range_low(level_now, range_info):
            if l15 < range_info["low"] and c15 < range_info["low"]:
                return

    # in messy mode, only allow strong entries
    if STATE.market_label == "Messy" and entry_mode != "Strong":
        return

    # score the setup
    score, reasons, trigger = score_setup(
        df_5m=df_5m,
        level_hit=level_hit,
        direction=direction,
        wick_info=wick_info,
        entry_mode=entry_mode,
    )
    if score <= 0:
        return

    for note in direction_notes:
        if note not in reasons:
            reasons.append(note)

    # confidence / trade type
    conf = confidence_percent(
        score=score,
        direction=direction,
        bias=bias,
        session=session,
        market_label=STATE.market_label,
        liq_state=liq_state,
        entry_mode=entry_mode,
    )

    classified_strength = classify_trade_strength(score, conf, entry_mode)
    trade_type = trade_type_from_entry_mode(entry_mode, classified_strength)

    # minimum score
    if score < CFG.score_threshold and entry_mode != "Early":
        return

    # offhours stricter
    if session in ("After-Hours", "Pre-Market"):
        if score < CFG.offhours_min_score:
            return
        if CFG.offhours_block_weak and trade_type == "Weak":
            return

    # momentum confirmation
    if CFG.require_momentum_confirmation:
        if direction == "BUY":
            if entry_mode == "Strong":
                if not (
                    strong_buy_confirmation(df_5m)
                    or trigger == "Break&Retest"
                    or trigger == "Strong Rejection"
                    or (STATE.market_label == "Clean Range" and price_near_range_low(level_now, range_info))
                ):
                    return
            else:
                if not (
                    early_setup_ready(df_5m, "BUY")
                    or (STATE.market_label == "Clean Range" and price_near_range_low(level_now, range_info))
                ):
                    return

        if direction == "SELL":
            if entry_mode == "Strong":
                if not (
                    strong_sell_confirmation(df_5m)
                    or trigger == "Break&Retest"
                    or trigger == "Strong Rejection"
                    or (STATE.market_label == "Clean Range" and price_near_range_high(level_now, range_info))
                ):
                    return
            else:
                if not (
                    early_setup_ready(df_5m, "SELL")
                    or (STATE.market_label == "Clean Range" and price_near_range_high(level_now, range_info))
                ):
                    return

    # clean range controls
    if STATE.market_label == "Clean Range":
        if CFG.enable_mid_range_skip and price_in_mid_range(level_now, range_info):
            return

        if direction == "BUY" and not price_near_range_low(level_now, range_info):
            return

        if direction == "SELL" and not price_near_range_high(level_now, range_info):
            return

    # trend protection final check
    if CFG.enable_trend_protection and STATE.market_label == "Trending":
        if trend_dir == "Bullish" and direction == "SELL":
            return
        if trend_dir == "Bearish" and direction == "BUY":
            return

    # compute trade plan
    plan = compute_trade_plan(
        df_5m=df_5m,
        df_15m=df_15m,
        levels=key_levels,
        level_hit=level_hit,
        direction=direction,
        trigger=trigger,
        trade_type=trade_type,
        entry_mode=entry_mode,
        exp_move_val=exp_move_val,
        range_info=range_info,
    )

    rr = plan.get("rr")
    if rr is not None and np.isfinite(float(rr)) and float(rr) < CFG.min_rr_to_t1:
        return

    # cooldown key
    key = f"{STATE.market_label}:{entry_mode}:{direction}:{round(level_hit,1)}:{trigger}:{trade_type}"
    if not STATE.can_signal(key):
        return

    # distances / probabilities
    dist_t1 = None
    dist_t2 = None
    if plan.get("t1") is not None:
        dist_t1 = abs(float(plan["t1"]) - level_now)
    if plan.get("t2") is not None:
        dist_t2 = abs(float(plan["t2"]) - level_now)

    p1, p2 = probability_t1_t2(
        conf=conf,
        rr=rr,
        dist_t1=dist_t1,
        dist_t2=dist_t2,
        exp_move_1h_val=exp_move_val,
        liq_state=liq_state,
        market_label=STATE.market_label,
        entry_mode=entry_mode,
    )

    eta = None
    if plan.get("t1") is not None:
        eta = eta_to_t1_minutes(df_5m, level_now, float(plan["t1"]), direction, liq_state)

    # send signal
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
        range_info=range_info,
    )
    send_telegram(msg)
    STATE.mark_signal(key)

    # open pending trade
    STATE.active_trade = {
        "direction": direction,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "initial_stop": float(plan["initial_stop"]),
        "t1": None if plan.get("t1") is None else float(plan["t1"]),
        "t2": None if plan.get("t2") is None else float(plan["t2"]),
        "t3": None if plan.get("t3") is None else float(plan["t3"]),
        "t4": None,
        "status": "pending",
        "created_utc": datetime.utcnow(),
        "trade_type": trade_type,
        "entry_mode": entry_mode,
        "risk_scale": float(plan.get("risk_scale", 1.0)),
        "market_label": STATE.market_label,
        "t1_hit": False,
        "t2_hit": False,
        "t3_hit": False,
        "t4_hit": False,
        "t3_defined": plan.get("t3") is not None,
        "t4_defined": False,
    }


# =========================================================
# Main
# =========================================================

def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- Source: TradingView ({CFG.tv_exchange}:{CFG.tv_symbol})\n"
        f"- 3-State Regime: ON\n"
        f"- States: Trending / Clean Range / Messy\n"
        f"- Trend Protection: {'ON' if CFG.enable_trend_protection else 'OFF'}\n"
        f"- Mid-Range Skip: {'ON' if CFG.enable_mid_range_skip else 'OFF'}\n"
        f"- Edge-only Range Entries: {'ON' if CFG.enable_edge_only_range else 'OFF'}\n"
        f"- Fake Break Filter: {'ON' if CFG.enable_fake_break_filter else 'OFF'}\n"
        f"- Hybrid Entries: {'ON' if CFG.enable_hybrid_entries else 'OFF'}\n"
        f"- Early in Clean Range: {'ON' if CFG.enable_early_entries_in_clean_range else 'OFF'}\n"
        f"- Early in Messy: {'ON' if CFG.enable_early_entries_in_messy else 'OFF'}\n"
        f"- Closed-bar Confirmation: {'ON' if CFG.require_closed_bar_confirmation else 'OFF'}\n"
        f"- Dynamic T3/T4 expansion: {'ON' if (CFG.enable_t3 or CFG.enable_dynamic_t4) else 'OFF'}\n"
        f"- Stop raising after targets: ON\n"
        f"- Stop: HardStop({CFG.hard_stop_buffer_pts:.1f} pts) + 5m close confirm"
    )

    while True:
        try:
            evaluate_once()

        except Exception as e:
            print("[ERROR]", repr(e))

            try:
                reinit_tv_client()
            except Exception as reinit_err:
                print("[WARN] TV client reinit failed:", repr(reinit_err))

            if STATE.should_notify_error():
                send_telegram(f"❌ {CFG.user_title}: خطأ بيانات TradingView - {repr(e)}")

            time.sleep(15)

        time.sleep(CFG.loop_sleep_seconds)


if __name__ == "__main__":
    main()
