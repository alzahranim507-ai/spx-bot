# -*- coding: utf-8 -*-
"""
SPX500 Trading Bot (Hunter WICK PRO Dynamic+ - FOREXCOM/TradingView)
For: دكتور محمد

Features:
- TradingView via tvdatafeed
- Range-quality filter
- Smart T2
- Dynamic target extension
- Reversal watch
- Daily and weekly stats
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date

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


# =========================
# Config
# =========================

@dataclass
class Config:
    # TradingView source
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
    loop_sleep_seconds: int = 25

    # Structure / pivots
    pivot_left: int = 3
    pivot_right: int = 3

    # Level detection / clustering
    level_touch_tolerance_frac: float = 0.0013
    level_cluster_tolerance_frac: float = 0.0010
    max_key_levels: int = 8

    # Scoring
    score_threshold: int = 3
    min_rr_to_t1: float = 1.30
    signal_cooldown_minutes: int = 14

    # Market state
    adx_window: int = 14
    market_state_update_minutes: int = 15
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # Liquidity
    liquidity_lookback_5m: int = 60
    liquidity_thresholds: tuple = (0.60, 1.25, 2.00)

    # Expected move
    expected_move_atr_window: int = 14

    # ETA
    eta_velocity_lookback_5m: int = 24
    eta_min_velocity_pts_per_min: float = 0.15

    # Wick cluster
    wick_cluster_lookback_5m: int = 10
    wick_ratio_strong: float = 0.45
    wick_cluster_min_hits: int = 3
    wick_near_level_tolerance_frac: float = 0.0010
    wick_min_abs_pts: float = 1.2

    # Level quality
    level_zone_near_pts: float = 18.0
    level_min_touch_count: int = 2
    level_strong_touch_count: int = 3

    # Trade strength
    strong_score_threshold: int = 5
    strong_conf_threshold: int = 62
    standard_conf_threshold: int = 50

    # Dynamic targets
    enable_t3: bool = True
    enable_dynamic_t4: bool = True
    atr_fallback_t1_mult_weak: float = 0.28
    atr_fallback_t2_mult_weak: float = 0.45
    atr_fallback_t1_mult_standard: float = 0.35
    atr_fallback_t2_mult_standard: float = 0.65
    atr_fallback_t3_mult_strong: float = 1.00
    dynamic_t3_atr_mult: float = 1.15
    dynamic_t4_atr_mult: float = 1.45

    # Entry logic
    aggressive_entry_for_strong_wick: bool = True

    # Trade tracking
    hourly_update: bool = True

    # Stop logic
    hard_stop_enabled: bool = True
    hard_stop_buffer_pts: float = 1.0
    stop_confirm_by_5m_close: bool = True
    stop_confirm_minutes: int = 5
    move_stop_to_be_on_t1: bool = True
    move_stop_to_t1_on_t2: bool = True
    move_stop_to_t2_on_t3: bool = True

    # Session filter
    offhours_min_score: int = 4
    offhours_block_weak: bool = True

    # Daily reset
    daily_reset_enabled: bool = True
    daily_reset_hour: int = 0
    daily_reset_minute: int = 0
    daily_reset_window_minutes: int = 5
    no_signal_after_reset_minutes: int = 8

    # Bars to fetch
    bars_5m: int = 900
    bars_15m: int = 600
    bars_1h: int = 400

    # TradingView resilience
    tv_retry_attempts: int = 3
    tv_retry_sleep_base: float = 2.0
    tv_fallback_bars: tuple = (800, 600, 500, 400, 300, 200)

    # Error reporting
    error_notify_cooldown_minutes: int = 15

    # Optional safety filters
    block_all_weak_trades: bool = False
    block_counter_trend_in_trending_market: bool = True
    require_momentum_confirmation: bool = True

    # Wick trade filter
    wick_trade_filter_enabled: bool = True
    allow_counter_trend_wick_if_strong: bool = True

    # NEW: range behavior filter
    range_quality_filter_enabled: bool = True
    range_filter_only_when_adx_below: float = 21.0
    range_min_space_pts: float = 14.0
    range_min_space_expected_move_ratio: float = 0.55
    range_max_nearby_levels_within_pts: float = 10.0
    range_max_nearby_levels_count: int = 2

    # NEW: adaptive stop in range
    range_stop_extra_atr_mult: float = 0.18
    range_stop_extra_min_pts: float = 0.8
    range_stop_extra_max_pts: float = 3.0

    # NEW: target map / probabilities
    target_probability_min_to_show: int = 45
    target_probability_min_to_extend: int = 50
    max_initial_targets: int = 5
    extension_rr_min_points: float = 4.0

    # NEW: smart T2
    smart_t2_enabled: bool = True
    smart_t2_atr_mult: float = 0.95
    smart_t2_expected_move_mult: float = 0.75
    smart_t2_max_distance_ratio_to_t1: float = 2.2

    # NEW: stats
    stats_daily_enabled: bool = True
    stats_weekly_enabled: bool = True
    stats_daily_send_hour: int = 23
    stats_daily_send_minute: int = 59
    stats_weekly_send_weekday: int = 4   # Friday = 4 in Python weekday()
    stats_weekly_send_hour: int = 23
    stats_weekly_send_minute: int = 59


CFG = Config()


# =========================
# TradingView client
# =========================

def make_tv_client() -> TvDatafeed:
    if CFG.tv_username and CFG.tv_password:
        return TvDatafeed(CFG.tv_username, CFG.tv_password)
    return TvDatafeed()

TV = make_tv_client()
LAST_GOOD_DATA = {}


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

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def level_bucket_x(level: float) -> str:
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
    payload = {
        "chat_id": CFG.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True
    }

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
# Stats models
# =========================

@dataclass
class ClosedTradeRecord:
    closed_at_riyadh: datetime
    direction: str
    entry: float
    stop: float
    initial_stop: float
    trade_type: str
    result_label: str
    max_target_hit: int
    r_result: float
    market_label: str
    session: str

@dataclass
class StatsState:
    closed_trades: list = field(default_factory=list)
    last_daily_stats_sent_date: date | None = None
    last_weekly_stats_sent_week_key: str | None = None

    def add_trade(self, rec: ClosedTradeRecord):
        self.closed_trades.append(rec)

    def get_day_records(self, target_date: date):
        return [x for x in self.closed_trades if x.closed_at_riyadh.date() == target_date]

    def get_week_records_mon_to_fri(self, target_date: date):
        weekday = target_date.weekday()  # Mon=0
        monday = target_date - timedelta(days=weekday)
        friday = monday + timedelta(days=4)
        return [
            x for x in self.closed_trades
            if monday <= x.closed_at_riyadh.date() <= friday
        ]


# =========================
# Stats helpers
# =========================

def summarize_records(records: list[ClosedTradeRecord]) -> dict:
    total = len(records)
    wins = sum(1 for r in records if r.r_result > 0)
    losses = sum(1 for r in records if r.r_result < 0)
    breakeven = sum(1 for r in records if abs(r.r_result) < 1e-9)
    net_r = float(sum(r.r_result for r in records))

    t1_only = sum(1 for r in records if r.max_target_hit == 1)
    t2_hit = sum(1 for r in records if r.max_target_hit == 2)
    t3_hit = sum(1 for r in records if r.max_target_hit == 3)
    t4_plus = sum(1 for r in records if r.max_target_hit >= 4)

    range_count = sum(1 for r in records if r.market_label == "Range")
    trending_count = sum(1 for r in records if r.market_label == "Trending")

    winrate = (wins / total * 100.0) if total > 0 else 0.0

    best_trade = max(records, key=lambda r: r.r_result, default=None)
    worst_trade = min(records, key=lambda r: r.r_result, default=None)

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "net_r": net_r,
        "winrate": winrate,
        "t1_only": t1_only,
        "t2_hit": t2_hit,
        "t3_hit": t3_hit,
        "t4_plus": t4_plus,
        "range_count": range_count,
        "trending_count": trending_count,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }

def daily_stats_message(summary: dict, target_date: date) -> str:
    best_txt = "N/A"
    worst_txt = "N/A"

    if summary["best_trade"] is not None:
        bt = summary["best_trade"]
        best_txt = f"{bt.direction} | {bt.result_label} | {bt.r_result:+.2f}R"
    if summary["worst_trade"] is not None:
        wt = summary["worst_trade"]
        worst_txt = f"{wt.direction} | {wt.result_label} | {wt.r_result:+.2f}R"

    return (
        f"📊 {CFG.user_title} — Daily Stats\n"
        f"📅 Date: {target_date.isoformat()} (Riyadh)\n\n"
        f"Total Trades: {summary['total']}\n"
        f"Wins: {summary['wins']}\n"
        f"Losses: {summary['losses']}\n"
        f"Breakeven: {summary['breakeven']}\n"
        f"Win Rate: {summary['winrate']:.1f}%\n"
        f"Net Result: {summary['net_r']:+.2f}R\n\n"
        f"T1 only: {summary['t1_only']}\n"
        f"T2 hit: {summary['t2_hit']}\n"
        f"T3 hit: {summary['t3_hit']}\n"
        f"T4+: {summary['t4_plus']}\n\n"
        f"Range Trades: {summary['range_count']}\n"
        f"Trending Trades: {summary['trending_count']}\n\n"
        f"Best Trade: {best_txt}\n"
        f"Worst Trade: {worst_txt}"
    )

def weekly_stats_message(summary: dict, monday: date, friday: date) -> str:
    best_txt = "N/A"
    worst_txt = "N/A"

    if summary["best_trade"] is not None:
        bt = summary["best_trade"]
        best_txt = f"{bt.direction} | {bt.result_label} | {bt.r_result:+.2f}R"
    if summary["worst_trade"] is not None:
        wt = summary["worst_trade"]
        worst_txt = f"{wt.direction} | {wt.result_label} | {wt.r_result:+.2f}R"

    return (
        f"📈 {CFG.user_title} — Weekly Stats\n"
        f"🗓 Period: {monday.isoformat()} → {friday.isoformat()} (Mon-Fri Riyadh)\n\n"
        f"Total Trades: {summary['total']}\n"
        f"Wins: {summary['wins']}\n"
        f"Losses: {summary['losses']}\n"
        f"Breakeven: {summary['breakeven']}\n"
        f"Win Rate: {summary['winrate']:.1f}%\n"
        f"Net Result: {summary['net_r']:+.2f}R\n\n"
        f"T1 only: {summary['t1_only']}\n"
        f"T2 hit: {summary['t2_hit']}\n"
        f"T3 hit: {summary['t3_hit']}\n"
        f"T4+: {summary['t4_plus']}\n\n"
        f"Range Trades: {summary['range_count']}\n"
        f"Trending Trades: {summary['trending_count']}\n\n"
        f"Best Trade: {best_txt}\n"
        f"Worst Trade: {worst_txt}"
    )


# =========================
# TradingView fetching
# =========================

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


# =========================
# Structure / Levels
# =========================

def find_pivots(series: pd.Series, left: int, right: int):
    arr = series.values
    piv_hi, piv_lo = [], []
    n = len(arr)
    for i in range(left, n - right):
        v = arr[i]
        wl = arr[i - left: i]
        wr = arr[i + 1: i + 1 + right]
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

def nearest_opposite_level_distance(levels: list, level_hit: float, direction: str) -> float | None:
    if not levels:
        return None
    if direction == "BUY":
        above = sorted([x for x in levels if x > level_hit])
        if not above:
            return None
        return float(above[0] - level_hit)
    below = sorted([x for x in levels if x < level_hit], reverse=True)
    if not below:
        return None
    return float(level_hit - below[0])

def nearby_levels_count(levels: list, level_hit: float, within_pts: float) -> int:
    count = 0
    for lvl in levels:
        if abs(float(lvl) - float(level_hit)) <= within_pts:
            count += 1
    return count

def range_level_is_clean(
    levels: list,
    level_hit: float,
    direction: str,
    level_info: dict,
    market_label: str,
    market_adx: float | None,
    exp_move_val: float | None,
) -> tuple[bool, str]:
    if not CFG.range_quality_filter_enabled:
        return True, "range filter off"

    if market_label != "Range":
        return True, "not range"

    if market_adx is not None and market_adx > CFG.range_filter_only_when_adx_below:
        return True, "range adx too strong"

    wick_info = level_info["wick_info"]
    has_clean_wick = (
        (direction == "BUY" and (wick_info.get("lower_hits", 0) >= 1 or wick_info.get("lower_cluster")))
        or
        (direction == "SELL" and (wick_info.get("upper_hits", 0) >= 1 or wick_info.get("upper_cluster")))
    )

    if not has_clean_wick and not level_info.get("strong", False):
        return False, "range level rejected: weak wick/level"

    space_pts = nearest_opposite_level_distance(levels, level_hit, direction)
    if space_pts is None:
        return False, "range level rejected: no opposite space"

    min_space_pts = CFG.range_min_space_pts
    if exp_move_val is not None and np.isfinite(exp_move_val):
        min_space_pts = max(min_space_pts, float(exp_move_val) * CFG.range_min_space_expected_move_ratio)

    if space_pts < min_space_pts:
        return False, f"range level rejected: too little space ({space_pts:.1f} pts)"

    nearby = nearby_levels_count(levels, level_hit, CFG.range_max_nearby_levels_within_pts)
    if nearby > CFG.range_max_nearby_levels_count:
        return False, f"range level rejected: congestion ({nearby} nearby lvls)"

    return True, "range level accepted"


# =========================
# Target map helpers
# =========================

def target_probability_estimate(
    direction: str,
    target_price: float,
    current_price: float,
    exp_move_val: float | None,
    market_label: str,
    market_dir: str,
    liq_state: str,
    df_5m: pd.DataFrame,
) -> int:
    if target_price is None or not np.isfinite(target_price):
        return 0

    dist = abs(target_price - current_price)
    score = 58.0

    if exp_move_val is not None and np.isfinite(exp_move_val) and exp_move_val > 0:
        ratio = dist / exp_move_val
        if ratio <= 0.5:
            score += 18
        elif ratio <= 0.85:
            score += 10
        elif ratio <= 1.1:
            score += 3
        elif ratio <= 1.35:
            score -= 8
        else:
            score -= 18

    if market_label == "Trending":
        score += 6
    elif market_label == "Range":
        score -= 4

    if (market_dir == "Bullish" and direction == "BUY") or (market_dir == "Bearish" and direction == "SELL"):
        score += 6
    elif market_dir in ("Bullish", "Bearish"):
        score -= 8

    if liq_state == "Low":
        score -= 6
    elif liq_state == "High":
        score += 2
    elif liq_state == "Extreme":
        score += 3

    try:
        c = float(df_5m["close"].iloc[-1])
        ema = float(df_5m["ema20"].iloc[-1])
        macd_hist = float(df_5m["macd_hist"].iloc[-1])
        if direction == "BUY":
            if c > ema:
                score += 4
            if macd_hist > 0:
                score += 4
        else:
            if c < ema:
                score += 4
            if macd_hist < 0:
                score += 4
    except Exception:
        pass

    return int(max(5, min(95, round(score))))

def classify_target_zone(target_idx: int, prob: int) -> str:
    if target_idx == 1:
        return "First reaction zone"
    if target_idx == 2:
        return "Continuation zone"
    if target_idx == 3:
        return "Weakening watch" if prob < 60 else "Continuation extension"
    if target_idx >= 4:
        return "Exhaustion / reversal watch" if prob < 55 else "Runner extension"
    return "Target zone"

def build_initial_target_map(
    direction: str,
    entry: float,
    targets: list[float],
    current_price: float,
    exp_move_val: float | None,
    market_label: str,
    market_dir: str,
    liq_state: str,
    df_5m: pd.DataFrame,
) -> list[dict]:
    out = []
    for i, tgt in enumerate(targets[:CFG.max_initial_targets], start=1):
        if tgt is None or not np.isfinite(tgt):
            continue
        prob = target_probability_estimate(
            direction=direction,
            target_price=float(tgt),
            current_price=current_price,
            exp_move_val=exp_move_val,
            market_label=market_label,
            market_dir=market_dir,
            liq_state=liq_state,
            df_5m=df_5m,
        )
        out.append({
            "name": f"T{i}",
            "price": float(tgt),
            "prob": int(prob),
            "zone": classify_target_zone(i, prob),
        })
    return out

def filter_target_map_for_message(target_map: list[dict]) -> list[dict]:
    if not target_map:
        return []
    kept = []
    for i, t in enumerate(target_map, start=1):
        if i <= 2:
            kept.append(t)
            continue
        if t["prob"] >= CFG.target_probability_min_to_show:
            kept.append(t)
    return kept
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


# =========================
# Liquidity / Market state
# =========================

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

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, prev_label: str):
    if len(df_1h) < (CFG.adx_window + 5):
        return "Weak", "Neutral", None

    adx = ADXIndicator(
        high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], window=CFG.adx_window
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
# Wick / Level quality
# =========================

def wick_cluster_near_level(df_5m: pd.DataFrame, level: float) -> dict:
    w = df_5m.tail(min(CFG.wick_cluster_lookback_5m, len(df_5m))).copy()
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
            (abs(h - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac)
            or (abs(l - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac)
            or (abs(c - level) / max(level, 1e-9) <= CFG.wick_near_level_tolerance_frac)
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


# =========================
# Trade strength / targets
# =========================

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
            above[1] if len(above) >= 2 else None
        )
    if direction == "SELL":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        return (
            below[0] if len(below) >= 1 else None,
            below[1] if len(below) >= 2 else None
        )
    return None, None

def compute_range_adaptive_extra_stop(
    market_label: str,
    exp_move_val: float | None,
) -> float:
    if market_label != "Range":
        return 0.0

    extra = CFG.range_stop_extra_min_pts
    if exp_move_val is not None and np.isfinite(exp_move_val):
        extra = max(extra, min(CFG.range_stop_extra_max_pts, float(exp_move_val) * CFG.range_stop_extra_atr_mult))

    return float(max(CFG.range_stop_extra_min_pts, min(CFG.range_stop_extra_max_pts, extra)))

def compute_smart_t2(
    entry: float,
    direction: str,
    t1: float | None,
    raw_t2: float | None,
    exp_move_val: float | None,
) -> float | None:
    if not CFG.smart_t2_enabled or t1 is None or not np.isfinite(t1):
        return raw_t2

    if exp_move_val is None or not np.isfinite(exp_move_val) or exp_move_val <= 0:
        return raw_t2

    if direction == "BUY":
        candidate_a = entry + (float(exp_move_val) * CFG.smart_t2_atr_mult)
        candidate_b = entry + (float(exp_move_val) * CFG.smart_t2_expected_move_mult)
        smart_candidate = max(candidate_a, candidate_b, t1 + max(2.0, (t1 - entry) * 0.35))
        chosen = smart_candidate if raw_t2 is None else min(raw_t2, smart_candidate)
        if (chosen - entry) / max((t1 - entry), 1e-9) > CFG.smart_t2_max_distance_ratio_to_t1:
            chosen = entry + ((t1 - entry) * CFG.smart_t2_max_distance_ratio_to_t1)
        if chosen <= t1:
            chosen = t1 + max(2.0, float(exp_move_val) * 0.18)
        return float(chosen)

    candidate_a = entry - (float(exp_move_val) * CFG.smart_t2_atr_mult)
    candidate_b = entry - (float(exp_move_val) * CFG.smart_t2_expected_move_mult)
    smart_candidate = min(candidate_a, candidate_b, t1 - max(2.0, (entry - t1) * 0.35))
    chosen = smart_candidate if raw_t2 is None else max(raw_t2, smart_candidate)
    if (entry - chosen) / max((entry - t1), 1e-9) > CFG.smart_t2_max_distance_ratio_to_t1:
        chosen = entry - ((entry - t1) * CFG.smart_t2_max_distance_ratio_to_t1)
    if chosen >= t1:
        chosen = t1 - max(2.0, float(exp_move_val) * 0.18)
    return float(chosen)

def compute_trade_plan(
    df_5m: pd.DataFrame,
    levels: list,
    level_hit: float,
    direction: str,
    trigger: str,
    trade_type: str,
    exp_move_val: float | None,
    market_label: str,
    market_dir: str,
    liq_state: str,
):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    buffer = max(price * 0.0002, 0.5)
    range_extra_stop = compute_range_adaptive_extra_stop(market_label, exp_move_val)

    if trigger == "Wick Rejection near Level":
        if direction == "BUY":
            if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                entry = float(max(price, level_hit + buffer * 0.4))
            else:
                entry = float(max(price, (last_high + level_hit) / 2.0))
            stop = float(min(last_low, level_hit) - buffer - range_extra_stop)
        else:
            if trade_type == "Strong" and CFG.aggressive_entry_for_strong_wick:
                entry = float(min(price, level_hit - buffer * 0.4))
            else:
                entry = float(min(price, (last_low + level_hit) / 2.0))
            stop = float(max(last_high, level_hit) + buffer + range_extra_stop)

    elif trigger == "Rejection":
        if direction == "BUY":
            entry = float(max(price, last_high - buffer * 0.4))
            stop = float(min(last_low, level_hit) - buffer - range_extra_stop)
        else:
            entry = float(min(price, last_low + buffer * 0.4))
            stop = float(max(last_high, level_hit) + buffer + range_extra_stop)
    else:
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer))
            stop = float(level_hit - (price * 0.0013) - buffer - range_extra_stop)
        else:
            entry = float(min(price, level_hit - buffer))
            stop = float(level_hit + (price * 0.0013) + buffer + range_extra_stop)

    raw_t1, raw_t2 = pick_targets(levels, entry, direction)
    atr_t1, atr_t2, atr_t3 = atr_fallback_targets(entry, direction, exp_move_val, trade_type)

    t1 = raw_t1 if raw_t1 is not None else atr_t1
    t2 = raw_t2 if raw_t2 is not None else atr_t2
    t2 = compute_smart_t2(entry=entry, direction=direction, t1=t1, raw_t2=t2, exp_move_val=exp_move_val)

    t3 = None
    if trade_type == "Strong" and atr_t3 is not None:
        if direction == "BUY":
            if t2 is None or atr_t3 > t2:
                t3 = atr_t3
        else:
            if t2 is None or atr_t3 < t2:
                t3 = atr_t3

    targets = [t for t in [t1, t2, t3] if t is not None and np.isfinite(t)]
    if direction == "BUY":
        targets = sorted(set(float(x) for x in targets))
    else:
        targets = sorted(set(float(x) for x in targets), reverse=True)

    t1 = targets[0] if len(targets) > 0 else None
    t2 = targets[1] if len(targets) > 1 else None
    t3 = targets[2] if len(targets) > 2 else None

    if t1 is not None and t2 is not None:
        if direction == "BUY" and t2 <= t1:
            t2 = t1 + max(2.0, (exp_move_val or 8.0) * 0.2)
        if direction == "SELL" and t2 >= t1:
            t2 = t1 - max(2.0, (exp_move_val or 8.0) * 0.2)

    rr = None
    if t1 is not None:
        risk = abs(entry - stop)
        reward = abs(t1 - entry)
        rr = (reward / risk) if risk > 0 else None

    initial_targets = [x for x in [t1, t2, t3] if x is not None and np.isfinite(x)]
    target_map = build_initial_target_map(
        direction=direction,
        entry=entry,
        targets=initial_targets,
        current_price=price,
        exp_move_val=exp_move_val,
        market_label=market_label,
        market_dir=market_dir,
        liq_state=liq_state,
        df_5m=df_5m,
    )
    message_target_map = filter_target_map_for_message(target_map)

    t4 = None
    if len(message_target_map) >= 4:
        t4 = float(message_target_map[3]["price"])

    return {
        "entry": float(entry),
        "stop": float(stop),
        "initial_stop": float(stop),
        "t1": None if t1 is None else float(t1),
        "t2": None if t2 is None else float(t2),
        "t3": None if t3 is None else float(t3),
        "t4": None if t4 is None else float(t4),
        "rr": rr,
        "trade_type": trade_type,
        "target_map": message_target_map,
        "all_target_candidates": target_map,
    }


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

def format_target_map_for_message(target_map: list[dict]) -> str:
    if not target_map:
        return "N/A"

    lines = []
    for t in target_map:
        lines.append(
            f"{t['name']}: {safe_f1(t['price'])} | {safe_int(t['prob'])}% | {t['zone']}"
        )
    return "\n".join(lines)

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
    t3_txt = safe_f1(plan.get("t3"))
    rr_txt = safe_f2(plan.get("rr"))
    exp_txt = f"±{float(exp_move):.0f} pts" if exp_move is not None and np.isfinite(float(exp_move)) else "N/A"
    eta_txt = "N/A" if eta_band is None else f"{eta_band[0]}–{eta_band[1]} min"
    trade_type = plan.get("trade_type", "N/A")
    target_map_txt = format_target_map_for_message(plan.get("target_map", []))

    return (
        f"🚨 {CFG.user_title} — فرصة دخول (Hunter Smart)\n\n"
        f"🕒 Time: {now_riyadh().strftime('%Y-%m-%d %H:%M')} (Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: TradingView | Symbol: {symbol}\n\n"
        f"📊 Market State: {market_state_str}\n"
        f"💧 Liquidity: {liq_state}\n"
        f"💰 Level Now: {safe_f1(level_now)}\n\n"
        f"📍 Direction: {direction}\n"
        f"🧱 Level: {safe_f1(level_hit)}\n"
        f"🧬 Trade Type: {trade_type}\n\n"
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
        f"🗺 Target Map\n"
        f"{target_map_txt}\n\n"
        f"📈 Expected Move (1H): {exp_txt}\n"
        f"⏱ ETA to T1: {eta_txt}\n\n"
        f"🧠 Reason: {', '.join(reasons)}"
    )

def hourly_update_message(session: str, symbol: str, bias: str, market_state_str: str,
                          levels: list, price: float, active_trade: dict | None) -> str:
    status = "None" if active_trade is None else active_trade.get("status", "Unknown")
    trade_line = "-"
    target_map_line = "N/A"
    if active_trade is not None:
        trade_line = (
            f"{active_trade.get('direction', '?')} | "
            f"Entry {safe_f1(active_trade.get('entry'))} | "
            f"Stop {safe_f1(active_trade.get('stop'))} | "
            f"T1 {safe_f1(active_trade.get('t1'))} | "
            f"T2 {safe_f1(active_trade.get('t2'))} | "
            f"T3 {safe_f1(active_trade.get('t3'))} | "
            f"T4 {safe_f1(active_trade.get('t4'))}"
        )
        target_map_line = format_target_map_for_message(active_trade.get("target_map", []))

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: TradingView | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): {bias}\n"
        f"📊 Market State: {market_state_str}\n"
        f"💵 Price: {safe_f1(price)}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"📌 Active Trade: {status}\n"
        f"🧾 Trade: {trade_line}\n"
        f"🗺 Targets:\n{target_map_line}"
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

        self.last_error_notify_utc = None
        self.stats = StatsState()

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

STATE = BotState()


# =========================
# Reset / Market state cache
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
            f"ملاحظة: الصفقة الحالية مستمرة ولن يتم حذفها، فقط بدأ يوم جديد."
        )
    else:
        send_telegram(
            f"🛎️ {CFG.user_title} — Daily Reset (Riyadh)\n"
            f"السبب: منتصف الليل بتوقيت السعودية.\n"
            f"لا توجد صفقة فعّالة — تم بدء يوم جديد."
        )

    STATE.last_reset_date_riyadh = today
    STATE.no_signal_until_utc = datetime.utcnow() + timedelta(minutes=CFG.no_signal_after_reset_minutes)

def maybe_update_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    nowu = datetime.utcnow()
    should = STATE.market_state_last_calc_utc is None or (
        nowu - STATE.market_state_last_calc_utc
    ) >= timedelta(minutes=CFG.market_state_update_minutes)

    if not should:
        return

    label, direction, adx_val = compute_market_state(df_1h, df_4h, STATE.market_label)
    STATE.market_label = label
    STATE.market_dir = direction
    STATE.market_adx = adx_val
    STATE.market_state_last_calc_utc = nowu


# =========================
# Wick validation
# =========================

def wick_trade_is_valid(direction: str, trigger: str, level_info: dict, market_label: str, market_dir: str,
                        df_5m: pd.DataFrame, score: int) -> bool:
    wick_info = level_info["wick_info"]
    touches = level_info["touches"]
    strong_level = level_info["strong"]

    is_wick_trade = (
        trigger == "Wick Rejection near Level"
        or wick_info.get("upper_cluster")
        or wick_info.get("lower_cluster")
    )
    if not is_wick_trade:
        return True

    if not level_info["tradable"]:
        return False

    if direction == "SELL":
        if wick_info["upper_hits"] < 1 and not wick_info["upper_cluster"]:
            return False
    if direction == "BUY":
        if wick_info["lower_hits"] < 1 and not wick_info["lower_cluster"]:
            return False

    if score <= 2 and not strong_level:
        return False

    if market_label == "Trending":
        if market_dir == "Bullish" and direction == "SELL":
            if score < 5:
                return False
            if wick_info["upper_hits"] < 2 and not wick_info["upper_cluster"]:
                return False
            if touches < CFG.level_strong_touch_count:
                return False
            if not strong_sell_confirmation(df_5m):
                return False

        if market_dir == "Bearish" and direction == "BUY":
            if score < 5:
                return False
            if wick_info["lower_hits"] < 2 and not wick_info["lower_cluster"]:
                return False
            if touches < CFG.level_strong_touch_count:
                return False
            if not strong_buy_confirmation(df_5m):
                return False

    return True


# =========================
# Score setup
# =========================

def score_setup(df_5m: pd.DataFrame, level_hit: float, direction: str, wick_info: dict) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    trigger = None

    wick_reason = None
    if wick_info.get("upper_cluster") and direction == "SELL":
        wick_reason = f"Upper-wick rejection cluster near {wick_info.get('bucket', 'N/A')}"
    if wick_info.get("lower_cluster") and direction == "BUY":
        wick_reason = f"Lower-wick cluster near support {wick_info.get('bucket', 'N/A')}"

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
# Dynamic expansion / target behavior
# =========================

def next_levels_beyond(levels: list, price_ref: float, direction: str) -> list:
    if direction == "BUY":
        return sorted([lvl for lvl in levels if lvl > price_ref])
    return sorted([lvl for lvl in levels if lvl < price_ref], reverse=True)

def dynamic_extension_allowed(df_5m: pd.DataFrame, direction: str) -> bool:
    if direction == "BUY":
        return strong_buy_confirmation(df_5m)
    return strong_sell_confirmation(df_5m)

def target_behavior_check(
    df_5m: pd.DataFrame,
    direction: str,
    market_label: str,
    exp_move_val: float | None,
    current_price: float,
    candidate_target: float | None,
) -> tuple[bool, str]:
    if candidate_target is None or not np.isfinite(candidate_target):
        return False, "invalid candidate target"

    if len(df_5m) < 10:
        return False, "not enough 5m data"

    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema20"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])
    macd_prev = float(df_5m["macd_hist"].iloc[-2])

    last = df_5m.iloc[-1]
    o, h, l, cl = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    dist = abs(candidate_target - current_price)

    if exp_move_val is not None and np.isfinite(exp_move_val):
        if dist > float(exp_move_val) * 1.35:
            return False, "target too far vs expected move"

    if direction == "BUY":
        if c < ema and macd_hist < macd_prev:
            return False, "buy extension weakening"
        if upper / rng >= 0.42 and cl < o:
            return False, "buy rejection candle"
    else:
        if c > ema and macd_hist > macd_prev:
            return False, "sell extension weakening"
        if lower / rng >= 0.42 and cl > o:
            return False, "sell rejection candle"

    if market_label == "Range" and dist < CFG.extension_rr_min_points:
        return False, "range extension too small"

    return True, "extension valid"

def dynamic_target_from_market_or_atr(levels: list, current_price: float, direction: str, exp_move_val: float | None,
                                      used_targets: list[float], stage: str) -> float | None:
    candidates = next_levels_beyond(levels, current_price, direction)

    for lvl in candidates:
        if all(abs(lvl - x) > 1.0 for x in used_targets):
            return float(lvl)

    if exp_move_val is None or not np.isfinite(exp_move_val):
        return None

    if stage == "T3":
        dist = exp_move_val * CFG.dynamic_t3_atr_mult
    elif stage == "T4":
        dist = exp_move_val * CFG.dynamic_t4_atr_mult
    else:
        dist = exp_move_val * (CFG.dynamic_t4_atr_mult + 0.18)

    if direction == "BUY":
        candidate = current_price + dist
    else:
        candidate = current_price - dist

    if all(abs(candidate - x) > 1.0 for x in used_targets):
        return float(candidate)

    return None

def probability_for_extension_target(
    df_5m: pd.DataFrame,
    direction: str,
    candidate_target: float,
    current_price: float,
    exp_move_val: float | None,
    market_label: str,
    market_dir: str,
    liq_state: str,
) -> int:
    return target_probability_estimate(
        direction=direction,
        target_price=float(candidate_target),
        current_price=float(current_price),
        exp_move_val=exp_move_val,
        market_label=market_label,
        market_dir=market_dir,
        liq_state=liq_state,
        df_5m=df_5m,
    )

def maybe_extend_next_target(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if tr is None or tr.get("status") != "live":
        return

    direction = tr["direction"]
    used_targets = [x for x in tr.get("all_target_prices", []) if x is not None]
    exp_move_val = expected_move_1h(df_1h)
    levels = extract_key_levels(df_15m, df_1h)

    if not dynamic_extension_allowed(df_5m, direction):
        return

    next_idx = int(tr.get("next_target_idx_to_define", 4))
    if next_idx < 3:
        next_idx = 3

    stage_name = f"T{next_idx}"
    candidate = dynamic_target_from_market_or_atr(
        levels=levels,
        current_price=last_price,
        direction=direction,
        exp_move_val=exp_move_val,
        used_targets=used_targets,
        stage=stage_name,
    )
    if candidate is None:
        return

    ok, reason = target_behavior_check(
        df_5m=df_5m,
        direction=direction,
        market_label=STATE.market_label,
        exp_move_val=exp_move_val,
        current_price=last_price,
        candidate_target=candidate,
    )
    if not ok:
        return

    prob = probability_for_extension_target(
        df_5m=df_5m,
        direction=direction,
        candidate_target=float(candidate),
        current_price=float(last_price),
        exp_move_val=exp_move_val,
        market_label=STATE.market_label,
        market_dir=STATE.market_dir,
        liq_state=liquidity_state(df_5m)[0],
    )
    if prob < CFG.target_probability_min_to_extend:
        return

    zone = classify_target_zone(next_idx, prob)

    tr["dynamic_targets"][stage_name] = {
        "price": float(candidate),
        "prob": int(prob),
        "zone": zone,
        "defined": True,
        "hit": False,
    }
    tr["all_target_prices"].append(float(candidate))
    tr["target_map"].append({
        "name": stage_name,
        "price": float(candidate),
        "prob": int(prob),
        "zone": zone,
    })
    tr["next_target_idx_to_define"] = next_idx + 1

    prev_hit_idx = int(tr.get("last_target_hit_idx", 2))
    prev_target_name = f"T{prev_hit_idx}"
    prev_target = tr["dynamic_targets"].get(prev_target_name, {})
    prev_target_price = prev_target.get("price")
    if prev_target_price is not None:
        tr["stop"] = float(prev_target_price)

    send_telegram(
        f"🧠 {CFG.user_title} — Dynamic Target Extension\n"
        f"New {stage_name}: {safe_f1(candidate)}\n"
        f"Probability: {prob}%\n"
        f"Zone: {zone}\n"
        f"Raised Stop: {safe_f1(tr['stop'])}"
    )

def detect_weakening_or_reversal_watch(df_5m: pd.DataFrame, direction: str) -> tuple[bool, str]:
    if len(df_5m) < 8:
        return False, "not enough data"

    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]

    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l

    ema = float(df_5m["ema20"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])
    macd_prev = float(df_5m["macd_hist"].iloc[-2])

    if direction == "BUY":
        if upper / rng >= 0.45 and c < o:
            return True, "upper rejection after target"
        if c < ema and macd_hist < macd_prev:
            return True, "buy momentum weakening"
        if c < pc and h <= ph:
            return True, "buy follow-through weakening"
    else:
        if lower / rng >= 0.45 and c > o:
            return True, "lower rejection after target"
        if c > ema and macd_hist > macd_prev:
            return True, "sell momentum weakening"
        if c > pc and l >= pl:
            return True, "sell follow-through weakening"

    return False, "no weakness"


# =========================
# Trade result recording
# =========================

def record_closed_trade(result_label: str, max_target_hit: int, r_result: float):
    tr = STATE.active_trade
    if tr is None:
        return

    rec = ClosedTradeRecord(
        closed_at_riyadh=now_riyadh(),
        direction=str(tr.get("direction", "?")),
        entry=float(tr.get("entry", np.nan)),
        stop=float(tr.get("stop", np.nan)),
        initial_stop=float(tr.get("initial_stop", np.nan)),
        trade_type=str(tr.get("trade_type", "N/A")),
        result_label=str(result_label),
        max_target_hit=int(max_target_hit),
        r_result=float(r_result),
        market_label=str(tr.get("market_label_at_entry", "Unknown")),
        session=str(tr.get("session_at_entry", "Unknown")),
    )
    STATE.stats.add_trade(rec)

def compute_r_result_for_stop_exit(last_price: float) -> tuple[float, str]:
    tr = STATE.active_trade
    if tr is None:
        return 0.0, "unknown"

    entry = float(tr["entry"])
    initial_stop = float(tr["initial_stop"])
    direction = tr["direction"]
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return 0.0, "invalid risk"

    pnl = (last_price - entry) if direction == "BUY" else (entry - last_price)
    r_result = pnl / risk

    if abs(r_result) < 0.08:
        return 0.0, "breakeven"
    if r_result > 0:
        return float(r_result), "stop in profit"
    return float(r_result), "stop loss"

def current_max_target_hit(tr: dict) -> int:
    if tr is None:
        return 0

    max_hit = 0
    dynamic_targets = tr.get("dynamic_targets", {})
    for name, info in dynamic_targets.items():
        if info.get("hit"):
            try:
                idx = int(name.replace("T", ""))
                max_hit = max(max_hit, idx)
            except Exception:
                pass
    return max_hit


# =========================
# Stats sending
# =========================

def maybe_send_daily_stats():
    if not CFG.stats_daily_enabled:
        return

    t = now_riyadh()
    target_date = t.date()

    should_window = (t.hour == CFG.stats_daily_send_hour and t.minute >= CFG.stats_daily_send_minute)
    if not should_window:
        return

    if STATE.stats.last_daily_stats_sent_date == target_date:
        return

    records = STATE.stats.get_day_records(target_date)
    summary = summarize_records(records)
    send_telegram(daily_stats_message(summary, target_date))
    STATE.stats.last_daily_stats_sent_date = target_date

def maybe_send_weekly_stats():
    if not CFG.stats_weekly_enabled:
        return

    t = now_riyadh()
    if t.weekday() != CFG.stats_weekly_send_weekday:
        return
    if not (t.hour == CFG.stats_weekly_send_hour and t.minute >= CFG.stats_weekly_send_minute):
        return

    monday = t.date() - timedelta(days=t.weekday())
    friday = monday + timedelta(days=4)
    week_key = f"{monday.isoformat()}_{friday.isoformat()}"

    if STATE.stats.last_weekly_stats_sent_week_key == week_key:
        return

    records = STATE.stats.get_week_records_mon_to_fri(t.date())
    summary = summarize_records(records)
    send_telegram(weekly_stats_message(summary, monday, friday))
    STATE.stats.last_weekly_stats_sent_week_key = week_key


# =========================
# Active trade management
# =========================

def update_active_trade(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, last_price: float):
    tr = STATE.active_trade
    if tr is None:
        return

    direction = tr["direction"]
    entry = float(tr["entry"])
    stop = float(tr["stop"])

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
                f"Targets:\n{format_target_map_for_message(tr.get('target_map', []))}"
            )
        return

    if tr["status"] != "live":
        return

    breached = (last_price <= stop) if direction == "BUY" else (last_price >= stop)
    if breached:
        if CFG.hard_stop_enabled:
            beyond = (stop - last_price) >= CFG.hard_stop_buffer_pts if direction == "BUY" else (last_price - stop) >= CFG.hard_stop_buffer_pts
            if beyond:
                r_result, label = compute_r_result_for_stop_exit(last_price)
                max_hit = current_max_target_hit(tr)
                send_telegram(
                    f"❌ {CFG.user_title} — وقف الخسارة (Hard Stop)\n"
                    f"Stop Hit ✅ | Direction: {direction}\n"
                    f"Stop: {safe_f1(stop)} | Price: {safe_f1(last_price)}\n"
                    f"Result: {r_result:+.2f}R\n"
                    f"ملاحظة: تم الإغلاق الفوري لتجنب انزلاق كبير."
                )
                record_closed_trade(label, max_hit, r_result)
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
                        r_result, label = compute_r_result_for_stop_exit(last_price)
                        max_hit = current_max_target_hit(tr)
                        send_telegram(
                            f"❌ {CFG.user_title} — وقف الخسارة تأكد\n"
                            f"Stop Hit ✅ | Direction: {direction}\n"
                            f"Stop: {safe_f1(stop)} | Price: {safe_f1(last_price)}\n"
                            f"Result: {r_result:+.2f}R\n"
                            f"تأكيد: إغلاق 5m ضد الصفقة."
                        )
                        record_closed_trade(label, max_hit, r_result)
                        STATE.active_trade = None
                        return
                    else:
                        tr["stop_pending"] = False
                        tr.pop("stop_pending_since_utc", None)
            return

    if tr.get("stop_pending"):
        tr["stop_pending"] = False
        tr.pop("stop_pending_since_utc", None)

    # Check target hits dynamically
    dynamic_targets = tr.get("dynamic_targets", {})
    sorted_targets = sorted(
        dynamic_targets.items(),
        key=lambda kv: int(kv[0].replace("T", "")),
        reverse=False
    )

    for name, info in sorted_targets:
        if info.get("hit"):
            continue

        tgt_price = float(info["price"])
        idx = int(name.replace("T", ""))

        hit = (last_price >= tgt_price) if direction == "BUY" else (last_price <= tgt_price)
        if not hit:
            continue

        info["hit"] = True
        tr["last_target_hit_idx"] = idx

        # Raise stop to previous target/entry
        if idx == 1 and CFG.move_stop_to_be_on_t1:
            tr["stop"] = float(entry)
        elif idx == 2 and CFG.move_stop_to_t1_on_t2:
            prev = dynamic_targets.get("T1", {})
            if prev.get("price") is not None:
                tr["stop"] = float(prev["price"])
        elif idx >= 3 and CFG.move_stop_to_t2_on_t3:
            prev_name = f"T{idx-1}"
            prev = dynamic_targets.get(prev_name, {})
            if prev.get("price") is not None:
                tr["stop"] = float(prev["price"])

        if idx == 1:
            send_telegram(
                f"🎯 {CFG.user_title} — {name} Hit\n"
                f"{name}: {safe_f1(tgt_price)}\n"
                f"Probability was: {safe_int(info.get('prob'))}%\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )
        else:
            weak, weak_reason = detect_weakening_or_reversal_watch(df_5m, direction)
            zone_note = info.get("zone", "Target zone")
            msg = (
                f"🏆 {CFG.user_title} — {name} Hit\n"
                f"{name}: {safe_f1(tgt_price)}\n"
                f"Probability was: {safe_int(info.get('prob'))}%\n"
                f"Zone: {zone_note}\n"
                f"Raised Stop: {safe_f1(tr['stop'])}"
            )
            if weak:
                msg += f"\n⚠️ Weakening Watch: {weak_reason}"
            send_telegram(msg)

        # Extend next target only after hitting the latest currently defined target
        highest_defined_idx = max(int(k.replace("T", "")) for k in dynamic_targets.keys())
        if idx >= highest_defined_idx:
            maybe_extend_next_target(df_5m, df_15m, df_1h, last_price)

       # If there is weakness after higher target, notify reversal watch
    # وإذا كانت الصفقة وصلت هدف متقدم ثم ظهر ضعف واضح، نعتبرها منتهية إحصائيًا إذا ما عاد فيه أهداف أقوى
    max_hit_idx = current_max_target_hit(tr)
    if max_hit_idx >= 2:
        weak, weak_reason = detect_weakening_or_reversal_watch(df_5m, direction)
        if weak and not tr.get("reversal_watch_sent"):
            tr["reversal_watch_sent"] = True
            send_telegram(
                f"🔄 {CFG.user_title} — Reversal Watch Zone\n"
                f"Last Hit Target: T{max_hit_idx}\n"
                f"Reason: {weak_reason}\n"
                f"ملاحظة: ليست صفقة عكسية مباشرة — فقط مراقبة لضعف الحركة."
            )

            # إذا ما فيه هدف جديد متوقع والصفقة أصلاً وصلت هدف متقدم،
            # سجلها كصفقة منتهية إحصائيًا عند آخر سعر حالي
            highest_defined_idx = 0
            if tr.get("dynamic_targets"):
                highest_defined_idx = max(int(k.replace('T', '')) for k in tr["dynamic_targets"].keys())

            if max_hit_idx >= highest_defined_idx:
                risk = abs(float(tr["entry"]) - float(tr["initial_stop"]))
                if risk > 0:
                    pnl = (last_price - float(tr["entry"])) if direction == "BUY" else (float(tr["entry"]) - last_price)
                    r_result = pnl / risk
                else:
                    r_result = 0.0

                record_closed_trade(
                    result_label=f"closed on weakness after T{max_hit_idx}",
                    max_target_hit=max_hit_idx,
                    r_result=float(r_result),
                )
                send_telegram(
                    f"✅ {CFG.user_title} — Trade Registered Closed\n"
                    f"Reason: weakness after T{max_hit_idx}\n"
                    f"Result: {r_result:+.2f}R"
                )
                STATE.active_trade = None
                return

# =========================
# Main evaluation
# =========================

def evaluate_once():
    maybe_daily_reset()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m = fetch_timeframes()
    df_5m = compute_indicators_5m(df_5m)
    level_now = float(df_5m["close"].iloc[-1])

    maybe_update_market_state(df_1h, df_4h)
    adx_txt = "N/A" if STATE.market_adx is None else safe_f1(STATE.market_adx)
    market_state_str = f"{STATE.market_label} | {STATE.market_dir} | ADX(1H): {adx_txt}"

    liq_state, _ = liquidity_state(df_5m)
    bias = structure_bias(df_1h, df_4h)
    key_levels = extract_key_levels(df_15m, df_1h)
    exp_move_val = expected_move_1h(df_1h)

    if STATE.active_trade is not None:
        update_active_trade(df_5m, df_15m, df_1h, level_now)

    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_update_message(
                session, symbol, bias, market_state_str, key_levels, level_now, STATE.active_trade
            ))
            STATE.last_hour_sent = current_hour

    maybe_send_daily_stats()
    maybe_send_weekly_stats()

    if STATE.no_signal_until_utc is not None and datetime.utcnow() < STATE.no_signal_until_utc:
        return

    if STATE.active_trade is not None:
        return

    if not key_levels:
        return

    level_hit, level_info = choose_best_level(df_5m, level_now, key_levels)
    wick_info = level_info["wick_info"]

    if bias == "Bullish":
        direction = "BUY"
    elif bias == "Bearish":
        direction = "SELL"
    else:
        direction = "BUY" if momentum_shift(df_5m, "BUY") else "SELL"

    sell_confirm = momentum_shift(df_5m, "SELL") or stoch_cross(df_5m, "SELL") or break_retest(df_5m, level_hit, "SELL")
    buy_confirm = momentum_shift(df_5m, "BUY") or stoch_cross(df_5m, "BUY") or break_retest(df_5m, level_hit, "BUY")

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
        return

    score, reasons, trigger = score_setup(df_5m, level_hit, direction, wick_info)
    if score <= 0:
        return

    conf = confidence_percent(score, direction, bias, session, STATE.market_label, liq_state)
    trade_type = classify_trade_strength(score, conf)

    if CFG.block_all_weak_trades and trade_type == "Weak":
        return

    if session in ("After-Hours", "Pre-Market"):
        if score < CFG.offhours_min_score:
            return
        if CFG.offhours_block_weak and trade_type == "Weak":
            return

    if CFG.block_counter_trend_in_trending_market and STATE.market_label == "Trending":
        if STATE.market_dir == "Bullish" and direction == "SELL":
            if not (CFG.allow_counter_trend_wick_if_strong and wick_trade_is_valid(
                direction=direction,
                trigger=trigger,
                level_info=level_info,
                market_label=STATE.market_label,
                market_dir=STATE.market_dir,
                df_5m=df_5m,
                score=score,
            )):
                return

        if STATE.market_dir == "Bearish" and direction == "BUY":
            if not (CFG.allow_counter_trend_wick_if_strong and wick_trade_is_valid(
                direction=direction,
                trigger=trigger,
                level_info=level_info,
                market_label=STATE.market_label,
                market_dir=STATE.market_dir,
                df_5m=df_5m,
                score=score,
            )):
                return

    if not wick_trade_is_valid(
        direction=direction,
        trigger=trigger,
        level_info=level_info,
        market_label=STATE.market_label,
        market_dir=STATE.market_dir,
        df_5m=df_5m,
        score=score,
    ):
        return

    # NEW: professional range filter
    range_ok, range_reason = range_level_is_clean(
        levels=key_levels,
        level_hit=level_hit,
        direction=direction,
        level_info=level_info,
        market_label=STATE.market_label,
        market_adx=STATE.market_adx,
        exp_move_val=exp_move_val,
    )
    if not range_ok:
        return

    if CFG.require_momentum_confirmation:
        if direction == "BUY" and not strong_buy_confirmation(df_5m):
            if not (
                trigger == "Wick Rejection near Level"
                and level_info["strong"]
                and wick_info["lower_hits"] >= 2
            ):
                return

        if direction == "SELL" and not strong_sell_confirmation(df_5m):
            if not (
                trigger == "Wick Rejection near Level"
                and level_info["strong"]
                and wick_info["upper_hits"] >= 2
            ):
                return

    if trade_type == "Weak" and trigger != "Wick Rejection near Level":
        if not level_info["strong"]:
            return

    plan = compute_trade_plan(
        df_5m=df_5m,
        levels=key_levels,
        level_hit=level_hit,
        direction=direction,
        trigger=trigger,
        trade_type=trade_type,
        exp_move_val=exp_move_val,
        market_label=STATE.market_label,
        market_dir=STATE.market_dir,
        liq_state=liq_state,
    )

    rr = plan.get("rr")
    if rr is not None and np.isfinite(float(rr)) and float(rr) < CFG.min_rr_to_t1:
        return

    key = f"{direction}:{round(level_hit, 1)}:{trigger}:{trade_type}"
    if not STATE.can_signal(key):
        return

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
        reasons=reasons + ([range_reason] if range_reason not in ("not range", "range filter off", "range adx too strong", "range level accepted") else []),
    )
    send_telegram(msg)
    STATE.mark_signal(key)

    # dynamic targets storage from target map
    dynamic_targets = {}
    for t in plan.get("target_map", []):
        dynamic_targets[t["name"]] = {
            "price": float(t["price"]),
            "prob": int(t["prob"]),
            "zone": str(t["zone"]),
            "defined": True,
            "hit": False,
        }

    all_target_prices = [float(t["price"]) for t in plan.get("target_map", [])]

    initial_highest_target_idx = 0
    if plan.get("target_map"):
        initial_highest_target_idx = max(int(t["name"].replace("T", "")) for t in plan["target_map"])

    STATE.active_trade = {
        "direction": direction,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "initial_stop": float(plan["initial_stop"]),
        "t1": None if plan.get("t1") is None else float(plan["t1"]),
        "t2": None if plan.get("t2") is None else float(plan["t2"]),
        "t3": None if plan.get("t3") is None else float(plan["t3"]),
        "t4": None if plan.get("t4") is None else float(plan["t4"]),
        "status": "pending",
        "created_utc": datetime.utcnow(),
        "trade_type": trade_type,
        "market_label_at_entry": STATE.market_label,
        "session_at_entry": session,
        "target_map": plan.get("target_map", []),
        "all_target_candidates": plan.get("all_target_candidates", []),
        "dynamic_targets": dynamic_targets,
        "all_target_prices": all_target_prices,
        "next_target_idx_to_define": max(3, initial_highest_target_idx + 1),
        "last_target_hit_idx": 0,
        "reversal_watch_sent": False,
    }


# =========================
# Main
# =========================

def main():
    send_telegram(
    f"✅ {CFG.user_title} — Bot started\n"
    f"Rules:\n"
    f"- Source: TradingView ({CFG.tv_exchange}:{CFG.tv_symbol})\n"
    f"- Direction: Bias(1H/4H) ➜ Wick-Cluster near Key Level\n"
    f"- Strongest Reason: Wick Rejection near Level\n"
    f"- Dynamic target map: ON\n"
    f"- Open-ended target extension: ON\n"
    f"- Range quality filter: ON\n"
    f"- Daily / Weekly stats: ON\n"
    f"- Counter-trend protection: {'ON' if CFG.block_counter_trend_in_trending_market else 'OFF'}\n"
    f"- Stop: HardStop({CFG.hard_stop_buffer_pts:.1f} pts) + 5m close confirm\n"
    f"- Midnight reset: keeps active trade tracking"
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
