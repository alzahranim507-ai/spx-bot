# -*- coding: utf-8 -*-
"""
SPX Backtest Bot — Refined Adaptive Version
For: محمد

الفكرة:
- Backtest فقط، بدون Telegram أو loop live
- يعتمد على:
  * 5m execution
  * 1H bias
  * 4H major regime
  * structure + key levels + wick/momentum triggers
  * simplified clean pipeline
- مناسب لـ GitHub / Railway

ملاحظات:
- الدخول يتم على Open الشمعة التالية بعد الإشارة
- إدارة الصفقة تتم باستخدام High/Low لكل شمعة
- الهدف من النسخة: اختبار منطق "مهذّب" وواضح وقابل للتعديل

تشغيل:
    pip install pandas numpy requests tvDatafeed ta
    python spx_backtest_refined.py

اختياري:
    export TV_USERNAME="..."
    export TV_PASSWORD="..."

مخرجات:
- backtest_results.csv
- trades.csv
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from tvDatafeed import TvDatafeed, Interval
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    tv_symbol: str = "SPX500"
    tv_exchange: str = "FOREXCOM"
    tv_username: str = os.getenv("TV_USERNAME", "")
    tv_password: str = os.getenv("TV_PASSWORD", "")

    bars_5m: int = 12000

    # timeframes
    entry_tf: str = "5min"
    bias_tf: str = "1H"
    trend_tf: str = "4H"
    level_tf: str = "15min"

    # structure / pivots
    pivot_left: int = 3
    pivot_right: int = 3
    max_key_levels: int = 8
    level_cluster_tolerance_frac: float = 0.0010
    level_touch_tolerance_frac: float = 0.0012
    level_zone_near_pts: float = 18.0
    level_min_touch_count: int = 2
    level_strong_touch_count: int = 3

    # regime
    adx_window: int = 14
    adx_trending_on: float = 25.0
    adx_range_on: float = 20.0

    # indicators
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_len: int = 14
    atr_len: int = 14

    # wick
    wick_cluster_lookback_5m: int = 10
    wick_ratio_strong: float = 0.45
    wick_cluster_min_hits: int = 3
    wick_near_level_tolerance_frac: float = 0.0010
    wick_min_abs_pts: float = 1.2

    # filters
    min_rr_to_t1: float = 1.20
    min_score: int = 3
    strong_score_threshold: int = 5
    require_momentum_confirmation: bool = True
    allow_counter_trend_wick_if_strong: bool = True
    block_counter_trend_in_trending_market: bool = True

    # risk / targets
    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 0.75
    slippage_pts: float = 0.25
    commission_per_trade: float = 0.0

    hard_stop_buffer_pts: float = 1.0
    stop_confirm_by_5m_close: bool = False

    tp1_rr: float = 1.00
    tp2_rr: float = 1.80
    tp3_rr: float = 2.60
    dynamic_t3_atr_mult: float = 1.15

    # backtest controls
    start_after_warmup_bars: int = 500
    export_trades_csv: str = "trades.csv"
    export_summary_csv: str = "backtest_results.csv"


CFG = Config()


# =========================================================
# HELPERS
# =========================================================

def make_tv_client(cfg: Config) -> TvDatafeed:
    if cfg.tv_username and cfg.tv_password:
        return TvDatafeed(cfg.tv_username, cfg.tv_password)
    return TvDatafeed()

def normalize_tv_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out.sort_index()
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).copy()
    return out

def find_pivots(series: pd.Series, left: int, right: int) -> Tuple[List[int], List[int]]:
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

def cluster_levels(levels: List[float], tol_frac: float, price_ref: float) -> List[float]:
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

def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame, cfg: Config) -> str:
    def bias_from(df):
        hi_idx, lo_idx = find_pivots(df["high"], cfg.pivot_left, cfg.pivot_right)
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

def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, cfg: Config) -> Tuple[str, str, Optional[float]]:
    if len(df_1h) < cfg.adx_window + 5:
        return "Weak", "Neutral", None

    adx = ADXIndicator(
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        window=cfg.adx_window,
    ).adx()
    adx_val = float(adx.iloc[-1]) if len(adx) else None

    bias = structure_bias(df_1h, df_4h, cfg)
    direction = "Neutral"
    if bias == "Bullish":
        direction = "Bullish"
    elif bias == "Bearish":
        direction = "Bearish"

    if adx_val is None or np.isnan(adx_val):
        return "Weak", direction, None

    if adx_val >= cfg.adx_trending_on:
        label = "Trending"
    elif adx_val <= cfg.adx_range_on:
        label = "Range"
    else:
        label = "Messy"

    return label, direction, adx_val

def extract_key_levels(df_15m: pd.DataFrame, df_1h: pd.DataFrame, cfg: Config) -> List[float]:
    price = float(df_15m["close"].iloc[-1])

    hi_idx, lo_idx = find_pivots(df_1h["high"], cfg.pivot_left, cfg.pivot_right)
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx][-14:] if hi_idx else []
    swing_lows = [float(df_1h["low"].iloc[i]) for i in lo_idx][-14:] if lo_idx else []

    recent = df_15m.tail(220)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())

    candidates = swing_highs + swing_lows + [range_hi, range_lo]
    candidates = [float(x) for x in candidates if np.isfinite(x)]
    merged = cluster_levels(candidates, cfg.level_cluster_tolerance_frac, price)
    merged = sorted(merged, key=lambda x: abs(x - price))[: max(cfg.max_key_levels * 4, 20)]
    merged = sorted(cluster_levels(merged, cfg.level_cluster_tolerance_frac, price))
    if len(merged) > cfg.max_key_levels:
        closest = sorted(merged, key=lambda x: abs(x - price))[: cfg.max_key_levels - 2]
        merged = sorted(cluster_levels(
            closest + [min(merged), max(merged)],
            cfg.level_cluster_tolerance_frac,
            price
        ))
    return merged

def count_level_touches(df_5m: pd.DataFrame, level: float, lookback: int = 40, abs_tol: float = 2.0) -> int:
    recent = df_5m.tail(min(lookback, len(df_5m)))
    touches = 0
    for _, r in recent.iterrows():
        h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        if abs(h - level) <= abs_tol or abs(l - level) <= abs_tol or abs(c - level) <= abs_tol:
            touches += 1
    return touches

def wick_cluster_near_level(df_5m: pd.DataFrame, level: float, cfg: Config) -> Dict:
    w = df_5m.tail(min(cfg.wick_cluster_lookback_5m, len(df_5m))).copy()
    if w.empty:
        return {"upper_cluster": False, "lower_cluster": False, "upper_hits": 0, "lower_hits": 0}

    upper_hits = 0
    lower_hits = 0
    for _, r in w.iterrows():
        o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        rng = max(h - l, 1e-9)
        upper = h - max(o, c)
        lower = min(o, c) - l

        near = (
            abs(h - level) / max(level, 1e-9) <= cfg.wick_near_level_tolerance_frac
            or abs(l - level) / max(level, 1e-9) <= cfg.wick_near_level_tolerance_frac
            or abs(c - level) / max(level, 1e-9) <= cfg.wick_near_level_tolerance_frac
        )
        if not near:
            continue

        if upper >= cfg.wick_min_abs_pts and (upper / rng) >= cfg.wick_ratio_strong:
            upper_hits += 1
        if lower >= cfg.wick_min_abs_pts and (lower / rng) >= cfg.wick_ratio_strong:
            lower_hits += 1

    return {
        "upper_cluster": upper_hits >= cfg.wick_cluster_min_hits,
        "lower_cluster": lower_hits >= cfg.wick_cluster_min_hits,
        "upper_hits": int(upper_hits),
        "lower_hits": int(lower_hits),
    }

def level_quality_info(df_5m: pd.DataFrame, level_now: float, level: float, cfg: Config) -> Dict:
    wick_info = wick_cluster_near_level(df_5m, level, cfg)
    touches = count_level_touches(df_5m, level)
    distance_pts = abs(level_now - level)
    strong = (
        touches >= cfg.level_strong_touch_count
        or wick_info["upper_hits"] >= 2
        or wick_info["lower_hits"] >= 2
        or wick_info["upper_cluster"]
        or wick_info["lower_cluster"]
    )
    tradable = (
        touches >= cfg.level_min_touch_count
        or wick_info["upper_hits"] >= 1
        or wick_info["lower_hits"] >= 1
    )
    score = 0.0
    score += max(0.0, 22.0 - distance_pts) * 0.14
    score += float(wick_info["upper_hits"] + wick_info["lower_hits"]) * 1.5
    score += float(touches) * 0.45
    return {
        "touches": touches,
        "distance_pts": distance_pts,
        "quality_score": float(score),
        "strong": bool(strong),
        "tradable": bool(tradable),
        "wick_info": wick_info,
    }

def choose_best_level(df_5m: pd.DataFrame, level_now: float, key_levels: List[float], cfg: Config) -> Tuple[float, Dict]:
    ranked = []
    for lvl in key_levels:
        info = level_quality_info(df_5m, level_now, float(lvl), cfg)
        ranked.append((info["quality_score"], float(lvl), info))
    ranked.sort(key=lambda x: x[0], reverse=True)
    near_candidates = [x for x in ranked if abs(x[1] - level_now) <= cfg.level_zone_near_pts]
    chosen = near_candidates[0] if near_candidates else ranked[0]
    return chosen[1], chosen[2]


# =========================================================
# INDICATORS
# =========================================================

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ema9"] = EMAIndicator(close=out["close"], window=cfg.ema_fast).ema_indicator()
    out["ema21"] = EMAIndicator(close=out["close"], window=cfg.ema_mid).ema_indicator()
    out["ema50"] = EMAIndicator(close=out["close"], window=cfg.ema_slow).ema_indicator()
    out["ema200"] = EMAIndicator(close=out["close"], window=cfg.ema_trend).ema_indicator()
    out["rsi"] = RSIIndicator(close=out["close"], window=cfg.rsi_len).rsi()

    st = StochRSIIndicator(close=out["close"], window=14, smooth1=3, smooth2=3)
    out["stoch_k"] = st.stochrsi_k()
    out["stoch_d"] = st.stochrsi_d()

    macd = MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=cfg.atr_len,
    ).average_true_range()
    out["atr"] = atr

    out["adx"] = ADXIndicator(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=cfg.adx_window,
    ).adx()

    return out

def stoch_cross(df_5m: pd.DataFrame, direction: str) -> bool:
    if len(df_5m) < 3:
        return False
    k = df_5m["stoch_k"].tail(3).values
    d = df_5m["stoch_d"].tail(3).values
    if np.isnan(k).any() or np.isnan(d).any():
        return False
    prev = k[-2] - d[-2]
    curr = k[-1] - d[-1]
    if direction == "BUY":
        return prev < 0 and curr > 0 and np.nanmin(k) < 0.25
    if direction == "SELL":
        return prev > 0 and curr < 0 and np.nanmax(k) > 0.75
    return False

def momentum_shift(df_5m: pd.DataFrame, direction: str) -> bool:
    if len(df_5m) < 5:
        return False
    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema21"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    hist_prev = float(df_5m["macd_hist"].iloc[-2])
    if direction == "BUY":
        return (c > ema) or (hist_prev < 0 and hist > 0)
    if direction == "SELL":
        return (c < ema) or (hist_prev > 0 and hist < 0)
    return False

def strong_buy_confirmation(df_5m: pd.DataFrame) -> bool:
    if len(df_5m) < 3:
        return False
    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema21"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    k = float(df_5m["stoch_k"].iloc[-1])
    d = float(df_5m["stoch_d"].iloc[-1])
    return (c > ema) and (hist > 0) and (k >= d)

def strong_sell_confirmation(df_5m: pd.DataFrame) -> bool:
    if len(df_5m) < 3:
        return False
    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema21"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    k = float(df_5m["stoch_k"].iloc[-1])
    d = float(df_5m["stoch_d"].iloc[-1])
    return (c < ema) and (hist < 0) and (k <= d)

def rejection_lite(df_5m: pd.DataFrame, direction: str) -> bool:
    if len(df_5m) < 1:
        return False
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

def momentum_breakout(df_5m: pd.DataFrame, level: float, direction: str) -> bool:
    if len(df_5m) < 10:
        return False

    closes = df_5m["close"].astype(float).tail(3).values
    ema = float(df_5m["ema21"].iloc[-1])
    macd_hist = float(df_5m["macd_hist"].iloc[-1])

    lookback_slice = df_5m.iloc[-7:-1]
    prev_high = float(lookback_slice["high"].max())
    prev_low = float(lookback_slice["low"].min())

    if direction == "BUY":
        breakout_ref = max(float(level), prev_high)
        return bool(closes[0] < closes[1] < closes[2] and closes[2] > breakout_ref and closes[2] > ema and macd_hist > 0)

    if direction == "SELL":
        breakout_ref = min(float(level), prev_low)
        return bool(closes[0] > closes[1] > closes[2] and closes[2] < breakout_ref and closes[2] < ema and macd_hist < 0)

    return False


# =========================================================
# SCORING / PLAN
# =========================================================

def score_setup(df_5m: pd.DataFrame, level_hit: float, direction: str, wick_info: Dict) -> Tuple[int, List[str], str]:
    score = 0
    reasons = []
    trigger = None

    if wick_info.get("upper_cluster") and direction == "SELL":
        score += 3
        reasons.append("Wick Rejection near Level")
        trigger = "Wick Rejection near Level"

    if wick_info.get("lower_cluster") and direction == "BUY":
        score += 3
        reasons.append("Wick Rejection near Level")
        trigger = "Wick Rejection near Level"

    br = break_retest(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction)
    st = stoch_cross(df_5m, direction)
    ms = momentum_shift(df_5m, direction)
    mb = momentum_breakout(df_5m, level_hit, direction)

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
    if mb:
        score += 2
        reasons.append("Momentum breakout")
        trigger = trigger or "Momentum Breakout"

    score = int(min(score, 6))
    if trigger is None:
        return 0, ["No trigger"], "None"
    return score, reasons, trigger

def pick_targets(levels: List[float], entry: float, direction: str) -> Tuple[Optional[float], Optional[float]]:
    if not levels:
        return None, None
    if direction == "BUY":
        above = sorted([lvl for lvl in levels if lvl > entry])
        return (above[0] if len(above) >= 1 else None, above[1] if len(above) >= 2 else None)
    if direction == "SELL":
        below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
        return (below[0] if len(below) >= 1 else None, below[1] if len(below) >= 2 else None)
    return None, None

def compute_trade_plan(
    df_5m: pd.DataFrame,
    levels: List[float],
    level_hit: float,
    direction: str,
    trigger: str,
    trade_type: str,
    cfg: Config,
) -> Dict:
    last = df_5m.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    buffer = max(price * 0.0002, 0.5)

    if trigger == "Wick Rejection near Level":
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer * 0.4))
            stop = float(min(last_low, level_hit) - buffer)
        else:
            entry = float(min(price, level_hit - buffer * 0.4))
            stop = float(max(last_high, level_hit) + buffer)
    elif trigger == "Momentum Breakout":
        if direction == "BUY":
            entry = float(max(price, level_hit + max(buffer, 1.0)))
            stop = float(min(last_low, level_hit) - buffer)
        else:
            entry = float(min(price, level_hit - max(buffer, 1.0)))
            stop = float(max(last_high, level_hit) + buffer)
    else:
        if direction == "BUY":
            entry = float(max(price, level_hit + buffer))
            stop = float(level_hit - (price * 0.0012) - buffer)
        else:
            entry = float(min(price, level_hit - buffer))
            stop = float(level_hit + (price * 0.0012) + buffer)

    raw_t1, raw_t2 = pick_targets(levels, entry, direction)
    risk = abs(entry - stop)
    if direction == "BUY":
        t1 = raw_t1 if raw_t1 is not None else entry + risk * cfg.tp1_rr
        t2 = raw_t2 if raw_t2 is not None else entry + risk * cfg.tp2_rr
        t3 = entry + max(risk * cfg.tp3_rr, atr * cfg.dynamic_t3_atr_mult)
    else:
        t1 = raw_t1 if raw_t1 is not None else entry - risk * cfg.tp1_rr
        t2 = raw_t2 if raw_t2 is not None else entry - risk * cfg.tp2_rr
        t3 = entry - max(risk * cfg.tp3_rr, atr * cfg.dynamic_t3_atr_mult)

    rr = (abs(t1 - entry) / risk) if risk > 0 else None

    return {
        "entry": float(entry),
        "stop": float(stop),
        "initial_stop": float(stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
        "rr": rr,
        "risk_pts": risk,
        "trade_type": trade_type,
    }

def classify_trade_strength(score: int, cfg: Config) -> str:
    if score >= cfg.strong_score_threshold:
        return "Strong"
    if score >= 4:
        return "Standard"
    return "Weak"

def position_size_units(capital: float, risk_pct: float, entry: float, stop: float) -> float:
    risk_cash = capital * (risk_pct / 100.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return risk_cash / risk_per_unit


# =========================================================
# BACKTEST
# =========================================================

def prepare_all_frames(cfg: Config):
    tv = make_tv_client(cfg)
    raw_5m = tv.get_hist(
        symbol=cfg.tv_symbol,
        exchange=cfg.tv_exchange,
        interval=Interval.in_5_minute,
        n_bars=cfg.bars_5m,
    )
    if raw_5m is None or raw_5m.empty:
        raise RuntimeError("TradingView returned empty 5m data.")
    df_5m = normalize_tv_df(raw_5m)
    df_5m = compute_indicators(df_5m, cfg)

    df_15m = compute_indicators(resample_ohlcv(df_5m[["open","high","low","close","volume"]], cfg.level_tf), cfg)
    df_1h = compute_indicators(resample_ohlcv(df_5m[["open","high","low","close","volume"]], cfg.bias_tf), cfg)
    df_4h = compute_indicators(resample_ohlcv(df_5m[["open","high","low","close","volume"]], cfg.trend_tf), cfg)

    return df_5m, df_15m, df_1h, df_4h

def closed_slice(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df.index < ts].copy()

def summarize_trades(trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if trades.empty:
        summary = pd.DataFrame([{
            "signals": 0,
            "triggered": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "winrate_pct": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "final_capital": cfg.initial_capital,
        }])
        return summary

    wins = (trades["r_result"] > 0).sum()
    losses = (trades["r_result"] < 0).sum()
    breakeven = (trades["r_result"].abs() < 1e-9).sum()
    total = len(trades)
    net_r = float(trades["r_result"].sum())
    avg_r = float(trades["r_result"].mean())

    gross_profit = float(trades.loc[trades["r_result"] > 0, "r_result"].sum())
    gross_loss = float(-trades.loc[trades["r_result"] < 0, "r_result"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    eq_curve = trades["equity_after"].astype(float)
    running_max = eq_curve.cummax()
    dd = eq_curve - running_max
    max_dd_cash = float(dd.min())
    risk_cash = cfg.initial_capital * (cfg.risk_per_trade_pct / 100.0)
    max_dd_r = max_dd_cash / risk_cash if risk_cash > 0 else 0.0

    summary = pd.DataFrame([{
        "signals": total,
        "triggered": total,
        "wins": int(wins),
        "losses": int(losses),
        "breakeven": int(breakeven),
        "winrate_pct": float((wins / total) * 100.0 if total else 0.0),
        "net_r": net_r,
        "avg_r": avg_r,
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999.0),
        "max_drawdown_r": max_dd_r,
        "final_capital": float(trades["equity_after"].iloc[-1]),
    }])
    return summary

def run_backtest(cfg: Config):
    df_5m, df_15m, df_1h, df_4h = prepare_all_frames(cfg)

    trades = []
    equity = cfg.initial_capital
    active_trade = None
    pending_signal = None

    start_idx = max(cfg.start_after_warmup_bars, 50)

    for i in range(start_idx, len(df_5m)):
        ts = df_5m.index[i]
        row = df_5m.iloc[i]

        # activate pending on current bar open
        if pending_signal is not None and active_trade is None:
            active_trade = pending_signal.copy()
            active_trade["entry_time"] = ts
            active_trade["entry_filled"] = float(row["open"]) + (cfg.slippage_pts if active_trade["direction"] == "BUY" else -cfg.slippage_pts)
            pending_signal = None

        # manage live trade
        if active_trade is not None:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            direction = active_trade["direction"]

            stop = active_trade["stop_live"]
            t1 = active_trade["t1"]
            t2 = active_trade["t2"]
            t3 = active_trade["t3"]
            entry = active_trade["entry_filled"]
            initial_stop = active_trade["initial_stop"]

            exit_price = None
            exit_reason = None

            if direction == "BUY":
                if low <= stop:
                    exit_price = stop - cfg.slippage_pts
                    exit_reason = "Stop"
                else:
                    if (not active_trade["tp1_hit"]) and high >= t1:
                        active_trade["tp1_hit"] = True
                        active_trade["stop_live"] = max(active_trade["stop_live"], entry)
                    if (not active_trade["tp2_hit"]) and high >= t2:
                        active_trade["tp2_hit"] = True
                        active_trade["stop_live"] = max(active_trade["stop_live"], t1)
                    if high >= t3:
                        exit_price = t3 - cfg.slippage_pts
                        exit_reason = "TP3"

            else:
                if high >= stop:
                    exit_price = stop + cfg.slippage_pts
                    exit_reason = "Stop"
                else:
                    if (not active_trade["tp1_hit"]) and low <= t1:
                        active_trade["tp1_hit"] = True
                        active_trade["stop_live"] = min(active_trade["stop_live"], entry)
                    if (not active_trade["tp2_hit"]) and low <= t2:
                        active_trade["tp2_hit"] = True
                        active_trade["stop_live"] = min(active_trade["stop_live"], t1)
                    if low <= t3:
                        exit_price = t3 + cfg.slippage_pts
                        exit_reason = "TP3"

            if exit_price is not None:
                risk_pts = abs(entry - initial_stop)
                pnl_pts = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
                r_result = pnl_pts / risk_pts if risk_pts > 0 else 0.0
                risk_cash = equity * (cfg.risk_per_trade_pct / 100.0)
                pnl_cash = (r_result * risk_cash) - cfg.commission_per_trade
                equity += pnl_cash

                trades.append({
                    "signal_time": active_trade["signal_time"],
                    "entry_time": active_trade["entry_time"],
                    "exit_time": ts,
                    "direction": direction,
                    "market_label": active_trade["market_label"],
                    "market_dir": active_trade["market_dir"],
                    "bias": active_trade["bias"],
                    "trigger": active_trade["trigger"],
                    "trade_type": active_trade["trade_type"],
                    "level_hit": active_trade["level_hit"],
                    "score": active_trade["score"],
                    "entry": entry,
                    "initial_stop": initial_stop,
                    "exit": exit_price,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "tp1_hit": active_trade["tp1_hit"],
                    "tp2_hit": active_trade["tp2_hit"],
                    "exit_reason": exit_reason,
                    "r_result": r_result,
                    "equity_after": equity,
                })
                active_trade = None
                continue

        # no new signal if trade exists or pending exists
        if active_trade is not None or pending_signal is not None:
            continue

        # use only closed data up to current timestamp
        hist_5m = closed_slice(df_5m.iloc[:i], ts)
        hist_15m = closed_slice(df_15m, ts)
        hist_1h = closed_slice(df_1h, ts)
        hist_4h = closed_slice(df_4h, ts)

        if min(len(hist_5m), len(hist_15m), len(hist_1h), len(hist_4h)) < 50:
            continue

        level_now = float(hist_5m["close"].iloc[-1])
        market_label, market_dir, _adx = compute_market_state(hist_1h, hist_4h, cfg)
        bias = structure_bias(hist_1h, hist_4h, cfg)
        key_levels = extract_key_levels(hist_15m, hist_1h, cfg)
        if not key_levels:
            continue

        level_hit, level_info = choose_best_level(hist_5m, level_now, key_levels, cfg)
        wick_info = level_info["wick_info"]

        # direction hierarchy
        if bias == "Bullish":
            direction = "BUY"
        elif bias == "Bearish":
            direction = "SELL"
        else:
            if momentum_breakout(hist_5m, level_hit, "BUY"):
                direction = "BUY"
            elif momentum_breakout(hist_5m, level_hit, "SELL"):
                direction = "SELL"
            else:
                direction = "BUY" if momentum_shift(hist_5m, "BUY") else "SELL"

        # block obvious conflict wick
        sell_confirm = (
            momentum_shift(hist_5m, "SELL")
            or stoch_cross(hist_5m, "SELL")
            or break_retest(hist_5m, level_hit, "SELL")
            or momentum_breakout(hist_5m, level_hit, "SELL")
        )
        buy_confirm = (
            momentum_shift(hist_5m, "BUY")
            or stoch_cross(hist_5m, "BUY")
            or break_retest(hist_5m, level_hit, "BUY")
            or momentum_breakout(hist_5m, level_hit, "BUY")
        )

        if direction == "BUY" and wick_info.get("upper_cluster") and sell_confirm:
            direction = "SELL"
        elif direction == "SELL" and wick_info.get("lower_cluster") and buy_confirm:
            direction = "BUY"

        score, reasons, trigger = score_setup(hist_5m, level_hit, direction, wick_info)
        if score < cfg.min_score:
            continue

        if cfg.require_momentum_confirmation:
            if direction == "BUY" and not strong_buy_confirmation(hist_5m):
                allow = trigger == "Wick Rejection near Level" and level_info["strong"] and wick_info["lower_hits"] >= 2
                if not allow:
                    continue
            if direction == "SELL" and not strong_sell_confirmation(hist_5m):
                allow = trigger == "Wick Rejection near Level" and level_info["strong"] and wick_info["upper_hits"] >= 2
                if not allow:
                    continue

        if cfg.block_counter_trend_in_trending_market and market_label == "Trending":
            if market_dir == "Bullish" and direction == "SELL":
                if not (cfg.allow_counter_trend_wick_if_strong and trigger == "Wick Rejection near Level" and level_info["strong"]):
                    continue
            if market_dir == "Bearish" and direction == "BUY":
                if not (cfg.allow_counter_trend_wick_if_strong and trigger == "Wick Rejection near Level" and level_info["strong"]):
                    continue

        trade_type = classify_trade_strength(score, cfg)
        if trade_type == "Weak" and not level_info["strong"]:
            continue

        plan = compute_trade_plan(
            df_5m=hist_5m,
            levels=key_levels,
            level_hit=level_hit,
            direction=direction,
            trigger=trigger,
            trade_type=trade_type,
            cfg=cfg,
        )

        rr = plan.get("rr")
        if rr is None or not np.isfinite(rr) or rr < cfg.min_rr_to_t1:
            continue

        # queue signal for next bar open
        pending_signal = {
            "signal_time": ts,
            "direction": direction,
            "market_label": market_label,
            "market_dir": market_dir,
            "bias": bias,
            "trigger": trigger,
            "trade_type": trade_type,
            "level_hit": level_hit,
            "score": score,
            "entry_planned": plan["entry"],
            "initial_stop": plan["initial_stop"],
            "stop_live": plan["initial_stop"],
            "t1": plan["t1"],
            "t2": plan["t2"],
            "t3": plan["t3"],
            "tp1_hit": False,
            "tp2_hit": False,
        }

    trades_df = pd.DataFrame(trades)
    summary_df = summarize_trades(trades_df, cfg)

    trades_df.to_csv(cfg.export_trades_csv, index=False)
    summary_df.to_csv(cfg.export_summary_csv, index=False)

    return trades_df, summary_df


def main():
    print("starting refined backtest...")
    trades_df, summary_df = run_backtest(CFG)

    print("\n------ BACKTEST SUMMARY ------")
    print(summary_df.to_string(index=False))

    print(f"\nSaved trades to: {CFG.export_trades_csv}")
    print(f"Saved summary to: {CFG.export_summary_csv}")

    if not trades_df.empty:
        print("\nLast 5 trades:")
        print(trades_df.tail(5).to_string(index=False))
    else:
        print("\nNo trades were generated.")


if __name__ == "__main__":
    main()
