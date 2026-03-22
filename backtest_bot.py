# -*- coding: utf-8 -*-
"""
SPX Backtest Bot — Yahoo Version
For: محمد

الفكرة:
- باك تست بدون TradingView login
- بيانات من Yahoo Finance
- يستخدم:
    * ^GSPC أثناء السوق الرسمي
    * ES=F كبديل عملي لبيانات 5m/15m/1h/4h
- منطق متوازن وأخف من النسخ السابقة
- مناسب لـ Railway و GitHub

تشغيل:
    pip install pandas numpy yfinance ta
    python spx_backtest_yahoo.py
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange


@dataclass
class Config:
    intraday_symbol: str = "ES=F"
    daily_symbol: str = "^GSPC"

    period_5m: str = "60d"
    interval_5m: str = "5m"

    use_rth_only: bool = False
    ny_session_start: str = "14:35"
    ny_session_end: str = "20:55"

    start_after_warmup_bars: int = 500
    cooldown_bars_after_exit: int = 4

    pivot_left: int = 3
    pivot_right: int = 3
    max_key_levels: int = 8
    level_cluster_tolerance_frac: float = 0.0010
    level_zone_near_pts: float = 22.0
    level_min_touch_count: int = 2
    level_strong_touch_count: int = 3

    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    adx_window: int = 14

    adx_trending_on: float = 22.0
    adx_range_on: float = 18.0
    min_entry_adx_5m: float = 14.0

    wick_cluster_lookback_5m: int = 10
    wick_ratio_strong: float = 0.45
    wick_cluster_min_hits: int = 2
    wick_near_level_tolerance_frac: float = 0.0010
    wick_min_abs_pts: float = 1.0

    min_score: int = 3
    strong_score_threshold: int = 5
    min_rr_to_t1: float = 1.15
    min_atr_points: float = 2.0
    min_body_ratio: float = 0.32

    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 0.75
    slippage_pts: float = 0.25
    commission_per_trade: float = 0.0

    tp1_rr: float = 1.00
    tp2_rr: float = 1.70
    tp3_rr: float = 2.50
    atr_stop_floor_mult: float = 1.10

    export_trades_csv: str = "trades_yahoo.csv"
    export_summary_csv: str = "backtest_results_yahoo.csv"


CFG = Config()


def fetch_yahoo(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty data for {symbol} {interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df.rename_axis("datetime")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return df[["open", "high", "low", "close", "volume"]].copy()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).copy()


def is_rth_bar(ts: pd.Timestamp, cfg: Config) -> bool:
    hhmm = ts.strftime("%H:%M")
    return cfg.ny_session_start <= hhmm <= cfg.ny_session_end


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

    out["atr"] = AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=cfg.atr_len
    ).average_true_range()

    out["adx"] = ADXIndicator(
        high=out["high"], low=out["low"], close=out["close"], window=cfg.adx_window
    ).adx()

    body = (out["close"] - out["open"]).abs()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_ratio"] = (body / rng).fillna(0)
    return out


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


def structure_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame, cfg: Config) -> str:
    def bias_from(df: pd.DataFrame):
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
    return "Weak"


def ema_trend_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
    r1 = df_1h.iloc[-1]
    r4 = df_4h.iloc[-1]

    long_ok = (
        r1["close"] > r1["ema200"] and r1["ema50"] > r1["ema200"]
        and r4["close"] > r4["ema200"] and r4["ema50"] > r4["ema200"]
    )
    short_ok = (
        r1["close"] < r1["ema200"] and r1["ema50"] < r1["ema200"]
        and r4["close"] < r4["ema200"] and r4["ema50"] < r4["ema200"]
    )

    if long_ok:
        return "Bullish"
    if short_ok:
        return "Bearish"
    return "Weak"


def compute_market_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame, cfg: Config):
    if len(df_1h) < cfg.adx_window + 5:
        return "Weak", "Neutral", None

    adx_val = float(
        ADXIndicator(
            high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], window=cfg.adx_window
        ).adx().iloc[-1]
    )

    sbias = structure_bias(df_1h, df_4h, cfg)
    ebias = ema_trend_bias(df_1h, df_4h)

    if adx_val >= cfg.adx_trending_on and sbias == ebias and sbias in ("Bullish", "Bearish"):
        return "Trending", sbias, adx_val
    if adx_val <= cfg.adx_range_on:
        return "Range", "Neutral", adx_val
    return "Messy", "Neutral", adx_val


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


def extract_key_levels(df_15m: pd.DataFrame, df_1h: pd.DataFrame, cfg: Config) -> List[float]:
    price = float(df_15m["close"].iloc[-1])
    hi_idx, lo_idx = find_pivots(df_1h["high"], cfg.pivot_left, cfg.pivot_right)
    swing_highs = [float(df_1h["high"].iloc[i]) for i in hi_idx][-14:] if hi_idx else []
    swing_lows = [float(df_1h["low"].iloc[i]) for i in lo_idx][-14:] if lo_idx else []
    recent = df_15m.tail(220)
    range_hi = float(recent["high"].max())
    range_lo = float(recent["low"].min())
    candidates = [x for x in swing_highs + swing_lows + [range_hi, range_lo] if np.isfinite(x)]
    merged = cluster_levels(candidates, cfg.level_cluster_tolerance_frac, price)
    merged = sorted(merged, key=lambda x: abs(x - price))[:max(cfg.max_key_levels * 4, 20)]
    merged = sorted(cluster_levels(merged, cfg.level_cluster_tolerance_frac, price))
    if len(merged) > cfg.max_key_levels:
        merged = sorted(merged, key=lambda x: abs(x - price))[:cfg.max_key_levels]
        merged = sorted(merged)
    return merged


def wick_cluster_near_level(df_5m: pd.DataFrame, level: float, cfg: Config) -> Dict:
    w = df_5m.tail(min(cfg.wick_cluster_lookback_5m, len(df_5m)))
    upper_hits = 0
    lower_hits = 0
    for _, r in w.iterrows():
        o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        rng = max(h - l, 1e-9)
        upper = h - max(o, c)
        lower = min(o, c) - l
        near = (
            abs(h - level)/max(level,1e-9) <= cfg.wick_near_level_tolerance_frac
            or abs(l - level)/max(level,1e-9) <= cfg.wick_near_level_tolerance_frac
            or abs(c - level)/max(level,1e-9) <= cfg.wick_near_level_tolerance_frac
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
        "upper_hits": upper_hits,
        "lower_hits": lower_hits,
    }


def count_level_touches(df_5m: pd.DataFrame, level: float, lookback: int = 50, abs_tol: float = 2.0) -> int:
    recent = df_5m.tail(min(lookback, len(df_5m)))
    touches = 0
    for _, r in recent.iterrows():
        h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        if abs(h - level) <= abs_tol or abs(l - level) <= abs_tol or abs(c - level) <= abs_tol:
            touches += 1
    return touches


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
        or wick_info["upper_cluster"]
        or wick_info["lower_cluster"]
    )
    quality_score = 0.0
    quality_score += max(0.0, 20.0 - distance_pts) * 0.16
    quality_score += float(wick_info["upper_hits"] + wick_info["lower_hits"]) * 1.8
    quality_score += float(touches) * 0.55
    return {
        "touches": touches,
        "distance_pts": distance_pts,
        "quality_score": quality_score,
        "strong": strong,
        "tradable": tradable,
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
    if len(df_5m) < 4:
        return False
    c = float(df_5m["close"].iloc[-1])
    ema = float(df_5m["ema21"].iloc[-1])
    hist = float(df_5m["macd_hist"].iloc[-1])
    hist_prev = float(df_5m["macd_hist"].iloc[-2])
    if direction == "BUY":
        return (c > ema and hist > hist_prev) or (hist_prev < 0 < hist)
    if direction == "SELL":
        return (c < ema and hist < hist_prev) or (hist_prev > 0 > hist)
    return False


def strong_buy_confirmation(df_5m: pd.DataFrame, cfg: Config) -> bool:
    r = df_5m.iloc[-1]
    return (
        float(r["close"]) > float(r["ema21"])
        and float(r["ema9"]) > float(r["ema21"])
        and float(r["macd_hist"]) > 0
        and float(r["rsi"]) >= 50
        and float(r["body_ratio"]) >= cfg.min_body_ratio
        and float(r["adx"]) >= cfg.min_entry_adx_5m
    )


def strong_sell_confirmation(df_5m: pd.DataFrame, cfg: Config) -> bool:
    r = df_5m.iloc[-1]
    return (
        float(r["close"]) < float(r["ema21"])
        and float(r["ema9"]) < float(r["ema21"])
        and float(r["macd_hist"]) < 0
        and float(r["rsi"]) <= 50
        and float(r["body_ratio"]) >= cfg.min_body_ratio
        and float(r["adx"]) >= cfg.min_entry_adx_5m
    )


def rejection_lite(df_5m: pd.DataFrame, direction: str, cfg: Config) -> bool:
    c = df_5m.iloc[-1]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    rng = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    if direction == "BUY":
        return lower / rng >= 0.22 and (abs(cl - o) / rng) >= cfg.min_body_ratio * 0.8
    if direction == "SELL":
        return upper / rng >= 0.22 and (abs(cl - o) / rng) >= cfg.min_body_ratio * 0.8
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
        return (window["close"] > level).sum() >= 2 and (abs(last_low - level) <= 2.5 or abs(prev_close - level) <= 2.5) and last_close > level
    if direction == "SELL":
        return (window["close"] < level).sum() >= 2 and (abs(last_high - level) <= 2.5 or abs(prev_close - level) <= 2.5) and last_close < level
    return False


def score_setup(df_5m: pd.DataFrame, level_hit: float, direction: str, wick_info: Dict, cfg: Config):
    score = 0
    reasons = []
    trigger = None

    if direction == "BUY" and wick_info.get("lower_cluster"):
        score += 3
        reasons.append("Wick rejection")
        trigger = "Wick Rejection"
    if direction == "SELL" and wick_info.get("upper_cluster"):
        score += 3
        reasons.append("Wick rejection")
        trigger = "Wick Rejection"

    br = break_retest(df_5m, level_hit, direction)
    rj = rejection_lite(df_5m, direction, cfg)
    st = stoch_cross(df_5m, direction)
    ms = momentum_shift(df_5m, direction)

    if br:
        score += 2
        reasons.append("Break retest")
        trigger = trigger or "Break Retest"
    if rj:
        score += 2
        reasons.append("Rejection candle")
        trigger = trigger or "Rejection"
    if st:
        score += 1
        reasons.append("Stoch cross")
    if ms:
        score += 1
        reasons.append("Momentum shift")

    return int(min(score, 6)), reasons, trigger


def pick_targets(levels: List[float], entry: float, direction: str):
    if direction == "BUY":
        above = sorted([lvl for lvl in levels if lvl > entry])
        return above[0] if len(above) >= 1 else None, above[1] if len(above) >= 2 else None
    below = sorted([lvl for lvl in levels if lvl < entry], reverse=True)
    return below[0] if len(below) >= 1 else None, below[1] if len(below) >= 2 else None


def compute_trade_plan(df_5m: pd.DataFrame, levels: List[float], level_hit: float, direction: str, trigger: str, cfg: Config):
    last = df_5m.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    buffer = max(price * 0.0002, 0.5)

    if direction == "BUY":
        entry = max(price, level_hit + buffer * (0.4 if trigger == "Wick Rejection" else 1.0))
        raw_stop = min(last_low, level_hit) - buffer
        atr_stop = entry - (atr * cfg.atr_stop_floor_mult)
        stop = min(raw_stop, atr_stop)
    else:
        entry = min(price, level_hit - buffer * (0.4 if trigger == "Wick Rejection" else 1.0))
        raw_stop = max(last_high, level_hit) + buffer
        atr_stop = entry + (atr * cfg.atr_stop_floor_mult)
        stop = max(raw_stop, atr_stop)

    risk = abs(entry - stop)
    raw_t1, raw_t2 = pick_targets(levels, entry, direction)
    if direction == "BUY":
        t1 = raw_t1 if raw_t1 is not None else entry + risk * cfg.tp1_rr
        t2 = raw_t2 if raw_t2 is not None else entry + risk * cfg.tp2_rr
        t3 = entry + risk * cfg.tp3_rr
    else:
        t1 = raw_t1 if raw_t1 is not None else entry - risk * cfg.tp1_rr
        t2 = raw_t2 if raw_t2 is not None else entry - risk * cfg.tp2_rr
        t3 = entry - risk * cfg.tp3_rr

    rr = abs(t1 - entry) / risk if risk > 0 else None
    return {
        "entry": float(entry),
        "initial_stop": float(stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
        "rr": rr,
    }


def closed_slice(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df.index < ts].copy()


def summarize_trades(trades_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame([{
            "signals": 0, "triggered": 0, "wins": 0, "losses": 0, "breakeven": 0,
            "winrate_pct": 0.0, "net_r": 0.0, "avg_r": 0.0, "profit_factor": 0.0,
            "max_drawdown_r": 0.0, "final_capital": cfg.initial_capital
        }])

    total = len(trades_df)
    wins = int((trades_df["r_result"] > 0).sum())
    losses = int((trades_df["r_result"] < 0).sum())
    breakeven = int((trades_df["r_result"].abs() < 1e-9).sum())

    gross_profit = float(trades_df.loc[trades_df["r_result"] > 0, "r_result"].sum())
    gross_loss = float(-trades_df.loc[trades_df["r_result"] < 0, "r_result"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    eq = trades_df["equity_after"].astype(float)
    run_max = eq.cummax()
    dd_cash = eq - run_max
    risk_cash_base = cfg.initial_capital * (cfg.risk_per_trade_pct / 100.0)
    max_dd_r = float(dd_cash.min() / risk_cash_base) if risk_cash_base > 0 else 0.0

    return pd.DataFrame([{
        "signals": total,
        "triggered": total,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "winrate_pct": float(wins / total * 100.0),
        "net_r": float(trades_df["r_result"].sum()),
        "avg_r": float(trades_df["r_result"].mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": max_dd_r,
        "final_capital": float(trades_df["equity_after"].iloc[-1]),
    }])


def run_backtest(cfg: Config):
    df_5m = fetch_yahoo(cfg.intraday_symbol, cfg.period_5m, cfg.interval_5m)
    df_5m = compute_indicators(df_5m, cfg)

    base = df_5m[["open", "high", "low", "close", "volume"]]
    df_15m = compute_indicators(resample_ohlcv(base, "15min"), cfg)
    df_1h = compute_indicators(resample_ohlcv(base, "1h"), cfg)
    df_4h = compute_indicators(resample_ohlcv(base, "4h"), cfg)

    trades = []
    equity = cfg.initial_capital
    active_trade = None
    pending_signal = None
    cooldown_until_idx = -1

    start_idx = max(cfg.start_after_warmup_bars, 50)

    for i in range(start_idx, len(df_5m)):
        ts = df_5m.index[i]
        row = df_5m.iloc[i]

        if cfg.use_rth_only and not is_rth_bar(ts, cfg):
            continue

        if pending_signal is not None and active_trade is None:
            active_trade = pending_signal.copy()
            active_trade["entry_time"] = ts
            active_trade["entry_filled"] = float(row["open"]) + (cfg.slippage_pts if active_trade["direction"] == "BUY" else -cfg.slippage_pts)
            active_trade["stop_live"] = active_trade["initial_stop"]
            pending_signal = None

        if active_trade is not None:
            high = float(row["high"])
            low = float(row["low"])
            direction = active_trade["direction"]

            stop = float(active_trade["stop_live"])
            entry = float(active_trade["entry_filled"])
            initial_stop = float(active_trade["initial_stop"])
            t1 = float(active_trade["t1"])
            t2 = float(active_trade["t2"])
            t3 = float(active_trade["t3"])

            exit_price = None
            exit_reason = None

            if direction == "BUY":
                if low <= stop:
                    exit_price = stop - cfg.slippage_pts
                    exit_reason = "Stop"
                else:
                    if not active_trade["tp1_hit"] and high >= t1:
                        active_trade["tp1_hit"] = True
                        active_trade["stop_live"] = max(active_trade["stop_live"], entry)
                    if not active_trade["tp2_hit"] and high >= t2:
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
                    if not active_trade["tp1_hit"] and low <= t1:
                        active_trade["tp1_hit"] = True
                        active_trade["stop_live"] = min(active_trade["stop_live"], entry)
                    if not active_trade["tp2_hit"] and low <= t2:
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
                cooldown_until_idx = i + cfg.cooldown_bars_after_exit
                continue

        if active_trade is not None or pending_signal is not None or i <= cooldown_until_idx:
            continue

        hist_5m = closed_slice(df_5m.iloc[:i], ts)
        hist_15m = closed_slice(df_15m, ts)
        hist_1h = closed_slice(df_1h, ts)
        hist_4h = closed_slice(df_4h, ts)

        if min(len(hist_5m), len(hist_15m), len(hist_1h), len(hist_4h)) < 80:
            continue

        level_now = float(hist_5m["close"].iloc[-1])
        atr_now = float(hist_5m["atr"].iloc[-1])
        if not np.isfinite(atr_now) or atr_now < cfg.min_atr_points:
            continue

        market_label, market_dir, _adx = compute_market_state(hist_1h, hist_4h, cfg)
        if market_label == "Messy":
            continue

        sbias = structure_bias(hist_1h, hist_4h, cfg)
        ebias = ema_trend_bias(hist_1h, hist_4h)
        if sbias != ebias or sbias not in ("Bullish", "Bearish"):
            continue

        direction = "BUY" if sbias == "Bullish" else "SELL"

        key_levels = extract_key_levels(hist_15m, hist_1h, cfg)
        if not key_levels:
            continue

        level_hit, level_info = choose_best_level(hist_5m, level_now, key_levels, cfg)
        if not level_info["tradable"]:
            continue
        if level_info["touches"] < cfg.level_min_touch_count:
            continue
        if abs(level_hit - level_now) > cfg.level_zone_near_pts:
            continue

        wick_info = level_info["wick_info"]
        score, reasons, trigger = score_setup(hist_5m, level_hit, direction, wick_info, cfg)
        if score < cfg.min_score or trigger is None:
            continue

        momentum_ok = momentum_shift(hist_5m, direction)
        if direction == "BUY":
            if not (momentum_ok or strong_buy_confirmation(hist_5m, cfg) or stoch_cross(hist_5m, direction)):
                continue
        else:
            if not (momentum_ok or strong_sell_confirmation(hist_5m, cfg) or stoch_cross(hist_5m, direction)):
                continue

        plan = compute_trade_plan(hist_5m, key_levels, level_hit, direction, trigger, cfg)
        rr = plan["rr"]
        if rr is None or not np.isfinite(rr) or rr < cfg.min_rr_to_t1:
            continue

        trade_type = "Strong" if score >= cfg.strong_score_threshold else "Standard"

        pending_signal = {
            "signal_time": ts,
            "direction": direction,
            "market_label": market_label,
            "market_dir": market_dir,
            "bias": sbias,
            "trigger": trigger,
            "trade_type": trade_type,
            "level_hit": level_hit,
            "score": score,
            "initial_stop": plan["initial_stop"],
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
    print("starting Yahoo backtest...")
    trades_df, summary_df = run_backtest(CFG)

    print("\n------ BACKTEST SUMMARY YAHOO ------")
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
