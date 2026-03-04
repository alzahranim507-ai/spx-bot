# -*- coding: utf-8 -*-
"""
ES Trading Bot — Hunter Smart + Smarter Liquidity Sweep + Trade Tracking + Hourly Trade Status + Daily Stats
(per Dr. Mohammed)

Data: Yahoo Finance (yfinance)
Symbol: ES=F always (Pre/Market/After)
Adjustment: -10 points always (match TradingView baseline)

Timeframes:
- 1H + 4H: context/structure + swings
- 15m: key levels + 24h highs/lows
- 5m: entries + confirmations + sweep detection + trade tracking

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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
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

    # Pivots (1H)
    pivot_left: int = 3
    pivot_right: int = 3

    # Key levels
    level_touch_tolerance: float = 0.0013       # 0.13%
    level_cluster_tolerance: float = 0.0010
    max_levels: int = 6

    # Hunter Smart
    signal_score_threshold: int = 3
    min_rr_to_t1: float = 1.6
    signal_cooldown_minutes: int = 20

    # Trade tracking
    progress_notify_frac: float = 0.50
    tracking_poll_seconds: int = 30

    # Smart stop confirmation (reduce false "stop hit")
    stop_confirm_seconds: int = 120
    stop_stoch_confirm: bool = True

    # Liquidity sweep (smarter)
    sweep_close_required: bool = True
    sweep_wick_min_points: float = 1.5
    sweep_stoch_confirm: bool = True
    sweep_priority: bool = True

    # Hourly update
    hourly_update: bool = True

    # Trade lifetime controls
    expire_at_rth_close: bool = True            # close trade at NY 16:00
    max_pending_minutes: int = 120              # cancel pending after X minutes
    max_live_hours: int = 8                     # safety: close live trade after X hours

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

def rth_close_dt_ny(ref: datetime | None = None) -> datetime:
    """RTH close = 16:00 New York time (same calendar day)."""
    t = ref or now_ny()
    return t.replace(hour=16, minute=0, second=0, microsecond=0)

def is_after_rth_close() -> bool:
    return now_ny() >= rth_close_dt_ny()

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
    df_1h  = apply_es_adjustment(_yf_download(sym, "60m", "90d"))

    agg = {"open":"first","high":"max","low":"min","close":"last"}
    df_4h = df_1h.resample("4h").agg(agg).dropna()
    return sym, df_4h, df_1h, df_15m, df_5m


# =========================
# Pivots / structure
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

    # Prev day hi/lo (Riyadh date)
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
    merged = cluster_levels(candidates, CFG.level_cluster_tolerance, price)

    # keep around current price + extremes
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
    """
    Sweep Up -> SELL idea: wick above level then close back below/at level
    Sweep Down -> BUY idea: wick below level then close back above/at level
    """
    c = df_5m.iloc[-1]
    h, l, cl = float(c["high"]), float(c["low"]), float(c["close"])
    min_wick = CFG.sweep_wick_min_points

    # Sweep Up
    if h >= level + min_wick:
        if (not CFG.sweep_close_required) or (cl <= level):
            return True, "SELL"

    # Sweep Down
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
# State (cooldown, active trade, daily stats)
# =========================

class BotState:
    def __init__(self):
        self.last_hour_sent = None
        self.last_signal_ts = {}
        self.active_trade = None

        self.last_day = now_riyadh().date()
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.daily_rr_sum = 0.0
        self.daily_rr_count = 0

    def can_signal(self, key: str) -> bool:
        last = self.last_signal_ts.get(key)
        if last is None:
            return True
        return (datetime.utcnow() - last) >= timedelta(minutes=CFG.signal_cooldown_minutes)

    def mark_signal(self, key: str):
        self.last_signal_ts[key] = datetime.utcnow()

    def roll_day_if_needed(self):
        today = now_riyadh().date()
        if today != self.last_day:
            self.send_daily_report(self.last_day)
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
# Trade status helpers
# =========================

def _fmt_level(x):
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:.1f}"
    return str(x)

def trade_details_line(tr: dict | None, price: float) -> tuple[str, str]:
    """
    Returns:
      status_line: "None" or "pending/live/..."
      details_line: short details
    """
    if not tr:
        return "None", "-"

    status = tr.get("status", "Unknown")
    direction = tr.get("direction", "?")
    entry = float(tr.get("entry", np.nan))
    stop = float(tr.get("stop", np.nan))
    t1 = tr.get("t1", None)
    t2 = tr.get("t2", None)

    # distances
    d_to_entry = (price - entry) if direction == "BUY" else (entry - price)
    d_to_t1 = None
    if t1 is not None and np.isfinite(entry):
        d_to_t1 = (t1 - price) if direction == "BUY" else (price - t1)

    details = (
        f"{direction} | E:{entry:.1f} SL:{stop:.1f} "
        f"T1:{_fmt_level(t1)} T2:{_fmt_level(t2)} | "
        f"ΔEntry:{d_to_entry:+.1f}"
    )
    if d_to_t1 is not None:
        details += f" | ΔT1:{d_to_t1:+.1f}"

    return status, details


# =========================
# Messages
# =========================

def hourly_msg(session: str, symbol: str, bias: str, levels: list[float], price: float) -> str:
    tr = STATE.active_trade
    status_line, details_line = trade_details_line(tr, price)

    extra = ""
    if CFG.expire_at_rth_close:
        extra = f"\n⏳ Trade auto-close: RTH 16:00 NY"

    return (
        f"👋 {CFG.user_title}\n"
        f"🕐 Hourly Update ({now_riyadh().strftime('%Y-%m-%d %H:%M')} Riyadh)\n"
        f"🏷️ Session: {session}\n"
        f"📌 Source: Yahoo | Symbol: {symbol}\n"
        f"🧭 Bias (1H/4H): <b>{bias}</b>\n"
        f"💵 Price: {price:.1f}\n"
        f"🧱 Key Levels: {fmt_levels(levels)}\n"
        f"📌 Active Trade: <b>{status_line}</b>\n"
        f"🧾 Trade: {details_line}{extra}\n"
        f"🧾 Note: ES adjusted -{CFG.es_points_adjust:.0f} pts (match TradingView)"
    )

def signal_msg(session: str, symbol: str, bias: str, direction: str, level_hit: float, score: int, reasons: list[str], plan: dict, mode: str) -> str:
    rr_txt = f"{plan['rr']:.2f}" if plan["rr"] is not None else "N/A"
    t1_txt = f"{plan['t1']:.1f}" if plan["t1"] is not None else "N/A"
    t2_txt = f"{plan['t2']:.1f}" if plan["t2"] is not None else "N/A"
    return (
        f"🚨 {CFG.user_title} — فرصة دخول ({mode})\n"
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


# =========================
# Scoring (trend = bonus only)
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
# Expiry rules (NEW)
# =========================

def expire_trade_if_needed():
    tr = STATE.active_trade
    if not tr:
        return

    # 1) Cancel long pending trades
    if tr.get("status") == "pending":
        created = tr.get("created_ts")
        if created and (datetime.utcnow() - created) >= timedelta(minutes=CFG.max_pending_minutes):
            send_telegram(
                f"⌛️ {CFG.user_title} — Trade Expired (Pending too long)\n"
                f"سبب: لم يتم تفعيل الدخول خلال {CFG.max_pending_minutes} دقيقة.\n"
                f"تم إلغاء الصفقة لتفادي تعليق البوت."
            )
            STATE.active_trade = None
            return

    # 2) Safety: cancel too-long live trades
    if tr.get("status") == "live":
        live_since = tr.get("live_since_ts")
        if live_since and (datetime.utcnow() - live_since) >= timedelta(hours=CFG.max_live_hours):
            send_telegram(
                f"🧯 {CFG.user_title} — Trade Closed (Time Limit)\n"
                f"سبب: تجاوزت الصفقة حد الزمن ({CFG.max_live_hours} ساعات).\n"
                f"تم الإغلاق لتفادي التعليق."
            )
            STATE.active_trade = None
            return

    # 3) Expire at RTH close (NY 16:00)
    if CFG.expire_at_rth_close and is_after_rth_close():
        send_telegram(
            f"🛎️ {CFG.user_title} — Trade Closed (Market Close)\n"
            f"سبب: إغلاق السوق (RTH 16:00 NY).\n"
            f"ملاحظة: تم إنهاء الصفقة لتفادي تعليق البوت."
        )
        STATE.active_trade = None
        return


# =========================
# Trade tracking
# =========================

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
            # recovered back inside -> likely sweep
            tr["stop_pending"] = False
            tr.pop("stop_first_breach_ts", None)
        else:
            dt = (datetime.utcnow() - tr["stop_first_breach_ts"]).total_seconds()
            if dt >= CFG.stop_confirm_seconds:
                stoch_ok = True
                if CFG.stop_stoch_confirm:
                    stoch_ok = stoch_cross(df_5m, "SELL" if direction == "BUY" else "BUY")
                if stoch_ok:
                    STATE.daily_trades += 1
                    STATE.daily_losses += 1
                    rr = tr.get("rr_to_t1")
                    if rr is not None:
                        STATE.daily_rr_sum += float(rr)
                        STATE.daily_rr_count += 1

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
            STATE.daily_trades += 1
            STATE.daily_wins += 1
            rr = tr.get("rr_to_t1")
            if rr is not None:
                STATE.daily_rr_sum += float(rr)
                STATE.daily_rr_count += 1

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
# Main evaluation
# =========================

def evaluate_once():
    STATE.roll_day_if_needed()

    session = session_label()
    symbol, df_4h, df_1h, df_15m, df_5m_raw = fetch_timeframes()
    df_5m = compute_indicators(df_5m_raw)

    price = float(df_5m["close"].iloc[-1])
    bias = structure_bias(df_1h, df_4h)
    key_levels = extract_key_levels(df_15m, df_1h)

    # Expire trade if needed (NEW)
    expire_trade_if_needed()

    # Track active trade (always)
    update_active_trade(df_5m, price)

    # Hourly update (includes trade status)
    if CFG.hourly_update:
        current_hour = now_riyadh().replace(minute=0, second=0, microsecond=0)
        if STATE.last_hour_sent is None or current_hour > STATE.last_hour_sent:
            send_telegram(hourly_msg(session, symbol, bias, key_levels, price))
            STATE.last_hour_sent = current_hour

    # One trade at a time
    if STATE.active_trade is not None:
        return

    # ---------- Sweep-first logic ----------
    sweep_levels = compute_sweep_levels(df_1h, df_15m, key_levels)
    sweep_hit = None
    sweep_dir = ""
    for lvl in sorted(sweep_levels, key=lambda x: abs(x - price))[:12]:
        is_sw, d = detect_sweep(df_5m, lvl)
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

        score, reasons, trigger = score_setup(df_5m, bias, level_hit, direction, is_sweep=True)
        if score >= CFG.signal_score_threshold:
            plan = compute_trade_plan(df_5m, key_levels, level_hit, direction, trigger="Sweep")
            if plan["rr"] is None or plan["rr"] >= CFG.min_rr_to_t1:
                key = f"SWEEP:{direction}:{round(level_hit,1)}"
                if STATE.can_signal(key):
                    mode = "Counter-Trend Scalp" if ((bias == "Bearish" and direction == "BUY") or (bias == "Bullish" and direction == "SELL")) else "Hunter Smart"
                    send_telegram(signal_msg(session, symbol, bias, direction, level_hit, score, reasons, plan, mode))
                    STATE.mark_signal(key)
                    STATE.active_trade = {
                        "direction": direction,
                        "level": float(level_hit),
                        "entry": float(plan["entry"]),
                        "stop": float(plan["stop"]),
                        "t1": None if plan["t1"] is None else float(plan["t1"]),
                        "t2": None if plan["t2"] is None else float(plan["t2"]),
                        "rr_to_t1": None if plan["rr"] is None else float(plan["rr"]),
                        "status": "pending",
                        "created_ts": datetime.utcnow(),
                    }
        return

    # ---------- Normal logic: must be near a key level ----------
    if not key_levels:
        return

    nearby = [lvl for lvl in key_levels if near_level(price, lvl, CFG.level_touch_tolerance)]
    if not nearby:
        return
    level_hit = float(min(nearby, key=lambda x: abs(x - price)))

    # Direction: prefer bias, else micro drift
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
    if plan["rr"] is not None and plan["rr"] < CFG.min_rr_to_t1:
        return

    key = f"NORM:{direction}:{round(level_hit,1)}"
    if not STATE.can_signal(key):
        return

    send_telegram(signal_msg(session, symbol, bias, direction, level_hit, score, reasons, plan, "Hunter Smart"))
    STATE.mark_signal(key)

    STATE.active_trade = {
        "direction": direction,
        "level": float(level_hit),
        "entry": float(plan["entry"]),
        "stop": float(plan["stop"]),
        "t1": None if plan["t1"] is None else float(plan["t1"]),
        "t2": None if plan["t2"] is None else float(plan["t2"]),
        "rr_to_t1": None if plan["rr"] is None else float(plan["rr"]),
        "status": "pending",
        "created_ts": datetime.utcnow(),
    }


def main():
    send_telegram(
        f"✅ {CFG.user_title} — Bot started\n"
        f"Rules:\n"
        f"- All Sessions: {CFG.es_symbol} (adjust -{CFG.es_points_adjust:.0f})\n"
        f"- Hourly Update: ON (includes Active Trade status)\n"
        f"- Signals: Hunter Smart (Score ≥ {CFG.signal_score_threshold}, RR ≥ {CFG.min_rr_to_t1})\n"
        f"- Sweep: Smarter (1H swings + 24h HL) + Stoch confirm\n"
        f"- Trade Tracking: ON (Entry/T1/T2/Stop confirm)\n"
        f"- Trade Expiry: Pending>{CFG.max_pending_minutes}m, RTH close auto-close\n"
        f"- Daily Stats: ON"
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
