import os
import time
import requests
import urllib.parse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

# ====== Railway Variables ======
TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID in Railway Variables")

def send_telegram(message: str):
    encoded = urllib.parse.quote(message)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={encoded}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

# ====== Settings ======
SPX_SYMBOL = "^SPX"
FUT_SYMBOL = "ES=F"

short_ema = 20
long_ema = 50

POLL_SECONDS = 300           # كل 5 دقائق
STATUS_INTERVAL = 21600      # كل 6 ساعات (6 × 60 × 60)

# ====== State ======
last_signal = None
last_status_time = 0.0

NY = ZoneInfo("America/New_York")

def is_market_open_now() -> bool:
    """
    سوق الأسهم الأمريكي (SPX cash):
    Mon–Fri 9:30am–4:00pm ET
    """
    now = datetime.now(NY)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = now.time()
    return dtime(9, 30) <= t <= dtime(16, 0)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def _ensure_numeric_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Close" not in df.columns:
        return df
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14) -> pd.Series:
    rsi = rsi_wilder(close, rsi_len)
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    return 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

def fetch_15m(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, interval="15m", period="2d", progress=False)
    df = _normalize_columns(df)
    df = _ensure_numeric_close(df)
    return df

def fetch_60m(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, interval="60m", period="10d", progress=False)
    df = _normalize_columns(df)
    df = _ensure_numeric_close(df)
    return df

# رسالة تشغيل
send_telegram("⚡ دكتور محمد: البوت اشتغل على Railway ✅")

while True:
    try:
        now_ts = time.time()

        # رسالة حالة كل 6 ساعات
        if now_ts - last_status_time >= STATUS_INTERVAL:
            send_telegram("⚡ النظام ماشي زي الحلاوة يا دكتور ✅")
            last_status_time = now_ts

        # اختيار المصدر: SPX أثناء السوق، ES خارج السوق
        market_open = is_market_open_now()
        symbol = SPX_SYMBOL if market_open else FUT_SYMBOL
        source_label = "SPX (Market Open)" if market_open else "ES Futures (Pre/After)"

        # ===== 15m (EMA) =====
        data_15 = fetch_15m(symbol)
        if data_15 is None or data_15.empty:
            time.sleep(POLL_SECONDS)
            continue

        data_15["EMA20"] = data_15["Close"].ewm(span=short_ema, adjust=False).mean()
        data_15["EMA50"] = data_15["Close"].ewm(span=long_ema, adjust=False).mean()

        last_close = data_15["Close"].iloc[-1].item()
        ema20 = data_15["EMA20"].iloc[-1].item()
        ema50 = data_15["EMA50"].iloc[-1].item()

        # ===== 60m (Stoch RSI) =====
        data_60 = fetch_60m(symbol)
        if data_60 is None or data_60.empty:
            time.sleep(POLL_SECONDS)
            continue

        srsi = stoch_rsi(data_60["Close"], rsi_len=14, stoch_len=14).dropna()
        if srsi.empty:
            time.sleep(POLL_SECONDS)
            continue

        last_srsi = srsi.iloc[-1].item()

        # ===== Signal =====
        signal = None
        tp_price = None

        if ema20 > ema50 and last_srsi < 20:
            signal = "CALL"
            tp_price = last_close * 1.002
        elif ema20 < ema50 and last_srsi > 80:
            signal = "PUT"
            tp_price = last_close * 0.998

        # إرسال فقط عند تغيّر الإشارة
        if signal and signal != last_signal:
            send_telegram(
                f"✅ دكتور محمد: {signal}\n"
                f"Source: {source_label}\n"
                f"Price: {last_close:.2f}\n"
                f"EMA20: {ema20:.2f} | EMA50: {ema50:.2f}\n"
                f"StochRSI(1H): {last_srsi:.1f}\n"
                f"Take Profit: {tp_price:.2f}"
            )
            last_signal = signal

        time.sleep(POLL_SECONDS)

    except Exception:
        # بدون سبام أخطاء — نكمّل بهدوء
        time.sleep(POLL_SECONDS)
