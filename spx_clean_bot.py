import os
import time
import requests
import urllib.parse
import yfinance as yf
import pandas as pd

# ====== Railway Variables ======
TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID in Railway Variables")

def send_telegram(message: str):
    encoded_msg = urllib.parse.quote(message)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={encoded_msg}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

# ====== Strategy Settings ======
symbol = "^SPX"
short_ema = 20
long_ema = 50

POLL_SECONDS = 300          # كل 5 دقائق
STATUS_INTERVAL = 1800      # كل 30 دقيقة

last_signal = None
last_status_time = 0.0

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix yfinance MultiIndex columns if they appear."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14) -> pd.Series:
    rsi = rsi_wilder(close, rsi_len)
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    return 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

# رسالة تشغيل
send_telegram("⚡ دكتور محمد: البوت اشتغل على Railway ✅")

while True:
    try:
        now = time.time()

        # ===== رسالة حالة كل 30 دقيقة =====
        if now - last_status_time >= STATUS_INTERVAL:
            send_telegram("⚡ النظام ماشي زي الحلاوة يا دكتور ✅")
            last_status_time = now

        # ===== بيانات 15 دقيقة (EMA) =====
        data_15 = yf.download(symbol, interval="15m", period="2d", progress=False)
        data_15 = _normalize_columns(data_15)

        if data_15 is None or data_15.empty or "Close" not in data_15.columns:
            time.sleep(POLL_SECONDS)
            continue

        data_15 = data_15.dropna(subset=["Close"])
        data_15["EMA20"] = data_15["Close"].ewm(span=short_ema, adjust=False).mean()
        data_15["EMA50"] = data_15["Close"].ewm(span=long_ema, adjust=False).mean()

        last_close = data_15["Close"].iloc[-1].item()
        ema20 = data_15["EMA20"].iloc[-1].item()
        ema50 = data_15["EMA50"].iloc[-1].item()

        # ===== بيانات 1 ساعة (Stoch RSI الحقيقي) =====
        data_60 = yf.download(symbol, interval="60m", period="10d", progress=False)
        data_60 = _normalize_columns(data_60)

        if data_60 is None or data_60.empty or "Close" not in data_60.columns:
            time.sleep(POLL_SECONDS)
            continue

        data_60 = data_60.dropna(subset=["Close"])

        srsi = stoch_rsi(data_60["Close"], rsi_len=14, stoch_len=14).dropna()
        if srsi.empty:
            time.sleep(POLL_SECONDS)
            continue

        last_srsi = srsi.iloc[-1].item()

        # ===== إشارات + TP =====
        signal = None
        tp_price = None

        if ema20 > ema50 and last_srsi < 20:
            signal = "CALL"
            tp_price = last_close * 1.002
        elif ema20 < ema50 and last_srsi > 80:
            signal = "PUT"
            tp_price = last_close * 0.998

        # ===== إرسال فقط عند تغير الإشارة =====
        if signal and signal != last_signal:
            send_telegram(
                f"✅ دكتور محمد: {signal} | Price: {last_close:.2f} | "
                f"StochRSI(1H): {last_srsi:.1f} | Take Profit: {tp_price:.2f}"
            )
            last_signal = signal

        time.sleep(POLL_SECONDS)

    except Exception as e:
        send_telegram(f"❌ دكتور محمد: خطأ - {str(e)}")
        time.sleep(POLL_SECONDS)
