import os
import time
import requests
import urllib.parse
import yfinance as yf
import pandas as pd

# ====== Railway Variables ======
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID in Railway Variables")

def send_telegram(message: str):
    encoded_msg = urllib.parse.quote(message)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={encoded_msg}"
    requests.get(url, timeout=20)

# ====== Strategy Settings ======
symbol = "^SPX"
short_ema = 20
long_ema = 50

last_signal = None
last_status_time = 0

def calculate_stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

while True:
    try:
        now = time.time()

        # ===== 15m data =====
        data_15 = yf.download(symbol, interval="15m", period="2d", progress=False)
        if data_15.empty:
            time.sleep(60)
            continue

        data_15["EMA20"] = data_15["Close"].ewm(span=short_ema, adjust=False).mean()
        data_15["EMA50"] = data_15["Close"].ewm(span=long_ema, adjust=False).mean()

        last_close = data_15["Close"].iloc[-1].item()
        ema20 = data_15["EMA20"].iloc[-1].item()
        ema50 = data_15["EMA50"].iloc[-1].item()

        # ===== 60m data for Stoch RSI =====
        data_60 = yf.download(symbol, interval="60m", period="5d", progress=False)
        if data_60.empty:
            time.sleep(60)
            continue

        stoch_rsi_series = calculate_stoch_rsi(data_60["Close"])
        last_stoch = stoch_rsi_series.iloc[-1].item()

        # ===== Signals + TP =====
        signal = None
        tp_price = None

        if ema20 > ema50 and last_stoch < 20:
            signal = "CALL"
            tp_price = last_close * 1.002
        elif ema20 < ema50 and last_stoch > 80:
            signal = "PUT"
            tp_price = last_close * 0.998

        # Send only on change
        if signal and signal != last_signal:
            send_telegram(f"✅ دكتور محمد: {signal} | Price: {last_close:.2f} | Take Profit: {tp_price:.2f}")
            last_signal = signal

        # Status message every 30 minutes
        if now - last_status_time >= 1800:
            send_telegram("⚡ دكتور محمد: النظام يعمل بشكل طبيعي ومستمر.")
            last_status_time = now

        time.sleep(60)

    except Exception as e:
        send_telegram(f"❌ دكتور محمد: خطأ - {str(e)}")
        time.sleep(60)
