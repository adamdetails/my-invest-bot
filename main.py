import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import os
from dotenv import load_dotenv
from tradingview_screener import Query, col

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
WATCHLIST = [s.strip() for s in os.getenv("WATCHLIST", "AAPL,MSFT,SPY").split(",") if s.strip()]

def fetch_data(symbols, period="1y"):
    close = pd.DataFrame()
    for sym in symbols:
        df = yf.download(sym, period=period, progress=False)
        if not df.empty:
            close[sym] = df["Close"]
    return close.dropna()

def max_sharpe(returns):
    n = len(returns.columns)
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    excess = mu
    try:
        w = np.linalg.inv(cov.values) @ excess.values
        w = np.clip(w, 0, None)
        w = w / w.sum()
        return pd.Series(w, index=returns.columns)
    except:
        return pd.Series(1/n, index=returns.columns)

def get_early_movers():
    try:
        result = (Query()
            .select('ticker', 'name', 'close', 'close|1', 'relative_volume_10d_calc', 'market_cap_basic')
            .where(
                col('market_cap_basic').between(100_000_000, 5_000_000_000),
                col('relative_volume_10d_calc') > 2,
                col('close') > 1
            )
            .order_by('relative_volume_10d_calc', ascending=False)
            .limit(5)
            .get_scanner_data())
        return result
    except Exception as e:
        return None

def build_message(prices, weights):
    latest = prices.iloc[-1]
    prev   = prices.iloc[-2]
    lines = ["📊 *Daily Digest — Suggestions*", ""]

    for sym in WATCHLIST:
        if sym not in prices.columns:
            continue
        price = prices[sym].iloc[-1]
        ma200 = prices[sym].rolling(200).mean().iloc[-1]
        rsi   = ta.momentum.RSIIndicator(prices[sym], window=14).rsi().iloc[-1]
        chg   = (price / prices[sym].iloc[-2] - 1) * 100
        alloc = weights.get(sym, 0)

                if price < ma200 and rsi < 30:
            action = "🟢 OVERWEIGHT — oversold, below trend"
        elif price > ma200 and rsi > 70:
            action = "🔴 UNDERWEIGHT — overbought, stretched"
        elif price > ma200 and rsi < 50:
            action = "🟡 ADD — uptrend, not yet overbought"
        elif price < ma200 and rsi > 50:
            action = "🟠 TRIM — downtrend, still elevated"
        else:
            action = "⚪ HOLD — neutral"   
