import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import os
from dotenv import load_dotenv

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
    excess = mu  # assume rf=0 for simplicity
    # Solve for max Sharpe: w = cov^-1 * excess
    try:
        w = np.linalg.inv(cov.values) @ excess.values
        w = np.clip(w, 0, None)
        w = w / w.sum()
        return pd.Series(w, index=returns.columns)
    except:
        return pd.Series(1/n, index=returns.columns)

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

        # Signal logic
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

        lines.append(
            f"{sym}: ${price:.2f} ({chg:+.1f}%)\n"
            f"  Suggested: {alloc:.1%} | {action}\n"
            f"  MA200: ${ma200:.2f} | RSI: {rsi:.0f}"
        )

    lines += ["", "*Model: Max Sharpe (1yr). Advisory only, not financial advice.*"]
    return "\n".join(lines)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    })

def main():
    prices  = fetch_data(WATCHLIST)
    returns = prices.pct_change().dropna()
    weights = max_sharpe(returns)
    msg     = build_message(prices, weights)
    send_telegram(msg)
    print("Digest sent.")

if __name__ == "__main__":
    main()   
