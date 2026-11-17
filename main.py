import yfinance as yf
import pandas as pd
import numpy as np
import riskfolio as rp
import ta
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
WATCHLIST = os.getenv("WATCHLIST", "AAPL,MSFT,SPY").split(",")

def fetch_data(symbols, period="1y"):
    data = yf.download(symbols, period=period, progress=False)["Close"]
    return data

def compute_indicators(df):
    out = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        out[f"{col}_price"] = df[col]
        out[f"{col}_ma200"] = df[col].rolling(200).mean()
        out[f"{col}_rsi"]   = ta.momentum.RSIIndicator(df[col], window=14).rsi()
    return out

def optimize_portfolio(returns):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    w = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=0, hist=True)
    return w.iloc[:, 0]

def build_message(ind, weights):
    latest = ind.iloc[-1]
    prev   = ind.iloc[-2]
    lines = ["📊 *Daily Investment Digest*", ""]

    for sym in WATCHLIST:
        price = latest[f"{sym}_price"]
        ma200 = latest[f"{sym}_ma200"]
        rsi   = latest[f"{sym}_rsi"]
        chg   = (price / prev[f"{sym}_price"] - 1) * 100

        signal = "🟢" if price < ma200 and rsi < 30 else "⚪"
        if price > ma200 and rsi > 70:
            signal = "🔴"

        lines.append(
            f"{signal} {sym}: ${price:.2f} ({chg:+.1f}%)\n"
            f"    MA200: ${ma200:.2f} | RSI: {rsi:.0f}"
        )

    lines += ["", "*Suggested Allocation (Risk Parity):*"]
    for sym, w in weights.items():
        lines.append(f"  {sym}: {w:.1%}")

    lines += ["", "_Advisory only — not financial advice._"]
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
    ind     = compute_indicators(prices)
    weights = optimize_portfolio(returns)
    msg     = build_message(ind, weights)
    send_telegram(msg)
    print("Digest sent.")

if __name__ == "__main__":
    main()   