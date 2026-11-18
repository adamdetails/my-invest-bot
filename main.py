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
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
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
        return pd.Series(1 / n, index=returns.columns)


def get_early_movers():
    try:
        _, df = (
            Query()
            .select(
                "ticker", "name", "close", "close|1",
                "relative_volume_10d_calc", "market_cap_basic"
            )
            .where(
                col("market_cap_basic").between(100_000_000, 5_000_000_000),
                col("relative_volume_10d_calc") > 2,
                col("close") > 1
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(5)
            .get_scanner_data()
        )
        return df
    except:
        return None


def build_message(prices, weights):
    lines = ["\U0001F4CA *Daily Digest — Suggestions*", ""]
    for sym in WATCHLIST:
        if sym not in prices.columns:
            continue
        price = prices[sym].iloc[-1]
        ma200 = prices[sym].rolling(200).mean().iloc[-1]
        rsi = ta.momentum.RSIIndicator(prices[sym], window=14).rsi().iloc[-1]
        chg = (price / prices[sym].iloc[-2] - 1) * 100
        alloc = weights.get(sym, 0)
        if price < ma200 and rsi < 30:
            action = "\U0001F7E2 OVERWEIGHT — oversold, below trend"
        elif price > ma200 and rsi > 70:
            action = "\U0001F534 UNDERWEIGHT — overbought, stretched"
        elif price > ma200 and rsi < 50:
            action = "\U0001F7E1 ADD — uptrend, not yet overbought"
        elif price < ma200 and rsi > 50:
            action = "\U0001F7E0 TRIM — downtrend, still elevated"
        else:
            action = "\u26AA HOLD — neutral"
        lines.append(
            f"{sym}: ${price:.2f} ({chg:+.1f}%)\n"
            f"  Suggested: {alloc:.1%} | {action}\n"
            f"  MA200: ${ma200:.2f} | RSI: {rsi:.0f}"
        )
    movers = get_early_movers()
    if movers is not None and not movers.empty:
        lines += ["", "\U0001F525 *Early Movers (Unusual Volume)*", ""]
        for i, row in movers.iterrows():
            ticker = row["ticker"]
            name = row["name"]
            price = row["close"]
            prev_close = row["close|1"]
            if prev_close and prev_close > 0:
                chg = (price / prev_close - 1) * 100
            else:
                chg = 0
            rel_vol = row["relative_volume_10d_calc"]
            mcap = row["market_cap_basic"]
            if mcap >= 1_000_000_000:
                mcap_str = f"${mcap / 1_000_000_000:.1f}B"
            else:
                mcap_str = f"${mcap / 1_000_000:.0f}M"
            lines.append(
                f"{i + 1}. {ticker} ({name}): ${price:.2f} ({chg:+.1f}%)\n"
                f"   Vol {rel_vol:.1f}x avg | Mkt Cap: {mcap_str}"
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
    prices = fetch_data(WATCHLIST)
    returns = prices.pct_change().dropna()
    weights = max_sharpe(returns)
    msg = build_message(prices, weights)
    send_telegram(msg)
    print("Digest sent.")


if __name__ == "__main__":
    main()
