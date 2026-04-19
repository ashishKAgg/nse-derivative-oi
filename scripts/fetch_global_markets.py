"""
fetch_global_markets.py
Fetches global market snapshot using yfinance (15-min delayed, free).

Tickers fetched:
  ^GSPC    S&P 500
  ^DJI     Dow Jones
  ^IXIC    Nasdaq Composite
  ^N225    Nikkei 225
  ^HSI     Hang Seng
  ^FTSE    FTSE 100
  BZ=F     Brent Crude Oil
  USDINR=X USD / INR
  ^VIX     CBOE VIX (US fear gauge)
  ^NSEI    Nifty 50 spot (pre-open reference / after-hours)
  DX-Y.NYB Dollar Index (DXY)  -- FII flow proxy, strong Nifty correlation
  ^TNX     US 10-Year Treasury Yield -- rising yield = FII outflows from India

Output:
  globalCues/global_latest.csv   — single-row snapshot (overwritten each run)
  globalCues/global_history.csv  — append-only history (one row per run)

NOTE: yfinance gives ~15-min delayed data for free.
      For truly live data you would need a paid market data feed.
      This is sufficient for pre-market gap analysis (run at 08:30 IST).
"""

import yfinance as yf
import pandas as pd
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

LATEST_FILE  = "globalCues/global_latest.csv"
HISTORY_FILE = "globalCues/global_history.csv"

TICKERS = {
    "sp500":       "^GSPC",
    "dow":         "^DJI",
    "nasdaq":      "^IXIC",
    "nikkei":      "^N225",
    "hang_seng":   "^HSI",
    "ftse":        "^FTSE",
    "crude_brent": "BZ=F",
    "usd_inr":     "USDINR=X",
    "us_vix":      "^VIX",
    # Tier-1 additions
    "nifty_spot":  "^NSEI",      # Nifty 50 last known close / pre-open ref
    "dxy":         "DX-Y.NYB",   # Dollar Index — inverse FII flow proxy
    "us_10y_yield":"^TNX",        # US 10-year yield — rising = FII outflows
}


def get_ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def fetch_snapshot() -> dict:
    """Download latest 2-day 1-min data and take the last available bar."""
    ist_now = get_ist_now()
    row = {"fetched_at": ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}

    # Download all tickers in one call to minimise requests
    symbols = list(TICKERS.values())
    data = yf.download(
        tickers=symbols,
        period="2d",
        interval="1m",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    for col_name, ticker in TICKERS.items():
        try:
            if len(symbols) == 1:
                close_series = data["Close"]
            else:
                close_series = data[ticker]["Close"]

            close_series = close_series.dropna()
            if close_series.empty:
                row[col_name] = None
                row[f"{col_name}_time"] = None
                continue

            last_val  = float(close_series.iloc[-1])
            prev_val  = float(close_series.iloc[-2]) if len(close_series) > 1 else last_val
            chg_pct   = (last_val - prev_val) / prev_val * 100 if prev_val else 0.0
            last_time = close_series.index[-1]

            row[col_name]             = round(last_val, 4)
            row[f"{col_name}_chg_pct"]= round(chg_pct, 3)
            row[f"{col_name}_time"]   = str(last_time)
        except Exception as e:
            print(f"  Warning: could not fetch {ticker} ({col_name}): {e}")
            row[col_name] = None
            row[f"{col_name}_chg_pct"] = None
            row[f"{col_name}_time"]    = None

    return row


def save(row: dict):
    os.makedirs("globalCues", exist_ok=True)
    df_new = pd.DataFrame([row])

    # Overwrite latest snapshot
    df_new.to_csv(LATEST_FILE, index=False)

    # Append to history
    if os.path.exists(HISTORY_FILE):
        df_new.to_csv(HISTORY_FILE, mode="a", index=False, header=False)
    else:
        df_new.to_csv(HISTORY_FILE, index=False)

    print(f"Saved snapshot at {row['fetched_at']}")


def print_summary(row: dict):
    print(f"\n{'─'*52}")
    print(f"  Global Market Snapshot  —  {row['fetched_at']}")
    print(f"{'─'*52}")
    labels = {
        "sp500":        "S&P 500      ",
        "dow":          "Dow Jones    ",
        "nasdaq":       "Nasdaq       ",
        "nikkei":       "Nikkei       ",
        "hang_seng":    "Hang Seng    ",
        "ftse":         "FTSE 100     ",
        "crude_brent":  "Brent ($/bbl)",
        "usd_inr":      "USD/INR      ",
        "us_vix":       "US VIX       ",
        "nifty_spot":   "Nifty Spot   ",
        "dxy":          "DXY          ",
        "us_10y_yield": "US 10Y Yield ",
    }
    for key, label in labels.items():
        val = row.get(key)
        chg = row.get(f"{key}_chg_pct")
        if val is not None:
            arrow = "▲" if (chg or 0) >= 0 else "▼"
            print(f"  {label}  {val:>12,.2f}   {arrow} {chg:+.2f}%")
        else:
            print(f"  {label}  {'N/A':>12}")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    print(f"Fetching global market snapshot…")
    snapshot = fetch_snapshot()
    print_summary(snapshot)
    save(snapshot)
