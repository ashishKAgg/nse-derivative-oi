"""
fetch_fii_dii.py
Fetches FII / DII daily cash + F&O activity from NSE and appends to
fiiDiiData/fii_dii_daily.csv

NSE publishes this after ~19:00 IST.  Run daily at 19:30 IST (14:00 UTC).

Columns saved:
  date, fii_cash_buy, fii_cash_sell, fii_cash_net,
  dii_cash_buy, dii_cash_sell, dii_cash_net,
  fii_fo_buy, fii_fo_sell, fii_fo_net
"""

import requests
import pandas as pd
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

OUTPUT_FILE = "fiiDiiData/fii_dii_daily.csv"
NSE_BASE    = "https://www.nseindia.com"

# NSE live FII/DII API (returns last ~30 days)
FII_DII_URL = f"{NSE_BASE}/api/fiidiiTradeReact"

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "referer": f"{NSE_BASE}/market-data/fii-dii-data",
    "accept":  "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
}


def get_ist_date():
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


def _safe_float(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return None


def fetch_fii_dii() -> pd.DataFrame:
    with requests.Session() as sess:
        sess.get(f"{NSE_BASE}/market-data/fii-dii-data",
                 headers=HEADERS, verify=False, timeout=15)
        resp = sess.get(FII_DII_URL, headers=HEADERS, verify=False, timeout=15)

    resp.raise_for_status()
    raw = resp.json()

    records = []
    for item in raw:
        # NSE returns two separate category entries per date: FII and DII
        cat  = str(item.get("category", "")).upper()
        date = item.get("date", "")

        row = {
            "date":     date,
            "category": cat,
            "buy":      _safe_float(item.get("buyValue")),
            "sell":     _safe_float(item.get("sellValue")),
            "net":      _safe_float(item.get("netValue")),
        }
        records.append(row)

    if not records:
        raise ValueError("Empty FII/DII response from NSE")

    raw_df = pd.DataFrame(records)
    raw_df["date"] = pd.to_datetime(raw_df["date"], format="%d-%b-%Y", errors="coerce")
    raw_df = raw_df.dropna(subset=["date"]).sort_values("date")

    # Pivot: one row per date with FII and DII columns
    fii = raw_df[raw_df["category"].str.contains("FII|FPI")].copy()
    dii = raw_df[raw_df["category"].str.contains("DII")].copy()

    fii = fii.rename(columns={"buy": "fii_cash_buy", "sell": "fii_cash_sell",
                               "net": "fii_cash_net"}).drop(columns=["category"])
    dii = dii.rename(columns={"buy": "dii_cash_buy", "sell": "dii_cash_sell",
                               "net": "dii_cash_net"}).drop(columns=["category"])

    df = pd.merge(fii, dii, on="date", how="outer").sort_values("date")
    df["combined_net"] = df["fii_cash_net"].fillna(0) + df["dii_cash_net"].fillna(0)
    return df.reset_index(drop=True)


def save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        last_date = existing["date"].max()
        df = df[df["date"] > last_date]
        if df.empty:
            print("No new FII/DII rows — already up to date.")
            return
        df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
        print(f"Appended {len(df)} new FII/DII rows (up to {df['date'].max().date()}).")
    else:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Created {OUTPUT_FILE} with {len(df)} rows.")


if __name__ == "__main__":
    print(f"[{get_ist_date()}] Fetching FII/DII data from NSE…")
    df = fetch_fii_dii()
    print(f"  Received {len(df)} rows, latest: {df['date'].max().date()}")
    print(df.tail(3).to_string(index=False))
    save(df)
