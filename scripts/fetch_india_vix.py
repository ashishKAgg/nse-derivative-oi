"""
fetch_india_vix.py
Fetches India VIX history from NSE and appends to vixData/india_vix.csv

NSE endpoint returns last 1-year of VIX data.
Run daily after 15:45 IST (10:15 UTC) on weekdays.
"""

import requests
import pandas as pd
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

OUTPUT_FILE = "vixData/india_vix.csv"
NSE_BASE    = "https://www.nseindia.com"
VIX_URL     = f"{NSE_BASE}/api/historical/vixhistory?data=1Y"

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "referer": f"{NSE_BASE}/market-data/india-vix",
    "accept":  "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
}


def get_ist_date():
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


def fetch_vix() -> pd.DataFrame:
    with requests.Session() as sess:
        # Warm up cookies — same pattern as nse_scraper.py
        sess.get(f"{NSE_BASE}/market-data/india-vix", headers=HEADERS, verify=False, timeout=15)
        resp = sess.get(VIX_URL, headers=HEADERS, verify=False, timeout=15)

    resp.raise_for_status()
    data = resp.json()

    rows = data.get("data", [])
    if not rows:
        raise ValueError("Empty data returned from NSE VIX endpoint")

    records = []
    for item in rows:
        records.append({
            "date":       item.get("EOD_TIMESTAMP", ""),
            "open":       item.get("EOD_OPEN_INDEX_VAL"),
            "high":       item.get("EOD_HIGH_INDEX_VAL"),
            "low":        item.get("EOD_LOW_INDEX_VAL"),
            "close":      item.get("EOD_CLOSE_INDEX_VAL"),
            "prev_close": item.get("EOD_PREV_CLOSE"),
            "change":     item.get("EOD_INDEX_CHANGE"),
            "change_pct": item.get("CHANGE_PERCENT"),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    for col in ["open", "high", "low", "close", "prev_close", "change", "change_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        last_date = existing["date"].max()
        df = df[df["date"] > last_date]
        if df.empty:
            print("No new VIX rows — already up to date.")
            return
        df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
        print(f"Appended {len(df)} new VIX rows (up to {df['date'].max().date()}).")
    else:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Created {OUTPUT_FILE} with {len(df)} rows.")


if __name__ == "__main__":
    print(f"[{get_ist_date()}] Fetching India VIX from NSE…")
    vix_df = fetch_vix()
    print(f"  Received {len(vix_df)} rows, latest: {vix_df['date'].max().date()}")
    save(vix_df)
