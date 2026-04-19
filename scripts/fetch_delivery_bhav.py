"""
fetch_delivery_bhav.py
----------------------
Downloads NSE CM Bhav Copy (EOD), extracts delivery % for Nifty 50
constituents, and saves to deliveryData/.

Delivery % interpretation:
  > 60%  High conviction — institutions accumulating / distributing
  30-60% Normal activity
  < 30%  Speculative / intraday heavy — move may not sustain

Outputs:
  deliveryData/delivery_latest.csv   — today's Nifty50 constituent delivery %
  deliveryData/delivery_history.csv  — append-only, one row per stock per date

Run: daily after 19:00 IST (14:30 UTC) on weekdays.
"""

import os, io, csv, zipfile, warnings
import urllib.request
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR   = "deliveryData"
LATEST_FILE  = os.path.join(OUTPUT_DIR, "delivery_latest.csv")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "delivery_history.csv")

BHAV_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Nifty 50 constituents (as of Apr 2026)
NIFTY50_SYMBOLS = {
    "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS",
    "BHARTIARTL","SBIN","KOTAKBANK","LT","HINDUNILVR",
    "AXISBANK","ITC","BAJFINANCE","MARUTI","NTPC",
    "WIPRO","HCLTECH","POWERGRID","ULTRACEMCO","TITAN",
    "ADANIENT","ADANIPORTS","BAJAJFINSV","BPCL","BRITANNIA",
    "CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT",
    "GRASIM","HEROMOTOCO","HINDALCO","INDUSINDBK","JSWSTEEL",
    "M&M","NESTLEIND","ONGC","SBILIFE","SHRIRAMFIN",
    "SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TECHM",
    "TRENT","ULTRACEMCO","UPL","VEDL","ZOMATO",
}


def _ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")


def _fetch_bhav(trade_date: date) -> pd.DataFrame:
    url     = BHAV_URL_TEMPLATE.format(date=trade_date.strftime("%Y%m%d"))
    req     = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        fname = z.namelist()[0]
        df    = pd.read_csv(z.open(fname))
    return df


def _find_bhav(lookback: int = 7) -> tuple[pd.DataFrame, date]:
    """Try last `lookback` weekdays to find the latest Bhav copy."""
    today = date.today()
    for i in range(lookback):
        dt = today - timedelta(days=i + 1)
        if dt.weekday() >= 5:   # skip weekend
            continue
        try:
            df = _fetch_bhav(dt)
            print(f"  Bhav copy fetched for {dt}")
            return df, dt
        except Exception as e:
            print(f"  {dt}: {e}")
    raise RuntimeError("Could not fetch Bhav copy for last 7 days")


def parse_delivery(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Extract delivery % for EQ series Nifty50 stocks.
    NSE Bhav copy columns include DELIV_QTY and TOTTRDQTY.
    """
    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Keep EQ series only
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"].copy()

    # Identify columns
    sym_col  = next((c for c in df.columns if "SYMBOL" in c), None)
    vol_col  = next((c for c in df.columns if c in ("TOTTRDQTY", "TTL_TRD_QNTY")), None)
    del_col  = next((c for c in df.columns
                     if "DELIV_QTY" in c or "DELVQTY" in c or "DELIV" in c), None)
    dpct_col = next((c for c in df.columns
                     if "DELIV_PER" in c or "DELVPER" in c or "%DELT" in c), None)

    if sym_col is None or vol_col is None:
        print(f"  WARNING: Could not find required columns. Got: {list(df.columns[:12])}")
        return pd.DataFrame()

    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    records = []
    for _, row in df[df[sym_col].isin(NIFTY50_SYMBOLS)].iterrows():
        symbol = str(row[sym_col]).strip()
        vol    = float(row[vol_col]) if pd.notna(row.get(vol_col)) else 0.0

        if dpct_col and pd.notna(row.get(dpct_col)):
            del_pct = float(row[dpct_col])
            del_qty = del_pct / 100 * vol
        elif del_col and pd.notna(row.get(del_col)):
            del_qty = float(row[del_col])
            del_pct = (del_qty / vol * 100) if vol > 0 else 0.0
        else:
            del_qty = del_pct = 0.0

        close_col = next((c for c in df.columns
                          if c in ("CLOSE_PRICE", "CLOSEPRICE", "CLOSE")), None)
        close = float(row[close_col]) if close_col and pd.notna(row.get(close_col)) else None

        signal = ("HIGH_CONVICTION" if del_pct >= 60
                  else "LOW_CONVICTION" if del_pct < 30
                  else "NORMAL")

        records.append({
            "date":        str(trade_date),
            "symbol":      symbol,
            "close":       close,
            "volume":      int(vol),
            "delivery_qty":int(del_qty),
            "delivery_pct":round(del_pct, 2),
            "signal":      signal,
        })

    result = pd.DataFrame(records).sort_values("delivery_pct", ascending=False)
    return result.reset_index(drop=True)


def _index_weighted_signal(df: pd.DataFrame) -> str:
    """
    Weight delivery % by approximate Nifty 50 index weight of top 10 stocks.
    Simplified: just use average of top-5 by market cap proxy (Reliance, HDFC, ICICI, etc.)
    """
    top5 = {"RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS"}
    sub  = df[df["symbol"].isin(top5)]
    if sub.empty:
        return "UNKNOWN"
    avg_del = sub["delivery_pct"].mean()
    if avg_del >= 55:
        return f"BULLISH CONVICTION ({avg_del:.1f}% avg del in top-5)"
    elif avg_del < 30:
        return f"LOW CONVICTION ({avg_del:.1f}% avg del in top-5) — move may not sustain"
    return f"NEUTRAL ({avg_del:.1f}% avg del in top-5)"


def save(df: pd.DataFrame, trade_date: date):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Latest (overwrite)
    df.to_csv(LATEST_FILE, index=False)

    # History (append only, skip if date already present)
    if os.path.exists(HISTORY_FILE):
        existing = pd.read_csv(HISTORY_FILE)
        if str(trade_date) in existing["date"].astype(str).values:
            print(f"  {trade_date} already in history — skipping append.")
            return
        df.to_csv(HISTORY_FILE, mode="a", index=False, header=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

    print(f"  Saved {len(df)} rows for {trade_date}")


if __name__ == "__main__":
    print(f"[{_ist_now()}] Fetching NSE Bhav Copy for delivery data...")

    bhav_df, trade_date = _find_bhav()
    delivery_df = parse_delivery(bhav_df, trade_date)

    if delivery_df.empty:
        print("No delivery data parsed — exiting.")
        exit(0)

    print(f"\n  Top 10 by Delivery %:")
    print(delivery_df[["symbol","close","delivery_pct","signal"]].head(10).to_string(index=False))
    print(f"\n  Bottom 5 (most speculative):")
    print(delivery_df[["symbol","close","delivery_pct","signal"]].tail(5).to_string(index=False))
    print(f"\n  Index signal: {_index_weighted_signal(delivery_df)}")

    save(delivery_df, trade_date)
