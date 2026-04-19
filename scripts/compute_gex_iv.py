"""
compute_gex_iv.py
-----------------
Computes TWO Tier-1 signals from the existing OI chain CSVs:

  1. Gamma Exposure (GEX) per strike
     GEX = Gamma × OI × Spot^2 × 0.01 × LotSize
     -- positive GEX at a strike = price-pinning (dealers buy dips, sell rallies)
     -- negative GEX = price-amplifying (dealers accelerate moves)
     Net GEX > 0  -> low realised vol, pinning
     Net GEX < 0  -> high realised vol, trending / explosive

  2. IV Percentile & IV Rank per ATM strike
     Uses pure-python Black-Scholes bisection (no numpy dependency issues).
     IV Rank   = (current IV - 52w low) / (52w high - 52w low) × 100
     IV %ile   = % of past sessions where IV was BELOW today's IV

Outputs (appended / overwritten each run):
  gexData/gex_latest.json      -- per-strike GEX snapshot + net GEX + flip point
  gexData/gex_history.csv      -- datetime, net_gex, gex_flip, spot
  ivData/iv_latest.json        -- ATM IV, IV rank, IV %ile, verdict
  ivData/iv_history.csv        -- datetime, atm_iv, iv_rank, iv_pctile, verdict

Run after every OI scrape (every 5 min during market hours).
"""

import os, json, csv, math, warnings
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

LOT_SIZE    = 65       # Nifty 50 lot size
RISK_FREE   = 0.065    # 6.5% annual
DATA_DIR    = "optionOIData/nifty50"
GEX_DIR     = "gexData"
IV_DIR      = "ivData"
GEX_LATEST  = os.path.join(GEX_DIR, "gex_latest.json")
GEX_HISTORY = os.path.join(GEX_DIR, "gex_history.csv")
IV_LATEST   = os.path.join(IV_DIR,  "iv_latest.json")
IV_HISTORY  = os.path.join(IV_DIR,  "iv_history.csv")


# ── Pure-python Black-Scholes ─────────────────────────────────────────────────
def _bs_price(S: float, K: float, T: float, r: float,
              sigma: float, flag: str) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        return max(0.0, (S - K) if flag == "c" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if flag == "c":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 1e-9 or sigma <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def implied_vol(price: float, S: float, K: float, T: float,
                r: float, flag: str,
                tol: float = 1e-5, max_iter: int = 150) -> float | None:
    """Bisection IV solver. Returns None if no solution found."""
    intrinsic = max(0.0, (S - K) if flag == "c" else (K - S))
    if price <= intrinsic + 1e-6 or T <= 1e-9:
        return None
    lo, hi = 1e-6, 20.0          # sigma range 0.001% – 2000%
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p   = _bs_price(S, K, T, r, mid, flag)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2
    return mid if 1e-5 < mid < 19.9 else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")


def _days_to_expiry(expiry_str: str, ref_dt: datetime) -> float:
    """Return fractional calendar days from ref_dt to expiry_str (dd-Mon-YYYY)."""
    try:
        exp = datetime.strptime(expiry_str, "%d-%b-%Y").replace(
            hour=15, minute=30,
            tzinfo=ZoneInfo("Asia/Kolkata"))
        return max(0.0, (exp - ref_dt).total_seconds() / 86400)
    except Exception:
        return 0.0


def _load_latest_snap() -> tuple[pd.DataFrame, str]:
    """Load the latest OI CSV (closest upcoming expiry) and return (df, expiry_str)."""
    files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")],
        key=lambda f: pd.to_datetime(
            f.replace("nifty50-", "").replace(".csv", ""),
            format="%d-%b-%Y", errors="coerce"))

    today = date.today()
    for fname in files:
        exp_str = fname.replace("nifty50-", "").replace(".csv", "")
        try:
            exp_date = pd.to_datetime(exp_str, format="%d-%b-%Y").date()
        except Exception:
            continue
        if exp_date >= today:
            path = os.path.join(DATA_DIR, fname)
            df   = pd.read_csv(path)
            df["datetime"]    = pd.to_datetime(df["datetime"], utc=False)
            df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
            df["openInterest"]= pd.to_numeric(df["openInterest"], errors="coerce")
            df["lastPrice"]   = pd.to_numeric(df["lastPrice"],   errors="coerce")
            return df, exp_str

    return pd.DataFrame(), ""


# ── GEX computation ───────────────────────────────────────────────────────────
def compute_gex(snap: pd.DataFrame, expiry_str: str,
                ref_dt: datetime) -> dict:
    """
    For each strike compute GEX = Gamma × OI × S² × 0.01 × LotSize.
    Dealer convention:
      - Call OI: dealers are SHORT calls → negative GEX contribution
      - Put  OI: dealers are SHORT puts  → positive GEX contribution
    Net GEX = sum(put_gex) - sum(call_gex) across all strikes.
    """
    spot = float(snap["underlyingValue"].iloc[0])
    T    = _days_to_expiry(expiry_str, ref_dt) / 365.0

    ce   = snap[snap["optionType"] == "Call"].groupby("strikePrice").agg(
               oi=("openInterest", "max"), price=("lastPrice", "last")).reset_index()
    pe   = snap[snap["optionType"] == "Put"].groupby("strikePrice").agg(
               oi=("openInterest", "max"), price=("lastPrice", "last")).reset_index()

    gex_by_strike = {}

    for _, row in ce.iterrows():
        K, oi, px = row["strikePrice"], row["oi"], row["price"]
        if oi <= 0 or px <= 0 or T <= 0:
            continue
        iv = implied_vol(px, spot, K, T, RISK_FREE, "c")
        if iv is None:
            continue
        g  = _bs_gamma(spot, K, T, RISK_FREE, iv)
        gex_by_strike.setdefault(K, {"ce_gex": 0, "pe_gex": 0, "ce_iv": None, "pe_iv": None})
        gex_by_strike[K]["ce_gex"] = -g * oi * spot ** 2 * 0.01 * LOT_SIZE
        gex_by_strike[K]["ce_iv"]  = round(iv * 100, 2)

    for _, row in pe.iterrows():
        K, oi, px = row["strikePrice"], row["oi"], row["price"]
        if oi <= 0 or px <= 0 or T <= 0:
            continue
        iv = implied_vol(px, spot, K, T, RISK_FREE, "p")
        if iv is None:
            continue
        g  = _bs_gamma(spot, K, T, RISK_FREE, iv)
        gex_by_strike.setdefault(K, {"ce_gex": 0, "pe_gex": 0, "ce_iv": None, "pe_iv": None})
        gex_by_strike[K]["pe_gex"] = g * oi * spot ** 2 * 0.01 * LOT_SIZE
        gex_by_strike[K]["pe_iv"]  = round(iv * 100, 2)

    records = []
    for K, v in sorted(gex_by_strike.items()):
        net = v["ce_gex"] + v["pe_gex"]
        records.append({
            "strike":  K,
            "ce_gex":  round(v["ce_gex"], 2),
            "pe_gex":  round(v["pe_gex"], 2),
            "net_gex": round(net, 2),
            "ce_iv":   v["ce_iv"],
            "pe_iv":   v["pe_iv"],
        })

    total_gex  = sum(r["net_gex"] for r in records)

    # GEX flip point = strike where cumulative GEX crosses zero (price magnet / repeller boundary)
    strikes_sorted = sorted(gex_by_strike.keys())
    cum = 0.0
    flip_point = None
    for K in strikes_sorted:
        prev_cum = cum
        cum += gex_by_strike[K]["ce_gex"] + gex_by_strike[K]["pe_gex"]
        if prev_cum < 0 <= cum or prev_cum >= 0 > cum:
            flip_point = K
            break

    regime = "PINNING" if total_gex >= 0 else "TRENDING"

    return {
        "computed_at": _ist_now(),
        "spot":        round(spot, 2),
        "expiry":      expiry_str,
        "net_gex":     round(total_gex, 2),
        "flip_point":  flip_point,
        "regime":      regime,
        "interpretation": (
            f"Net GEX {total_gex:+,.0f}: dealers will BUY dips & SELL rallies "
            f"(price-pinning near {flip_point or 'N/A'})"
            if total_gex >= 0 else
            f"Net GEX {total_gex:+,.0f}: dealers AMPLIFY moves "
            f"(trending / explosive — flip level {flip_point or 'N/A'})"
        ),
        "strikes": records,
    }


# ── IV Rank / Percentile ──────────────────────────────────────────────────────
def compute_iv_rank(snap: pd.DataFrame, expiry_str: str,
                    ref_dt: datetime) -> dict:
    """
    Compute ATM IV for the current snapshot, then compare against
    historical ATM IVs stored in iv_history.csv.
    """
    spot = float(snap["underlyingValue"].iloc[0])
    T    = _days_to_expiry(expiry_str, ref_dt) / 365.0

    all_strikes = sorted(snap["strikePrice"].dropna().unique())
    if not all_strikes:
        return {}

    atm = min(all_strikes, key=lambda x: abs(x - spot))

    # Compute ATM CE and PE IV
    atm_iv_list = []
    for flag, otype in [("c", "Call"), ("p", "Put")]:
        row = (snap[(snap["strikePrice"] == atm) &
                    (snap["optionType"] == otype)]
               .sort_values("datetime").iloc[-1]
               if not snap[(snap["strikePrice"] == atm) &
                            (snap["optionType"] == otype)].empty
               else None)
        if row is None:
            continue
        px = float(row["lastPrice"])
        if px <= 0 or T <= 0:
            continue
        iv = implied_vol(px, spot, atm, T, RISK_FREE, flag)
        if iv is not None:
            atm_iv_list.append(iv * 100)

    if not atm_iv_list:
        return {}

    atm_iv = round(sum(atm_iv_list) / len(atm_iv_list), 2)  # avg CE+PE ATM IV

    # Load historical IV to compute rank / percentile
    iv_rank   = None
    iv_pctile = None

    if os.path.exists(IV_HISTORY):
        hist = pd.read_csv(IV_HISTORY)
        if "atm_iv" in hist.columns and len(hist) >= 5:
            hist_ivs = hist["atm_iv"].dropna().values
            iv_52w_hi  = hist_ivs.max()
            iv_52w_lo  = hist_ivs.min()
            iv_rank    = round((atm_iv - iv_52w_lo) /
                                (iv_52w_hi - iv_52w_lo) * 100, 1) \
                         if iv_52w_hi > iv_52w_lo else 50.0
            iv_pctile  = round((hist_ivs < atm_iv).sum() / len(hist_ivs) * 100, 1)

    if iv_rank is None:
        verdict = "INSUFFICIENT HISTORY"
        color   = "#6c757d"
    elif iv_rank >= 80:
        verdict = "IV HIGH — Options expensive, mean-reversion likely"
        color   = "#dc3545"
    elif iv_rank <= 20:
        verdict = "IV LOW — Options cheap, big move expected"
        color   = "#28a745"
    elif iv_rank >= 60:
        verdict = "IV ELEVATED — Slight premium in options"
        color   = "#fd7e14"
    else:
        verdict = "IV NORMAL — No edge from volatility alone"
        color   = "#6c757d"

    return {
        "computed_at": _ist_now(),
        "spot":        round(spot, 2),
        "atm_strike":  int(atm),
        "atm_iv":      atm_iv,
        "iv_rank":     iv_rank,
        "iv_pctile":   iv_pctile,
        "verdict":     verdict,
        "verdict_color": color,
    }


# ── Persistence ───────────────────────────────────────────────────────────────
def _save_gex(result: dict):
    os.makedirs(GEX_DIR, exist_ok=True)

    # latest JSON (overwrite)
    with open(GEX_LATEST, "w") as f:
        json.dump(result, f, indent=2)

    # history CSV (append one summary row)
    row = {
        "datetime":  result["computed_at"],
        "spot":      result["spot"],
        "net_gex":   result["net_gex"],
        "flip_point":result["flip_point"],
        "regime":    result["regime"],
    }
    write_hdr = not os.path.exists(GEX_HISTORY)
    with open(GEX_HISTORY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _save_iv(result: dict):
    os.makedirs(IV_DIR, exist_ok=True)

    with open(IV_LATEST, "w") as f:
        json.dump(result, f, indent=2)

    row = {k: result.get(k) for k in
           ["computed_at", "spot", "atm_strike", "atm_iv",
            "iv_rank", "iv_pctile", "verdict"]}
    write_hdr = not os.path.exists(IV_HISTORY)
    with open(IV_HISTORY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ref_dt = datetime.now(ZoneInfo("Asia/Kolkata"))
    print(f"[{_ist_now()}] Loading latest OI snapshot...")

    df, expiry_str = _load_latest_snap()
    if df.empty:
        print("No OI data found — exiting.")
        exit(0)

    latest_dt = df["datetime"].max()
    snap      = df[df["datetime"] == latest_dt].copy()
    spot      = float(snap["underlyingValue"].iloc[0])

    print(f"  Expiry: {expiry_str}  |  Snap: {latest_dt}  |  Spot: {spot:.1f}")
    print(f"  Strikes: {snap['strikePrice'].nunique()}  |  Rows: {len(snap)}")

    # -- GEX ------------------------------------------------------------------
    print("\n[GEX] Computing Gamma Exposure...")
    gex = compute_gex(snap, expiry_str, ref_dt)
    _save_gex(gex)
    print(f"  Net GEX : {gex['net_gex']:+,.0f}")
    print(f"  Regime  : {gex['regime']}")
    print(f"  Flip    : {gex['flip_point']}")
    print(f"  -> {gex['interpretation']}")

    # -- IV Rank --------------------------------------------------------------
    print("\n[IV] Computing ATM IV Rank / Percentile...")
    iv_res = compute_iv_rank(snap, expiry_str, ref_dt)
    if iv_res:
        _save_iv(iv_res)
        print(f"  ATM Strike : {iv_res['atm_strike']}")
        print(f"  ATM IV     : {iv_res['atm_iv']:.2f}%")
        print(f"  IV Rank    : {iv_res.get('iv_rank', 'N/A')}")
        print(f"  IV %ile    : {iv_res.get('iv_pctile', 'N/A')}")
        print(f"  Verdict    : {iv_res['verdict']}")
    else:
        print("  Could not compute ATM IV (insufficient data or market closed)")

    print(f"\nDone. Files written to {GEX_DIR}/ and {IV_DIR}/")
