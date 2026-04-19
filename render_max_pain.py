import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from render_live_data import render_live_data_tab

def set_layout():
    # ─── Light theme CSS ────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #f8f9fa;
        color: #212529;
    }
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; color: #212529 !important; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600; color: #212529; }
    .metric-sub   { font-size: 12px; color: #adb5bd; margin-top: 2px; }

    .stDataFrame { background: #ffffff !important; border-radius: 10px; }
    .block-container { padding-top: 1.5rem; }

    div[data-testid="stSidebarContent"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }
    .tab-header {
        font-size: 13px; font-weight: 600; letter-spacing: 0.5px;
        padding: 6px 0; border-bottom: 2px solid #dee2e6;
        margin-bottom: 18px; color: #495057;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Colour palette (Light Theme) ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Sora, sans-serif", color="#495057", size=12),
    xaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6", showgrid=True),
    yaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6", showgrid=True),
    legend=dict(bgcolor="#ffffff", bordercolor="#dee2e6", borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=60),
)
CE_COLOR   = "#007bff"  # Blue
PE_COLOR   = "#dc3545"  # Red
MAX_COLOR  = "#28a745"  # Green
LOSS_COLOR = "#dc3545"
GAIN_COLOR = "#28a745"
PALETTE    = [CE_COLOR, PE_COLOR, "#fd7e14", MAX_COLOR, "#6f42c1",
              "#17a2b8", "#ffc107", LOSS_COLOR, "#6610f2", "#20c997",
              "#d63384", "#0dcaf0", "#ffc107", "#e83e8c", "#052c65"]


# ─── Data helpers ────────────────────────────────────────────────────────────
DATA_DIR = "optionOIData/nifty50/"

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
    df["openInterest"]= pd.to_numeric(df["openInterest"], errors="coerce")
    df["expiryDate"]  = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y", errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_data(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
        return transform_data(df)
    return pd.DataFrame()


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows between 09:00 and 15:30 IST (inclusive)."""
    minutes = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    return df[(minutes >= 9 * 60) & (minutes <= 15 * 60 + 30)].copy()


def date_selector(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render a date dropdown and return market-hours-filtered df for that date."""
    available_dates = sorted(df["datetime"].dt.date.unique())
    date_labels     = [d.strftime("%d-%b-%Y") for d in available_dates]
    sel_lbl = st.selectbox("Select Date", date_labels,
                           index=len(date_labels) - 1, key=key)
    sel_date = available_dates[date_labels.index(sel_lbl)]
    return filter_market_hours(df[df["datetime"].dt.date == sel_date])


def compute_max_pain(df_snap: pd.DataFrame):
    """
    Standard max-pain calculation.
    For each potential expiry price S:
      CE buyers at strike K gain max(0, S-K) → writers lose (S-K)*OI*lot_size
      PE buyers at strike K gain max(0, K-S) → writers lose (K-S)*OI*lot_size
    Max pain = S that minimises total buyers' gain (= maximises writers' gain).
    Lot size: Nifty 50 = 65.
    """
    lot_size = 65
    ce = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].max()
    pe = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].max()
    strikes = sorted(set(ce.index) | set(pe.index))
    if not strikes:
        return None, pd.DataFrame()

    records = []
    for s in strikes:
        ce_buyers_gain = sum((s - k) * ce.get(k, 0) * lot_size for k in strikes if k < s)
        pe_buyers_gain = sum((k - s) * pe.get(k, 0) * lot_size for k in strikes if k > s)
        records.append({"strikePrice": s,
                        "CE_loss": ce_buyers_gain, "PE_loss": pe_buyers_gain,
                        "total_loss": ce_buyers_gain + pe_buyers_gain})

    pain_df = pd.DataFrame(records)
    mp = int(pain_df.loc[pain_df["total_loss"].idxmin(), "strikePrice"])
    return mp, pain_df


@st.cache_data(ttl=300)
def compute_max_pain_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute max pain for every datetime snapshot in df.
    Uses numpy broadcasting per snapshot so it's fast even with many strikes.
    Returns DataFrame with columns: datetime, max_pain, spot.
    """
    records = []
    for dt, snap in df.groupby("datetime", sort=True):
        ce = snap[snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].max()
        pe = snap[snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].max()
        strikes = np.array(sorted(set(ce.index) | set(pe.index)), dtype=float)
        if len(strikes) < 2:
            continue
        ce_oi = np.array([ce.get(k, 0) for k in strikes])
        pe_oi = np.array([pe.get(k, 0) for k in strikes])
        # diff[i,j] = strikes[i] - strikes[j]  (row = potential expiry, col = strike)
        diff = strikes[:, None] - strikes[None, :]
        total = (np.maximum(0, diff) * ce_oi).sum(axis=1) + \
                (np.maximum(0, -diff) * pe_oi).sum(axis=1)
        mp = int(strikes[np.argmin(total)])
        spot = snap["underlyingValue"].iloc[0]
        records.append({"datetime": dt, "max_pain": mp, "spot": spot})
    return pd.DataFrame(records)


@st.cache_data(ttl=1800)   # file refreshes every ~30 min
def load_futures_data() -> pd.DataFrame:
    """
    Load Nifty50 futures OHLCV + OI data from the API-populated JSON file.

    Expected JSON layout:
        { "status": "success",
          "data":   { "candles": [ [datetime, open, high, low, close, volume, oi], ... ] } }

    Columns returned: datetime, open, high, low, close, volume, oi
    """
    import json as _json

    json_path = os.path.join("futuresData", "nifty50", "nifty50_futures_apr_2026.json")

    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        return pd.DataFrame()

    with open(json_path) as f:
        raw = _json.load(f)

    candles = raw.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()

    COLS = ["datetime", "open", "high", "low", "close", "volume", "oi"]
    df   = pd.DataFrame(candles, columns=COLS[:len(candles[0])])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    for c in COLS[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=300)
def compute_pcr_history(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PCR (Put/Call OI ratio) for every datetime snapshot in df."""
    records = []
    for dt, snap in df.groupby("datetime", sort=True):
        ce_oi = snap[snap["optionType"] == "Call"]["openInterest"].sum()
        pe_oi = snap[snap["optionType"] == "Put"]["openInterest"].sum()
        if ce_oi > 0:
            records.append({
                "datetime": dt,
                "pcr":      pe_oi / ce_oi,
                "ce_oi":    ce_oi,
                "pe_oi":    pe_oi,
                "spot":     snap["underlyingValue"].iloc[0],
            })
    return pd.DataFrame(records)


def compute_trap_signals(df_snap: pd.DataFrame,
                         pcr_hist: pd.DataFrame,
                         mp_hist:  pd.DataFrame,
                         fut_df:   pd.DataFrame = None) -> dict:
    """
    Compute all operator-trap indicators and return a flat signal dict.

    Convention:
      Bull Trap  → operators trap retail bulls; price will fall
      Bear Trap  → operators trap retail bears; price will rise
    Positive score values always indicate a trap (bull or bear).
    """
    out = {}
    spot = float(df_snap["underlyingValue"].iloc[0])
    out["spot"] = spot

    ce_by_strike = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].max()
    pe_by_strike = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].max()
    total_ce_oi  = ce_by_strike.sum()
    total_pe_oi  = pe_by_strike.sum()

    # ── 1. Put-Call Ratio (PCR) ───────────────────────────────────────────────
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
    out["pcr"] = pcr
    if pcr >= 1.5:
        pcr_bull, pcr_bear = 0, 70
        pcr_signal = "Extreme PUT writing → Strong bullish operator positioning"
        pcr_bias   = "BEAR TRAP"
    elif pcr >= 1.2:
        pcr_bull, pcr_bear = 0, 40
        pcr_signal = "High PCR → Moderate bullish operator bias"
        pcr_bias   = "BEAR TRAP"
    elif pcr <= 0.70:
        pcr_bull, pcr_bear = 70, 0
        pcr_signal = "Extreme CALL writing → Strong bearish operator positioning"
        pcr_bias   = "BULL TRAP"
    elif pcr <= 0.85:
        pcr_bull, pcr_bear = 40, 0
        pcr_signal = "Low PCR → Moderate bearish operator bias"
        pcr_bias   = "BULL TRAP"
    else:
        pcr_bull, pcr_bear = 0, 0
        pcr_signal = "PCR near neutral — no directional bias"
        pcr_bias   = "NEUTRAL"
    out.update(pcr_signal=pcr_signal, pcr_bias=pcr_bias,
               pcr_bull_score=pcr_bull, pcr_bear_score=pcr_bear)

    # ── 2. Max Pain Divergence ────────────────────────────────────────────────
    max_pain, _ = compute_max_pain(df_snap)
    out["max_pain"] = max_pain
    div_pct = (spot - max_pain) / max_pain * 100 if max_pain else 0.0
    out["divergence_pct"] = div_pct
    abs_div = abs(div_pct)
    div_score = 80 if abs_div >= 2.0 else 50 if abs_div >= 1.0 else 25 if abs_div >= 0.5 else 0
    if div_pct > 0:
        div_bull, div_bear = div_score, 0
        div_signal = f"Spot +{div_pct:.2f}% above max pain → Gravitational DOWNWARD pull"
        div_bias   = "BULL TRAP"
    elif div_pct < 0:
        div_bull, div_bear = 0, div_score
        div_signal = f"Spot {div_pct:.2f}% below max pain → Gravitational UPWARD pull"
        div_bias   = "BEAR TRAP"
    else:
        div_bull, div_bear = 0, 0
        div_signal = "Spot at max pain — no gravitational pull"
        div_bias   = "NEUTRAL"
    out.update(div_signal=div_signal, div_bias=div_bias,
               div_bull_score=div_bull, div_bear_score=div_bear)

    # ── 3. OI Walls ───────────────────────────────────────────────────────────
    ce_wall = int(ce_by_strike.idxmax()) if not ce_by_strike.empty else int(spot)
    pe_wall = int(pe_by_strike.idxmax()) if not pe_by_strike.empty else int(spot)
    out["ce_wall"] = ce_wall
    out["pe_wall"] = pe_wall
    ce_wall_dist = (ce_wall - spot) / spot * 100   # +ve = above spot
    pe_wall_dist = (spot - pe_wall) / spot * 100   # +ve = below spot
    out["ce_wall_dist_pct"] = ce_wall_dist
    out["pe_wall_dist_pct"] = pe_wall_dist

    if -0.5 <= ce_wall_dist < 0:
        wall_bull, wall_bear = 80, 0
        wall_signal = f"Spot BREACHED CE wall {ce_wall:,} → Fake breakout risk"
        wall_bias   = "BULL TRAP"
    elif 0 <= ce_wall_dist <= 1.0:
        wall_bull, wall_bear = 65, 0
        wall_signal = f"CE wall {ce_wall:,} only {ce_wall_dist:.1f}% above spot → Strong resistance ahead"
        wall_bias   = "BULL TRAP"
    elif -0.5 <= pe_wall_dist < 0:
        wall_bull, wall_bear = 0, 80
        wall_signal = f"Spot BREACHED PE wall {pe_wall:,} → Fake breakdown risk"
        wall_bias   = "BEAR TRAP"
    elif 0 <= pe_wall_dist <= 1.0:
        wall_bull, wall_bear = 0, 65
        wall_signal = f"PE wall {pe_wall:,} only {pe_wall_dist:.1f}% below spot → Strong support below"
        wall_bias   = "BEAR TRAP"
    else:
        wall_bull, wall_bear = 0, 0
        wall_signal = (f"CE wall {ce_wall_dist:+.1f}% / PE wall {pe_wall_dist:+.1f}% — "
                       "no immediate pressure")
        wall_bias   = "NEUTRAL"
    out.update(wall_signal=wall_signal, wall_bias=wall_bias,
               wall_bull_score=wall_bull, wall_bear_score=wall_bear)

    # ── 4. ATM OI Skew ────────────────────────────────────────────────────────
    all_stk = sorted(set(ce_by_strike.index) | set(pe_by_strike.index))
    if all_stk:
        atm     = min(all_stk, key=lambda x: abs(x - spot))
        ai      = all_stk.index(atm)
        near    = all_stk[max(0, ai - 3): ai + 4]
        atm_ce  = float(ce_by_strike[ce_by_strike.index.isin(near)].sum())
        atm_pe  = float(pe_by_strike[pe_by_strike.index.isin(near)].sum())
        atm_rat = atm_ce / atm_pe if atm_pe > 0 else 1.0
    else:
        atm, atm_ce, atm_pe, atm_rat = int(spot), 0.0, 0.0, 1.0
    out.update(atm=atm, atm_ce_oi=atm_ce, atm_pe_oi=atm_pe, atm_skew_ratio=atm_rat)
    if atm_rat >= 2.0:
        skew_bull, skew_bear = 60, 0
        skew_signal = f"CE OI {atm_rat:.1f}× PE OI near ATM → Heavy call writing → capped upside"
        skew_bias   = "BULL TRAP"
    elif atm_rat <= 0.50:
        skew_bull, skew_bear = 0, 60
        skew_signal = f"PE OI {1/atm_rat:.1f}× CE OI near ATM → Heavy put writing → capped downside"
        skew_bias   = "BEAR TRAP"
    else:
        skew_bull, skew_bear = 0, 0
        skew_signal = f"ATM skew ratio {atm_rat:.2f} — balanced near-ATM writing"
        skew_bias   = "NEUTRAL"
    out.update(skew_signal=skew_signal, skew_bias=skew_bias,
               skew_bull_score=skew_bull, skew_bear_score=skew_bear)

    # ── 5. PCR Trend (rolling 5-snapshot direction) ───────────────────────────
    trend_bull = trend_bear = 0
    trend_signal  = "Insufficient history for PCR trend"
    trend_bias    = "NEUTRAL"
    pcr_trend_val = 0.0
    if pcr_hist is not None and len(pcr_hist) >= 5:
        recent        = pcr_hist["pcr"].iloc[-5:].values
        pcr_trend_val = float(recent[-1] - recent[0])
        if pcr_trend_val >= 0.2:
            trend_bull, trend_bear = 0, 55
            trend_signal = f"PCR +{pcr_trend_val:.2f} over last 5 snaps → Increasing put writing"
            trend_bias   = "BEAR TRAP"
        elif pcr_trend_val <= -0.2:
            trend_bull, trend_bear = 55, 0
            trend_signal = f"PCR {pcr_trend_val:.2f} over last 5 snaps → Increasing call writing"
            trend_bias   = "BULL TRAP"
        else:
            trend_signal = f"PCR trend {pcr_trend_val:+.3f} — stable operator positioning"
    out.update(pcr_trend_val=pcr_trend_val, trend_signal=trend_signal, trend_bias=trend_bias,
               trend_bull_score=trend_bull, trend_bear_score=trend_bear)

    # ── 6. Max Pain Velocity (convergence with spot) ──────────────────────────
    vel_bull = vel_bear = 0
    vel_signal = "Insufficient history for velocity analysis"
    vel_bias   = "NEUTRAL"
    if mp_hist is not None and len(mp_hist) >= 3:
        recent_mp = mp_hist.iloc[-3:]
        mp_vel    = float(recent_mp["max_pain"].diff().mean())
        convergence = spot - max_pain   # +ve = spot above max pain
        if abs(mp_vel) > 5:
            if convergence > 0 and mp_vel > 0:
                vel_bull, vel_bear = 40, 0
                vel_signal = f"Max Pain velocity +{mp_vel:.0f} pts/snap — chasing spot UP → bull trap closing"
                vel_bias   = "BULL TRAP"
            elif convergence < 0 and mp_vel < 0:
                vel_bull, vel_bear = 0, 40
                vel_signal = f"Max Pain velocity {mp_vel:.0f} pts/snap — chasing spot DOWN → bear trap closing"
                vel_bias   = "BEAR TRAP"
            else:
                vel_signal = f"Max Pain velocity {mp_vel:+.0f} pts/snap — diverging from spot"
                vel_bias   = "DIVERGING"
        else:
            vel_signal = f"Max Pain near stationary ({mp_vel:+.1f} pts/snap)"
    out.update(vel_signal=vel_signal, vel_bias=vel_bias,
               vel_bull_score=vel_bull, vel_bear_score=vel_bear)

    # ── 7. Futures Basis ──────────────────────────────────────────────────────
    basis_bull = basis_bear = 0
    basis_signal = "No futures data available"
    basis_bias   = "NEUTRAL"
    basis_val    = 0.0
    basis_pct    = 0.0
    fut_close    = None
    fut_vol      = None
    fut_vol_avg  = None
    snap_time    = df_snap["datetime"].max()
    fut_before   = pd.DataFrame()
    out["fut_oi_now"] = out["fut_oi_delta"] = None

    if fut_df is not None and not fut_df.empty:
        # Align futures to market-hours window of the selected day
        snap_date  = pd.Timestamp(snap_time).date()
        fut_day    = fut_df[fut_df["datetime"].dt.date == snap_date].sort_values("datetime")
        fut_before = fut_day[fut_day["datetime"] <= snap_time]
        if not fut_before.empty:
            fut_close = float(fut_before.iloc[-1]["close"])
            fut_vol   = float(fut_before.iloc[-1]["volume"])
            basis_val = fut_close - spot
            basis_pct = basis_val / spot * 100
            # Futures OI (if available)
            if "oi" in fut_before.columns:
                out["fut_oi_now"]   = float(fut_before.iloc[-1]["oi"])
                out["fut_oi_delta"] = (float(fut_before.iloc[-1]["oi"]) -
                                       float(fut_before.iloc[-2]["oi"])
                                       if len(fut_before) >= 2 else 0.0)
            else:
                out["fut_oi_now"] = out["fut_oi_delta"] = None
            if basis_pct <= -0.5:
                basis_bull, basis_bear = 60, 0
                basis_signal = (f"Backwardation {basis_pct:.2f}% — futures discount to spot → "
                                "operators SHORT futures → BULL TRAP")
                basis_bias   = "BULL TRAP"
            elif basis_pct <= -0.2:
                basis_bull, basis_bear = 35, 0
                basis_signal = f"Mild backwardation {basis_pct:.2f}% — slight bearish futures bias"
                basis_bias   = "BULL TRAP"
            elif basis_pct >= 0.5:
                basis_bull, basis_bear = 0, 40
                basis_signal = (f"Strong contango {basis_pct:.2f}% — futures premium → "
                                "operators LONG futures → BEAR TRAP")
                basis_bias   = "BEAR TRAP"
            elif basis_pct >= 0.2:
                basis_bull, basis_bear = 0, 20
                basis_signal = f"Mild contango {basis_pct:.2f}% — slight bullish futures bias"
                basis_bias   = "BEAR TRAP"
            else:
                basis_signal = f"Basis {basis_pct:+.3f}% — futures near parity with spot"
        else:
            basis_signal = "No futures bar found for this date / snapshot time"
    out.update(fut_close=fut_close, basis_val=basis_val, basis_pct=basis_pct,
               basis_signal=basis_signal, basis_bias=basis_bias,
               basis_bull_score=basis_bull, basis_bear_score=basis_bear)

    # ── 8. Futures Volume Spike ────────────────────────────────────────────────
    vol_bull = vol_bear = 0
    vol_signal  = "No futures data available"
    vol_bias    = "NEUTRAL"
    vol_ratio   = 1.0

    if fut_close is not None and len(fut_before) >= 5:
        avg_window  = fut_before["volume"].iloc[-6:-1]           # 5 bars before current
        fut_vol_avg = float(avg_window.mean()) if len(avg_window) > 0 else 1.0
        vol_ratio   = fut_vol / fut_vol_avg if fut_vol_avg > 0 else 1.0
        # Augment with futures OI change direction if available
        if "oi" in fut_before.columns:
            oi_now   = float(fut_before["oi"].iloc[-1])
            oi_prev  = float(fut_before["oi"].iloc[-2]) if len(fut_before) >= 2 else oi_now
            oi_delta = oi_now - oi_prev
            # Rising futures OI + backwardation = fresh short build = BULL TRAP
            # Rising futures OI + contango     = fresh long build  = BEAR TRAP
            if oi_delta > 0 and basis_pct <= -0.2:
                vol_bull = max(vol_bull, 35)
                vol_bear = 0
            elif oi_delta > 0 and basis_pct >= 0.2:
                vol_bear = max(vol_bear, 35)
                vol_bull = 0

        if vol_ratio >= 2.5:
            if basis_pct <= -0.2:
                vol_bull, vol_bear = 50, 0
                vol_signal = (f"Volume spike {vol_ratio:.1f}× avg with backwardation → "
                              "Aggressive institutional selling → BULL TRAP")
                vol_bias   = "BULL TRAP"
            elif basis_pct >= 0.2:
                vol_bull, vol_bear = 0, 50
                vol_signal = (f"Volume spike {vol_ratio:.1f}× avg with contango → "
                              "Aggressive institutional buying → BEAR TRAP")
                vol_bias   = "BEAR TRAP"
            else:
                vol_signal = f"Volume spike {vol_ratio:.1f}× avg — heavy activity, direction unclear"
        elif vol_ratio >= 1.5:
            if basis_pct <= -0.2:
                vol_bull, vol_bear = 25, 0
                vol_signal = f"Elevated volume {vol_ratio:.1f}× avg with backwardation → operator pressure"
                vol_bias   = "BULL TRAP"
            elif basis_pct >= 0.2:
                vol_bull, vol_bear = 0, 25
                vol_signal = f"Elevated volume {vol_ratio:.1f}× avg with contango → operator accumulation"
                vol_bias   = "BEAR TRAP"
            else:
                vol_signal = f"Elevated volume {vol_ratio:.1f}× avg — monitoring needed"
        else:
            vol_signal = f"Normal volume ({vol_ratio:.1f}× 5-bar avg) — no unusual activity"
    elif fut_close is not None:
        vol_signal = "Insufficient futures history for volume spike analysis"
    out.update(fut_vol=fut_vol, fut_vol_avg=fut_vol_avg, vol_ratio=vol_ratio,
               vol_signal=vol_signal, vol_bias=vol_bias,
               vol_bull_score=vol_bull, vol_bear_score=vol_bear)

    # ── Final Verdict ─────────────────────────────────────────────────────────
    total_bull = (pcr_bull + div_bull + wall_bull + skew_bull + trend_bull + vel_bull +
                  basis_bull + vol_bull)
    total_bear = (pcr_bear + div_bear + wall_bear + skew_bear + trend_bear + vel_bear +
                  basis_bear + vol_bear)
    max_possible = 80 + 80 + 70 + 60 + 55 + 40 + 60 + 50   # sum of per-signal max scores
    bull_conf = min(100, round(total_bull / max_possible * 100))
    bear_conf = min(100, round(total_bear / max_possible * 100))

    if bull_conf >= 60:
        verdict, v_color, v_bg, v_conf = (
            "🔴 BULL TRAP DETECTED", "#dc3545", "#f8d7da", bull_conf)
        v_detail = ("Strong signals that operators are trapping retail bulls. "
                    "Current price strength may be artificial — expect a reversal downward.")
    elif bear_conf >= 60:
        verdict, v_color, v_bg, v_conf = (
            "🟢 BEAR TRAP DETECTED", "#198754", "#d1e7dd", bear_conf)
        v_detail = ("Strong signals that operators are trapping retail bears. "
                    "Current price weakness may be artificial — expect a reversal upward.")
    elif bull_conf >= 40:
        verdict, v_color, v_bg, v_conf = (
            "⚠️ POSSIBLE BULL TRAP", "#fd7e14", "#fff3cd", bull_conf)
        v_detail = "Moderate signals suggest a bull trap. Exercise caution on long positions."
    elif bear_conf >= 40:
        verdict, v_color, v_bg, v_conf = (
            "⚠️ POSSIBLE BEAR TRAP", "#fd7e14", "#fff3cd", bear_conf)
        v_detail = "Moderate signals suggest a bear trap. Exercise caution on short positions."
    else:
        # Confidence in "no trap" = inverse of how strong any trap signal is
        no_trap_conf = max(50, min(90, 100 - max(bull_conf, bear_conf)))
        verdict, v_color, v_bg, v_conf = (
            "✅ NO VISIBLE TRAP", "#6c757d", "#f8f9fa", no_trap_conf)
        v_detail = ("All signals are neutral or weak. Market appears to be trading without "
                    "a visible operator trap at this time. Operators likely pinning near max pain.")

    out.update(
        total_bull_score=total_bull, total_bear_score=total_bear,
        bull_confidence=bull_conf,   bear_confidence=bear_conf,
        verdict=verdict, verdict_color=v_color, verdict_bg=v_bg,
        final_confidence=v_conf,     verdict_detail=v_detail,
    )
    return out


def build_oi_table(df_snap: pd.DataFrame, max_pain: int, n: int = 10) -> pd.DataFrame:
    lot_size = 65  # Nifty 50 lot size

    # Per-strike OI for display columns
    ce_oi = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].max().rename("CE_OI")
    pe_oi = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].max().rename("PE_OI")

    # Full chain OI + lastPrice across ALL strikes (needed for aggregate P&L)
    ce_all = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice").agg(
        OI=("openInterest", "max"), Price=("lastPrice", "last")
    ).fillna(0)
    pe_all = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice").agg(
        OI=("openInterest", "max"), Price=("lastPrice", "last")
    ).fillna(0)

    # Total premium already collected by ALL option writers across the full chain
    total_ce_prem = float((ce_all["OI"] * ce_all["Price"] * lot_size).sum())
    total_pe_prem = float((pe_all["OI"] * pe_all["Price"] * lot_size).sum())

    ce_k = ce_all.index.values
    ce_ois = ce_all["OI"].values
    pe_k = pe_all.index.values
    pe_ois = pe_all["OI"].values

    merged = pd.concat([ce_oi, pe_oi], axis=1).fillna(0).reset_index()
    merged["CE_OI"] = merged["CE_OI"].astype(int)
    merged["PE_OI"] = merged["PE_OI"].astype(int)

    # For each row's strike as the potential expiry price S:
    #   CE writers' P&L = total CE premium collected - payout on all ITM calls (strike < S)
    #   PE writers' P&L = total PE premium collected - payout on all ITM puts (strike > S)
    ce_pnl_list, pe_pnl_list = [], []
    for s in merged["strikePrice"]:
        ce_payout = sum((s - k) * oi * lot_size for k, oi in zip(ce_k, ce_ois) if k < s)
        pe_payout = sum((k - s) * oi * lot_size for k, oi in zip(pe_k, pe_ois) if k > s)
        ce_pnl_list.append(int(total_ce_prem - ce_payout))
        pe_pnl_list.append(int(total_pe_prem - pe_payout))

    merged["CE_PnL"]          = np.array(ce_pnl_list, dtype=np.int64)
    merged["PE_PnL"]          = np.array(pe_pnl_list, dtype=np.int64)
    merged["Total_PnL"]       = (merged["CE_PnL"] + merged["PE_PnL"]).astype(np.int64)
    merged["Operator_PnL_70"] = (merged["Total_PnL"] * 0.70).astype(np.int64)

    strikes = sorted(merged["strikePrice"].unique())
    nearest = min(strikes, key=lambda x: abs(x - max_pain))
    idx     = strikes.index(nearest)
    lo, hi  = max(0, idx - n), min(len(strikes) - 1, idx + n)
    window  = strikes[lo: hi + 1]

    result = merged[merged["strikePrice"].isin(window)].copy()
    result["isMaxPain"] = result["strikePrice"] == nearest
    return result.sort_values("strikePrice").reset_index(drop=True)


def render_oi_analytics():
    set_layout()
    # ─── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        if not os.path.exists(DATA_DIR):
            st.error(f"Directory not found: {DATA_DIR}")
            st.stop()

        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if not all_files:
            st.error("No CSV files found in the data directory.")
            st.stop()

        raw_expiries = [f.replace('nifty50-', '').replace('.csv', '') for f in all_files]
        def _parse_exp(s):
            try:
                return pd.to_datetime(s, format="%d-%b-%Y").date()
            except Exception:
                return pd.Timestamp.min.date()
        expiry_options = sorted(raw_expiries, key=_parse_exp, reverse=True)
        today = pd.Timestamp.now().date()
        parsed_exp_dates = [_parse_exp(e) for e in expiry_options]
        default_exp_idx = min(range(len(parsed_exp_dates)), key=lambda i: abs((parsed_exp_dates[i] - today).days))
        selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_options, index=default_exp_idx, key='expiry_date')

        target_file = f"nifty50-{selected_expiry}.csv"
        df_raw = load_data(target_file)
        
        if df_raw.empty:
            st.error("Loaded data is empty.")
            st.stop()

        expiries       = sorted(df_raw["expiryDate"].dropna().unique())
        expiry_labels  = [pd.Timestamp(e).strftime("%d-%b-%Y") for e in expiries]
        sel_expiry_lbl = selected_expiry
        
        try:
            sel_expiry = expiries[expiry_labels.index(sel_expiry_lbl)]
        except ValueError:
            sel_expiry = expiries

        df = df_raw[df_raw["expiryDate"] == sel_expiry].copy()

        all_strikes = sorted(df["strikePrice"].dropna().unique().astype(int).tolist())

        st.divider()
        auto_refresh = st.toggle("Auto-refresh every 3 minutes", value=False)
        if auto_refresh:
            time.sleep(3)
            st.rerun()


    # ─── Header ──────────────────────────────────────────────────────────────────
    underlying_spot = df["underlyingValue"].iloc[-1] if len(df) else 0
    latest_ts       = df["datetime"].max() if len(df) else "—"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
    <div>
        <h1 style="margin:0;font-size:28px;color:#212529;">📊 Options OI Analytics</h1>
        <p style="margin:0;color:#6c757d;font-size:13px;">
        Expiry: <b style="color:#495057;">{sel_expiry_lbl}</b>&nbsp;|&nbsp;
        {len(df):,} records&nbsp;|&nbsp;
        Underlying: <b style="color:#fd7e14;">₹{underlying_spot:,.2f}</b>&nbsp;|&nbsp;
        Last snap: <b style="color:#495057;">{pd.Timestamp(latest_ts).strftime('%H:%M:%S') if latest_ts != '—' else '—'}</b>
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈  OI Profile & Trend",
        "🕯  OHLC Single Strike",
        "⚖️  Max Pain & Table",
        "📡  Max Pain Velocity",
        "🎯  Operator Trap Detector",
        "📡  Live Market Data",
    ])


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 1 — OI Change Over Time
    # ══════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="tab-header">Open Interest Change Over Time</div>', unsafe_allow_html=True)

        # ── Controls ─────────────────────────────────────────────────────────────
        cc1, cc2, cc3, cc4 = st.columns([1.5, 1, 1, 1])
        with cc1:
            df_day1 = date_selector(df, key="t1_date")
        with cc2:
            t1_opt = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="t1_opt")
        with cc3:
            n_side1 = st.slider("Strikes each side", 3, min(20, len(all_strikes) // 2), 10, key="ns1")
        with cc4:
            t1_resample = st.selectbox("OHLC bar size", ["5min", "10min", "15min", "30min"], index=1, key="t1_rs")

        if df_day1.empty:
            st.warning("No market-hours data for this date.")
        else:
            # Spot = latest underlying value in selected day
            spot1   = df_day1["underlyingValue"].iloc[-1]
            center1 = min(all_strikes, key=lambda x: abs(x - spot1)) if all_strikes else 0
            ci1     = all_strikes.index(center1) if center1 in all_strikes else 0
            disp_strikes1 = all_strikes[max(0, ci1 - n_side1): min(len(all_strikes), ci1 + n_side1 + 1)]
            t1_clr  = CE_COLOR if t1_opt == "Call" else PE_COLOR

            df_t1 = (df_day1[df_day1["optionType"] == t1_opt]
                             [df_day1["strikePrice"].isin(disp_strikes1)]
                             .sort_values(["strikePrice", "datetime"]))

            all_times1 = sorted(df_t1["datetime"].unique())

            if df_t1.empty or len(all_times1) < 1:
                st.warning("No data for this selection.")
            else:
                pivot = (df_t1.pivot_table(index="datetime", columns="strikePrice",
                                           values="openInterest", aggfunc="max")
                             .sort_index())
                pivot.columns = [int(c) for c in pivot.columns]
                pivot = pivot.ffill()

                oi_first   = pivot.iloc[0]
                delta_cum  = pivot - oi_first
                delta_last = (pivot.iloc[-1] - pivot.iloc[-2]
                              if len(all_times1) >= 2
                              else pd.Series(0, index=pivot.columns))

                # ── Row 1: Current OI snapshot | Latest interval Δ OI ────────
                col_snap, col_delta = st.columns(2)

                with col_snap:
                    oi_now   = pivot.iloc[-1].sort_index()
                    bar_cols = [MAX_COLOR if s == center1 else t1_clr for s in oi_now.index]
                    fig_snap = go.Figure(go.Bar(
                        x=[f"{int(s):,}" for s in oi_now.index],
                        y=oi_now.values,
                        marker_color=bar_cols,
                        hovertemplate="Strike %{x}<br>OI: <b>%{y:,.0f}</b><extra></extra>"
                    ))
                    fig_snap.update_layout(**PLOTLY_LAYOUT,
                        title=dict(text=f"{t1_opt} OI — Latest Snapshot  (ATM = green)",
                                   font=dict(color="#212529", size=13)),
                        height=380)
                    fig_snap.update_xaxes(title="Strike", tickangle=-45, gridcolor="#e9ecef")
                    fig_snap.update_yaxes(title="OI", gridcolor="#e9ecef")
                    st.plotly_chart(fig_snap, width='stretch')

                with col_delta:
                    dl = delta_last.sort_index()
                    bar_d_cols = [
                        "#fd7e14" if s == center1 else (GAIN_COLOR if v >= 0 else LOSS_COLOR)
                        for s, v in zip(dl.index, dl.values)
                    ]
                    fig_delta = go.Figure(go.Bar(
                        x=[f"{int(s):,}" for s in dl.index],
                        y=dl.values,
                        marker_color=bar_d_cols,
                        hovertemplate="Strike %{x}<br>Δ OI: <b>%{y:+,.0f}</b><extra></extra>"
                    ))
                    fig_delta.add_hline(y=0, line_color="#dee2e6", line_width=1)
                    fig_delta.update_layout(**PLOTLY_LAYOUT,
                        title=dict(text=f"{t1_opt} OI Δ — Latest Interval  (ATM = orange)",
                                   font=dict(color="#212529", size=13)),
                        height=380)
                    fig_delta.update_xaxes(title="Strike", tickangle=-45, gridcolor="#e9ecef")
                    fig_delta.update_yaxes(title="Δ OI", gridcolor="#e9ecef")
                    st.plotly_chart(fig_delta, width='stretch')

                # ── Row 2: OI OHLC Candlestick per strike ─────────────────────
                st.markdown("---")
                st.markdown(f"#### {t1_opt} OI — OHLC Candlestick per Strike  `({t1_resample} bars)`")
                st.caption("Each candle = OHLC of Open Interest values within the bar period. "
                           "Green candle = OI increased, Red = OI decreased.")

                n_cols_ohlc = 3
                strike_rows = [disp_strikes1[i:i+n_cols_ohlc]
                               for i in range(0, len(disp_strikes1), n_cols_ohlc)]

                for row_strikes in strike_rows:
                    cols_ohlc = st.columns(n_cols_ohlc)
                    for col_o, sk in zip(cols_ohlc, row_strikes):
                        s_data = (df_t1[df_t1["strikePrice"] == sk]
                                      .sort_values("datetime")
                                      .set_index("datetime")["openInterest"])
                        ohlc = s_data.resample(t1_resample).ohlc().dropna()
                        if ohlc.empty:
                            col_o.info(f"{sk}: no data")
                            continue
                        is_atm = (sk == center1)
                        with col_o:
                            fig_o = go.Figure(go.Candlestick(
                                x=ohlc.index,
                                open=ohlc["open"], high=ohlc["high"],
                                low=ohlc["low"],   close=ohlc["close"],
                                increasing_line_color=GAIN_COLOR,
                                decreasing_line_color=LOSS_COLOR,
                                name=str(sk),
                            ))
                            fig_o.update_layout(**PLOTLY_LAYOUT,
                                title=dict(
                                    text=f"{'⭐ ATM  ' if is_atm else ''}{sk}",
                                    font=dict(color=MAX_COLOR if is_atm else "#212529", size=12)
                                ),
                                height=260,
                                xaxis_rangeslider_visible=False)
                            fig_o.update_layout(margin=dict(l=40, r=10, t=40, b=40))
                            fig_o.update_xaxes(tickformat="%H:%M", gridcolor="#e9ecef")
                            fig_o.update_yaxes(tickformat=",", gridcolor="#e9ecef")
                            st.plotly_chart(fig_o, width='stretch')

                # ── Row 3: Cumulative Δ OI multi-line ─────────────────────────
                st.markdown("---")
                fig_cum = go.Figure()
                for i, sk in enumerate(sorted(pivot.columns)):
                    series  = delta_cum[sk].dropna()
                    is_atm  = (sk == center1)
                    fig_cum.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        mode="lines+markers", name=str(sk),
                        line=dict(color=PALETTE[i % len(PALETTE)],
                                  width=3 if is_atm else 1.5),
                        marker=dict(size=6 if is_atm else 3),
                        hovertemplate=f"Strike {sk}<br>%{{x|%H:%M}}<br>Δ OI: <b>%{{y:+,.0f}}</b><extra></extra>"
                    ))
                fig_cum.add_hline(y=0, line_color="#dee2e6", line_width=1,
                                  annotation_text="No change",
                                  annotation_position="bottom right",
                                  annotation_font_color="#6c757d")
                fig_cum.update_layout(**PLOTLY_LAYOUT,
                    title=dict(
                        text=f"{t1_opt} — Cumulative OI Change from Day Open  (+ve = buildup, −ve = unwinding)",
                        font=dict(color="#212529", size=14)
                    ),
                    xaxis_title="Time", yaxis_title="Δ OI from first snapshot",
                    height=500)
                fig_cum.update_layout(legend=dict(
                    bgcolor="#ffffff", bordercolor="#dee2e6", borderwidth=1,
                    orientation="v", x=1.01, y=1, font=dict(size=10)
                ))
                st.plotly_chart(fig_cum, width='stretch')

                # ── Row 4: Interval Δ OI stacked bar ──────────────────────────
                st.markdown("---")
                fig_iv = go.Figure()
                interval_delta = pivot.diff()
                for i, sk in enumerate(sorted(pivot.columns)):
                    series     = interval_delta[sk].dropna()
                    bar_colors = [GAIN_COLOR if v >= 0 else LOSS_COLOR for v in series.values]
                    fig_iv.add_trace(go.Bar(
                        x=series.index, y=series.values,
                        name=str(sk), marker_color=bar_colors, opacity=0.75,
                        hovertemplate=f"Strike {sk}<br>%{{x|%H:%M}}<br>Δ OI: <b>%{{y:+,.0f}}</b><extra></extra>",
                    ))
                fig_iv.add_hline(y=0, line_color="#dee2e6", line_width=1)
                fig_iv.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text=f"{t1_opt} — OI Change Per Interval",
                               font=dict(color="#212529", size=14)),
                    barmode="relative",
                    xaxis_title="Time", yaxis_title="Δ OI per interval",
                    height=450)
                fig_iv.update_layout(legend=dict(
                    bgcolor="#ffffff", bordercolor="#dee2e6", borderwidth=1,
                    orientation="v", x=1.01, y=1, font=dict(size=10)
                ))
                st.plotly_chart(fig_iv, width='stretch')


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 2 — OHLC for a single strike
    # ══════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="tab-header">OHLC — Open Interest & Price for a Single Strike</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_strike = st.selectbox("Strike Price", all_strikes,
                                    index=min(len(all_strikes) // 2, len(all_strikes) - 1) if all_strikes else 0)
        with c2:
            sel_opt = st.selectbox("Option Type", ["Call", "Put"])
        with c3:
            resample_freq = st.selectbox("Bar size", ["5min", "10min", "15min", "30min"], index=0)

        df_s = df[
            (df["strikePrice"] == sel_strike) &
            (df["optionType"]  == sel_opt)
        ].sort_values("datetime").set_index("datetime")

        if df_s.empty:
            st.warning("No data for this selection.")
        else:
            ohlc_oi    = df_s["openInterest"].resample(resample_freq).ohlc().dropna()
            ohlc_price = df_s["lastPrice"].resample(resample_freq).ohlc().dropna()
            ohlc_vol   = df_s["volume"].resample(resample_freq).sum()

            fig2 = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.3, 0.2],
                vertical_spacing=0.04,
                subplot_titles=["Open Interest (OHLC Bars)",
                                "Last Price (OHLC Bars)",
                                "Volume"]
            )
            clr = CE_COLOR if sel_opt == "Call" else PE_COLOR

            fig2.add_trace(go.Candlestick(
                x=ohlc_oi.index, open=ohlc_oi["open"], high=ohlc_oi["high"],
                low=ohlc_oi["low"], close=ohlc_oi["close"],
                name="OI",
                increasing_line_color=GAIN_COLOR, decreasing_line_color=LOSS_COLOR,
            ), row=1, col=1)

            fig2.add_trace(go.Candlestick(
                x=ohlc_price.index, open=ohlc_price["open"], high=ohlc_price["high"],
                low=ohlc_price["low"], close=ohlc_price["close"],
                name="Price",
                increasing_line_color=GAIN_COLOR, decreasing_line_color=LOSS_COLOR,
            ), row=2, col=1)

            fig2.add_trace(go.Bar(
                x=ohlc_vol.index, y=ohlc_vol.values,
                name="Volume", marker_color=clr, opacity=0.7,
            ), row=3, col=1)

            fig2.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"{sel_opt} {int(sel_strike)} | {sel_expiry_lbl} | {resample_freq} bars",
                        font=dict(color=clr, size=15)),
                height=640,
                xaxis3_rangeslider_visible=False,
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
            )
            st.plotly_chart(fig2, width='stretch')

            # Quick stats
            latest = df_s.iloc[-1]
            cols = st.columns(4)
            for col, (lbl, val, sub) in zip(cols, [
                ("Current OI",   f"{int(latest['openInterest']):,}", "contracts"),
                ("Last Price",   f"₹{latest['lastPrice']:,.2f}", ""),
                ("Volume",       f"{int(latest['volume']):,}", "today"),
                ("Underlying",   f"₹{latest['underlyingValue']:,.2f}", "spot"),
            ]):
                col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 3 — Max Pain + Table
    # ══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="tab-header">Max Pain Analysis</div>', unsafe_allow_html=True)

        df_day3   = date_selector(df, key="t3_date")
        if df_day3.empty:
            st.warning("No market-hours data for this date.")
            st.stop()

        df_snap         = df_day3[df_day3["datetime"] == df_day3["datetime"].max()].copy()
        max_pain, pain_df = compute_max_pain(df_snap)

        if max_pain is None:
            st.warning("Not enough data to compute max pain.")
            st.stop()

        spot = df_snap["underlyingValue"].iloc[0]
        delta_mp = spot - max_pain

        # ── Top metrics ─────────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (lbl, val, sub, color) in zip(
            [mc1, mc2, mc3, mc4],
            [
                ("Max Pain Strike",      f"{max_pain:,}",     "Writers' preferred close", MAX_COLOR),
                ("Spot vs Max Pain",     f"₹{delta_mp:+,.0f}", f"Spot ₹{spot:,.2f}",
                GAIN_COLOR if delta_mp >= 0 else LOSS_COLOR),
                ("Total CE OI",
                f"{df_snap[df_snap['optionType']=='Call']['openInterest'].sum()/1e5:.2f}L",
                "lots", CE_COLOR),
                ("Total PE OI",
                f"{df_snap[df_snap['optionType']=='Put']['openInterest'].sum()/1e5:.2f}L",
                "lots", PE_COLOR),
            ]
        ):
            col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Controls ─────────────────────────────────────────────────────────────
        n_side = st.slider("Strikes on EACH side of max pain", 5, 20, 10, key="nside")

        # Build window strikes
        table_df = build_oi_table(df_snap, max_pain, n=n_side)

        # Pain chart: CE/PE OI grouped bar — max pain strike highlighted
        fig_mp = go.Figure()
        colors_ce = [MAX_COLOR if s == max_pain else CE_COLOR for s in table_df["strikePrice"]]
        colors_pe = ["#20c997"  if s == max_pain else PE_COLOR for s in table_df["strikePrice"]]

        fig_mp.add_trace(go.Bar(
            x=table_df["strikePrice"], y=table_df["CE_OI"],
            name="Call OI", marker_color=colors_ce, opacity=0.85,
            hovertemplate="Strike: %{x:,}<br>CE OI: %{y:,.0f}<extra></extra>"
        ))
        fig_mp.add_trace(go.Bar(
            x=table_df["strikePrice"], y=table_df["PE_OI"],
            name="Put OI", marker_color=colors_pe, opacity=0.85,
            hovertemplate="Strike: %{x:,}<br>PE OI: %{y:,.0f}<extra></extra>"
        ))
        fig_mp.add_vline(x=max_pain, line_dash="dash", line_color=MAX_COLOR,
                        annotation_text=f"Max Pain {max_pain:,}", annotation_font_color=MAX_COLOR)
        fig_mp.add_vline(x=spot, line_dash="dot", line_color="#fd7e14",
                        annotation_text=f"Spot {int(spot):,}",
                        annotation_font_color="#fd7e14", annotation_position="bottom right")

        fig_mp.update_layout(
            **(PLOTLY_LAYOUT | dict(
            barmode="group",
            title=dict(text="Call / Put Open Interest | Max Pain View",
                    font=dict(color="#212529", size=15)),
            xaxis=dict(title="Strike Price",
                    tickmode="array",
                    tickvals=table_df["strikePrice"].tolist(),
                    ticktext=[f"{int(s):,}" for s in table_df["strikePrice"]],
                    tickangle=-45, gridcolor="#e9ecef"),
            yaxis=dict(title="Open Interest", tickformat=",", gridcolor="#e9ecef"),
            height=500,
            ))
        )
        st.plotly_chart(fig_mp, width='stretch')

        # ── Total buyers' loss curve (full chain) ─────────────────────────────
        fig_pain = go.Figure()
        fig_pain.add_trace(go.Bar(
            x=pain_df["strikePrice"], y=pain_df["total_loss"],
            marker_color=[MAX_COLOR if s == max_pain else "#adb5bd"
                        for s in pain_df["strikePrice"]],
            name="Total Buyers' Loss",
            hovertemplate="Strike: %{x}<br>Buyers' Loss: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_pain.add_vline(x=max_pain, line_dash="dash", line_color=MAX_COLOR,
                        annotation_text=f"Min loss (Max Pain) = {max_pain}",
                        annotation_font_color=MAX_COLOR)
        fig_pain.update_layout(
            **(PLOTLY_LAYOUT | dict(
            title=dict(text="Total Option Buyers' Loss at Each Expiry Price",
                    font=dict(color="#212529", size=14)),
            xaxis=dict(title="Potential Expiry Price (Strike)",
                    tickmode="array",
                    tickvals=pain_df["strikePrice"].tolist(),
                    ticktext=[f"{int(s):,}" for s in pain_df["strikePrice"]],
                    tickangle=-45, gridcolor="#e9ecef"),
            yaxis=dict(title="Buyers' Total Loss (₹)", gridcolor="#e9ecef"),
            height=380,
            ))
        )
        st.plotly_chart(fig_pain, width='stretch')

        # ── Detailed strike table ─────────────────────────────────────────────
        st.markdown("#### 📋 Strike-wise OI & Seller P&L Detail")
        st.caption(
            f"Max pain strike **{max_pain}** highlighted in green · "
            "P&L = total premium collected by ALL writers minus payout if market closes at each strike · "
            "Operator P&L = 70% of total (institutional writers)"
        )

        # 1. Update column names to indicate Crores
        display_cols = {
            "strikePrice":     "Strike",
            "CE_OI":           "CE OI",
            "PE_OI":           "PE OI",
            "CE_PnL":          "CE Writers P&L (₹ Cr)",
            "PE_PnL":          "PE Writers P&L (₹ Cr)",
            "Total_PnL":       "Total Writers P&L (₹ Cr)",
            "Operator_PnL_70": "Operator P&L 70% (₹ Cr)",
        }
        tbl_display = table_df[list(display_cols.keys())].rename(columns=display_cols).copy()
        tbl_display["Strike"] = tbl_display["Strike"].astype(int)

        # 2. Divide the P&L columns by 1 Crore (10,000,000)
        pnl_columns = ["CE Writers P&L (₹ Cr)", "PE Writers P&L (₹ Cr)", "Total Writers P&L (₹ Cr)", "Operator P&L 70% (₹ Cr)"]
        for col in pnl_columns:
            tbl_display[col] = tbl_display[col] / 10000000

        def color_pnl(val):
            if pd.isna(val):
                return ''
            if val > 0:
                return 'background-color: #d1e7dd; color: #0f5132' # Light Green
            elif val < 0:
                return 'background-color: #f8d7da; color: #842029' # Light Red
            return ''

        def highlight_max_pain(row):
            i     = tbl_display.index.get_loc(row.name)
            is_mp = table_df["isMaxPain"].iloc[i]
            if is_mp:
                return ["background-color:#c3e6cb; color:#155724; font-weight:700;"] * len(row)
            return [""] * len(row)

        # 3. Update formatting to show 2 decimal places for Crores
        fmt = {
            "CE OI": "{:,.0f}", "PE OI": "{:,.0f}",
            "CE Writers P&L (₹ Cr)": "{:+,.2f}", "PE Writers P&L (₹ Cr)": "{:+,.2f}",
            "Total Writers P&L (₹ Cr)": "{:+,.2f}", "Operator P&L 70% (₹ Cr)": "{:+,.2f}",
        }
        
        # 4. Apply the map subset to the new column names
        styled = (
            tbl_display.style
            .apply(highlight_max_pain, axis=1)
            .format(fmt)
            .map(color_pnl, subset=pnl_columns)
            .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "12px"})
        )
        st.dataframe(styled, width='stretch', height=440)

        # Download
        csv_dl = table_df.drop(columns=["isMaxPain"]).to_csv(index=False)
        ts_str = pd.Timestamp(df["datetime"].max()).strftime("%H%M")
        st.download_button(
            label="⬇ Download Table as CSV",
            data=csv_dl,
            file_name=f"maxpain_{sel_expiry_lbl}_{ts_str}.csv",
            mime="text/csv"
        )


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 4 — Max Pain Velocity & Acceleration
    # ══════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="tab-header">Max Pain Velocity & Acceleration Over Time</div>', unsafe_allow_html=True)

        t4c1, t4c2 = st.columns([2, 1])
        with t4c1:
            df_day4 = date_selector(df, key="t4_date")
        with t4c2:
            t4_resample = st.selectbox("OHLC bar size", ["5min", "10min", "15min", "30min"], index=1, key="t4_rs")

        if df_day4.empty:
            st.warning("No market-hours data for this date.")
            st.stop()

        hist = compute_max_pain_history(df_day4)

        if hist.empty or len(hist) < 3:
            st.info("Not enough snapshots yet to compute velocity. Need at least 3 data points.")
        else:
            # ── Derive interval in minutes from data ─────────────────────────
            intervals = hist["datetime"].diff().dt.total_seconds().dropna() / 60
            interval_min = round(intervals.median())

            hist["velocity"]     = hist["max_pain"].diff()          # pts per interval
            hist["acceleration"] = hist["velocity"].diff()          # pts per interval²

            # Optional smoothing
            smooth = st.slider("Smoothing window (snapshots)", 1, max(2, min(10, len(hist) // 2)), 1, key="smooth_vel")
            if smooth > 1:
                hist["velocity_s"]     = hist["velocity"].rolling(smooth, center=True).mean()
                hist["acceleration_s"] = hist["acceleration"].rolling(smooth, center=True).mean()
            else:
                hist["velocity_s"]     = hist["velocity"]
                hist["acceleration_s"] = hist["acceleration"]

            # ── Metrics row ──────────────────────────────────────────────────
            latest = hist.iloc[-1]
            m1, m2, m3, m4, m5 = st.columns(5)
            for col, (lbl, val, sub, color) in zip(
                [m1, m2, m3, m4, m5],
                [
                    ("Current Max Pain",  f"{int(latest['max_pain']):,}", "strike", MAX_COLOR),
                    ("Current Spot",      f"₹{latest['spot']:,.0f}",     "underlying", "#fd7e14"),
                    ("Velocity",          f"{latest['velocity_s']:+.0f}" if pd.notna(latest['velocity_s']) else "—",
                                          f"pts / {interval_min:.0f}m",
                                          GAIN_COLOR if latest['velocity_s'] >= 0 else LOSS_COLOR),
                    ("Acceleration",      f"{latest['acceleration_s']:+.0f}" if pd.notna(latest['acceleration_s']) else "—",
                                          "Δ velocity",
                                          GAIN_COLOR if pd.notna(latest['acceleration_s']) and latest['acceleration_s'] >= 0 else LOSS_COLOR),
                    ("# Snapshots",       str(len(hist)), f"every ~{interval_min:.0f}m", "#6c757d"),
                ]
            ):
                col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Chart 1: Max Pain OHLC + Spot line ───────────────────────────
            hist_idx = hist.set_index("datetime")
            mp_ohlc  = hist_idx["max_pain"].resample(t4_resample).ohlc().dropna()
            sp_ohlc  = hist_idx["spot"].resample(t4_resample).last().dropna()

            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(
                x=mp_ohlc.index,
                open=mp_ohlc["open"], high=mp_ohlc["high"],
                low=mp_ohlc["low"],   close=mp_ohlc["close"],
                name="Max Pain",
                increasing_line_color=GAIN_COLOR,
                decreasing_line_color=LOSS_COLOR,
            ))
            fig1.add_trace(go.Scatter(
                x=sp_ohlc.index, y=sp_ohlc.values,
                mode="lines", name="Spot",
                line=dict(color="#fd7e14", width=1.5, dash="dot"),
                hovertemplate="%{x|%H:%M}<br>Spot: <b>₹%{y:,.0f}</b><extra></extra>"
            ))
            fig1.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Max Pain OHLC ({t4_resample}) vs Spot",
                           font=dict(color="#212529", size=14)),
                xaxis_title="Time", yaxis_title="Price (₹)",
                xaxis_rangeslider_visible=False,
                height=480,
            )
            st.plotly_chart(fig1, width='stretch')

            # ── Chart 2: Velocity ─────────────────────────────────────────────
            vel_colors = [GAIN_COLOR if v >= 0 else LOSS_COLOR
                          for v in hist["velocity"].fillna(0)]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=hist["datetime"], y=hist["velocity"],
                name="Velocity (raw)", marker_color=vel_colors, opacity=0.45,
                hovertemplate="%{x|%H:%M}<br>Velocity: <b>%{y:+.0f} pts</b><extra></extra>"
            ))
            if smooth > 1:
                fig2.add_trace(go.Scatter(
                    x=hist["datetime"], y=hist["velocity_s"],
                    mode="lines", name=f"Velocity ({smooth}-snap avg)",
                    line=dict(color=MAX_COLOR, width=2),
                    hovertemplate="%{x|%H:%M}<br>Smoothed: <b>%{y:+.2f} pts</b><extra></extra>"
                ))
            fig2.add_hline(y=0, line_color="#dee2e6", line_width=1)
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Max Pain Velocity  (Δ per {interval_min:.0f}-min interval)",
                           font=dict(color="#212529", size=14)),
                xaxis_title="Time", yaxis_title="Velocity (pts)", height=280,
            )
            st.plotly_chart(fig2, width='stretch')

            # ── Chart 3: Acceleration ─────────────────────────────────────────
            acc_colors = [GAIN_COLOR if a >= 0 else LOSS_COLOR
                          for a in hist["acceleration"].fillna(0)]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=hist["datetime"], y=hist["acceleration"],
                name="Acceleration (raw)", marker_color=acc_colors, opacity=0.45,
                hovertemplate="%{x|%H:%M}<br>Accel: <b>%{y:+.0f} pts²</b><extra></extra>"
            ))
            if smooth > 1:
                fig3.add_trace(go.Scatter(
                    x=hist["datetime"], y=hist["acceleration_s"],
                    mode="lines", name=f"Acceleration ({smooth}-snap avg)",
                    line=dict(color="#6f42c1", width=2),
                    hovertemplate="%{x|%H:%M}<br>Smoothed: <b>%{y:+.2f} pts²</b><extra></extra>"
                ))
            fig3.add_hline(y=0, line_color="#dee2e6", line_width=1)
            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Max Pain Acceleration  (Δ velocity per {interval_min:.0f}-min interval)",
                           font=dict(color="#212529", size=14)),
                xaxis_title="Time", yaxis_title="Acceleration (pts²)", height=280,
            )
            st.plotly_chart(fig3, width='stretch')

            # ── Raw data table ────────────────────────────────────────────────
            with st.expander("Raw history data"):
                disp = hist[["datetime", "max_pain", "spot", "velocity", "acceleration"]].copy()
                disp["datetime"]    = disp["datetime"].dt.strftime("%H:%M:%S")
                disp["velocity"]    = disp["velocity"].map(lambda x: f"{x:+.0f}" if pd.notna(x) else "—")
                disp["acceleration"]= disp["acceleration"].map(lambda x: f"{x:+.0f}" if pd.notna(x) else "—")
                st.dataframe(disp, width='stretch', height=300)


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 5 — Operator Trap Detector
    # ══════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="tab-header">Operator Trap Detector — Signal Analysis & Verdict</div>',
                    unsafe_allow_html=True)

        df_day5 = date_selector(df, key="t5_date")
        if df_day5.empty:
            st.warning("No market-hours data for this date.")
            st.stop()

        df_snap5  = df_day5[df_day5["datetime"] == df_day5["datetime"].max()].copy()
        pcr_hist5 = compute_pcr_history(df_day5)
        mp_hist5  = compute_max_pain_history(df_day5)
        fut_df5   = load_futures_data()
        sig       = compute_trap_signals(df_snap5, pcr_hist5, mp_hist5, fut_df=fut_df5)

        # ── Top metrics ──────────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        pcr_color = (GAIN_COLOR if sig["pcr"] > 1.2 else
                     LOSS_COLOR if sig["pcr"] < 0.85 else "#6c757d")
        div_color = (LOSS_COLOR if sig["divergence_pct"] > 0 else
                     GAIN_COLOR if sig["divergence_pct"] < 0 else "#6c757d")
        for col, (lbl, val, sub, clr) in zip(
            [mc1, mc2, mc3, mc4, mc5],
            [
                ("PCR",             f"{sig['pcr']:.3f}",
                 "Put / Call OI ratio",                    pcr_color),
                ("Max Pain Gap",    f"{sig['divergence_pct']:+.2f}%",
                 f"Spot vs MP {sig['max_pain']:,}",         div_color),
                ("CE Wall",         f"{sig['ce_wall']:,}",
                 f"{sig['ce_wall_dist_pct']:+.1f}% from spot", CE_COLOR),
                ("PE Wall",         f"{sig['pe_wall']:,}",
                 f"{sig['pe_wall_dist_pct']:+.1f}% from spot", PE_COLOR),
                ("ATM CE/PE Ratio", f"{sig['atm_skew_ratio']:.2f}×",
                 f"ATM strike {sig['atm']:,}",             "#6f42c1"),
            ]
        ):
            col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{clr};">{val}</div>
            <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        # ── Futures metrics row ───────────────────────────────────────────────────
        fm1, fm2, fm3, fm4, fm5 = st.columns(5)
        _no_fut = sig["fut_close"] is None
        basis_clr = ("#6c757d" if _no_fut else
                     LOSS_COLOR if sig["basis_pct"] <= -0.2 else
                     GAIN_COLOR if sig["basis_pct"] >= 0.2 else "#6c757d")
        vol_clr   = ("#6c757d" if _no_fut else
                     LOSS_COLOR if sig["vol_ratio"] >= 2.5 else
                     "#fd7e14" if sig["vol_ratio"] >= 1.5 else "#6c757d")
        for col, (lbl, val, sub, clr) in zip(
            [fm1, fm2, fm3, fm4, fm5],
            [
                ("Futures Price",
                 f"₹{sig['fut_close']:,.2f}" if not _no_fut else "—",
                 "Latest futures close", "#17a2b8"),
                ("Futures Basis",
                 f"{sig['basis_pct']:+.3f}%" if not _no_fut else "—",
                 f"₹{sig['basis_val']:+.1f} (fut − spot)" if not _no_fut else "No data",
                 basis_clr),
                ("Futures OI",
                 f"{int(sig['fut_oi_now']):,}" if sig.get('fut_oi_now') is not None else "—",
                 (f"Δ {int(sig['fut_oi_delta']):+,}" if sig.get('fut_oi_delta') is not None else "—"),
                 GAIN_COLOR if (sig.get('fut_oi_delta') or 0) > 0 else
                 LOSS_COLOR if (sig.get('fut_oi_delta') or 0) < 0 else "#6c757d"),
                ("Futures Volume",
                 f"{int(sig['fut_vol']):,}" if sig["fut_vol"] is not None else "—",
                 f"5-bar avg {int(sig['fut_vol_avg']):,}" if sig["fut_vol_avg"] is not None else "—",
                 vol_clr),
                ("Vol Spike Ratio",
                 f"{sig['vol_ratio']:.2f}×" if not _no_fut else "—",
                 "vs 5-bar average", vol_clr),
            ]
        ):
            col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{clr};">{val}</div>
            <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Charts row: PCR history | Max Pain divergence history ────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            if not pcr_hist5.empty:
                fig_pcr = go.Figure()
                fig_pcr.add_trace(go.Scatter(
                    x=pcr_hist5["datetime"], y=pcr_hist5["pcr"],
                    mode="lines+markers", name="PCR",
                    line=dict(color="#6f42c1", width=2), marker=dict(size=5),
                    hovertemplate="%{x|%H:%M}<br>PCR: <b>%{y:.3f}</b><extra></extra>"
                ))
                fig_pcr.add_hline(y=1.2,  line_dash="dash", line_color=GAIN_COLOR,
                                  annotation_text="1.2 Bear-trap zone",
                                  annotation_font_color=GAIN_COLOR)
                fig_pcr.add_hline(y=0.85, line_dash="dash", line_color=LOSS_COLOR,
                                  annotation_text="0.85 Bull-trap zone",
                                  annotation_font_color=LOSS_COLOR)
                fig_pcr.add_hline(y=1.0,  line_dash="dot",  line_color="#adb5bd",
                                  annotation_text="Neutral 1.0",
                                  annotation_font_color="#adb5bd")
                fig_pcr.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="Put-Call Ratio (PCR) — Intraday History",
                               font=dict(color="#212529", size=14)),
                    xaxis_title="Time", yaxis_title="PCR", height=360)
                st.plotly_chart(fig_pcr, width='stretch')
            else:
                st.info("Not enough snapshots for PCR history.")

        with ch2:
            if not mp_hist5.empty:
                div_series  = (mp_hist5["spot"] - mp_hist5["max_pain"]) / mp_hist5["max_pain"] * 100
                bar_clrs    = [LOSS_COLOR if v > 0 else GAIN_COLOR for v in div_series]
                fig_div = go.Figure()
                fig_div.add_trace(go.Bar(
                    x=mp_hist5["datetime"], y=div_series,
                    name="Spot−MaxPain %", marker_color=bar_clrs, opacity=0.82,
                    hovertemplate="%{x|%H:%M}<br>Gap: <b>%{y:+.2f}%</b><extra></extra>"
                ))
                fig_div.add_hline(y=0,    line_color="#dee2e6", line_width=1.5)
                fig_div.add_hline(y=1.0,  line_dash="dash", line_color=LOSS_COLOR,
                                  annotation_text="+1% Bull-trap zone",
                                  annotation_font_color=LOSS_COLOR)
                fig_div.add_hline(y=-1.0, line_dash="dash", line_color=GAIN_COLOR,
                                  annotation_text="−1% Bear-trap zone",
                                  annotation_font_color=GAIN_COLOR)
                fig_div.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="Spot vs Max Pain Divergence % (History)",
                               font=dict(color="#212529", size=14)),
                    xaxis_title="Time",
                    yaxis_title="(Spot − MaxPain) / MaxPain %",
                    height=360)
                st.plotly_chart(fig_div, width='stretch')

        # ── Futures Basis + Volume chart ──────────────────────────────────────────
        if not fut_df5.empty:
            snap_date5 = pd.Timestamp(df_day5["datetime"].max()).date()
            fut_day5   = fut_df5[fut_df5["datetime"].dt.date == snap_date5].sort_values("datetime")
            if not fut_day5.empty:
                st.markdown("---")
                st.markdown("#### Futures Basis & Volume — Intraday")
                fc1, fc2 = st.columns(2)

                with fc1:
                    # Basis line: need spot at each futures bar time
                    # Approximate spot from max pain history (spot column)
                    if not mp_hist5.empty:
                        mp_idx = mp_hist5.set_index("datetime")["spot"]
                        basis_vals = []
                        for _, row in fut_day5.iterrows():
                            # nearest spot at or before this bar
                            sp_before = mp_idx[mp_idx.index <= row["datetime"]]
                            s = float(sp_before.iloc[-1]) if not sp_before.empty else float(row["close"])
                            basis_vals.append(row["close"] - s)
                    else:
                        basis_vals = [0.0] * len(fut_day5)

                    b_clrs = [LOSS_COLOR if v < 0 else GAIN_COLOR for v in basis_vals]
                    fig_bas = go.Figure()
                    fig_bas.add_trace(go.Bar(
                        x=fut_day5["datetime"], y=basis_vals,
                        name="Basis (Fut−Spot)", marker_color=b_clrs, opacity=0.8,
                        hovertemplate="%{x|%H:%M}<br>Basis: <b>₹%{y:+.1f}</b><extra></extra>"
                    ))
                    fig_bas.add_hline(y=0, line_color="#dee2e6", line_width=1.5)
                    fig_bas.update_layout(**PLOTLY_LAYOUT,
                        title=dict(text="Futures Basis (Futures − Spot) — Backwardation = Bearish",
                                   font=dict(color="#212529", size=13)),
                        xaxis_title="Time", yaxis_title="Basis (₹)", height=320)
                    st.plotly_chart(fig_bas, width='stretch')

                with fc2:
                    vol_avg_line = fut_day5["volume"].expanding().mean()
                    v_clrs = [LOSS_COLOR if v >= 2.5 * a else
                              "#fd7e14"  if v >= 1.5 * a else CE_COLOR
                              for v, a in zip(fut_day5["volume"], vol_avg_line)]
                    if "oi" in fut_day5.columns:
                        fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                row_heights=[0.55, 0.45],
                                                vertical_spacing=0.06,
                                                subplot_titles=["Futures Volume", "Futures OI"])
                        fig_vol.add_trace(go.Bar(
                            x=fut_day5["datetime"], y=fut_day5["volume"],
                            name="Volume", marker_color=v_clrs, opacity=0.8,
                            hovertemplate="%{x|%H:%M}<br>Vol: <b>%{y:,}</b><extra></extra>"
                        ), row=1, col=1)
                        fig_vol.add_trace(go.Scatter(
                            x=fut_day5["datetime"], y=vol_avg_line,
                            mode="lines", name="Vol Avg",
                            line=dict(color="#6f42c1", width=2, dash="dot"),
                            hovertemplate="%{x|%H:%M}<br>Avg: <b>%{y:,.0f}</b><extra></extra>"
                        ), row=1, col=1)
                        oi_delta_colors = [GAIN_COLOR if d >= 0 else LOSS_COLOR
                                           for d in fut_day5["oi"].diff().fillna(0)]
                        fig_vol.add_trace(go.Bar(
                            x=fut_day5["datetime"], y=fut_day5["oi"],
                            name="Futures OI", marker_color=oi_delta_colors, opacity=0.75,
                            hovertemplate="%{x|%H:%M}<br>OI: <b>%{y:,}</b><extra></extra>"
                        ), row=2, col=1)
                    else:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(
                            x=fut_day5["datetime"], y=fut_day5["volume"],
                            name="Volume", marker_color=v_clrs, opacity=0.8,
                            hovertemplate="%{x|%H:%M}<br>Vol: <b>%{y:,}</b><extra></extra>"
                        ))
                        fig_vol.add_trace(go.Scatter(
                            x=fut_day5["datetime"], y=vol_avg_line,
                            mode="lines", name="Cumulative Avg",
                            line=dict(color="#6f42c1", width=2, dash="dot"),
                            hovertemplate="%{x|%H:%M}<br>Avg: <b>%{y:,.0f}</b><extra></extra>"
                        ))
                    fig_vol.update_layout(**PLOTLY_LAYOUT,
                        title=dict(
                            text="Futures Volume + OI — Red=2.5× spike, Orange=1.5× elevated",
                            font=dict(color="#212529", size=13)),
                        xaxis_title="Time", height=380,
                        xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_vol, width='stretch')

        # ── OI Wall Visualization ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### OI Distribution — CE Wall, PE Wall & Spot")

        TOP_N       = 20
        ce_top      = (df_snap5[df_snap5["optionType"] == "Call"]
                       .groupby("strikePrice")["openInterest"].max()
                       .nlargest(TOP_N))
        pe_top      = (df_snap5[df_snap5["optionType"] == "Put"]
                       .groupby("strikePrice")["openInterest"].max()
                       .nlargest(TOP_N))
        wall_stk    = sorted(set(ce_top.index) | set(pe_top.index))
        ce_vals     = [float(ce_top.get(s, 0)) for s in wall_stk]
        pe_vals     = [float(pe_top.get(s, 0)) for s in wall_stk]
        ce_clrs_w   = [MAX_COLOR if s == sig["ce_wall"] else CE_COLOR for s in wall_stk]
        pe_clrs_w   = ["#20c997"  if s == sig["pe_wall"] else PE_COLOR for s in wall_stk]

        fig_wall = go.Figure()
        fig_wall.add_trace(go.Bar(
            x=wall_stk, y=ce_vals, name="CE OI",
            marker_color=ce_clrs_w, opacity=0.85,
            hovertemplate="Strike %{x:,}<br>CE OI: <b>%{y:,.0f}</b><extra></extra>"
        ))
        fig_wall.add_trace(go.Bar(
            x=wall_stk, y=pe_vals, name="PE OI",
            marker_color=pe_clrs_w, opacity=0.85,
            hovertemplate="Strike %{x:,}<br>PE OI: <b>%{y:,.0f}</b><extra></extra>"
        ))
        fig_wall.add_vline(x=sig["spot"],     line_dash="dot",      line_color="#fd7e14",
                           annotation_text=f"Spot {int(sig['spot']):,}",
                           annotation_font_color="#fd7e14")
        fig_wall.add_vline(x=sig["max_pain"], line_dash="dash",     line_color=MAX_COLOR,
                           annotation_text=f"Max Pain {sig['max_pain']:,}",
                           annotation_font_color=MAX_COLOR)
        fig_wall.add_vline(x=sig["ce_wall"],  line_dash="longdash", line_color=CE_COLOR,
                           annotation_text=f"CE Wall {sig['ce_wall']:,}",
                           annotation_font_color=CE_COLOR,
                           annotation_position="top left")
        fig_wall.add_vline(x=sig["pe_wall"],  line_dash="longdash", line_color=PE_COLOR,
                           annotation_text=f"PE Wall {sig['pe_wall']:,}",
                           annotation_font_color=PE_COLOR,
                           annotation_position="top left")
        fig_wall.update_layout(**(PLOTLY_LAYOUT | dict(
            barmode="group",
            title=dict(
                text="Top OI Strikes — CE Wall (★ highlighted) & PE Wall (★ highlighted)",
                font=dict(color="#212529", size=14)),
            xaxis=dict(title="Strike Price", tickmode="array",
                       tickvals=wall_stk,
                       ticktext=[f"{int(s):,}" for s in wall_stk],
                       tickangle=-45, gridcolor="#e9ecef"),
            yaxis=dict(title="Open Interest", tickformat=",", gridcolor="#e9ecef"),
            height=430,
        )))
        st.plotly_chart(fig_wall, width='stretch')

        # ── Signal Scorecard ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Signal Scorecard")
        st.caption("Each signal contributes an independent score toward the final verdict. "
                   "Scores ≥ 60 are highlighted as strong. Threshold for verdict: combined confidence ≥ 60%.")

        scorecard = pd.DataFrame([
            {
                "Signal":           "Put-Call Ratio (PCR)",
                "Value":            f"{sig['pcr']:.3f}",
                "Bull Trap Score":  sig["pcr_bull_score"],
                "Bear Trap Score":  sig["pcr_bear_score"],
                "Bias":             sig["pcr_bias"],
                "Interpretation":   sig["pcr_signal"],
            },
            {
                "Signal":           "Max Pain Divergence",
                "Value":            f"{sig['divergence_pct']:+.2f}%",
                "Bull Trap Score":  sig["div_bull_score"],
                "Bear Trap Score":  sig["div_bear_score"],
                "Bias":             sig["div_bias"],
                "Interpretation":   sig["div_signal"],
            },
            {
                "Signal":           "OI Wall Proximity",
                "Value":            f"CE {sig['ce_wall_dist_pct']:+.1f}% / PE {sig['pe_wall_dist_pct']:+.1f}%",
                "Bull Trap Score":  sig["wall_bull_score"],
                "Bear Trap Score":  sig["wall_bear_score"],
                "Bias":             sig["wall_bias"],
                "Interpretation":   sig["wall_signal"],
            },
            {
                "Signal":           "ATM OI Skew (CE÷PE)",
                "Value":            f"{sig['atm_skew_ratio']:.2f}×",
                "Bull Trap Score":  sig["skew_bull_score"],
                "Bear Trap Score":  sig["skew_bear_score"],
                "Bias":             sig["skew_bias"],
                "Interpretation":   sig["skew_signal"],
            },
            {
                "Signal":           "PCR Trend (5-snap Δ)",
                "Value":            f"{sig['pcr_trend_val']:+.3f}",
                "Bull Trap Score":  sig["trend_bull_score"],
                "Bear Trap Score":  sig["trend_bear_score"],
                "Bias":             sig["trend_bias"],
                "Interpretation":   sig["trend_signal"],
            },
            {
                "Signal":           "Max Pain Velocity",
                "Value":            "—",
                "Bull Trap Score":  sig["vel_bull_score"],
                "Bear Trap Score":  sig["vel_bear_score"],
                "Bias":             sig["vel_bias"],
                "Interpretation":   sig["vel_signal"],
            },
            {
                "Signal":           "Futures Basis",
                "Value":            (f"{sig['basis_pct']:+.3f}%"
                                     if sig["fut_close"] is not None else "—"),
                "Bull Trap Score":  sig["basis_bull_score"],
                "Bear Trap Score":  sig["basis_bear_score"],
                "Bias":             sig["basis_bias"],
                "Interpretation":   sig["basis_signal"],
            },
            {
                "Signal":           "Futures Vol Spike",
                "Value":            (f"{sig['vol_ratio']:.2f}×"
                                     if sig["fut_close"] is not None else "—"),
                "Bull Trap Score":  sig["vol_bull_score"],
                "Bear Trap Score":  sig["vol_bear_score"],
                "Bias":             sig["vol_bias"],
                "Interpretation":   sig["vol_signal"],
            },
            {
                "Signal":           "━━ TOTAL ━━",
                "Value":            "",
                "Bull Trap Score":  sig["total_bull_score"],
                "Bear Trap Score":  sig["total_bear_score"],
                "Bias":             ("BULL TRAP" if sig["bull_confidence"] >= sig["bear_confidence"]
                                     else "BEAR TRAP"),
                "Interpretation":   (f"Bull confidence: {sig['bull_confidence']}%  |  "
                                     f"Bear confidence: {sig['bear_confidence']}%"),
            },
        ])

        def _style_bias(val):
            s = str(val)
            if "BULL TRAP" in s:
                return "background-color:#f8d7da; color:#842029; font-weight:600;"
            if "BEAR TRAP" in s:
                return "background-color:#d1e7dd; color:#0f5132; font-weight:600;"
            if s == "NEUTRAL":
                return "color:#6c757d;"
            if s == "DIVERGING":
                return "color:#6f42c1;"
            return ""

        def _style_score(val):
            if isinstance(val, (int, float)):
                if val >= 60:
                    return "font-weight:700; color:#dc3545;"
                if val >= 30:
                    return "font-weight:600; color:#fd7e14;"
            return ""

        def _highlight_total(row):
            if str(row["Signal"]).startswith("━"):
                return ["font-weight:700; border-top:2px solid #dee2e6;"] * len(row)
            return [""] * len(row)

        styled_sc = (
            scorecard.style
            .apply(_highlight_total, axis=1)
            .map(_style_bias,  subset=["Bias"])
            .map(_style_score, subset=["Bull Trap Score", "Bear Trap Score"])
            .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "12px"})
        )
        st.dataframe(styled_sc, width='stretch', height=390)

        # ── Final Verdict ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Final Verdict")

        st.markdown(f"""
        <div style="
            background:{sig['verdict_bg']};
            border:2px solid {sig['verdict_color']};
            border-radius:16px;
            padding:28px 32px;
            text-align:center;
        ">
            <div style="font-size:28px; font-weight:800;
                        color:{sig['verdict_color']};
                        font-family:'Sora',sans-serif; margin-bottom:8px;">
                {sig['verdict']}
            </div>
            <div style="font-size:15px; color:{sig['verdict_color']};
                        opacity:0.85; font-weight:500; margin-bottom:12px;">
                Confidence: <b>{sig['final_confidence']}%</b>
            </div>
            <div style="font-size:13px; color:#495057; line-height:1.6;">
                {sig['verdict_detail']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        gc1, gc2 = st.columns(2)
        for gcol, (trap_lbl, conf_val, bar_clr, steps_clrs) in zip(
            [gc1, gc2],
            [
                ("Bull Trap Confidence", sig["bull_confidence"], LOSS_COLOR,
                 [{"range": [0, 40],  "color": "#f8f9fa"},
                  {"range": [40, 60], "color": "#fff3cd"},
                  {"range": [60, 100],"color": "#f8d7da"}]),
                ("Bear Trap Confidence", sig["bear_confidence"], GAIN_COLOR,
                 [{"range": [0, 40],  "color": "#f8f9fa"},
                  {"range": [40, 60], "color": "#fff3cd"},
                  {"range": [60, 100],"color": "#d1e7dd"}]),
            ]
        ):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_val,
                title={"text": trap_lbl, "font": {"color": bar_clr, "size": 15}},
                gauge={
                    "axis":        {"range": [0, 100], "tickcolor": "#495057",
                                    "tickfont": {"size": 12}},
                    "bar":         {"color": bar_clr, "thickness": 0.25},
                    "bgcolor":     "#ffffff",
                    "bordercolor": "#dee2e6",
                    "borderwidth": 2,
                    "steps":       steps_clrs,
                    "threshold":   {"line": {"color": bar_clr, "width": 4}, "value": 60},
                },
                number={"suffix": "%", "font": {"color": bar_clr, "size": 36}},
            ))
            fig_g.update_layout(
                paper_bgcolor="#ffffff",
                font=dict(family="Sora, sans-serif", color="#495057"),
                height=340,
                margin=dict(l=30, r=30, t=60, b=20),
            )
            gcol.plotly_chart(fig_g, width='stretch')

        st.caption(
            "Build in India."
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 6 — Live Market Data (Angel One SmartAPI)
    # ══════════════════════════════════════════════════════════════════════════════
    with tab6:
        render_live_data_tab()
