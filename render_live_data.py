"""
render_live_data.py
Tab 6 — Live Market Data (Angel One SmartAPI)

Fetches per refresh (4 sequential calls, 0.4s apart → well under 3 req/s):
  1. Nifty 50 Spot  OHLCV  (NSE, token 99926000)
  2. Nifty Futures  OHLCV  (NFO, near-month token from scrip master)
  3. India VIX      OHLCV  (NSE, token 99919000)
  4. Bank Nifty     OHLCV  (NSE, token 99926009)  — extra context

Auto-saves futures candles to futuresData/nifty50/ in the same JSON format
that the Operator Trap Detector already reads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import json

from api_wrapper import (get_api_client, get_future_token,
                         fetch_candle_data, get_seconds_until_8am)

# ── Angel One token constants ─────────────────────────────────────────────────
TOKEN_NIFTY_SPOT   = "99926000"
TOKEN_INDIA_VIX    = "99919000"
TOKEN_BANKNIFTY    = "99926009"
EXCHANGE_NSE       = "NSE"
EXCHANGE_NFO       = "NFO"

# ── Futures output path (matches Operator Trap Detector) ─────────────────────
FUTURES_JSON_PATH = os.path.join("futuresData", "nifty50",
                                 "nifty50_futures_apr_2026.json")

# ── UI interval map ───────────────────────────────────────────────────────────
INTERVAL_MAP = {
    "5 min":  "FIVE_MINUTE",
    "10 min": "TEN_MINUTE",
    "15 min": "FIFTEEN_MINUTE",
    "30 min": "THIRTY_MINUTE",
    "1 hour": "ONE_HOUR",
    "1 day":  "ONE_DAY",
}

# Reuse layout/colours from parent module
CE_COLOR   = "#007bff"
PE_COLOR   = "#dc3545"
MAX_COLOR  = "#28a745"
GAIN_COLOR = "#28a745"
LOSS_COLOR = "#dc3545"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(family="Sora, sans-serif", color="#495057", size=12),
    xaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6", showgrid=True),
    yaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6", showgrid=True),
    legend=dict(bgcolor="#ffffff", bordercolor="#dee2e6", borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=60),
)


# ═════════════════════════════════════════════════════════════════════════════
# Authentication — cached until next 8 AM (session valid all day)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(ttl=get_seconds_until_8am())   # recalculated at module load = until next 8 AM
def _get_client():
    api_key    = st.secrets["API_KEY"]
    client_id  = st.secrets["CLIENT_ID"]
    mpin       = st.secrets["MPIN"]
    totp_seed  = st.secrets["TOTP_SEED"]
    return get_api_client(api_key, client_id, mpin, totp_seed)


@st.cache_data(ttl=3600)           # scrip master rarely changes intraday
def _get_futures_token(index_name: str = "NIFTY") -> dict:
    return get_future_token(index_name)


# ═════════════════════════════════════════════════════════════════════════════
# Core fetch — single function, all 4 calls, sequential with 0.4 s gap
# ═════════════════════════════════════════════════════════════════════════════
def _make_params(token: str, exchange: str, interval: str,
                 from_dt: datetime, to_dt: datetime) -> dict:
    return {
        "exchange":    exchange,
        "symboltoken": token,
        "interval":    interval,
        "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
    }


def _candles_to_df(candles: list, extra_col: str = None) -> pd.DataFrame:
    """Convert Angel One candle list → DataFrame with datetime index."""
    if not candles:
        return pd.DataFrame()
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if extra_col:
        cols.append(extra_col)
    df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    for c in cols[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)   # 5-minute data cache
def fetch_all_market_data(interval: str, lookback_days: int,
                          futures_token: str) -> dict:
    """
    Fetch spot, futures, VIX, Bank Nifty in one call with 0.4 s spacing.
    Returns dict of DataFrames keyed by name.
    TTL=300 s → data auto-expires every 5 minutes.
    """
    client = _get_client()
    if client is None:
        return {}

    to_dt   = datetime.now()
    from_dt = to_dt - timedelta(days=lookback_days)

    results = {}
    fetch_plan = [
        ("spot",      TOKEN_NIFTY_SPOT, EXCHANGE_NSE),
        ("futures",   futures_token,    EXCHANGE_NFO),
        ("vix",       TOKEN_INDIA_VIX,  EXCHANGE_NSE),
        ("banknifty", TOKEN_BANKNIFTY,  EXCHANGE_NSE),
    ]

    for name, token, exchange in fetch_plan:
        params = _make_params(token, exchange, interval, from_dt, to_dt)
        raw    = fetch_candle_data(client, params)
        if raw and raw.get("data"):
            results[name] = _candles_to_df(raw["data"])
        else:
            results[name] = pd.DataFrame()
        time.sleep(0.4)          # ← rate-limit guard: ≤ 2.5 req/s

    # Derive basis where both exist
    if not results["spot"].empty and not results["futures"].empty:
        spot_cl = (results["spot"]
                   .set_index("datetime")["close"]
                   .rename("spot_close"))
        fut_cl  = (results["futures"]
                   .set_index("datetime")["close"]
                   .rename("fut_close"))
        basis   = pd.concat([spot_cl, fut_cl], axis=1).dropna()
        basis["basis"]     = basis["fut_close"] - basis["spot_close"]
        basis["basis_pct"] = basis["basis"] / basis["spot_close"] * 100
        results["basis"]   = basis.reset_index()

    results["fetched_at"] = datetime.now().strftime("%H:%M:%S")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Technical indicators
# ═════════════════════════════════════════════════════════════════════════════
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI-14
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # VWAP (resets each calendar day)
    if "volume" in df.columns:
        df["_typical"] = (df["high"] + df["low"] + df["close"]) / 3
        df["_date"]    = df["datetime"].dt.date
        df["_cum_tv"]  = df.groupby("_date").apply(
            lambda g: (g["_typical"] * g["volume"]).cumsum()
        ).reset_index(level=0, drop=True)
        df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()
        df["vwap"]     = df["_cum_tv"] / df["_cum_vol"]
        df.drop(columns=["_typical", "_date", "_cum_tv", "_cum_vol"],
                inplace=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Chart builders
# ═════════════════════════════════════════════════════════════════════════════
def _ohlc_chart(df: pd.DataFrame, title: str,
                show_vwap: bool = True,
                show_oi: bool = False,
                height: int = 500) -> go.Figure:
    """Candlestick + EMAs + VWAP + Volume (+ OI if present)."""
    rows      = 3 if (show_oi and "oi" in df.columns) else 2
    row_h     = [0.55, 0.25, 0.20] if rows == 3 else [0.65, 0.35]
    sub_titles = (["Price", "Volume", "Futures OI"] if rows == 3
                  else ["Price", "Volume"])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_h, vertical_spacing=0.03,
                        subplot_titles=sub_titles)

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="OHLC",
        increasing_line_color=GAIN_COLOR,
        decreasing_line_color=LOSS_COLOR,
    ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    for span, color, dash in [(9, "#fd7e14", "solid"),
                               (21, "#6f42c1", "solid"),
                               (50, "#17a2b8", "dash")]:
        col_name = f"ema{span}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col_name],
                mode="lines", name=f"EMA{span}",
                line=dict(color=color, width=1.5, dash=dash),
                hovertemplate=f"EMA{span}: <b>%{{y:,.1f}}</b><extra></extra>",
            ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if show_vwap and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["vwap"],
            mode="lines", name="VWAP",
            line=dict(color="#e83e8c", width=1.5, dash="dot"),
            hovertemplate="VWAP: <b>%{y:,.1f}</b><extra></extra>",
        ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────────
    if "volume" in df.columns:
        vol_clrs = [GAIN_COLOR if c >= o else LOSS_COLOR
                    for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["volume"],
            name="Volume", marker_color=vol_clrs, opacity=0.7,
            hovertemplate="%{x|%d-%b %H:%M}<br>Vol: <b>%{y:,}</b><extra></extra>",
        ), row=2, col=1)

    # ── Futures OI ───────────────────────────────────────────────────────────
    if show_oi and "oi" in df.columns and rows == 3:
        oi_delta_clrs = [GAIN_COLOR if d >= 0 else LOSS_COLOR
                         for d in df["oi"].diff().fillna(0)]
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["oi"],
            name="Futures OI", marker_color=oi_delta_clrs, opacity=0.75,
            hovertemplate="%{x|%d-%b %H:%M}<br>OI: <b>%{y:,}</b><extra></extra>",
        ), row=3, col=1)

    fig.update_layout(**(PLOTLY_LAYOUT | dict(
        title=dict(text=title, font=dict(color="#212529", size=14)),
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )))
    fig.update_xaxes(tickformat="%d-%b\n%H:%M", gridcolor="#e9ecef")
    fig.update_yaxes(tickformat=",", gridcolor="#e9ecef")
    return fig


def _rsi_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if "rsi14" not in df.columns:
        return fig
    rsi_clrs = [LOSS_COLOR if v >= 70 else GAIN_COLOR if v <= 30
                else "#6c757d" for v in df["rsi14"].fillna(50)]
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["rsi14"],
        mode="lines", name="RSI 14",
        line=dict(color="#6f42c1", width=2),
        hovertemplate="%{x|%d-%b %H:%M}<br>RSI: <b>%{y:.1f}</b><extra></extra>",
    ))
    for level, label, color in [
        (70, "Overbought", LOSS_COLOR),
        (30, "Oversold",   GAIN_COLOR),
        (50, "Mid",        "#adb5bd"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="right")
    fig.update_layout(**(PLOTLY_LAYOUT | dict(
        title=dict(text=title, font=dict(color="#212529", size=13)),
        yaxis=dict(range=[0, 100], gridcolor="#e9ecef"),
        height=220,
    )))
    return fig


def _basis_chart(basis_df: pd.DataFrame) -> go.Figure:
    if basis_df.empty:
        return go.Figure()
    b_clrs = [LOSS_COLOR if v < 0 else GAIN_COLOR for v in basis_df["basis"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=basis_df["datetime"], y=basis_df["basis"],
        name="Basis (₹)", marker_color=b_clrs, opacity=0.8,
        hovertemplate="%{x|%d-%b %H:%M}<br>Basis: <b>₹%{y:+.1f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=basis_df["datetime"], y=basis_df["basis_pct"],
        mode="lines", name="Basis %",
        yaxis="y2",
        line=dict(color="#6f42c1", width=1.5, dash="dot"),
        hovertemplate="%{x|%d-%b %H:%M}<br>Basis%%: <b>%{y:+.3f}%%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#dee2e6", line_width=1.5)
    fig.update_layout(**(PLOTLY_LAYOUT | dict(
        title=dict(text="Futures Basis — Futures minus Spot  (negative = backwardation)",
                   font=dict(color="#212529", size=14)),
        yaxis=dict(title="Basis (₹)", gridcolor="#e9ecef"),
        yaxis2=dict(title="Basis %", overlaying="y", side="right",
                    showgrid=False, tickformat=".3f"),
        height=300,
    )))
    return fig


def _vix_chart(vix_df: pd.DataFrame) -> go.Figure:
    if vix_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vix_df["datetime"], y=vix_df["close"],
        mode="lines+markers", name="India VIX",
        line=dict(color="#fd7e14", width=2),
        marker=dict(size=4),
        hovertemplate="%{x|%d-%b %H:%M}<br>VIX: <b>%{y:.2f}</b><extra></extra>",
    ))
    for level, label, color in [
        (25, "High fear (>25)", LOSS_COLOR),
        (15, "Low fear (<15)", GAIN_COLOR),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="right")
    fig.update_layout(**(PLOTLY_LAYOUT | dict(
        title=dict(text="India VIX — Volatility Index  (higher = more fear = wider ranges)",
                   font=dict(color="#212529", size=14)),
        height=280,
    )))
    return fig


def _banknifty_chart(bn_df: pd.DataFrame, nifty_df: pd.DataFrame) -> go.Figure:
    """Bank Nifty vs Nifty normalized to 100 for relative strength."""
    if bn_df.empty or nifty_df.empty:
        return go.Figure()
    bn_close    = bn_df.set_index("datetime")["close"]
    nf_close    = nifty_df.set_index("datetime")["close"]
    common      = bn_close.index.intersection(nf_close.index)
    if len(common) < 2:
        return go.Figure()
    bn_norm = bn_close[common] / float(bn_close[common].iloc[0]) * 100
    nf_norm = nf_close[common] / float(nf_close[common].iloc[0]) * 100
    rs      = bn_norm - nf_norm    # positive = BankNifty outperforming

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05,
                        subplot_titles=["Normalized Price (base=100)",
                                        "Bank Nifty Relative Strength vs Nifty"])
    fig.add_trace(go.Scatter(x=common, y=nf_norm.values, mode="lines",
                             name="Nifty 50",
                             line=dict(color=CE_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=common, y=bn_norm.values, mode="lines",
                             name="Bank Nifty",
                             line=dict(color="#fd7e14", width=2)), row=1, col=1)
    rs_clrs = [GAIN_COLOR if v >= 0 else LOSS_COLOR for v in rs]
    fig.add_trace(go.Bar(x=common, y=rs.values, name="BN−Nifty RS",
                         marker_color=rs_clrs, opacity=0.75,
                         hovertemplate="%{x|%d-%b %H:%M}<br>RS: <b>%{y:+.2f}</b><extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color="#dee2e6", row=2, col=1)
    fig.update_layout(**(PLOTLY_LAYOUT | dict(
        title=dict(text="Bank Nifty vs Nifty 50 — Relative Strength",
                   font=dict(color="#212529", size=14)),
        height=400,
        xaxis_rangeslider_visible=False,
    )))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Save futures data → trap detector JSON
# ═════════════════════════════════════════════════════════════════════════════
def _save_futures_json(fut_df: pd.DataFrame):
    """Save futures DataFrame to the format expected by the Trap Detector."""
    if fut_df.empty:
        return
    os.makedirs(os.path.dirname(FUTURES_JSON_PATH), exist_ok=True)
    candles = []
    for _, row in fut_df.iterrows():
        entry = [row["datetime"].strftime("%Y-%m-%dT%H:%M:%S+0530"),
                 row["open"], row["high"], row["low"], row["close"],
                 int(row["volume"]) if "volume" in row else 0]
        if "oi" in row:
            entry.append(int(row["oi"]))
        candles.append(entry)
    payload = {"status": "success", "data": {"candles": candles}}
    with open(FUTURES_JSON_PATH, "w") as f:
        json.dump(payload, f)
    st.toast(f"Futures data saved → {FUTURES_JSON_PATH}", icon="💾")


# ═════════════════════════════════════════════════════════════════════════════
# Main render function (called from render_max_pain.py tab6)
# ═════════════════════════════════════════════════════════════════════════════
def render_live_data_tab():
    st.markdown('<div class="tab-header">Live Market Data — Angel One SmartAPI</div>',
                unsafe_allow_html=True)

    # ── Check credentials ────────────────────────────────────────────────────
    try:
        _ = st.secrets["API_KEY"]
    except Exception:
        st.error("Angel One credentials not found in `.streamlit/secrets.toml`. "
                 "Add API_KEY, CLIENT_ID, MPIN, TOTP_SEED.")
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1])
    with ctrl1:
        interval_label = st.selectbox("Interval", list(INTERVAL_MAP.keys()),
                                      index=3, key="ld_interval")   # default 30 min
    with ctrl2:
        lookback_label = st.selectbox(
            "Lookback",
            ["Today", "2 days", "5 days", "10 days", "30 days"],
            index=2, key="ld_lookback")
    with ctrl3:
        auto_refresh = st.toggle("Auto-refresh (5 min)", value=False,
                                 key="ld_auto")
    with ctrl4:
        manual_refresh = st.button("⟳ Refresh Now", use_container_width=True,
                                   key="ld_refresh")

    lookback_days_map = {"Today": 1, "2 days": 2, "5 days": 5,
                         "10 days": 10, "30 days": 30}
    lookback_days = lookback_days_map[lookback_label]
    interval      = INTERVAL_MAP[interval_label]

    # Clear cache on manual refresh so fresh data is fetched immediately
    if manual_refresh:
        fetch_all_market_data.clear()

    # ── Session & futures token ───────────────────────────────────────────────
    with st.spinner("Authenticating…"):
        client = _get_client()
    if client is None:
        st.error("Authentication failed. Check your credentials.")
        return

    with st.spinner("Resolving near-month futures token…"):
        try:
            fut_info = _get_futures_token("NIFTY")
        except Exception as e:
            st.error(f"Could not resolve futures token: {e}")
            return

    st.caption(f"Trading: **{fut_info['symbol']}** · Expiry: **{fut_info['expiry']}**")

    # ── Fetch all data (rate-limited inside) ──────────────────────────────────
    with st.spinner("Fetching market data (4 calls, ~2 s)…"):
        data = fetch_all_market_data(interval, lookback_days,
                                     fut_info["token"])

    if not data:
        st.error("No data returned. Session may have expired — try refreshing.")
        return

    fetched_at = data.get("fetched_at", "—")
    spot_df    = _add_indicators(data.get("spot",      pd.DataFrame()))
    fut_df     = _add_indicators(data.get("futures",   pd.DataFrame()))
    vix_df     = data.get("vix",       pd.DataFrame())
    bn_df      = _add_indicators(data.get("banknifty", pd.DataFrame()))
    basis_df   = data.get("basis",     pd.DataFrame())

    # ── Top metrics ───────────────────────────────────────────────────────────
    def _last(df, col, default=None):
        return float(df[col].iloc[-1]) if not df.empty and col in df.columns else default

    spot_close  = _last(spot_df, "close")
    spot_prev   = _last(spot_df.iloc[:-1], "close") if len(spot_df) > 1 else spot_close
    fut_close   = _last(fut_df,  "close")
    vix_close   = _last(vix_df,  "close")
    bn_close    = _last(bn_df,   "close")
    spot_chg    = (spot_close - spot_prev) if (spot_close and spot_prev) else 0.0
    spot_chg_pct= spot_chg / spot_prev * 100 if spot_prev else 0.0
    basis_last  = (fut_close - spot_close) if (fut_close and spot_close) else None
    rsi_last    = _last(spot_df, "rsi14")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, (lbl, val, sub, color) in zip(
        [m1, m2, m3, m4, m5, m6],
        [
            ("Nifty 50 Spot",
             f"₹{spot_close:,.1f}" if spot_close else "—",
             f"{spot_chg:+.1f} ({spot_chg_pct:+.2f}%)" if spot_close else "—",
             GAIN_COLOR if spot_chg >= 0 else LOSS_COLOR),
            ("Nifty Futures",
             f"₹{fut_close:,.1f}" if fut_close else "—",
             fut_info["expiry"], CE_COLOR),
            ("Basis (Fut−Spot)",
             f"₹{basis_last:+.1f}" if basis_last is not None else "—",
             ("Backwardation" if (basis_last or 0) < 0
              else "Contango" if (basis_last or 0) > 0
              else "At Parity"),
             LOSS_COLOR if (basis_last or 0) < 0 else
             GAIN_COLOR if (basis_last or 0) > 0 else "#6c757d"),
            ("India VIX",
             f"{vix_close:.2f}" if vix_close else "—",
             ("High fear" if (vix_close or 0) > 25
              else "Low fear" if (vix_close or 0) < 15
              else "Moderate"),
             LOSS_COLOR if (vix_close or 0) > 25 else
             GAIN_COLOR if (vix_close or 0) < 15 else "#fd7e14"),
            ("Bank Nifty",
             f"₹{bn_close:,.1f}" if bn_close else "—",
             "spot index", "#fd7e14"),
            ("RSI 14 (Spot)",
             f"{rsi_last:.1f}" if rsi_last else "—",
             ("Overbought" if (rsi_last or 50) >= 70
              else "Oversold" if (rsi_last or 50) <= 30
              else "Neutral"),
             LOSS_COLOR if (rsi_last or 50) >= 70 else
             GAIN_COLOR if (rsi_last or 50) <= 30 else "#6c757d"),
        ]
    ):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{lbl}</div>
        <div class="metric-value" style="color:{color}; font-size:20px;">{val}</div>
        <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Last fetched: **{fetched_at}** · Cache TTL: 5 min · "
               f"Interval: **{interval_label}** · Lookback: **{lookback_label}**")
    st.divider()

    # ── Section 1: Nifty Spot ─────────────────────────────────────────────────
    st.markdown("#### Nifty 50 Spot")
    if spot_df.empty:
        st.warning("No spot data returned.")
    else:
        fig_spot = _ohlc_chart(spot_df,
                               title=f"Nifty 50 Spot — {interval_label} OHLC  |  EMA 9/21/50  |  VWAP",
                               show_vwap=True, show_oi=False, height=520)
        st.plotly_chart(fig_spot, width="stretch")
        st.plotly_chart(_rsi_chart(spot_df, "RSI 14 — Nifty Spot"),
                        width="stretch")

    st.divider()

    # ── Section 2: Nifty Futures ──────────────────────────────────────────────
    st.markdown(f"#### Nifty Futures — {fut_info['symbol']}")
    if fut_df.empty:
        st.warning("No futures data returned.")
    else:
        has_oi = "oi" in fut_df.columns
        fig_fut = _ohlc_chart(
            fut_df,
            title=f"Nifty Futures — {interval_label} OHLC  |  EMA 9/21/50"
                  + ("  |  Futures OI" if has_oi else ""),
            show_vwap=True, show_oi=has_oi, height=560)
        st.plotly_chart(fig_fut, width="stretch")
        st.plotly_chart(_rsi_chart(fut_df, "RSI 14 — Nifty Futures"),
                        width="stretch")

        sa1, sa2 = st.columns(2)
        with sa1:
            if st.button("💾 Save Futures Data → Trap Detector",
                         use_container_width=True, key="ld_save"):
                _save_futures_json(fut_df)
        with sa2:
            csv_data = fut_df.to_csv(index=False)
            st.download_button(
                "⬇ Download Futures CSV",
                data=csv_data,
                file_name=f"nifty_futures_{interval_label.replace(' ','')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="ld_dl_fut",
            )

    st.divider()

    # ── Section 3: Basis ──────────────────────────────────────────────────────
    st.markdown("#### Futures Basis (Futures − Spot)")
    if basis_df.empty:
        st.info("Basis requires both spot and futures data.")
    else:
        st.plotly_chart(_basis_chart(basis_df), width="stretch")
        latest_basis = basis_df.iloc[-1]
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Current Basis (₹)",
                   f"₹{latest_basis['basis']:+.1f}",
                   f"{latest_basis['basis_pct']:+.3f}%")
        max_b = basis_df["basis"].max()
        min_b = basis_df["basis"].min()
        bc2.metric("Period High Basis", f"₹{max_b:+.1f}")
        bc3.metric("Period Low Basis",  f"₹{min_b:+.1f}")

    st.divider()

    # ── Section 4: India VIX ──────────────────────────────────────────────────
    st.markdown("#### India VIX")
    if vix_df.empty:
        st.info("No VIX data returned.")
    else:
        st.plotly_chart(_vix_chart(vix_df), width="stretch")
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Current VIX", f"{_last(vix_df,'close'):.2f}")
        vc2.metric("Period High", f"{vix_df['high'].max():.2f}" if "high" in vix_df.columns else "—")
        vc3.metric("Period Low",  f"{vix_df['low'].min():.2f}"  if "low"  in vix_df.columns else "—")

    st.divider()

    # ── Section 5: Bank Nifty relative strength ───────────────────────────────
    st.markdown("#### Bank Nifty vs Nifty 50 — Relative Strength")
    if bn_df.empty:
        st.info("No Bank Nifty data returned.")
    else:
        st.plotly_chart(_banknifty_chart(bn_df, spot_df), width="stretch")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(300)    # 5 minutes
        st.rerun()
