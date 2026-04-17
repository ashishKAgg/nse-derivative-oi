import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def set_layout():
    st.set_page_config(
        page_title="Options OI Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ─── Dark theme CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    .stApp { background-color: #0d0f14; }
    h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; }

    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1e2535 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #718096; margin-bottom: 4px; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600; color: #e2e8f0; }
    .metric-sub   { font-size: 12px; color: #a0aec0; margin-top: 2px; }

    .stDataFrame { background: #161b27 !important; border-radius: 10px; }
    .block-container { padding-top: 1.5rem; }

    div[data-testid="stSidebarContent"] {
        background-color: #101420;
        border-right: 1px solid #2d3748;
    }
    .tab-header {
        font-size: 13px; font-weight: 600; letter-spacing: 0.5px;
        padding: 6px 0; border-bottom: 2px solid #2d3748;
        margin-bottom: 18px; color: #a0aec0;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Colour palette ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font=dict(family="Sora, sans-serif", color="#a0aec0", size=12),
    xaxis=dict(gridcolor="#1e2535", zerolinecolor="#2d3748", showgrid=True),
    yaxis=dict(gridcolor="#1e2535", zerolinecolor="#2d3748", showgrid=True),
    legend=dict(bgcolor="#161b27", bordercolor="#2d3748", borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=60),
)
CE_COLOR   = "#63b3ed"
PE_COLOR   = "#f687b3"
MAX_COLOR  = "#9ae6b4"
LOSS_COLOR = "#fc8181"
GAIN_COLOR = "#68d391"
PALETTE    = [CE_COLOR, PE_COLOR, "#f6ad55", MAX_COLOR, "#b794f4",
              "#76e4f7", "#fbd38d", LOSS_COLOR, "#90cdf4", "#9ae6b4",
              "#e9d8a6", "#94d2bd", "#ee9b00", "#ae2012", "#005f73"]


# ─── Data helpers ────────────────────────────────────────────────────────────
# Path to your data folder relative to app.py
DATA_DIR = "optionOIData/nifty50/"

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
    df["openInterest"]= pd.to_numeric(df["openInterest"], errors="coerce")
    df["expiryDate"]  = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y", errors="coerce")
    return df


# --- 2. Data Loading Logic ---
@st.cache_data(ttl=300)
def load_data(file_name):
    """Reads the specific CSV for the selected expiry from local disk."""
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
        return transform_data(df)
    return pd.DataFrame()


# @st.cache_data(ttl=30)
# def load_data(filepath: str) -> pd.DataFrame:
#     df = pd.read_csv(filepath)
#     df["datetime"]    = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
#     df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
#     df["openInterest"]= pd.to_numeric(df["openInterest"], errors="coerce")
#     df["expiryDate"]  = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y", errors="coerce")
#     return df


# def parse_upload(uploaded_file) -> pd.DataFrame:
#     df = pd.read_csv(uploaded_file)
#     df["datetime"]    = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
#     df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
#     df["openInterest"]= pd.to_numeric(df["openInterest"], errors="coerce")
#     df["expiryDate"]  = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y", errors="coerce")
#     return df


def compute_max_pain(df_snap: pd.DataFrame):
    """
    Classic max-pain calculation.
    For each potential expiry price S:
      CE writers' gain = sum over K < S of (S-K)*CE_OI[K]   (buyers lose)
      PE writers' gain = sum over K > S of (K-S)*PE_OI[K]   (buyers lose)
    Max pain = S that minimises total buyers' loss (maximises writers' gain).
    Returns (max_pain_strike, full pain DataFrame).
    """
    lot_size = 65
    ce = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].last()
    pe = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].last()
    strikes = sorted(set(ce.index) | set(pe.index))
    if not strikes:
        return None, pd.DataFrame()

    records = []
    for s in strikes:
        ce_loss = sum((s - k) * ce.get(k, 0) for k in strikes if k < s)
        pe_loss = sum((k - s) * pe.get(k, 0) for k in strikes if k > s)
        records.append({"strikePrice": s,
                        "CE_loss": ce_loss, "PE_loss": pe_loss,
                        "total_loss": ce_loss + pe_loss})

    pain_df = pd.DataFrame(records)
    mp = int(pain_df.loc[pain_df["total_loss"].idxmin(), "strikePrice"])
    return mp, pain_df


def build_oi_table(df_snap: pd.DataFrame, max_pain: int, n: int = 10) -> pd.DataFrame:
    ce = df_snap[df_snap["optionType"] == "Call"].groupby("strikePrice")["openInterest"].last().rename("CE_OI")
    pe = df_snap[df_snap["optionType"] == "Put"].groupby("strikePrice")["openInterest"].last().rename("PE_OI")
    merged = pd.concat([ce, pe], axis=1).fillna(0).reset_index()
    merged["CE_OI"] = merged["CE_OI"].astype(int)
    merged["PE_OI"] = merged["PE_OI"].astype(int)

    # P&L from BUYERS' perspective if spot expires at max_pain
    merged["CE_PnL"]          = merged.apply(lambda r: max(0, max_pain - r["strikePrice"]) * r["CE_OI"], axis=1).astype(int)
    merged["PE_PnL"]          = merged.apply(lambda r: max(0, r["strikePrice"] - max_pain) * r["PE_OI"], axis=1).astype(int)
    merged["Total_PnL"]        = (merged["CE_PnL"] + merged["PE_PnL"]).astype(int)
    merged["Operator_PnL_70"] = (merged["Total_PnL"] * 0.70).astype(int)

    strikes = sorted(merged["strikePrice"].unique())
    nearest = min(strikes, key=lambda x: abs(x - max_pain))
    idx     = strikes.index(nearest)
    lo, hi  = max(0, idx - n), min(len(strikes) - 1, idx + n)
    window  = strikes[lo: hi + 1]

    result = merged[merged["strikePrice"].isin(window)].copy()
    result["isMaxPain"] = result["strikePrice"] == nearest
    return result.sort_values("strikePrice").reset_index(drop=True)


def render_oi_analytics():
    # set_layout()
    # ─── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        expiry_options = sorted([f.replace('nifty50-', '').replace('.csv', '') for f in all_files])

        selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_options, key='expiry_date')

        target_file = f"nifty50-{selected_expiry}.csv"
        df_raw = load_data(target_file)


        # uploaded = st.file_uploader("Upload CSV", type=["csv"])
        # default_path = "options_data.csv"

        # if uploaded:
        #     df_raw = parse_upload(uploaded)
        # elif os.path.exists(default_path):
        #     df_raw = load_data(default_path)
        # else:
        #     st.info("Upload a CSV or place **options_data.csv** in the working directory.")
        #     st.stop()

        expiries       = sorted(df_raw["expiryDate"].dropna().unique())
        expiry_labels  = [pd.Timestamp(e).strftime("%d-%b-%Y") for e in expiries]
        # sel_expiry_lbl = st.selectbox("Expiry Date", expiry_labels, key='expiry_date')
        sel_expiry_lbl = selected_expiry
        sel_expiry     = expiries[expiry_labels.index(sel_expiry_lbl)]

        df = df_raw[df_raw["expiryDate"] == sel_expiry].copy()

        all_strikes = sorted(df["strikePrice"].dropna().unique().astype(int).tolist())

        st.divider()
        auto_refresh = st.toggle("Auto-refresh every 30s", value=False)
        if auto_refresh:
            import time; time.sleep(0.5); st.rerun()


    # ─── Header ──────────────────────────────────────────────────────────────────
    underlying_spot = df["underlyingValue"].iloc[-1] if len(df) else 0
    latest_ts       = df["datetime"].max() if len(df) else "—"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
    <div>
        <h1 style="margin:0;font-size:28px;color:#e2e8f0;">📊 Options OI Analytics</h1>
        <p style="margin:0;color:#718096;font-size:13px;">
        Expiry: <b style="color:#a0aec0;">{sel_expiry_lbl}</b>&nbsp;|&nbsp;
        {len(df):,} records&nbsp;|&nbsp;
        Underlying: <b style="color:#f6ad55;">₹{underlying_spot:,.2f}</b>&nbsp;|&nbsp;
        Last snap: <b style="color:#a0aec0;">{pd.Timestamp(latest_ts).strftime('%H:%M:%S') if latest_ts != '—' else '—'}</b>
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈  OI Heatmap / Multi-Strike", "🕯  OHLC Single Strike", "⚖️  Max Pain & Table"])


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 1 — Multi-Strike OI Heatmap + Line Chart
    # ══════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="tab-header">Open Interest Across Strike Prices Over Time</div>', unsafe_allow_html=True)

        ctrl_col, chart_col = st.columns([1, 4])
        with ctrl_col:
            view_type    = st.radio("OI View", ["Absolute OI", "OI Change (Δ)"], key="oi_view")
            opt_tab1     = st.radio("Option Type", ["Call", "Put", "Both"], key="ot1")
            n_strikes    = st.slider("# strikes around spot", 5, min(40, len(all_strikes)),
                                    min(20, len(all_strikes)), key="ns1")

        center = min(all_strikes, key=lambda x: abs(x - underlying_spot)) if all_strikes else 0
        ci     = all_strikes.index(center)
        half   = n_strikes // 2
        disp_strikes = all_strikes[max(0, ci - half): min(len(all_strikes), ci + half + 1)]

        df_tab1 = df[df["strikePrice"].isin(disp_strikes)].copy()

        def make_heatmap(opt_type: str):
            sub   = df_tab1[df_tab1["optionType"] == opt_type]
            pivot = sub.pivot_table(index="strikePrice", columns="datetime",
                                    values="openInterest", aggfunc="last")
            pivot = pivot.reindex(sorted(pivot.index))
            pivot.columns = [pd.Timestamp(c).strftime("%H:%M") for c in pivot.columns]
            if "OI Change" in view_type:
                pivot = pivot.diff(axis=1)
            return pivot

        with chart_col:
            for opt, cscale, zmid, badge_color in [
                ("Call", "Blues",     None, CE_COLOR),
                ("Put",  "RdPu",      None, PE_COLOR),
            ]:
                if opt_tab1 not in [opt, "Both"]:
                    continue
                pv = make_heatmap(opt)
                if pv.empty:
                    st.info(f"No {opt} data.")
                    continue
                if "OI Change" in view_type:
                    cscale = "RdYlGn" if opt == "Call" else "RdYlGn_r"
                    zmid   = 0

                fig = go.Figure(go.Heatmap(
                    z=pv.values, x=pv.columns.tolist(), y=pv.index.tolist(),
                    colorscale=cscale, zmid=zmid,
                    colorbar=dict(title="OI", tickfont=dict(color="#a0aec0")),
                    hovertemplate="Strike: %{y}<br>Time: %{x}<br>OI: %{z:,.0f}<extra></extra>"
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"{opt.upper()} — {view_type}",
                            font=dict(color=badge_color, size=14)),
                    xaxis_title="Time", yaxis_title="Strike Price", height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Multi-strike Line Chart ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### OI Over Time — Multi-Strike Line Chart")
        lc1, lc2 = st.columns([1, 4])
        with lc1:
            line_opt = st.radio("Type", ["Call", "Put"], key="lo2")
            default_sel = disp_strikes[:min(6, len(disp_strikes))]
            sel_multi = st.multiselect("Pick strikes", disp_strikes, default=default_sel, key="ms2")

        with lc2:
            if sel_multi:
                sub_l = df_tab1[
                    (df_tab1["optionType"] == line_opt) &
                    (df_tab1["strikePrice"].isin(sel_multi))
                ].sort_values("datetime")

                fig_l = go.Figure()
                for i, sk in enumerate(sel_multi):
                    s = sub_l[sub_l["strikePrice"] == sk]
                    fig_l.add_trace(go.Scatter(
                        x=s["datetime"], y=s["openInterest"],
                        mode="lines+markers", name=str(int(sk)),
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"Strike {int(sk)}<br>%{{x|%H:%M}}<br>OI: %{{y:,.0f}}<extra></extra>"
                    ))
                fig_l.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"{line_opt} OI — Selected Strikes",
                            font=dict(color="#e2e8f0", size=14)),
                    xaxis_title="Time", yaxis_title="Open Interest", height=360,
                )
                st.plotly_chart(fig_l, use_container_width=True)
            else:
                st.info("Select at least one strike above.")


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 2 — OHLC for a single strike
    # ══════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="tab-header">OHLC — Open Interest & Price for a Single Strike</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_strike = st.selectbox("Strike Price", all_strikes,
                                    index=min(len(all_strikes) // 2, len(all_strikes) - 1))
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
            st.plotly_chart(fig2, use_container_width=True)

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
        st.markdown('<div class="tab-header">Max Pain Analysis — Latest Snapshot</div>', unsafe_allow_html=True)

        df_snap         = df[df["datetime"] == df["datetime"].max()].copy()
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

        # Pain chart: CE/PE OI mirrored bar
        fig_mp = go.Figure()
        colors_ce = [MAX_COLOR if s == max_pain else CE_COLOR for s in table_df["strikePrice"]]
        colors_pe = [MAX_COLOR if s == max_pain else PE_COLOR for s in table_df["strikePrice"]]

        fig_mp.add_trace(go.Bar(
            x=table_df["strikePrice"], y=table_df["CE_OI"],
            name="Call OI", marker_color=colors_ce, opacity=0.85,
            hovertemplate="Strike: %{x}<br>CE OI: %{y:,.0f}<extra></extra>"
        ))
        fig_mp.add_trace(go.Bar(
            x=table_df["strikePrice"], y=-table_df["PE_OI"],
            name="Put OI", marker_color=colors_pe, opacity=0.85,
            hovertemplate="Strike: %{x}<br>PE OI: %{y:,.0f}<extra></extra>"
        ))
        fig_mp.add_vline(x=max_pain, line_dash="dash", line_color=MAX_COLOR,
                        annotation_text=f"Max Pain {max_pain}", annotation_font_color=MAX_COLOR)
        fig_mp.add_vline(x=spot,     line_dash="dot",  line_color="#f6ad55",
                        annotation_text=f"Spot {spot:.0f}",
                        annotation_font_color="#f6ad55", annotation_position="bottom right")

        fig_mp.update_layout(
            **(PLOTLY_LAYOUT | dict(
            barmode="overlay",
            title=dict(text="Call / Put Open Interest | Max Pain View",
                    font=dict(color="#e2e8f0", size=15)),
            xaxis=dict(title="Strike Price",
                    tickmode="array",
                    tickvals=table_df["strikePrice"].tolist(),
                    ticktext=[str(int(s)) for s in table_df["strikePrice"]],
                    tickangle=-45, gridcolor="#1e2535"),
            yaxis=dict(title="OI (CE ↑ / PE ↓)", tickformat=",", gridcolor="#1e2535"),
            height=460,
            ))
        )
        st.plotly_chart(fig_mp, use_container_width=True)

        # ── Total buyers' loss curve (full chain) ─────────────────────────────
        fig_pain = go.Figure()
        fig_pain.add_trace(go.Bar(
            x=pain_df["strikePrice"], y=pain_df["total_loss"],
            marker_color=[MAX_COLOR if s == max_pain else "#374151"
                        for s in pain_df["strikePrice"]],
            name="Total Buyers' Loss",
            hovertemplate="Strike: %{x}<br>Buyers' Loss: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_pain.add_vline(x=max_pain, line_dash="dash", line_color=MAX_COLOR,
                        annotation_text=f"Min loss (Max Pain) = {max_pain}",
                        annotation_font_color=MAX_COLOR)
        fig_pain.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Total Option Buyers' Loss at Each Expiry Price",
                    font=dict(color="#e2e8f0", size=14)),
            xaxis_title="Potential Expiry Price (Strike)",
            yaxis_title="Buyers' Total Loss (₹)",
            height=320,
        )
        st.plotly_chart(fig_pain, use_container_width=True)

        # ── Detailed strike table ─────────────────────────────────────────────
        st.markdown("#### 📋 Strike-wise OI & P&L Detail")
        st.caption(
            f"Max pain strike **{max_pain}** highlighted in green · "
            "Buyer P&L calculated assuming expiry AT max pain · "
            "Operator P&L = 70% of total (institutional writers)"
        )

        display_cols = {
            "strikePrice":     "Strike",
            "CE_OI":           "CE OI",
            "PE_OI":           "PE OI",
            "CE_PnL":          "CE P&L (₹)",
            "PE_PnL":          "PE P&L (₹)",
            "Total_PnL":       "Total P&L (₹)",
            "Operator_PnL_70": "Operator P&L 70% (₹)",
        }
        tbl_display = table_df[list(display_cols.keys())].rename(columns=display_cols).copy()
        tbl_display["Strike"] = tbl_display["Strike"].astype(int)

        def color_pnl(val):
            if val > 0:
                return 'background-color: #d1fae5; color: #065f46' # Light Green
            elif val < 0:
                return 'background-color: #fee2e2; color: #991b1b' # Light Red
            return ''

        def highlight_max_pain(row):
            i    = tbl_display.index.get_loc(row.name)
            is_mp = table_df["isMaxPain"].iloc[i]
            if is_mp:
                return ["background-color:#1a3028; color:#9ae6b4; font-weight:700;"] * len(row)
            return [""] * len(row)

        fmt = {
            "CE OI": "{:,.0f}", "PE OI": "{:,.0f}",
            "CE P&L (₹)": "₹{:,.0f}", "PE P&L (₹)": "₹{:,.0f}",
            "Total P&L (₹)": "₹{:,.0f}", "Operator P&L 70% (₹)": "₹{:,.0f}",
        }
        styled = (
            tbl_display.style
            .apply(highlight_max_pain, axis=1)
            .format(fmt)
            .map(color_pnl, subset=["Total P&L (₹)"]) 
            # .background_gradient(subset=["Total P&L (₹)"], cmap="RdYlGn_r")
            .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "12px"})
        )
        st.dataframe(styled, use_container_width=True, height=440)

        # Download
        csv_dl = table_df.drop(columns=["isMaxPain"]).to_csv(index=False)
        ts_str = pd.Timestamp(df["datetime"].max()).strftime("%H%M")
        st.download_button(
            label="⬇ Download Table as CSV",
            data=csv_dl,
            file_name=f"maxpain_{sel_expiry_lbl}_{ts_str}.csv",
            mime="text/csv"
        )