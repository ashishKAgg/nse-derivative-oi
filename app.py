import streamlit as st
import pandas as pd
import os
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. Configuration & Setup ---
st.set_page_config(layout="wide", page_title="Nifty50 OI OHLC")

# Path to your data folder relative to app.py
DATA_DIR = "optionOIData/nifty50/"

# --- 2. Data Loading Logic ---
@st.cache_data(ttl=300)
def load_data(file_name):
    """Reads the specific CSV for the selected expiry from local disk."""
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
        return df
    return pd.DataFrame()

# --- 3. Sidebar Filters ---
st.sidebar.header("Chart Settings")

try:
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    expiry_options = sorted([f.replace('nifty50-', '').replace('.csv', '') for f in all_files])

    selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_options)

    target_file = f"nifty50-{selected_expiry}.csv"
    df_raw = load_data(target_file)

    if not df_raw.empty:
        # Option Type Filter
        opt_type = st.sidebar.radio("Option Type", ["Call", "Put"])

        # Strike Price Filter
        available_strikes = sorted(df_raw[df_raw['optionType'] == opt_type]['strikePrice'].unique())
        selected_strike = st.sidebar.selectbox("Strike Price", available_strikes)

        # Timeframe Filter
        tf_map = {"5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min", "60m": "60min"}
        selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=0)

        # --- 4. Data Processing ---
        mask = (df_raw['optionType'] == opt_type) & (df_raw['strikePrice'] == selected_strike)
        df_filtered = df_raw[mask].copy()

        ohlc = df_filtered.resample(tf_map[selected_tf], on='datetime')['openInterest'].ohlc().dropna()
        ohlc = ohlc.reset_index()
        ohlc.columns = ['time', 'open', 'high', 'low', 'close']

        # Convert to Unix timestamp (seconds) as native Python int
        ohlc['time'] = (ohlc['time'].dt.tz_convert('UTC') - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
        ohlc['time'] = ohlc['time'].astype(int)
        # Add IST offset (5.5 hours = 19800 seconds) to shift display
        # ohlc['time'] = ohlc['time'] + 19800

        # Convert OHLC values to native Python float for JSON serialization
        for col in ['open', 'high', 'low', 'close']:
            ohlc[col] = ohlc[col].astype(float)

        # --- 5. Rendering ---
        st.subheader(f"NIFTY {selected_strike} {opt_type} | Expiry: {selected_expiry} | TF: {selected_tf}")

        charts = [
            {
                "chart": {
                    "layout": {
                        "backgroundColor": "#131722",
                        "textColor": "#d1d4dc",
                        "fontSize": 12,
                    },
                    "grid": {
                        "vertLines": {"color": "#2B2B43"},
                        "horzLines": {"color": "#2B2B43"},
                    },
                    "timeScale": {
                        "timeVisible": True,
                        "secondsVisible": False,
                        "timezone": "Asia/Kolkata"
                    },
                    "height": 600,
                },
                "series": [
                    {
                        "type": "Candlestick",
                        "data": ohlc.to_dict('records'),
                        "options": {
                            "upColor": "#26a69a",
                            "downColor": "#ef5350",
                            "borderVisible": False,
                            "wickVisible": True,
                        }
                    }
                ]
            }
        ]

        renderLightweightCharts(charts, key="nifty_chart")

        with st.expander("View Raw Data Snippet"):
            st.dataframe(df_filtered.tail(10))

    else:
        st.warning("No data found for the selected expiry.")

except FileNotFoundError:
    st.error(f"Directory not found: `{DATA_DIR}`. Please check your folder structure.")
except Exception as e:
    st.error(f"An error occurred: {e}")
