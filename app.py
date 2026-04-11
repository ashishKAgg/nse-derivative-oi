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
        # Standardize column names and types
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    return pd.DataFrame()

# --- 3. Sidebar Filters ---
st.sidebar.header("Chart Settings")

# Dynamic Expiry Selection based on files in the folder
try:
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    # Extracts '13-Apr-2026' from 'nifty50-13-Apr-2026.csv'
    expiry_options = sorted([f.replace('nifty50-', '').replace('.csv', '') for f in all_files])
    
    selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_options)
    
    # Load data for that specific expiry
    target_file = f"nifty50-{selected_expiry}.csv"
    df_raw = load_data(target_file)

    if not df_raw.empty:
        # Option Type Filter
        opt_type = st.sidebar.radio("Option Type", ["Call", "Put"])
        
        # Strike Price Filter (Updates based on Option Type)
        available_strikes = sorted(df_raw[df_raw['optionType'] == opt_type]['strikePrice'].unique())
        selected_strike = st.sidebar.selectbox("Strike Price", available_strikes)
        
        # Timeframe Filter
        tf_map = {"5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min", "60m": "60min"}
        selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=0)

        # --- 4. Data Processing ---
        # Filter for the specific contract
        mask = (df_raw['optionType'] == opt_type) & (df_raw['strikePrice'] == selected_strike)
        df_filtered = df_raw[mask].copy()

        # Resample 'openInterest' to create OHLC candles
        ohlc = df_filtered.resample(tf_map[selected_tf], on='datetime')['openInterest'].ohlc().dropna()
        
        # Format for Lightweight Charts
        ohlc = ohlc.reset_index()
        ohlc.columns = ['time', 'open', 'high', 'low', 'close']
        # Convert datetime to Unix timestamp (seconds)
        ohlc['time'] = ohlc['time'].view('int64') // 10**9

        # --- 5. Rendering ---
        st.subheader(f"NIFTY {selected_strike} {opt_type} | Expiry: {selected_expiry} | TF: {selected_tf}")
        
        series_data = [
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
        
        renderLightweightCharts(series_data, {
            "layout": {
                "backgroundColor": "#131722",
                "textColor": "#d1d4dc",
                "fontSize": 12,
            },
            "grid": {
                "vertLines": {"color": "#2B2B43"},
                "horzLines": {"color": "#2B2B43"},
            },
            "timeScale": {"timeVisible": True, "secondsVisible": False},
            "height": 600,
        })
        
        # Display raw data snapshot
        with st.expander("View Raw Data Snippet"):
            st.dataframe(df_filtered.tail(10))

    else:
        st.warning("No data found for the selected expiry.")

except FileNotFoundError:
    st.error(f"Directory not found: `{DATA_DIR}`. Please check your folder structure.")
except Exception as e:
    st.error(f"An error occurred: {e}")
