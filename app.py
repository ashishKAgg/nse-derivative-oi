import streamlit as st
import pandas as pd
import os
from streamlit_lightweight_charts import renderLightweightCharts
from render_options import render_oi_chart
# from render_futures import render_futures_chart
from render_max_pain import render_oi_analytics

# --- 1. Configuration & Setup ---
st.set_page_config(layout="wide", page_title="Nifty50 OI OHLC")

# options_tab, futures_tab, options_analytics = st.tabs(["📊 Option OI Analysis", "📈 Nifty Futures Chart", "Options Analytics"])
options_tab, options_analytics = st.tabs(["📊 Option OI Analysis", "Options Analytics"])

with options_tab:
    render_oi_chart()

# with futures_tab:
#     render_futures_chart()

with options_analytics:
    render_oi_analytics()
