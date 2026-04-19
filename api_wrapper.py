import pyotp
from SmartApi import SmartConnect
from datetime import datetime, timedelta
import streamlit as st
import requests
import pandas as pd

ANGELONE_MASTER_SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def get_api_client(api_key, client_id, mpin, totp_seed):
    # Regular TOTP login flow
    totp = pyotp.TOTP(totp_seed).now()
    smart_api = SmartConnect(api_key=api_key)
    data = smart_api.generateSession(client_id, mpin, totp)
    
    if data['status']:
        return smart_api
    return None


def get_future_token(index_name = 'NIFTY'):
    """
    Fetches the near-month Nifty Future token and symbol from Angel One Scrip Master.
    """
    response = requests.get(ANGELONE_MASTER_SCRIP_URL)
    scrip_master = response.json()
    
    # Convert to DataFrame for easy filtering
    df = pd.DataFrame(scrip_master)
    
    # Filter for Nifty Futures in the NFO segment
    nifty_futs = df[(df['name'] == index_name) & 
                    (df['exch_seg'] == 'NFO') & 
                    (df['instrumenttype'] == 'FUTIDX')]
    
    # Convert expiry strings to datetime objects to find the closest one
    nifty_futs['expiry_dt'] = pd.to_datetime(nifty_futs['expiry'])
    
    # Sort by expiry and pick the first one (Near Month)
    near_month = nifty_futs.sort_values(by='expiry_dt').iloc[0]
    
    return {
        "token": near_month['token'],
        "symbol": near_month['symbol'],
        "expiry": near_month['expiry']
    }


def get_seconds_until_8am():
    now = datetime.now()
    next_8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now >= next_8am:
        next_8am += timedelta(days=1)
    return (next_8am - now).total_seconds()


def fetch_candle_data(api_client, params):
    try:
        data = api_client.getCandleData(params)
        if data.get('message') in ["Invalid Token", "Session Expired", "Token Expired"]:
            raise Exception("SESSION_EXPIRED")
        return data
    except Exception as e:
        if str(e) == "SESSION_EXPIRED":
            # Clear cache and re-authenticate
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error(f"Data Fetch Error: {e}")
            return None
