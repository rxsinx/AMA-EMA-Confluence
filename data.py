"""
data.py — Zerodha Kite Connect data layer for paper trading bot
Uses kiteconnect library for historical + live price data
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from kiteconnect import KiteConnect


# ── Session helpers ──────────────────────────────────────────────────────────

def get_kite() -> KiteConnect | None:
    """Return authenticated KiteConnect instance from session state."""
    return st.session_state.get("kite", None)


def init_kite(api_key: str, access_token: str) -> KiteConnect:
    """Initialise and store KiteConnect in session."""
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    st.session_state["kite"] = kite
    st.session_state["kite_api_key"] = api_key
    return kite


def is_authenticated() -> bool:
    return get_kite() is not None


# ── Instrument lookup ────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_instruments(_kite: KiteConnect) -> pd.DataFrame:
    """
    Download full NSE instrument list and cache for the session day.
    Returns DataFrame with columns: tradingsymbol, instrument_token, exchange.
    """
    instruments = _kite.instruments("NSE")
    df = pd.DataFrame(instruments)
    return df[["tradingsymbol", "instrument_token", "exchange", "name", "lot_size"]]


def get_instrument_token(symbol: str, _kite: KiteConnect) -> int | None:
    """Resolve NSE symbol -> instrument_token."""
    df = load_instruments(_kite)
    row = df[df["tradingsymbol"] == symbol.upper()]
    if row.empty:
        return None
    return int(row.iloc[0]["instrument_token"])


# ── Historical data ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_ohlcv(
    symbol: str,
    interval: str = "day",
    days: int = 365,
    _kite: KiteConnect = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Kite Connect.

    Kite interval strings:
        intraday  -> "minute" / "3minute" / "5minute" / "15minute" / "30minute" / "60minute"
        daily     -> "day"
        weekly    -> "week"

    Returns DataFrame indexed by datetime with columns: open, high, low, close, volume.
    """
    if _kite is None:
        _kite = get_kite()
    if _kite is None:
        st.error("Kite not authenticated. Please log in.")
        return pd.DataFrame()

    token = get_instrument_token(symbol, _kite)
    if token is None:
        st.warning(f"Symbol '{symbol}' not found on NSE.")
        return pd.DataFrame()

    to_date   = datetime.now()
    from_date = to_date - timedelta(days=days)

    # Kite max 60-day limit for intraday intervals
    if interval != "day" and days > 60:
        from_date = to_date - timedelta(days=60)

    try:
        records = _kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval=interval,
            continuous=False,
            oi=False,
        )
    except Exception as e:
        st.error(f"Kite historical_data error for {symbol}: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    return df.sort_index()


# ── Live / real-time price ───────────────────────────────────────────────────

def fetch_ltp(symbols: list, _kite: KiteConnect = None) -> dict:
    """
    Fetch Last Traded Price for a list of NSE symbols.
    Returns {symbol: ltp} dict.
    """
    if _kite is None:
        _kite = get_kite()
    if _kite is None:
        return {}

    instruments = [f"NSE:{s}" for s in symbols]
    try:
        quotes = _kite.ltp(instruments)
        return {
            s: quotes.get(f"NSE:{s}", {}).get("last_price", 0.0)
            for s in symbols
        }
    except Exception as e:
        st.error(f"LTP fetch error: {e}")
        return {}


def fetch_quote(symbol: str, _kite: KiteConnect = None) -> dict:
    """Full market quote for a single symbol."""
    if _kite is None:
        _kite = get_kite()
    if _kite is None:
        return {}
    try:
        q = _kite.quote([f"NSE:{symbol}"])
        return q.get(f"NSE:{symbol}", {})
    except Exception as e:
        st.error(f"Quote error for {symbol}: {e}")
        return {}


# ── Universes ────────────────────────────────────────────────────────────────

NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
    "HINDUNILVR","SBIN","BHARTIARTL","ITC","KOTAKBANK",
    "LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
    "SUNPHARMA","ULTRACEMCO","BAJFINANCE","WIPRO","HCLTECH",
    "NESTLEIND","POWERGRID","NTPC","TECHM","ADANIENT",
    "JSWSTEEL","TATASTEEL","ONGC","BPCL","COALINDIA",
    "DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP","BAJAJFINSV",
    "TATAMOTORS","M&M","HEROMOTOCO","EICHERMOT","BAJAJ-AUTO",
    "BRITANNIA","HDFCLIFE","SBILIFE","INDUSINDBK","GRASIM",
    "ADANIPORTS","SHREECEM","UPL","TATACONSUM","HINDALCO",
]

BANK_NIFTY = [
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK",
    "INDUSINDBK","BANDHANBNK","FEDERALBNK","IDFCFIRSTB","PNB",
    "AUBANK","CANBK",
]

NIFTY_IT = [
    "TCS","INFY","WIPRO","HCLTECH","TECHM",
    "MPHASIS","LTIM","COFORGE","PERSISTENT","OFSS",
]

UNIVERSES = {
    "NIFTY 50":   NIFTY_50,
    "BANK NIFTY": BANK_NIFTY,
    "NIFTY IT":   NIFTY_IT,
}


def get_universe(name: str) -> list:
    return UNIVERSES.get(name, NIFTY_50)


# ── Demo / fallback ──────────────────────────────────────────────────────────

def generate_synthetic_data(symbol: str, days: int = 365, base_price: float = 1000.0) -> pd.DataFrame:
    """Synthetic OHLCV for demo mode when Kite is not authenticated."""
    np.random.seed(hash(symbol) % 10000)
    dates   = pd.date_range(end=datetime.today(), periods=days, freq="B")
    returns = np.random.normal(0.0003, 0.018, days)
    prices  = base_price * np.exp(np.cumsum(returns))
    opens   = prices * (1 + np.random.normal(0, 0.003, days))
    highs   = np.maximum(prices, opens) * (1 + np.abs(np.random.normal(0, 0.006, days)))
    lows    = np.minimum(prices, opens) * (1 - np.abs(np.random.normal(0, 0.006, days)))
    volumes = np.random.lognormal(15, 0.5, days)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=dates,
    )


def get_price_data(symbol: str, interval: str = "day", days: int = 365, demo_mode: bool = False) -> pd.DataFrame:
    """Unified entry point. Uses Kite if authenticated, else synthetic data."""
    kite = get_kite()
    if demo_mode or kite is None:
        return generate_synthetic_data(symbol, days=days)
    return fetch_ohlcv(symbol, interval=interval, days=days, _kite=kite)


# ── Kite login UI (called from app.py sidebar) ───────────────────────────────

def render_kite_login():
    """
    Renders Kite Connect login flow in Streamlit sidebar.
    Step 1: Enter API key -> get redirected to Zerodha login
    Step 2: Paste request token from redirect URL -> generate access token
    """
    st.sidebar.markdown("### 🔑 Kite Connect")

    if is_authenticated():
        st.sidebar.success("Connected ✓")
        if st.sidebar.button("Disconnect"):
            st.session_state.pop("kite", None)
            st.rerun()
        return

    with st.sidebar.expander("Login", expanded=True):
        api_key    = st.text_input("API Key",    type="password", key="kite_api_key_input")
        api_secret = st.text_input("API Secret", type="password", key="kite_api_secret_input")

        if api_key:
            login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
            st.markdown(f"[**Step 1 → Authorise on Zerodha ↗**]({login_url})")

        request_token = st.text_input("Request Token (from redirect URL)", key="kite_req_token")

        if st.button("Generate Access Token"):
            if not all([api_key, api_secret, request_token]):
                st.error("All three fields required.")
            else:
                try:
                    kite    = KiteConnect(api_key=api_key)
                    session = kite.generate_session(request_token, api_secret=api_secret)
                    init_kite(api_key, session["access_token"])
                    st.success("Authenticated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Auth failed: {e}")

        st.caption("Access token resets at market close each day.")
