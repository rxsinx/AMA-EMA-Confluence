"""
app.py — Paper Trading Bot · AMA × EMA × RSI × Vol Avg Confluence
Zerodha Kite Connect · Paper simulation · Auto-refresh every N minutes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import re

from streamlit_autorefresh import st_autorefresh

from data import (
    render_kite_login, get_price_data, fetch_ltp,
    get_universe, is_authenticated, UNIVERSES
)
from engine import compute_indicators, run_signals, COV_WEIGHTS
from bot import PaperTradingBot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Bot · AMA Confluence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
code, .stMetric { font-family: 'JetBrains Mono', monospace !important; }
.main { background: #0D0F14; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.signal-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
    letter-spacing: 0.05em;
}
.sig-buy  { background:#0D3B2E; color:#1DB97B; border:1px solid #1DB97B; }
.sig-sell { background:#3B0D0D; color:#E24B4A; border:1px solid #E24B4A; }
.sig-hold { background:#1E2129; color:#7C8CA0; border:1px solid #2C3340; }
.strength-strong   { color:#1DB97B; font-weight:700; }
.strength-moderate { color:#F5A623; font-weight:600; }
.strength-weak     { color:#7C8CA0; }
.stat-card {
    background:#141820; border:1px solid #1E2533; border-radius:8px;
    padding:14px 18px; margin-bottom:8px;
}
.stat-label { font-size:11px; color:#5A6B80; letter-spacing:0.08em; text-transform:uppercase; }
.stat-value { font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:700; color:#E8EDF3; }
.score-bar-wrap { background:#1E2533; border-radius:4px; height:6px; width:100%; margin-top:4px; }
.score-bar      { height:6px; border-radius:4px; }
.section-header {
    font-family:'JetBrains Mono',monospace; font-size:11px;
    color:#3A7BD5; letter-spacing:0.12em; text-transform:uppercase;
    border-bottom:1px solid #1E2533; padding-bottom:6px; margin-bottom:12px;
}
div[data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "bot": None,
        "backtest_done": False,
        "bt_results": {},
        "live_signals": [],
        "scan_meta": {},
        "scan_running": False,
        "selected_symbol": "RELIANCE",
        "uploaded_symbols": [],
        # auto-refresh
        "auto_refresh_enabled": False,
        "auto_refresh_interval": 15,   # minutes
        "last_scan_time": None,
        "last_scan_symbols": [],
        "last_scan_params": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_symbols(raw: str) -> list:
    parts = re.split(r"[,;\n\t]+", raw)
    seen, out = set(), []
    for p in parts:
        s = re.sub(r"(\.NS|\.BSE|-EQ)$", "", p.strip().upper())
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out


def should_auto_refresh() -> bool:
    if not st.session_state.get("auto_refresh_enabled"):
        return False
    last = st.session_state.get("last_scan_time")
    if last is None:
        return False
    elapsed_min = (datetime.now() - last).total_seconds() / 60
    return elapsed_min >= st.session_state.get("auto_refresh_interval", 15)


def run_scan(symbols, params, lookback_days, demo_mode, show_progress=True):
    """Run scan, update session state, return (results, meta)."""
    results, skipped = [], []
    t0 = time.time()
    n  = len(symbols)

    status = st.empty() if show_progress else None
    bar    = st.progress(0) if show_progress else None

    for idx, sym in enumerate(symbols):
        if show_progress:
            elapsed = time.time() - t0
            status.caption(f"Scanning **{sym}** ({idx+1}/{n}) · {elapsed:.0f}s elapsed")
            bar.progress((idx + 1) / n)

        df = get_price_data(sym, days=lookback_days, demo_mode=demo_mode)
        if df.empty or len(df) < 60:
            skipped.append(sym); continue

        sigs = run_signals(df, params)
        if sigs:
            last = sigs[-1]
            results.append({
                "Symbol":    sym,
                "Price":     round(last["close"], 2),
                "AMA":       round(last["ama"], 2),
                "EMA":       round(last["ema"], 2),
                "RSI":       round(last["rsi"], 1),
                "Vol Ratio": round(last["vol_ratio"], 2),
                "Score":     round(last["weighted_score"], 3),
                "Signal":    last["signal"],
                "Strength":  last["strength"],
                "Crossover": last["crossover"],
            })

    if show_progress:
        status.empty(); bar.empty()

    elapsed = time.time() - t0
    meta = {"total": n, "scanned": len(results),
            "skipped": skipped, "elapsed": elapsed,
            "at": datetime.now().strftime("%H:%M:%S")}

    st.session_state["live_signals"]      = results
    st.session_state["scan_meta"]         = meta
    st.session_state["last_scan_time"]    = datetime.now()
    st.session_state["last_scan_symbols"] = symbols[:]
    st.session_state["last_scan_params"]  = params.copy()
    return results, meta

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_kite_login()
    st.markdown("---")

    demo_mode = not is_authenticated()
    if demo_mode:
        st.info("📊 Demo mode — synthetic data")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    st.markdown("### 🔄 Auto-Refresh")
    ar_col1, ar_col2 = st.columns([1, 2])
    with ar_col1:
        ar_on = st.toggle("ON", value=st.session_state["auto_refresh_enabled"])
        st.session_state["auto_refresh_enabled"] = ar_on
    with ar_col2:
        ar_interval = st.selectbox(
            "Interval", [5, 10, 15, 30, 60],
            index=[5,10,15,30,60].index(st.session_state["auto_refresh_interval"]),
            format_func=lambda x: f"{x} min",
            label_visibility="collapsed",
        )
        st.session_state["auto_refresh_interval"] = ar_interval

    # Countdown
    last_t = st.session_state.get("last_scan_time")
    if ar_on and last_t:
        elapsed_s   = (datetime.now() - last_t).total_seconds()
        remaining_s = max(0.0, ar_interval * 60 - elapsed_s)
        pct         = min(1.0, elapsed_s / (ar_interval * 60))
        mins, secs  = divmod(int(remaining_s), 60)
        st.progress(pct, text=f"Next scan {mins:02d}:{secs:02d}")
        at = st.session_state["scan_meta"].get("at", "")
        if at:
            st.caption(f"Last scan: {at}")
    elif ar_on:
        st.caption("Run a scan first to start the timer.")

    # ── streamlit-autorefresh heartbeat ───────────────────────────────────────
    # Fires every (interval) minutes; triggers a Streamlit rerun.
    # When should_auto_refresh() returns True the scan will execute automatically.
    if ar_on:
        st_autorefresh(
            interval=ar_interval * 60 * 1000,   # milliseconds
            limit=None,
            key="ar_heartbeat",
        )

    st.markdown("---")

    # ── Universe ──────────────────────────────────────────────────────────────
    st.markdown("### Universe")
    universe_source = st.radio(
        "Source", ["Preset index", "Upload file", "Type list"],
        horizontal=True, label_visibility="collapsed"
    )

    if universe_source == "Preset index":
        universe_name = st.selectbox("Index", list(UNIVERSES.keys()),
                                     label_visibility="collapsed")
        symbols = get_universe(universe_name)

    elif universe_source == "Upload file":
        universe_name = "Uploaded"
        uploaded = st.file_uploader(
            "Upload .csv or .txt", type=["csv","txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw_bytes = uploaded.read().decode("utf-8", errors="ignore")
            if uploaded.name.lower().endswith(".csv"):
                import io
                df_up = pd.read_csv(io.StringIO(raw_bytes))
                sym_col = next(
                    (c for c in df_up.columns
                     if c.strip().upper() in
                     ["SYMBOL","TICKER","NSE","SCRIP","STOCK","CODE","NAME"]),
                    None
                )
                raw_text = (",".join(df_up[sym_col].dropna().astype(str).tolist())
                            if sym_col else raw_bytes)
            else:
                raw_text = raw_bytes
            syms = _parse_symbols(raw_text)
            if syms:
                st.session_state["uploaded_symbols"] = syms
                st.success(f"{len(syms)} symbols loaded")
                with st.expander(f"Preview ({min(10,len(syms))} shown)"):
                    st.write(", ".join(syms[:10]) + ("…" if len(syms)>10 else ""))
            else:
                st.error("No valid symbols found.")
        symbols = st.session_state.get("uploaded_symbols", ["RELIANCE"])
        if not symbols:
            st.info("Upload a .csv or .txt with NSE symbols.")
            symbols = ["RELIANCE"]
        else:
            st.caption(f"{len(symbols)} symbols active")

    else:
        universe_name = "Custom"
        raw_input = st.text_area(
            "Symbols (comma / newline / semicolon)",
            value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK",
            height=130, label_visibility="collapsed",
        )
        symbols = _parse_symbols(raw_input)
        if not symbols:
            st.warning("Enter at least one symbol.")
            symbols = ["RELIANCE"]
        st.caption(f"{len(symbols)} symbol{'s' if len(symbols)!=1 else ''} loaded")

    if len(symbols) > 1:
        st.caption(f"🔍 Scan will run on **{len(symbols)}** symbols")

    # ── Crossover ─────────────────────────────────────────────────────────────
    st.markdown("### Crossover")
    xo_lookback = st.slider("Lookback (bars)", 1, 10, 3,
        help="Bars to scan for AMA–EMA crossover. 3 bars on 1day = 3 trading days.")
    st.caption("Runs on the same interval as the chart tab.")

    # ── Indicator params ──────────────────────────────────────────────────────
    st.markdown("### Indicator Params")
    with st.expander("AMA"):
        ama_fast = st.slider("Fast period", 2, 20, 9)
        ama_slow = st.slider("Slow period", 10, 60, 30)
        ama_er   = st.slider("ER period",   5, 20, 10)
    with st.expander("EMA"):
        ema_period = st.slider("EMA period", 5, 50, 20)
    with st.expander("RSI"):
        rsi_period = st.slider("RSI period", 5, 21, 14)
        rsi_ob     = st.slider("Overbought", 55, 80, 65)
        rsi_os     = st.slider("Oversold",   20, 45, 35)
    with st.expander("Volume"):
        vol_period    = st.slider("Vol MA period",      5, 50, 20)
        vol_threshold = st.slider("Confirm threshold", 1.0, 3.0, 1.2, step=0.1)
        vol_strong    = st.slider("Strong threshold",  1.5, 5.0, 2.0, step=0.1)

    # ── Risk controls ─────────────────────────────────────────────────────────
    st.markdown("### Risk Controls")
    capital       = st.number_input("Capital (INR)", 100_000, 10_000_000, 1_000_000, step=100_000)
    position_size = st.slider("Position size %",  1, 20,  5) / 100
    stop_loss_pct = st.slider("Stop loss %",      1, 10,  3) / 100
    target_pct    = st.slider("Target %",         2, 20,  6) / 100
    max_positions = st.slider("Max positions",    1, 20, 10)
    min_score     = st.slider("Min confluence",  0.20, 0.80, 0.35, step=0.05)
    min_strength  = st.selectbox("Min strength", ["WEAK","MODERATE","STRONG"])
    lookback_days = st.slider("Lookback (days)", 60, 500, 252)

    params = dict(
        ama_fast=ama_fast, ama_slow=ama_slow, ama_er=ama_er,
        ema_period=ema_period,
        rsi_period=rsi_period, rsi_ob=rsi_ob, rsi_os=rsi_os,
        vol_period=vol_period, vol_threshold=vol_threshold, vol_strong=vol_strong,
        xo_lookback=xo_lookback,
    )
    bot_config = dict(
        capital=capital, position_size_pct=position_size,
        stop_loss_pct=stop_loss_pct, target_pct=target_pct,
        max_positions=max_positions, min_score=min_score,
        min_strength=min_strength,
    )

# ── Auto-refresh trigger (runs AFTER sidebar so params are ready) ─────────────
if should_auto_refresh():
    st.toast(f"🔄 Auto-scanning {len(symbols)} symbols…", icon="🔄")
    run_scan(symbols, params, lookback_days, demo_mode, show_progress=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("## 🤖 Paper Trading Bot — AMA · EMA · RSI · Vol Avg")
tabs = st.tabs(["📡 Live Scan", "📈 Single Stock", "🔁 Backtest", "📊 Portfolio", "⚖️ Signal Matrix"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · LIVE SCAN
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    n_syms  = len(symbols)
    est_sec = n_syms * (1.2 if is_authenticated() else 0.05)
    est_str = f"~{int(est_sec)}s" if est_sec < 120 else f"~{int(est_sec/60)}m"

    st.markdown(
        f'<div class="section-header">CONFLUENCE SCAN · {universe_name} · {n_syms} symbols</div>',
        unsafe_allow_html=True,
    )

    # Last scan timestamp
    last_t = st.session_state.get("last_scan_time")
    if last_t:
        age_s = (datetime.now() - last_t).total_seconds()
        age_str = f"{int(age_s//60)}m {int(age_s%60)}s ago" if age_s >= 60 else f"{int(age_s)}s ago"
        col_ts, col_ar = st.columns([3,1])
        col_ts.caption(f"Last scan: **{st.session_state['scan_meta'].get('at','')}**  ({age_str})")
        if ar_on:
            remaining_s = max(0, ar_interval*60 - age_s)
            col_ar.caption(f"Next in {int(remaining_s//60):02d}:{int(remaining_s%60):02d}")

    col_run, col_sig_f, col_str_f = st.columns([1, 2, 2])
    with col_run:
        manual_scan = st.button(
            f"▶ Run Scan  ({est_str})",
            use_container_width=True, type="primary",
        )
    with col_sig_f:
        filter_signal = st.multiselect(
            "Signals", ["BUY","SELL","HOLD"], default=["BUY","SELL"],
            label_visibility="collapsed",
        )
    with col_str_f:
        filter_strength = st.multiselect(
            "Strength", ["STRONG","MODERATE","WEAK"],
            default=["STRONG","MODERATE","WEAK"],
            label_visibility="collapsed",
        )

    if manual_scan:
        run_scan(symbols, params, lookback_days, demo_mode, show_progress=True)
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    results = st.session_state.get("live_signals", [])
    meta    = st.session_state.get("scan_meta", {})

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False)

        # Metadata row
        if meta:
            mc = st.columns([1,1,1,1,2])
            mc[0].metric("Total",   meta.get("total", 0))
            mc[1].metric("OK",      meta.get("scanned", 0))
            mc[2].metric("Skipped", len(meta.get("skipped",[])))
            mc[3].metric("Time",    f"{meta.get('elapsed',0):.1f}s")
            if meta.get("skipped"):
                with mc[4].expander(f"Skipped ({len(meta['skipped'])})"):
                    st.write(", ".join(meta["skipped"]))

        buys  = len(df_res[df_res["Signal"]=="BUY"])
        sells = len(df_res[df_res["Signal"]=="SELL"])
        holds = len(df_res[df_res["Signal"]=="HOLD"])
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🟢 BUY",  buys)
        c2.metric("🔴 SELL", sells)
        c3.metric("⚪ HOLD", holds)
        c4.metric("Total",   len(df_res))

        # Filter
        df_filtered = df_res[
            df_res["Signal"].isin(filter_signal) &
            df_res["Strength"].isin(filter_strength + ["HOLD"])
        ]
        if len(df_filtered) < len(df_res):
            st.caption(f"Showing {len(df_filtered)} of {len(df_res)} after filter")

        # Distribution chart
        fig_h = px.histogram(df_res, x="Score", nbins=30,
                             color_discrete_sequence=["#3A7BD5"],
                             title="Confluence Score Distribution")
        fig_h.add_vline(x=0.35,  line_dash="dot", line_color="#1DB97B", line_width=1)
        fig_h.add_vline(x=-0.35, line_dash="dot", line_color="#E24B4A", line_width=1)
        fig_h.update_layout(paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                            font_color="#8A9BB0", title_font_color="#E8EDF3",
                            height=200, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_h, use_container_width=True)

        # Table
        def signal_html(s):
            cls = {"BUY":"sig-buy","SELL":"sig-sell","HOLD":"sig-hold"}.get(s,"sig-hold")
            return f'<span class="signal-pill {cls}">{s}</span>'
        def strength_html(s):
            cls = {"STRONG":"strength-strong","MODERATE":"strength-moderate",
                   "WEAK":"strength-weak"}.get(s,"")
            return f'<span class="{cls}">{s}</span>'

        disp = df_filtered.copy()
        disp["Signal"]   = disp["Signal"].apply(signal_html)
        disp["Strength"] = disp["Strength"].apply(strength_html)

        PAGE = 100
        total_pages = max(1, (len(disp)-1)//PAGE + 1)
        if total_pages > 1:
            page = st.number_input("Page", 1, total_pages, 1,
                                   label_visibility="collapsed")
            disp = disp.iloc[(page-1)*PAGE : page*PAGE]
            st.caption(f"Page {page}/{total_pages}")

        st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        csv_out = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV", data=csv_out,
            file_name=f"scan_{universe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        if ar_on:
            st.info(f"🔄 Auto-refresh ON every {ar_interval} min. Click **▶ Run Scan** for the first scan.")
        else:
            st.info("Click **▶ Run Scan** to scan the universe for confluence signals.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · SINGLE STOCK
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_sym, col_int = st.columns([2,1])
    with col_sym:
        symbol = st.selectbox("Symbol", symbols,
                              index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
    with col_int:
        interval = st.selectbox("Interval", ["day","week","60minute","15minute"])

    df_raw = get_price_data(symbol, interval=interval, days=lookback_days, demo_mode=demo_mode)

    if not df_raw.empty:
        df_ind      = compute_indicators(df_raw, params)
        signals_df  = pd.DataFrame(run_signals(df_raw, params))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.50,0.18,0.16,0.16],
                            vertical_spacing=0.02,
                            subplot_titles=("Price · AMA · EMA","Volume","RSI","Score"))

        fig.add_trace(go.Candlestick(
            x=df_ind.index, open=df_ind["open"], high=df_ind["high"],
            low=df_ind["low"], close=df_ind["close"], name="Price",
            increasing_line_color="#1DB97B", decreasing_line_color="#E24B4A",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ama"], name="AMA",
                                 line=dict(color="#3A7BD5",width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema"], name="EMA",
                                 line=dict(color="#F5A623",width=1.5,dash="dot")), row=1, col=1)

        if not signals_df.empty:
            b = signals_df[signals_df["signal"]=="BUY"]
            s = signals_df[signals_df["signal"]=="SELL"]
            if not b.empty:
                fig.add_trace(go.Scatter(x=b["date"], y=b["close"]*0.985, mode="markers",
                    name="BUY", marker=dict(symbol="triangle-up",size=10,color="#1DB97B")), row=1,col=1)
            if not s.empty:
                fig.add_trace(go.Scatter(x=s["date"], y=s["close"]*1.015, mode="markers",
                    name="SELL", marker=dict(symbol="triangle-down",size=10,color="#E24B4A")), row=1,col=1)

        colors = ["#1DB97B" if c>=o else "#E24B4A"
                  for c,o in zip(df_ind["close"],df_ind["open"])]
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["volume"], name="Volume",
                             marker_color=colors, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["vol_avg"], name="Vol Avg",
                                 line=dict(color="#F5A623",width=1.5)), row=2, col=1)

        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["rsi"], name="RSI",
                                 line=dict(color="#A78BFA",width=1.5)), row=3, col=1)
        for lvl, col in [(rsi_ob,"#E24B4A"),(50,"#5A6B80"),(rsi_os,"#1DB97B")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col, line_width=1, row=3, col=1)

        if not signals_df.empty:
            sc = signals_df["signal"].map({"BUY":"#1DB97B","SELL":"#E24B4A","HOLD":"#3A7BD5"})
            fig.add_trace(go.Bar(x=signals_df["date"], y=signals_df["weighted_score"],
                                 name="Score", marker_color=sc, opacity=0.8), row=4, col=1)
            fig.add_hline(y=min_score,  line_dash="dot", line_color="#1DB97B", line_width=1, row=4,col=1)
            fig.add_hline(y=-min_score, line_dash="dot", line_color="#E24B4A", line_width=1, row=4,col=1)

        fig.update_layout(paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                          font=dict(color="#8A9BB0",size=11),
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h",y=1.02,font_size=11),
                          height=750, margin=dict(l=0,r=0,t=40,b=0))
        fig.update_xaxes(gridcolor="#1A2030", zeroline=False)
        fig.update_yaxes(gridcolor="#1A2030", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

        if not signals_df.empty:
            last = signals_df.iloc[-1]
            sig_cls = {"BUY":"sig-buy","SELL":"sig-sell","HOLD":"sig-hold"}.get(last["signal"],"sig-hold")
            sv = last["weighted_score"]
            bw = min(abs(sv)*100, 100)
            bc = "#1DB97B" if sv > 0 else "#E24B4A"
            st.markdown(f"""
            <div class="stat-card">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div class="stat-label">Latest · {last['date'].strftime('%d %b %Y')}</div>
                  <div style="margin-top:6px;">
                    <span class="signal-pill {sig_cls}">{last['signal']}</span>
                    &nbsp;<span style="font-size:12px;color:#7C8CA0;">{last['strength']}</span>
                    &nbsp;|&nbsp;<span style="font-family:JetBrains Mono;font-size:13px;color:#E8EDF3;">Score: {sv:+.3f}</span>
                    &nbsp;|&nbsp;<span style="font-size:12px;color:#7C8CA0;">XO: {last['crossover']}</span>
                  </div>
                </div>
                <div style="text-align:right;"><div class="stat-label">RSI</div>
                  <div class="stat-value" style="font-size:18px;">{last['rsi']:.1f}</div></div>
                <div style="text-align:right;"><div class="stat-label">Vol Ratio</div>
                  <div class="stat-value" style="font-size:18px;">{last['vol_ratio']:.2f}x</div></div>
              </div>
              <div class="score-bar-wrap" style="margin-top:10px;">
                <div class="score-bar" style="width:{bw}%;background:{bc};"></div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("No data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">BACKTEST · Paper Simulation</div>',
                unsafe_allow_html=True)
    col_bs, col_rb = st.columns([3,1])
    with col_bs:
        bt_symbol = st.selectbox("Symbol", symbols, key="bt_sym")
    with col_rb:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, type="primary")

    if run_bt:
        with st.spinner(f"Running backtest on {bt_symbol}…"):
            df_bt = get_price_data(bt_symbol, days=lookback_days, demo_mode=demo_mode)
            if not df_bt.empty:
                bot = PaperTradingBot(bot_config)
                sig_rows = run_signals(df_bt, params)
                result   = bot.backtest(bt_symbol, sig_rows)
                st.session_state["bt_results"]   = {"symbol": bt_symbol,
                                                     "result": result,
                                                     "signal_rows": sig_rows,
                                                     "bot": bot}
                st.session_state["backtest_done"] = True

    if st.session_state["backtest_done"] and st.session_state["bt_results"]:
        btr    = st.session_state["bt_results"]
        result = btr["result"]
        stats  = result["stats"]
        trades = result["trades"]
        eq     = result["equity_curve"]

        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        kpis = [
            (c1,"Total Trades", str(stats["total_trades"]), ""),
            (c2,"Win Rate",     f"{stats['win_rate']:.1f}%", ""),
            (c3,"P&L",          f"₹{stats['realised_pnl']:,.0f}",
             "green" if stats["realised_pnl"]>0 else "red"),
            (c4,"Profit Factor",f"{stats['profit_factor']:.2f}", ""),
            (c5,"Max DD",       f"{stats['max_drawdown_pct']:.1f}%","red"),
            (c6,"Avg Win",      f"₹{stats['avg_win']:,.0f}","green"),
            (c7,"Sharpe",       f"{stats['sharpe']:.2f}", ""),
        ]
        for col, lbl, val, color in kpis:
            with col:
                cc = {"green":"color:#1DB97B","red":"color:#E24B4A","":""}.get(color,"")
                st.markdown(f'<div class="stat-card"><div class="stat-label">{lbl}</div>'
                            f'<div class="stat-value" style="font-size:18px;{cc}">{val}</div></div>',
                            unsafe_allow_html=True)

        if len(eq) > 1:
            eq_dates = [e[0] for e in eq]
            eq_vals  = [e[1] for e in eq]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_dates, y=eq_vals, fill="tozeroy",
                                        line=dict(color="#3A7BD5",width=2),
                                        fillcolor="rgba(58,123,213,0.10)"))
            fig_eq.add_hline(y=capital, line_dash="dot", line_color="#5A6B80", line_width=1)
            fig_eq.update_layout(paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                                 font=dict(color="#8A9BB0"), title="Equity Curve",
                                 title_font_color="#E8EDF3", height=280,
                                 margin=dict(l=0,r=0,t=35,b=0))
            fig_eq.update_xaxes(gridcolor="#1A2030")
            fig_eq.update_yaxes(gridcolor="#1A2030")
            st.plotly_chart(fig_eq, use_container_width=True)

        if trades:
            st.markdown('<div class="section-header" style="margin-top:1rem;">TRADE LOG</div>',
                        unsafe_allow_html=True)
            trade_data = [{"Entry": t.entry_date.strftime("%d %b %y"),
                           "Exit":  t.exit_date.strftime("%d %b %y"),
                           "Entry ₹": f"{t.entry_price:,.2f}",
                           "Exit ₹":  f"{t.exit_price:,.2f}",
                           "Qty": t.quantity,
                           "P&L ₹": f"{t.pnl:+,.0f}",
                           "Net ₹":  f"{t.net_pnl:+,.0f}",
                           "Ret %":  f"{t.pnl_pct:+.2f}%",
                           "Reason": t.exit_reason,
                           "Score":  f"{t.entry_score:.3f}"}
                          for t in trades]
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

            pnls = [t.net_pnl for t in trades]
            fig_p = px.bar(x=list(range(1,len(pnls)+1)), y=pnls, color=pnls,
                           color_continuous_scale=["#E24B4A","#1E2533","#1DB97B"],
                           title="Trade P&L",
                           labels={"x":"Trade #","y":"Net P&L (₹)"})
            fig_p.update_layout(paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                                font_color="#8A9BB0", title_font_color="#E8EDF3",
                                height=250, margin=dict(l=0,r=0,t=35,b=0),
                                coloraxis_showscale=False)
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("No trades triggered. Lower min confluence score or adjust params.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 · PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">PAPER PORTFOLIO · Live Positions</div>',
                unsafe_allow_html=True)
    bot_obj = st.session_state.get("bt_results", {}).get("bot")

    if bot_obj and bot_obj.portfolio.positions:
        port      = bot_obj.portfolio
        open_syms = list(port.positions.keys())
        ltps = (fetch_ltp(open_syms) if is_authenticated() else
                {s: port.positions[s].entry_price * np.random.uniform(0.97,1.06)
                 for s in open_syms})

        pos_rows = []
        for sym, pos in port.positions.items():
            ltp = ltps.get(sym, pos.entry_price)
            pos_rows.append({
                "Symbol":   sym,
                "Entry ₹":  f"{pos.entry_price:,.2f}",
                "LTP ₹":    f"{ltp:,.2f}",
                "Qty":      pos.quantity,
                "Open P&L": f"{pos.current_pnl(ltp):+,.0f}",
                "Ret %":    f"{pos.current_pnl_pct(ltp):+.2f}%",
                "SL ₹":     f"{pos.stop_loss:,.2f}",
                "Tgt ₹":    f"{pos.target:,.2f}",
                "Score":    f"{pos.entry_score:.3f}",
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        total_pnl = sum(port.positions[s].current_pnl(ltps.get(s,port.positions[s].entry_price))
                        for s in open_syms)
        ca, cb, cc = st.columns(3)
        ca.metric("Open Positions", len(open_syms))
        cb.metric("Open P&L",       f"₹{total_pnl:+,.0f}")
        cc.metric("Cash",           f"₹{port.cash:,.0f}")

        st.markdown("---")
        exit_sym = st.selectbox("Close position", open_syms, key="exit_sym")
        if st.button(f"Close {exit_sym}"):
            ltp_exit = ltps.get(exit_sym, port.positions[exit_sym].entry_price)
            t = bot_obj.try_exit(exit_sym, ltp_exit, datetime.now(), "MANUAL")
            if t:
                st.success(f"Closed {exit_sym} @ ₹{ltp_exit:,.2f} | Net P&L: ₹{t.net_pnl:+,.0f}")
                st.rerun()

    elif bot_obj:
        st.info("No open positions.")
    else:
        st.info("Run a backtest first.")

    if bot_obj and bot_obj.portfolio.trade_log:
        st.markdown('<div class="section-header" style="margin-top:1rem;">REALISED SUMMARY</div>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Realised P&L",  f"₹{bot_obj.portfolio.realised_pnl():+,.0f}")
        c2.metric("Win Rate",      f"{bot_obj.portfolio.win_rate():.1f}%")
        c3.metric("Profit Factor", f"{bot_obj.portfolio.profit_factor():.2f}")
        c4.metric("Max Drawdown",  f"{bot_obj.portfolio.max_drawdown():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · SIGNAL MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">COVARIANCE-WEIGHTED SIGNAL MATRIX</div>',
                unsafe_allow_html=True)

    st.markdown("#### Indicator Weights (from inverse-covariance)")
    labels_w = ["AMA","EMA","RSI","Vol Avg"]
    keys_w   = ["ama","ema","rsi","vol"]
    w_cols = st.columns(4)
    for col, lbl, key in zip(w_cols, labels_w, keys_w):
        w = COV_WEIGHTS[key]
        col.markdown(
            f'<div class="stat-card" style="text-align:center;">'
            f'<div class="stat-label">{lbl}</div>'
            f'<div class="stat-value">{w:.0%}</div>'
            f'<div class="score-bar-wrap"><div class="score-bar" '
            f'style="width:{w*100:.0f}%;background:#3A7BD5;"></div></div></div>',
            unsafe_allow_html=True)

    st.markdown("**AMA & EMA** combined weight capped at 40% (ρ≈0.90). "
                "**Vol Avg** highest individual weight — near-orthogonal (ρ<0.08).")

    if st.session_state["backtest_done"] and st.session_state["bt_results"]:
        sig_rows = st.session_state["bt_results"].get("signal_rows",[])
        sym_hm   = st.session_state["bt_results"]["symbol"]

        if sig_rows:
            last_sig = sig_rows[-1]
            comp = last_sig.get("components",{})
            if comp:
                st.markdown(f"#### Component Breakdown — {sym_hm} (latest bar)")
                comp_df = pd.DataFrame([
                    {"Indicator":lbl,
                     "Raw Score": comp.get(key,0),
                     "Weight":    COV_WEIGHTS.get(key,0),
                     "Contribution": comp.get(key,0)*COV_WEIGHTS.get(key,0)}
                    for lbl,key in zip(labels_w, keys_w)
                ])
                fig_c = px.bar(comp_df, x="Indicator", y="Contribution",
                               color="Contribution",
                               color_continuous_scale=["#E24B4A","#1E2533","#1DB97B"],
                               title=f"Score: {last_sig['weighted_score']:+.3f}  ({last_sig['signal']})")
                fig_c.update_layout(paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                                    font_color="#8A9BB0", title_font_color="#E8EDF3",
                                    height=300, margin=dict(l=0,r=0,t=40,b=0),
                                    coloraxis_showscale=False)
                st.plotly_chart(fig_c, use_container_width=True)

        df_hm = get_price_data(sym_hm, days=lookback_days, demo_mode=demo_mode)
        if not df_hm.empty:
            df_hm   = compute_indicators(df_hm, params)
            corr_df = df_hm[["ama","ema","rsi","vol_ratio"]].dropna().corr()
            fig_heat = go.Figure(go.Heatmap(
                z=corr_df.values, x=corr_df.columns, y=corr_df.index,
                colorscale=[[0,"#E24B4A"],[0.5,"#1E2533"],[1,"#1DB97B"]],
                zmin=-1, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                font_color="#8A9BB0", title=f"Live correlation — {sym_hm}",
                title_font_color="#E8EDF3", height=320,
                margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_heat, use_container_width=True)
