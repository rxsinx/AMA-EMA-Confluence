"""
app.py — Paper Trading Bot · AMA × EMA × RSI × Vol Avg Confluence
Zerodha Kite Connect data · Full paper simulation · No real orders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

from data import (
    render_kite_login, get_price_data, fetch_ltp,
    get_universe, is_authenticated, UNIVERSES
)
from engine import compute_indicators, run_signals, COV_WEIGHTS
from bot import PaperTradingBot, Portfolio

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Bot · AMA Confluence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
code, .metric-value, .stMetric { font-family: 'JetBrains Mono', monospace !important; }

.main { background: #0D0F14; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

.signal-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
    letter-spacing: 0.05em;
}
.sig-buy  { background: #0D3B2E; color: #1DB97B; border: 1px solid #1DB97B; }
.sig-sell { background: #3B0D0D; color: #E24B4A; border: 1px solid #E24B4A; }
.sig-hold { background: #1E2129; color: #7C8CA0; border: 1px solid #2C3340; }

.strength-strong   { color: #1DB97B; font-weight: 700; }
.strength-moderate { color: #F5A623; font-weight: 600; }
.strength-weak     { color: #7C8CA0; }

.stat-card {
    background: #141820; border: 1px solid #1E2533; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 8px;
}
.stat-label { font-size: 11px; color: #5A6B80; letter-spacing: 0.08em; text-transform: uppercase; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #E8EDF3; }
.stat-sub   { font-size: 11px; color: #5A6B80; margin-top: 2px; }

.score-bar-wrap { background: #1E2533; border-radius: 4px; height: 6px; width: 100%; margin-top: 4px; }
.score-bar      { height: 6px; border-radius: 4px; }

.section-header {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #3A7BD5; letter-spacing: 0.12em; text-transform: uppercase;
    border-bottom: 1px solid #1E2533; padding-bottom: 6px; margin-bottom: 12px;
}

.trade-row-win  { background: #0D1F19 !important; }
.trade-row-loss { background: #1F0D0D !important; }

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "bot": None,
        "backtest_done": False,
        "bt_results": {},
        "live_signals": {},
        "scan_running": False,
        "selected_symbol": "RELIANCE",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Kite auth
    render_kite_login()

    st.markdown("---")
    demo_mode = not is_authenticated()
    if demo_mode:
        st.info("📊 Demo mode — synthetic data")

    st.markdown("### Universe")
    universe_source = st.radio(
        "Source", ["Preset index", "Upload file", "Type list"],
        horizontal=True, label_visibility="collapsed"
    )

    def _parse_symbols(raw: str) -> list:
        """Split on comma, semicolon, newline, tab — strip, uppercase, deduplicate."""
        import re
        parts = re.split(r"[,;\n\t]+", raw)
        seen, out = set(), []
        for p in parts:
            s = p.strip().upper()
            # Strip common suffixes users paste: .NS .BSE -EQ
            s = re.sub(r"(\.NS|\.BSE|-EQ)$", "", s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    if universe_source == "Preset index":
        universe_name = st.selectbox("Index", list(UNIVERSES.keys()), label_visibility="collapsed")
        symbols = get_universe(universe_name)

    elif universe_source == "Upload file":
        universe_name = "Uploaded"
        uploaded = st.file_uploader(
            "Upload .csv or .txt (one symbol per line, or comma-separated)",
            type=["csv", "txt"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                import io, pandas as pd
                raw_bytes = uploaded.read().decode("utf-8", errors="ignore")

                # CSV: try reading as dataframe first — handle headers like Symbol/Ticker/NSE Code
                if uploaded.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(io.StringIO(raw_bytes))
                    # Look for a column that likely holds symbols
                    sym_col = None
                    for col in df_up.columns:
                        if col.strip().upper() in ["SYMBOL","TICKER","NSE","SCRIP","STOCK","CODE","NAME","SCRIP CODE","NSE CODE"]:
                            sym_col = col
                            break
                    if sym_col:
                        raw_text = ",".join(df_up[sym_col].dropna().astype(str).tolist())
                    else:
                        # No recognised header — treat whole file as flat text
                        raw_text = raw_bytes
                else:
                    raw_text = raw_bytes

                symbols = _parse_symbols(raw_text)
                if symbols:
                    st.success(f"{len(symbols)} symbols loaded from **{uploaded.name}**")
                    with st.expander(f"Preview ({min(10, len(symbols))} of {len(symbols)})", expanded=False):
                        st.write(", ".join(symbols[:10]) + ("…" if len(symbols) > 10 else ""))
                else:
                    st.error("No valid symbols found. Check file format.")
                    symbols = ["RELIANCE"]
            except Exception as e:
                st.error(f"Parse error: {e}")
                symbols = ["RELIANCE"]
            # Persist across reruns
            st.session_state["uploaded_symbols"] = symbols
        else:
            # Reuse last upload if available
            symbols = st.session_state.get("uploaded_symbols", [])
            if not symbols:
                st.info("Upload a .csv or .txt file with NSE symbols.")
                symbols = ["RELIANCE"]
            else:
                st.caption(f"{len(symbols)} symbols from last upload still active.")

    else:  # Type list
        universe_name = "Custom"
        raw_input = st.text_area(
            "Enter symbols (comma, newline, or semicolon separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK",
            height=130,
            label_visibility="collapsed",
            help="NSE trading symbols. Paste up to ~200. Suffixes like .NS are auto-stripped.",
        )
        symbols = _parse_symbols(raw_input)
        if symbols:
            st.caption(f"{len(symbols)} symbol{'s' if len(symbols) != 1 else ''} loaded")
        else:
            st.warning("Enter at least one symbol.")
            symbols = ["RELIANCE"]

    # Show total count for all modes
    if symbols and len(symbols) > 1:
        st.caption(f"🔍 Scan will run on **{len(symbols)}** symbols")

    st.markdown("### Crossover")
    xo_lookback = st.slider(
        "Crossover lookback (bars)", 1, 10, 3,
        help="How many bars back to scan for an AMA-EMA crossover. On 1day interval, 3 = last 3 trading days."
    )
    st.caption(
        "Crossover runs on the same interval as the chart tab. "
        "On 1day: 3 bars = 3 trading days. On 60min: 3 bars = 3 hours."
    )

    st.markdown("### Indicator Params")
    with st.expander("AMA", expanded=False):
        ama_fast = st.slider("Fast period",   2, 20, 9)
        ama_slow = st.slider("Slow period",  10, 60, 30)
        ama_er   = st.slider("ER period",     5, 20, 10)

    with st.expander("EMA", expanded=False):
        ema_period = st.slider("EMA period", 5, 50, 20)

    with st.expander("RSI", expanded=False):
        rsi_period = st.slider("RSI period", 5, 21, 14)
        rsi_ob     = st.slider("Overbought", 55, 80, 65)
        rsi_os     = st.slider("Oversold",   20, 45, 35)

    with st.expander("Volume", expanded=False):
        vol_period    = st.slider("Vol MA period",       5, 50, 20)
        vol_threshold = st.slider("Confirm threshold", 1.0, 3.0, 1.2, step=0.1)
        vol_strong    = st.slider("Strong threshold",  1.5, 5.0, 2.0, step=0.1)

    st.markdown("### Risk Controls")
    capital         = st.number_input("Capital (INR)", 100_000, 10_000_000, 1_000_000, step=100_000)
    position_size   = st.slider("Position size %",   1, 20, 5) / 100
    stop_loss_pct   = st.slider("Stop loss %",       1, 10,  3) / 100
    target_pct      = st.slider("Target %",          2, 20,  6) / 100
    max_positions   = st.slider("Max open positions", 1, 20, 10)
    min_score       = st.slider("Min confluence score", 0.20, 0.80, 0.35, step=0.05)
    min_strength    = st.selectbox("Min signal strength", ["WEAK","MODERATE","STRONG"])

    lookback_days = st.slider("Backtest lookback (days)", 60, 500, 252)

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


# ── Main layout ──────────────────────────────────────────────────────────────
st.markdown("## 🤖 Paper Trading Bot — AMA · EMA · RSI · Vol Avg")

tabs = st.tabs(["📡 Live Scan", "📈 Single Stock", "🔁 Backtest", "📊 Portfolio", "⚖️ Signal Matrix"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 · LIVE SCAN
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    n_syms = len(symbols)
    # estimate: ~1.2s per symbol (Kite API) or ~0.05s demo
    est_sec = n_syms * (1.2 if is_authenticated() else 0.05)
    est_str = f"~{int(est_sec)}s" if est_sec < 120 else f"~{int(est_sec/60)}m"

    st.markdown(
        f'<div class="section-header">CONFLUENCE SCAN · {universe_name} · {n_syms} symbols</div>',
        unsafe_allow_html=True
    )

    col_run, col_sig_filter, col_str_filter = st.columns([1, 2, 2])
    with col_run:
        run_scan = st.button(
            f"▶ Run Scan  ({est_str})",
            use_container_width=True, type="primary",
            help=f"Will scan {n_syms} symbols. {est_str} estimated."
        )
    with col_sig_filter:
        filter_signal = st.multiselect(
            "Show signals", ["BUY", "SELL", "HOLD"],
            default=["BUY", "SELL"], label_visibility="collapsed"
        )
    with col_str_filter:
        filter_strength = st.multiselect(
            "Min strength", ["STRONG", "MODERATE", "WEAK"],
            default=["STRONG", "MODERATE", "WEAK"], label_visibility="collapsed"
        )

    if run_scan:
        scan_results = []
        skipped = []
        t_start = time.time()
        status_box = st.empty()
        progress = st.progress(0)

        for idx, sym in enumerate(symbols):
            elapsed = time.time() - t_start
            remaining = max(0, est_sec - elapsed)
            pct = (idx + 1) / n_syms
            status_box.caption(
                f"Scanning **{sym}** ({idx+1}/{n_syms}) · "
                f"{elapsed:.0f}s elapsed · ~{remaining:.0f}s remaining"
            )
            progress.progress(pct)

            df = get_price_data(sym, days=lookback_days, demo_mode=demo_mode)
            if df.empty or len(df) < 60:
                skipped.append(sym)
                continue
            signals = run_signals(df, params)
            if signals:
                last = signals[-1]
                scan_results.append({
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

        progress.empty()
        status_box.empty()
        total_time = time.time() - t_start
        st.session_state["live_signals"] = scan_results
        st.session_state["scan_meta"] = {
            "total": n_syms, "scanned": len(scan_results),
            "skipped": skipped, "elapsed": total_time,
        }

    results = st.session_state.get("live_signals", [])
    meta    = st.session_state.get("scan_meta", {})

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False)

        # ── Scan metadata bar ──────────────────────────────────────────────
        if meta:
            mcols = st.columns([1,1,1,1,2])
            mcols[0].metric("Total symbols", meta.get("total", len(df_res)))
            mcols[1].metric("Scanned OK",    meta.get("scanned", len(df_res)))
            mcols[2].metric("Skipped",       len(meta.get("skipped", [])))
            mcols[3].metric("Time",          f"{meta.get('elapsed', 0):.1f}s")
            if meta.get("skipped"):
                with mcols[4].expander(f"Skipped symbols ({len(meta['skipped'])})"):
                    st.write(", ".join(meta["skipped"]))

        # ── Signal summary ─────────────────────────────────────────────────
        buys  = len(df_res[df_res["Signal"] == "BUY"])
        sells = len(df_res[df_res["Signal"] == "SELL"])
        holds = len(df_res[df_res["Signal"] == "HOLD"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  buys)
        c2.metric("🔴 SELL", sells)
        c3.metric("⚪ HOLD", holds)
        c4.metric("Total",   len(df_res))

        # ── Apply sidebar filters ──────────────────────────────────────────
        df_filtered = df_res[
            df_res["Signal"].isin(filter_signal) &
            df_res["Strength"].isin(filter_strength + ["HOLD"])
        ]
        if len(df_filtered) < len(df_res):
            st.caption(f"Showing {len(df_filtered)} of {len(df_res)} after filter")

        # ── Score distribution chart ───────────────────────────────────────
        fig_hist = px.histogram(
            df_res, x="Score", nbins=30, color_discrete_sequence=["#3A7BD5"],
            title=f"Confluence Score Distribution — {len(df_res)} symbols",
        )
        fig_hist.add_vline(x=0.35,  line_dash="dot", line_color="#1DB97B", line_width=1)
        fig_hist.add_vline(x=-0.35, line_dash="dot", line_color="#E24B4A", line_width=1)
        fig_hist.update_layout(
            paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
            font_color="#8A9BB0", title_font_color="#E8EDF3",
            height=200, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────
        def signal_html(sig):
            cls = {"BUY": "sig-buy", "SELL": "sig-sell", "HOLD": "sig-hold"}.get(sig, "sig-hold")
            return f'<span class="signal-pill {cls}">{sig}</span>'

        def strength_html(s):
            cls = {"STRONG": "strength-strong", "MODERATE": "strength-moderate", "WEAK": "strength-weak"}.get(s, "")
            return f'<span class="{cls}">{s}</span>'

        display_df = df_filtered.copy()
        display_df["Signal"]   = display_df["Signal"].apply(signal_html)
        display_df["Strength"] = display_df["Strength"].apply(strength_html)

        # Paginate for large lists
        PAGE = 100
        total_pages = max(1, (len(display_df) - 1) // PAGE + 1)
        if total_pages > 1:
            page = st.number_input("Page", 1, total_pages, 1, label_visibility="collapsed")
            display_df = display_df.iloc[(page-1)*PAGE : page*PAGE]
            st.caption(f"Page {page}/{total_pages}  ·  {PAGE} rows per page")

        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # ── Download filtered results ──────────────────────────────────────
        csv_out = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download results CSV",
            data=csv_out,
            file_name=f"scan_{universe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **▶ Run Scan** to scan the universe for confluence signals.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 · SINGLE STOCK CHART
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_sym, col_int = st.columns([2, 1])
    with col_sym:
        symbol = st.selectbox("Symbol", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
    with col_int:
        interval = st.selectbox("Interval", ["day","week","60minute","15minute"], index=0)

    df_raw = get_price_data(symbol, interval=interval, days=lookback_days, demo_mode=demo_mode)

    if not df_raw.empty:
        df_ind = compute_indicators(df_raw, params)
        signals_list = run_signals(df_raw, params)
        signals_df = pd.DataFrame(signals_list)

        # Main chart
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.18, 0.16, 0.16],
            vertical_spacing=0.02,
            subplot_titles=("Price · AMA · EMA", "Volume", "RSI", "Confluence Score"),
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_ind.index, open=df_ind["open"], high=df_ind["high"],
            low=df_ind["low"], close=df_ind["close"],
            name="Price", increasing_line_color="#1DB97B", decreasing_line_color="#E24B4A",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind["ama"], name="AMA",
            line=dict(color="#3A7BD5", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind["ema"], name="EMA",
            line=dict(color="#F5A623", width=1.5, dash="dot"),
        ), row=1, col=1)

        # Buy/sell markers
        if not signals_df.empty:
            buys_df  = signals_df[signals_df["signal"] == "BUY"]
            sells_df = signals_df[signals_df["signal"] == "SELL"]

            if not buys_df.empty:
                fig.add_trace(go.Scatter(
                    x=buys_df["date"], y=buys_df["close"] * 0.985,
                    mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=10, color="#1DB97B"),
                ), row=1, col=1)

            if not sells_df.empty:
                fig.add_trace(go.Scatter(
                    x=sells_df["date"], y=sells_df["close"] * 1.015,
                    mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=10, color="#E24B4A"),
                ), row=1, col=1)

        # Volume
        colors = ["#1DB97B" if c >= o else "#E24B4A"
                  for c, o in zip(df_ind["close"], df_ind["open"])]
        fig.add_trace(go.Bar(
            x=df_ind.index, y=df_ind["volume"], name="Volume", marker_color=colors, opacity=0.6,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind["vol_avg"], name="Vol Avg",
            line=dict(color="#F5A623", width=1.5),
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind["rsi"], name="RSI",
            line=dict(color="#A78BFA", width=1.5),
        ), row=3, col=1)
        for level, color in [(params["rsi_ob"], "#E24B4A"), (50, "#5A6B80"), (params["rsi_os"], "#1DB97B")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1, row=3, col=1)

        # Confluence score
        if not signals_df.empty:
            score_colors = signals_df["signal"].map({"BUY":"#1DB97B","SELL":"#E24B4A","HOLD":"#3A7BD5"})
            fig.add_trace(go.Bar(
                x=signals_df["date"], y=signals_df["weighted_score"],
                name="Score", marker_color=score_colors, opacity=0.8,
            ), row=4, col=1)
            fig.add_hline(y=min_score,  line_dash="dot", line_color="#1DB97B", line_width=1, row=4, col=1)
            fig.add_hline(y=-min_score, line_dash="dot", line_color="#E24B4A", line_width=1, row=4, col=1)

        fig.update_layout(
            paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
            font=dict(color="#8A9BB0", size=11),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, font_size=11),
            height=750, margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_xaxes(gridcolor="#1A2030", zeroline=False)
        fig.update_yaxes(gridcolor="#1A2030", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

        # Latest signal card
        if not signals_df.empty:
            last = signals_df.iloc[-1]
            sig_cls = {"BUY": "sig-buy", "SELL": "sig-sell", "HOLD": "sig-hold"}.get(last["signal"], "sig-hold")
            score_val = last["weighted_score"]
            bar_color = "#1DB97B" if score_val > 0 else "#E24B4A"
            bar_width = min(abs(score_val) / 1.0 * 100, 100)

            st.markdown(f"""
            <div class="stat-card">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div class="stat-label">Latest signal · {last['date'].strftime('%d %b %Y')}</div>
                  <div style="margin-top:6px;">
                    <span class="signal-pill {sig_cls}">{last['signal']}</span>
                    &nbsp;<span style="font-size:12px;color:#7C8CA0;">{last['strength']}</span>
                    &nbsp;|&nbsp;<span style="font-family:JetBrains Mono;font-size:13px;color:#E8EDF3;">Score: {score_val:+.3f}</span>
                    &nbsp;|&nbsp;<span style="font-size:12px;color:#7C8CA0;">XO: {last['crossover']}</span>
                  </div>
                </div>
                <div style="text-align:right;">
                  <div class="stat-label">RSI</div>
                  <div class="stat-value" style="font-size:18px;">{last['rsi']:.1f}</div>
                </div>
                <div style="text-align:right;">
                  <div class="stat-label">Vol Ratio</div>
                  <div class="stat-value" style="font-size:18px;">{last['vol_ratio']:.2f}x</div>
                </div>
              </div>
              <div class="score-bar-wrap" style="margin-top:10px;">
                <div class="score-bar" style="width:{bar_width}%;background:{bar_color};margin-left:{'0' if score_val > 0 else str(50-bar_width/2)+'%'};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No data available for this symbol.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 · BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">BACKTEST · Paper Simulation</div>', unsafe_allow_html=True)

    col_bsym, col_run_bt = st.columns([3, 1])
    with col_bsym:
        bt_symbol = st.selectbox("Symbol to backtest", symbols, key="bt_sym")
    with col_run_bt:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, type="primary")

    if run_bt:
        with st.spinner(f"Running backtest on {bt_symbol}..."):
            df_bt = get_price_data(bt_symbol, days=lookback_days, demo_mode=demo_mode)
            if not df_bt.empty:
                bot = PaperTradingBot(bot_config)
                signal_rows = run_signals(df_bt, params)
                result = bot.backtest(bt_symbol, signal_rows)
                st.session_state["bt_results"] = {
                    "symbol": bt_symbol, "result": result,
                    "signal_rows": signal_rows, "bot": bot
                }
                st.session_state["backtest_done"] = True

    if st.session_state["backtest_done"] and st.session_state["bt_results"]:
        btr    = st.session_state["bt_results"]
        result = btr["result"]
        stats  = result["stats"]
        trades = result["trades"]
        eq_curve = result["equity_curve"]

        # ── KPI row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        kpis = [
            (c1, "Total Trades",    str(stats["total_trades"]),     ""),
            (c2, "Win Rate",        f"{stats['win_rate']:.1f}%",    ""),
            (c3, "Realised P&L",    f"₹{stats['realised_pnl']:,.0f}",
             "green" if stats["realised_pnl"] > 0 else "red"),
            (c4, "Profit Factor",   f"{stats['profit_factor']:.2f}", ""),
            (c5, "Max Drawdown",    f"{stats['max_drawdown_pct']:.1f}%", "red"),
            (c6, "Avg Win",         f"₹{stats['avg_win']:,.0f}",    "green"),
            (c7, "Sharpe",          f"{stats['sharpe']:.2f}",       ""),
        ]
        for col, label, val, color in kpis:
            with col:
                color_css = {"green":"color:#1DB97B","red":"color:#E24B4A","":""}.get(color,"")
                st.markdown(f"""
                <div class="stat-card">
                  <div class="stat-label">{label}</div>
                  <div class="stat-value" style="font-size:18px;{color_css}">{val}</div>
                </div>""", unsafe_allow_html=True)

        # ── Equity curve ─────────────────────────────────────────────────────
        if len(eq_curve) > 1:
            eq_dates = [e[0] for e in eq_curve]
            eq_vals  = [e[1] for e in eq_curve]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_dates, y=eq_vals, name="Portfolio Value",
                fill="tozeroy", line=dict(color="#3A7BD5", width=2),
                fillcolor="rgba(58,123,213,0.10)",
            ))
            fig_eq.add_hline(y=capital, line_dash="dot", line_color="#5A6B80", line_width=1)
            fig_eq.update_layout(
                paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                font=dict(color="#8A9BB0"), title="Equity Curve",
                title_font_color="#E8EDF3", height=280,
                margin=dict(l=0, r=0, t=35, b=0),
            )
            fig_eq.update_xaxes(gridcolor="#1A2030")
            fig_eq.update_yaxes(gridcolor="#1A2030")
            st.plotly_chart(fig_eq, use_container_width=True)

        # ── Trade log ────────────────────────────────────────────────────────
        if trades:
            st.markdown('<div class="section-header" style="margin-top:1rem;">TRADE LOG</div>', unsafe_allow_html=True)
            trade_data = []
            for t in trades:
                trade_data.append({
                    "Entry Date":  t.entry_date.strftime("%d %b %y"),
                    "Exit Date":   t.exit_date.strftime("%d %b %y"),
                    "Entry ₹":     f"{t.entry_price:,.2f}",
                    "Exit ₹":      f"{t.exit_price:,.2f}",
                    "Qty":         t.quantity,
                    "P&L ₹":       f"{t.pnl:+,.0f}",
                    "Net P&L ₹":   f"{t.net_pnl:+,.0f}",
                    "Return %":    f"{t.pnl_pct:+.2f}%",
                    "Exit Reason": t.exit_reason,
                    "Score":       f"{t.entry_score:.3f}",
                })
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(
                trade_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Net P&L ₹": st.column_config.TextColumn("Net P&L ₹"),
                }
            )

            # P&L distribution
            pnls = [t.net_pnl for t in trades]
            fig_pnl = px.bar(
                x=list(range(1, len(pnls)+1)), y=pnls,
                color=pnls, color_continuous_scale=["#E24B4A","#1E2533","#1DB97B"],
                title="Trade-by-trade Net P&L",
                labels={"x":"Trade #","y":"Net P&L (₹)"},
            )
            fig_pnl.update_layout(
                paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                font_color="#8A9BB0", title_font_color="#E8EDF3",
                height=250, margin=dict(l=0, r=0, t=35, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trades triggered. Adjust params or lower min confluence score.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 · PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">PAPER PORTFOLIO · Live Positions</div>', unsafe_allow_html=True)

    bot_obj = st.session_state.get("bt_results", {}).get("bot")

    if bot_obj and bot_obj.portfolio.positions:
        port = bot_obj.portfolio
        open_syms = list(port.positions.keys())

        # Fetch live LTPs
        if is_authenticated():
            ltps = fetch_ltp(open_syms)
        else:
            ltps = {sym: port.positions[sym].entry_price * np.random.uniform(0.97, 1.06)
                    for sym in open_syms}

        pos_rows = []
        for sym, pos in port.positions.items():
            ltp   = ltps.get(sym, pos.entry_price)
            pnl   = pos.current_pnl(ltp)
            pnl_p = pos.current_pnl_pct(ltp)
            sl_dist = (ltp - pos.stop_loss) / ltp * 100
            tgt_dist = (pos.target - ltp) / ltp * 100
            pos_rows.append({
                "Symbol":       sym,
                "Entry ₹":      f"{pos.entry_price:,.2f}",
                "LTP ₹":        f"{ltp:,.2f}",
                "Qty":          pos.quantity,
                "Open P&L ₹":   f"{pnl:+,.0f}",
                "Return %":     f"{pnl_p:+.2f}%",
                "Stop Loss ₹":  f"{pos.stop_loss:,.2f}",
                "Target ₹":     f"{pos.target:,.2f}",
                "SL dist %":    f"{sl_dist:.1f}%",
                "Tgt dist %":   f"{tgt_dist:.1f}%",
                "Score @entry": f"{pos.entry_score:.3f}",
            })

        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        total_pnl = sum(
            port.positions[sym].current_pnl(ltps.get(sym, port.positions[sym].entry_price))
            for sym in open_syms
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Open Positions",  len(open_syms))
        col_b.metric("Open P&L ₹",     f"₹{total_pnl:+,.0f}")
        col_c.metric("Cash Available",  f"₹{port.cash:,.0f}")

        # Add exit controls
        st.markdown("---")
        st.markdown("**Manual Exit**")
        exit_sym = st.selectbox("Select position to close", open_syms, key="exit_sym")
        if st.button(f"Close {exit_sym}"):
            ltp_exit = ltps.get(exit_sym, bot_obj.portfolio.positions[exit_sym].entry_price)
            trade = bot_obj.try_exit(exit_sym, ltp_exit, datetime.now(), reason="MANUAL")
            if trade:
                st.success(f"Closed {exit_sym} @ ₹{ltp_exit:,.2f} | P&L: ₹{trade.net_pnl:+,.0f}")
                st.rerun()

    elif bot_obj and not bot_obj.portfolio.positions:
        st.info("No open positions. Run a backtest to populate the portfolio.")
    else:
        st.info("Run a backtest first to see portfolio state.")

    # Realised trades summary
    if bot_obj and bot_obj.portfolio.trade_log:
        st.markdown('<div class="section-header" style="margin-top:1rem;">REALISED SUMMARY</div>', unsafe_allow_html=True)
        total_real   = bot_obj.portfolio.realised_pnl()
        win_rate     = bot_obj.portfolio.win_rate()
        pf           = bot_obj.portfolio.profit_factor()
        max_dd       = bot_obj.portfolio.max_drawdown()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Realised P&L", f"₹{total_real:+,.0f}")
        c2.metric("Win Rate",     f"{win_rate:.1f}%")
        c3.metric("Profit Factor",f"{pf:.2f}")
        c4.metric("Max Drawdown", f"{max_dd:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 · SIGNAL MATRIX
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">COVARIANCE-WEIGHTED SIGNAL MATRIX</div>', unsafe_allow_html=True)

    # Weights display
    st.markdown("#### Indicator Weights (from inverse-covariance)")
    w_cols = st.columns(4)
    labels_w = ["AMA", "EMA", "RSI", "Vol Avg"]
    keys_w   = ["ama", "ema", "rsi", "vol"]
    for col, lbl, key in zip(w_cols, labels_w, keys_w):
        w = COV_WEIGHTS[key]
        col.markdown(f"""
        <div class="stat-card" style="text-align:center;">
          <div class="stat-label">{lbl}</div>
          <div class="stat-value">{w:.0%}</div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:{w*100:.0f}%;background:#3A7BD5;"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("**Rationale:** AMA & EMA are highly correlated (ρ≈0.90) so their combined weight is capped at 40%. "
                "Vol Avg is near-orthogonal (ρ<0.08) and gets the highest individual weight as the most independent signal.")

    # Component breakdown for last signal
    if st.session_state["backtest_done"] and st.session_state["bt_results"]:
        sig_rows = st.session_state["bt_results"].get("signal_rows", [])
        if sig_rows:
            last_sig = sig_rows[-1]
            comp = last_sig.get("components", {})
            if comp:
                st.markdown(f"#### Component Breakdown — {st.session_state['bt_results']['symbol']} (latest bar)")
                comp_df = pd.DataFrame([
                    {"Indicator": lbl, "Raw Score": comp.get(key, 0), "Weight": COV_WEIGHTS.get(key, 0),
                     "Contribution": comp.get(key, 0) * COV_WEIGHTS.get(key, 0)}
                    for lbl, key in zip(labels_w, keys_w)
                ])

                fig_comp = px.bar(
                    comp_df, x="Indicator", y="Contribution",
                    color="Contribution",
                    color_continuous_scale=["#E24B4A","#1E2533","#1DB97B"],
                    title=f"Weighted contributions → Score: {last_sig['weighted_score']:+.3f}  ({last_sig['signal']})",
                )
                fig_comp.update_layout(
                    paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                    font_color="#8A9BB0", title_font_color="#E8EDF3",
                    height=300, margin=dict(l=0, r=0, t=40, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

    # Rolling correlation heatmap (if backtest has been run)
    if st.session_state["backtest_done"] and st.session_state["bt_results"]:
        sym_hm = st.session_state["bt_results"]["symbol"]
        df_hm  = get_price_data(sym_hm, days=lookback_days, demo_mode=demo_mode)
        if not df_hm.empty:
            df_hm = compute_indicators(df_hm, params)
            corr_cols = ["ama","ema","rsi","vol_ratio"]
            corr_df   = df_hm[corr_cols].dropna().corr()

            fig_heat = go.Figure(go.Heatmap(
                z=corr_df.values, x=corr_df.columns, y=corr_df.index,
                colorscale=[[0,"#E24B4A"],[0.5,"#1E2533"],[1,"#1DB97B"]],
                zmin=-1, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                paper_bgcolor="#0D0F14", plot_bgcolor="#141820",
                font_color="#8A9BB0", title=f"Live correlation matrix — {sym_hm}",
                title_font_color="#E8EDF3", height=320,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
