# ── Auto-refresh engine ───────────────────────────────────────────────────────
def run_scan_core(symbols: list, params: dict, lookback_days: int, demo_mode: bool) -> tuple:
    """
    Execute a full scan and return (results, meta).
    Extracted so both manual scan button and auto-refresh can call it identically.
    """
    scan_results = []
    skipped = []
    t_start = time.time()
 
    for sym in symbols:
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
 
    elapsed = time.time() - t_start
    meta = {
        "total":   len(symbols),
        "scanned": len(scan_results),
        "skipped": skipped,
        "elapsed": elapsed,
    }
    return scan_results, meta
 
 
def should_auto_refresh() -> bool:
    """Return True if auto-refresh is ON and interval has elapsed."""
    if not st.session_state.get("auto_refresh_enabled"):
        return False
    last = st.session_state.get("last_scan_time")
    if last is None:
        return False
    interval_min = st.session_state.get("auto_refresh_interval", 15)
    elapsed_min = (datetime.now() - last).total_seconds() / 60
    return elapsed_min >= interval_min
 
 
