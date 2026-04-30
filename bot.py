"""
bot.py — Paper trading simulator
Portfolio management, order execution, P&L, risk controls
No real orders are placed — all simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from engine import Signal


# ── Risk defaults ────────────────────────────────────────────────────────────
DEFAULT_CAPITAL = 1_000_000       # INR 10L
DEFAULT_POSITION_SIZE = 0.05      # 5% of capital per trade
DEFAULT_STOP_LOSS = 0.03          # 3% stop loss
DEFAULT_TARGET = 0.06             # 6% profit target (2:1 R:R)
DEFAULT_MAX_POSITIONS = 10        # max concurrent open positions
DEFAULT_BROKERAGE = 20.0          # flat INR per trade (Zerodha-like)
DEFAULT_STT = 0.001               # 0.1% STT on sell side
DEFAULT_MIN_SCORE = 0.35          # minimum confluence score to enter


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: int
    entry_date: datetime
    stop_loss: float
    target: float
    entry_score: float
    entry_signal_strength: str
    direction: str = "LONG"       # LONG only for now

    @property
    def investment(self) -> float:
        return self.entry_price * self.quantity

    def current_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity

    def current_pnl_pct(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price * 100

    def is_stopped_out(self, price: float) -> bool:
        return price <= self.stop_loss

    def is_target_hit(self, price: float) -> bool:
        return price >= self.target


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_date: datetime
    exit_date: datetime
    exit_reason: str           # SIGNAL / STOP_LOSS / TARGET / FORCED
    entry_score: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    brokerage: float = 0.0
    net_pnl: float = 0.0


@dataclass
class Portfolio:
    capital: float = DEFAULT_CAPITAL
    cash: float = field(init=False)
    positions: dict = field(default_factory=dict)    # symbol → Position
    trade_log: list = field(default_factory=list)    # list of Trade
    equity_curve: list = field(default_factory=list) # [(date, equity)]
    signal_log: list = field(default_factory=list)   # all signals seen

    def __post_init__(self):
        self.cash = self.capital
        self.equity_curve = [(datetime.now(), self.capital)]

    # ── Portfolio metrics ────────────────────────────────────────────────────
    def portfolio_value(self, prices: dict) -> float:
        pos_value = sum(
            p.entry_price * p.quantity  # use cost basis; price passed separately
            for p in self.positions.values()
        )
        mark_to_market = sum(
            prices.get(sym, p.entry_price) * p.quantity
            for sym, p in self.positions.items()
        )
        return self.cash + mark_to_market

    def open_pnl(self, prices: dict) -> float:
        return sum(
            p.current_pnl(prices.get(sym, p.entry_price))
            for sym, p in self.positions.items()
        )

    def realised_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trade_log)

    def total_trades(self) -> int:
        return len(self.trade_log)

    def win_rate(self) -> float:
        wins = [t for t in self.trade_log if t.net_pnl > 0]
        return len(wins) / len(self.trade_log) * 100 if self.trade_log else 0.0

    def avg_win(self) -> float:
        wins = [t.net_pnl for t in self.trade_log if t.net_pnl > 0]
        return np.mean(wins) if wins else 0.0

    def avg_loss(self) -> float:
        losses = [t.net_pnl for t in self.trade_log if t.net_pnl <= 0]
        return np.mean(losses) if losses else 0.0

    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl for t in self.trade_log if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trade_log if t.net_pnl <= 0))
        return gross_win / gross_loss if gross_loss > 0 else np.inf

    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for e in equities:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100

    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 10:
            return 0.0
        equities = np.array([e for _, e in self.equity_curve])
        returns = np.diff(equities) / equities[:-1]
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)


class PaperTradingBot:
    def __init__(self, config: dict):
        self.config = config
        self.portfolio = Portfolio(capital=config.get("capital", DEFAULT_CAPITAL))
        self.position_size_pct = config.get("position_size_pct", DEFAULT_POSITION_SIZE)
        self.stop_loss_pct = config.get("stop_loss_pct", DEFAULT_STOP_LOSS)
        self.target_pct = config.get("target_pct", DEFAULT_TARGET)
        self.max_positions = config.get("max_positions", DEFAULT_MAX_POSITIONS)
        self.min_score = config.get("min_score", DEFAULT_MIN_SCORE)
        self.min_strength = config.get("min_strength", "WEAK")  # WEAK/MODERATE/STRONG
        self.brokerage = config.get("brokerage", DEFAULT_BROKERAGE)
        self.stt = config.get("stt", DEFAULT_STT)

    # ── Order sizing ─────────────────────────────────────────────────────────
    def _calculate_quantity(self, price: float) -> int:
        allocation = self.portfolio.cash * self.position_size_pct
        qty = int(allocation // price)
        return max(qty, 0)

    def _calculate_cost(self, price: float, qty: int, side: str) -> float:
        """Total cost including brokerage and STT."""
        gross = price * qty
        stt = gross * self.stt if side == "SELL" else 0
        return gross + self.brokerage + stt

    # ── Execution ────────────────────────────────────────────────────────────
    def try_enter(self, symbol: str, price: float, date: datetime,
                  score: float, strength: str, signal: str) -> Optional[str]:
        """
        Attempt to open a position. Returns reason string or None on success.
        """
        if symbol in self.portfolio.positions:
            return "already_open"

        if len(self.portfolio.positions) >= self.max_positions:
            return "max_positions"

        if signal != "BUY":
            return "not_buy_signal"

        if score < self.min_score:
            return f"score_too_low ({score:.3f} < {self.min_score})"

        strength_rank = {"WEAK": 1, "MODERATE": 2, "STRONG": 3}
        if strength_rank.get(strength, 0) < strength_rank.get(self.min_strength, 1):
            return f"strength_below_min ({strength})"

        qty = self._calculate_quantity(price)
        if qty == 0:
            return "insufficient_capital"

        cost = self._calculate_cost(price, qty, "BUY")
        if cost > self.portfolio.cash:
            return "insufficient_cash"

        # Open position
        sl = round(price * (1 - self.stop_loss_pct), 2)
        tgt = round(price * (1 + self.target_pct), 2)

        pos = Position(
            symbol=symbol,
            entry_price=price,
            quantity=qty,
            entry_date=date,
            stop_loss=sl,
            target=tgt,
            entry_score=score,
            entry_signal_strength=strength,
        )
        self.portfolio.positions[symbol] = pos
        self.portfolio.cash -= cost

        return None  # success

    def try_exit(self, symbol: str, price: float, date: datetime,
                 reason: str = "SIGNAL") -> Optional[Trade]:
        """
        Close an open position. Returns the Trade record.
        """
        if symbol not in self.portfolio.positions:
            return None

        pos = self.portfolio.positions.pop(symbol)
        qty = pos.quantity
        gross_proceeds = price * qty
        cost = self.brokerage + gross_proceeds * self.stt

        pnl = (price - pos.entry_price) * qty
        # cost already contains exit-side brokerage + STT
        # entry-side brokerage was deducted from cash at entry via _calculate_cost()
        # so net_pnl = gross pnl - entry brokerage - exit brokerage - STT
        entry_brokerage = self.brokerage
        net_pnl = pnl - entry_brokerage - cost  # cost = exit_brokerage + stt

        self.portfolio.cash += gross_proceeds - cost

        trade = Trade(
            symbol=symbol,
            direction="LONG",
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=qty,
            entry_date=pos.entry_date,
            exit_date=date,
            exit_reason=reason,
            entry_score=pos.entry_score,
            pnl=round(pnl, 2),
            pnl_pct=round((price - pos.entry_price) / pos.entry_price * 100, 3),
            brokerage=round(cost, 2),
            net_pnl=round(net_pnl, 2),
        )
        self.portfolio.trade_log.append(trade)
        return trade

    # ── Risk management check ────────────────────────────────────────────────
    def check_exits(self, prices: dict, date: datetime) -> list[Trade]:
        """
        Check all open positions for stop loss / target hits.
        Returns list of triggered exits.
        """
        exits = []
        for sym in list(self.portfolio.positions.keys()):
            price = prices.get(sym)
            if price is None:
                continue
            pos = self.portfolio.positions[sym]
            if pos.is_stopped_out(price):
                t = self.try_exit(sym, price, date, reason="STOP_LOSS")
                if t:
                    exits.append(t)
            elif pos.is_target_hit(price):
                t = self.try_exit(sym, price, date, reason="TARGET")
                if t:
                    exits.append(t)
        return exits

    # ── Backtest engine ──────────────────────────────────────────────────────
    def backtest(self, symbol: str, signal_rows: list[dict]) -> dict:
        """
        Run backtest on a single symbol's signal sequence.
        Returns dict of trades and equity curve.
        """
        trades = []

        for row in signal_rows:
            date = row["date"]
            price = row["close"]
            signal = row["signal"]
            score = row["weighted_score"]
            strength = row["strength"]

            # Check SL/target first
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                if pos.is_stopped_out(price):
                    t = self.try_exit(symbol, price, date, "STOP_LOSS")
                    if t:
                        trades.append(t)
                elif pos.is_target_hit(price):
                    t = self.try_exit(symbol, price, date, "TARGET")
                    if t:
                        trades.append(t)
                elif signal == "SELL":
                    t = self.try_exit(symbol, price, date, "SIGNAL")
                    if t:
                        trades.append(t)
            else:
                self.try_enter(symbol, price, date, score, strength, signal)

            # Track equity
            eq = self.portfolio.cash + (
                self.portfolio.positions[symbol].entry_price *
                self.portfolio.positions[symbol].quantity
                if symbol in self.portfolio.positions else 0
            )
            self.portfolio.equity_curve.append((date, eq))

        # Force close at end
        if symbol in self.portfolio.positions:
            last = signal_rows[-1]
            t = self.try_exit(symbol, last["close"], last["date"], "FORCED")
            if t:
                trades.append(t)

        # Append final equity point AFTER any force-close so curve reflects true cash
        if signal_rows:
            final_date = signal_rows[-1]["date"]
            self.portfolio.equity_curve.append((final_date, self.portfolio.cash))

        return {
            "trades": trades,
            "equity_curve": self.portfolio.equity_curve,
            "stats": {
                "total_trades": len(trades),
                "win_rate": self.portfolio.win_rate(),
                "realised_pnl": self.portfolio.realised_pnl(),
                "profit_factor": self.portfolio.profit_factor(),
                "max_drawdown_pct": self.portfolio.max_drawdown(),
                "avg_win": self.portfolio.avg_win(),
                "avg_loss": self.portfolio.avg_loss(),
                "sharpe": self.portfolio.sharpe_ratio(),
            }
        }
