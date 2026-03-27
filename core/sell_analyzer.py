"""
Sell signal analyzer for portfolio positions
Detects stop-loss, trailing stops, and technical breakdowns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
from rich.console import Console

console = Console()


class SellAnalyzer:
    """
    Technical sell signal detection

    Signals:
    - Stop-loss: Fixed percentage below entry
    - Trailing stop: ATR-based dynamic stop
    - Technical breakdown: MA death cross + RSI breakdown + MACD bearish
    """

    def __init__(self, stop_loss_pct: float = 0.08, trailing_atr_mult: float = 2.0):
        """
        Args:
            stop_loss_pct: Fixed stop-loss percentage (0.08 = 8%)
            trailing_atr_mult: ATR multiplier for trailing stop (2.0 = 2x ATR)
        """
        self.stop_loss_pct = stop_loss_pct
        self.trailing_atr_mult = trailing_atr_mult

    def analyze_position(
        self, symbol: str, entry_price: float, entry_date: str = ""
    ) -> Dict:
        """
        Check all sell signals for a position

        Args:
            symbol: Stock ticker
            entry_price: Entry price for position
            entry_date: Optional entry date (not used currently)

        Returns:
            {
                'symbol': 'AAPL',
                'sell_signal': True/False,
                'reasons': ['stop_loss', 'tech_breakdown'],
                'current_price': 150.0,
                'gain_loss': -0.08,
                'recommendation': 'SELL' | 'HOLD'
            }
        """
        try:
            # Download recent data (6 months)
            df = yf.download(symbol, period="6mo", progress=False)

            if df.empty or len(df) < 50:
                console.print(f"[yellow]⚠ {symbol}: Insufficient data[/yellow]")
                return {
                    "symbol": symbol,
                    "sell_signal": False,
                    "reasons": [],
                    "error": "insufficient_data",
                }

            # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Current price
            current_price = float(df["Close"].iloc[-1])
            gain_loss = (current_price - entry_price) / entry_price

            # Check all signals
            signals = []

            # 1. Stop Loss (fixed %)
            if current_price < entry_price * (1 - self.stop_loss_pct):
                signals.append("stop_loss")

            # 2. Trailing Stop (ATR-based)
            if self._check_trailing_stop(df):
                signals.append("trailing_stop")

            # 3. Technical Breakdown
            if self._check_technical_breakdown(df):
                signals.append("technical_breakdown")

            # 4. Profit Target (trim recommendation, not sell)
            if gain_loss > 1.0:
                signals.append("profit_target_trim")

            # Recommendation
            hard_sells = ["stop_loss", "trailing_stop", "technical_breakdown"]
            recommendation = "SELL" if any(s in signals for s in hard_sells) else "HOLD"

            if "profit_target_trim" in signals and recommendation == "HOLD":
                recommendation = "TRIM"  # Consider taking partial profits

            return {
                "symbol": symbol,
                "sell_signal": len(signals) > 0,
                "reasons": signals,
                "current_price": float(current_price),
                "entry_price": entry_price,
                "gain_loss": float(gain_loss),
                "gain_loss_pct": f"{gain_loss * 100:.2f}%",
                "recommendation": recommendation,
            }

        except Exception as e:
            console.print(f"[red]✗ {symbol}: Error - {e}[/red]")
            return {
                "symbol": symbol,
                "sell_signal": False,
                "reasons": [],
                "error": str(e),
            }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        ATR = EMA of True Range
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range components
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))

        # True Range = max of the three
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # ATR = EMA of True Range
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def _check_trailing_stop(self, df: pd.DataFrame) -> bool:
        """
        ATR-based trailing stop

        Stop level = 20-day high - (2 * ATR)
        Triggered if current price < stop level
        """
        # Calculate ATR
        atr = self._calculate_atr(df, period=14)

        # 20-day high
        high_20 = df["Close"].rolling(20).max()

        # Trailing stop level
        trailing_level = high_20 - (self.trailing_atr_mult * atr)

        # Check if triggered
        current_price = df["Close"].iloc[-1]
        stop_level = trailing_level.iloc[-1]

        if pd.isna(stop_level):
            return False

        return current_price < stop_level

    def _check_technical_breakdown(self, df: pd.DataFrame) -> bool:
        """
        Technical breakdown signals:
        - Death cross: 50 MA < 200 MA
        - RSI breakdown: RSI < 30 (oversold, not bouncing)
        - MACD bearish: MACD < Signal line

        All three must align for breakdown
        """
        # Calculate indicators
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df["RSI"] = self._calculate_rsi(df["Close"], period=14)
        macd_data = self._calculate_macd(df["Close"])
        df["MACD"] = macd_data["macd"]
        df["MACD_signal"] = macd_data["signal"]

        # Check latest values
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Death cross
        death_cross = last["SMA_50"] < last["SMA_200"]

        # RSI breakdown (crossed below 30)
        rsi_break = last["RSI"] < 30 and prev["RSI"] >= 30

        # MACD bearish
        macd_bear = last["MACD"] < last["MACD_signal"]

        # All three must align
        return death_cross and macd_bear and rsi_break

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD and signal line"""
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()

        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        return {"macd": macd, "signal": signal_line, "histogram": macd - signal_line}

    def check_fundamental_deterioration(self, symbol: str) -> Dict:
        """
        Check if fundamental metrics have deteriorated

        Warning signals:
        - Negative earnings growth (< -10%)
        - Low ROE (< 5%)
        - High debt (Debt/Equity > 2.0)
        - Overvaluation (P/E > 30 or PEG > 2)

        Returns recommendation to review if 2+ issues detected
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            signals = []

            # 1. Negative earnings growth
            earnings_growth = info.get("earningsGrowth", 0)
            if earnings_growth < -0.10:
                signals.append("earnings_decline")

            # 2. Low ROE
            roe = info.get("returnOnEquity", 0)
            if roe < 0.05:
                signals.append("low_roe")

            # 3. High debt
            debt_to_equity = info.get("debtToEquity", 0)
            if debt_to_equity > 200:  # yfinance returns as percentage
                signals.append("high_debt")

            # 4. Overvaluation
            pe = info.get("forwardPE", 0)
            peg = info.get("pegRatio", 999)
            if pe > 30 or (0 < peg < 999 and peg > 2):
                signals.append("overvalued")

            return {
                "symbol": symbol,
                "fundamental_issues": signals,
                "recommend_review": len(signals) >= 2,
                "metrics": {
                    "earnings_growth": earnings_growth,
                    "roe": roe,
                    "debt_to_equity": debt_to_equity / 100 if debt_to_equity else 0,
                    "pe": pe,
                    "peg": peg if peg < 999 else None,
                },
            }

        except Exception as e:
            console.print(f"[red]✗ {symbol}: Fundamental check error - {e}[/red]")
            return {
                "symbol": symbol,
                "fundamental_issues": [],
                "error": str(e),
            }

    def batch_analyze(self, holdings: List[Dict]) -> List[Dict]:
        """
        Analyze multiple positions

        Args:
            holdings: [
                {'symbol': 'AAPL', 'entry_price': 150, 'shares': 10},
                {'symbol': 'MSFT', 'entry_price': 300, 'shares': 5}
            ]

        Returns:
            List of analysis results with sell recommendations
        """
        results = []

        console.print(f"[blue]Analyzing {len(holdings)} positions for sell signals...[/blue]")

        for holding in holdings:
            symbol = holding["symbol"]
            entry_price = holding["entry_price"]
            shares = holding.get("shares", 0)

            # Technical analysis
            tech_result = self.analyze_position(
                symbol, entry_price, holding.get("entry_date", "")
            )
            tech_result["shares"] = shares

            # Fundamental analysis
            fund_result = self.check_fundamental_deterioration(symbol)

            # Combine results
            tech_result["fundamental_issues"] = fund_result.get("fundamental_issues", [])
            tech_result["fundamental_metrics"] = fund_result.get("metrics", {})

            # Overall recommendation
            if tech_result.get("recommendation") == "SELL":
                tech_result["priority"] = "HIGH"
            elif fund_result.get("recommend_review", False):
                tech_result["priority"] = "MEDIUM"
                if tech_result.get("recommendation") == "HOLD":
                    tech_result["recommendation"] = "REVIEW"
            else:
                tech_result["priority"] = "LOW"

            results.append(tech_result)

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        results.sort(key=lambda x: priority_order.get(x.get("priority", "LOW"), 2))

        return results
