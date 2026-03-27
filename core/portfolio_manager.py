"""
Portfolio-level risk and position management
Handles position sizing, concentration risk, and rebalancing triggers
"""

from typing import Dict, List
from rich.console import Console

console = Console()


class PortfolioManager:
    """
    Portfolio-level position and risk management

    Manages:
    - Position size limits (max % per stock)
    - Concentration risk (top N positions)
    - Rebalancing triggers (drift from target)
    """

    def __init__(
        self,
        max_position: float = 0.15,
        max_top3: float = 0.50,
        drift_threshold: float = 0.05,
    ):
        """
        Args:
            max_position: Maximum weight per position (0.15 = 15%)
            max_top3: Maximum weight for top 3 positions combined (0.50 = 50%)
            drift_threshold: Rebalance trigger (0.05 = 5% drift)
        """
        self.max_position = max_position
        self.max_top3 = max_top3
        self.drift_threshold = drift_threshold

    def check_position_limits(self, portfolio: Dict[str, float]) -> List[Dict]:
        """
        Check if any positions exceed size limits

        Args:
            portfolio: {'AAPL': 0.35, 'MSFT': 0.25, 'GOOGL': 0.20, ...}

        Returns:
            [
                {'symbol': 'AAPL', 'current': 0.35, 'max': 0.15,
                 'action': 'TRIM', 'amount': 0.20, 'reason': 'position_limit'},
                ...
            ]
        """
        violations = []

        for symbol, weight in portfolio.items():
            if weight > self.max_position:
                violations.append(
                    {
                        "symbol": symbol,
                        "current": weight,
                        "max": self.max_position,
                        "action": "TRIM",
                        "amount": weight - self.max_position,
                        "reason": "position_limit",
                    }
                )

        return violations

    def check_concentration_risk(self, portfolio: Dict[str, float]) -> Dict:
        """
        Check if portfolio is too concentrated in top positions

        Args:
            portfolio: {'AAPL': 0.35, 'MSFT': 0.25, 'GOOGL': 0.20, ...}

        Returns:
            {
                'concentrated': True/False,
                'top3_weight': 0.80,
                'max_allowed': 0.50,
                'top3_positions': [('AAPL', 0.35), ('MSFT', 0.25), ('GOOGL', 0.20)],
                'recommendation': 'Reduce top 3 positions'
            }
        """
        # Sort positions by weight
        sorted_positions = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)

        # Top 3 positions
        top3 = sorted_positions[:3]
        top3_weight = sum([w for _, w in top3])

        concentrated = top3_weight > self.max_top3

        return {
            "concentrated": concentrated,
            "top3_weight": top3_weight,
            "max_allowed": self.max_top3,
            "top3_positions": top3,
            "recommendation": f"Reduce top 3 positions to below {self.max_top3:.0%}"
            if concentrated
            else "Concentration acceptable",
        }

    def check_rebalancing_needed(
        self, current_portfolio: Dict[str, float], target_portfolio: Dict[str, float]
    ) -> Dict:
        """
        Check if portfolio has drifted from target allocation

        Args:
            current_portfolio: {'AAPL': 0.35, 'MSFT': 0.25, ...}
            target_portfolio: {'AAPL': 0.30, 'MSFT': 0.25, ...}

        Returns:
            {
                'needs_rebalance': True/False,
                'actions': [
                    {'symbol': 'AAPL', 'action': 'SELL', 'current': 0.35,
                     'target': 0.30, 'amount': 0.05},
                    {'symbol': 'GOOGL', 'action': 'BUY', 'current': 0.18,
                     'target': 0.20, 'amount': 0.02}
                ],
                'total_drift': 0.07,
                'reason': 'drift'
            }
        """
        actions = []
        total_drift = 0

        # Check all positions in target
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())

        for symbol in all_symbols:
            current = current_portfolio.get(symbol, 0)
            target = target_portfolio.get(symbol, 0)
            drift = abs(current - target)

            if drift > self.drift_threshold:
                action = "BUY" if current < target else "SELL"
                actions.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "current": current,
                        "target": target,
                        "drift": drift,
                        "amount": abs(current - target),
                    }
                )
                total_drift += drift

        return {
            "needs_rebalance": len(actions) > 0,
            "actions": actions,
            "total_drift": total_drift,
            "threshold": self.drift_threshold,
            "reason": "drift" if actions else "within_threshold",
        }

    def comprehensive_check(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float] = None,
    ) -> Dict:
        """
        Run all portfolio checks

        Args:
            current_portfolio: Current allocations
            target_portfolio: Optional target allocations

        Returns:
            {
                'position_violations': [...],
                'concentration': {...},
                'rebalancing': {...},
                'overall_recommendation': 'REBALANCE' | 'TRIM' | 'OK'
            }
        """
        # Check position limits
        position_violations = self.check_position_limits(current_portfolio)

        # Check concentration
        concentration = self.check_concentration_risk(current_portfolio)

        # Check rebalancing (if target provided)
        rebalancing = None
        if target_portfolio:
            rebalancing = self.check_rebalancing_needed(
                current_portfolio, target_portfolio
            )

        # Overall recommendation
        if position_violations:
            recommendation = "TRIM"
        elif concentration["concentrated"]:
            recommendation = "REBALANCE"
        elif rebalancing and rebalancing["needs_rebalance"]:
            recommendation = "REBALANCE"
        else:
            recommendation = "OK"

        return {
            "position_violations": position_violations,
            "concentration": concentration,
            "rebalancing": rebalancing,
            "overall_recommendation": recommendation,
        }

    def calculate_profit_targets(
        self, holdings: List[Dict], trim_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Identify positions that have hit profit targets

        Args:
            holdings: [
                {'symbol': 'AAPL', 'entry_price': 150, 'current_price': 350, 'shares': 10},
                ...
            ]
            trim_threshold: Gain threshold to trigger trim (1.0 = 100% gain)

        Returns:
            [
                {'symbol': 'AAPL', 'gain': 1.33, 'recommendation': 'TRIM_50',
                 'reason': 'Hit 100%+ gain threshold'},
                ...
            ]
        """
        profit_targets = []

        for holding in holdings:
            entry = holding["entry_price"]
            current = holding.get("current_price", entry)
            gain = (current - entry) / entry

            if gain > trim_threshold:
                # Recommend trimming 50% to lock in profits
                profit_targets.append(
                    {
                        "symbol": holding["symbol"],
                        "gain": gain,
                        "gain_pct": f"{gain * 100:.1f}%",
                        "recommendation": "TRIM_50",
                        "reason": f"Hit {trim_threshold * 100:.0f}%+ gain threshold",
                        "shares": holding.get("shares", 0),
                        "trim_shares": holding.get("shares", 0) // 2,
                    }
                )

        return profit_targets
