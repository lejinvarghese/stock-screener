#!/usr/bin/env python3
"""
Analyze your portfolio from my_stocks.csv for sell signals
"""

import pandas as pd
import requests
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Read your stocks
df = pd.read_csv('data/inputs/my_stocks.csv')

# Group by symbol and get weighted average entry price
holdings = []
for symbol in df['Symbol'].unique():
    symbol_data = df[df['Symbol'] == symbol]

    # Calculate weighted average entry price
    total_qty = symbol_data['Quantity'].sum()
    weighted_price = (symbol_data['Purchase Price'] * symbol_data['Quantity']).sum() / total_qty

    holdings.append({
        'symbol': symbol,
        'entry_price': float(weighted_price),
        'shares': float(total_qty),
        'entry_date': str(symbol_data['Trade Date'].min())
    })

console.print(f"[blue]Found {len(holdings)} unique positions in your portfolio[/blue]")

# Call API to check sells
response = requests.post(
    'http://localhost:8000/check_sells/',
    headers={'Content-Type': 'application/json'},
    json={'holdings': holdings}
)

if response.status_code == 200:
    data = response.json()
    recommendations = data.get('sell_recommendations', [])

    # Separate by priority
    high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
    medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
    low_priority = [r for r in recommendations if r.get('priority') == 'LOW']

    # Display HIGH priority (SELL signals)
    if high_priority:
        console.print("\n")
        console.print(Panel(
            f"[bold red]🚨 {len(high_priority)} SELL SIGNALS DETECTED 🚨[/bold red]",
            border_style="red"
        ))

        table = Table(title="HIGH PRIORITY - SELL NOW", show_header=True, header_style="bold red")
        table.add_column("Symbol", style="cyan")
        table.add_column("Entry", style="white")
        table.add_column("Current", style="white")
        table.add_column("Gain/Loss", style="white")
        table.add_column("Reasons", style="yellow")
        table.add_column("Shares", style="white")

        for r in high_priority:
            gain_loss = r.get('gain_loss', 0)
            color = "green" if gain_loss > 0 else "red"

            table.add_row(
                r['symbol'],
                f"${r.get('entry_price', 0):.2f}",
                f"${r.get('current_price', 0):.2f}",
                f"[{color}]{gain_loss*100:.1f}%[/{color}]",
                ", ".join(r.get('reasons', [])),
                str(int(r.get('shares', 0)))
            )

        console.print(table)

    # Display MEDIUM priority (REVIEW)
    if medium_priority:
        console.print("\n")
        table = Table(title="MEDIUM PRIORITY - REVIEW RECOMMENDED", show_header=True, header_style="bold yellow")
        table.add_column("Symbol", style="cyan")
        table.add_column("Entry", style="white")
        table.add_column("Current", style="white")
        table.add_column("Gain/Loss", style="white")
        table.add_column("Issues", style="yellow")

        for r in medium_priority:
            gain_loss = r.get('gain_loss', 0)
            color = "green" if gain_loss > 0 else "red"

            issues = r.get('fundamental_issues', []) + r.get('reasons', [])

            table.add_row(
                r['symbol'],
                f"${r.get('entry_price', 0):.2f}",
                f"${r.get('current_price', 0):.2f}",
                f"[{color}]{gain_loss*100:.1f}%[/{color}]",
                ", ".join(issues) if issues else "Review fundamentals"
            )

        console.print(table)

    # Summary of HOLD positions
    if low_priority:
        console.print(f"\n[green]✓ {len(low_priority)} positions: HOLD (no issues detected)[/green]")

        # Show top gainers
        top_gainers = sorted(low_priority, key=lambda x: x.get('gain_loss', 0), reverse=True)[:5]
        if top_gainers:
            console.print("\n[bold green]Top Gainers (HOLD):[/bold green]")
            for r in top_gainers:
                gain_loss = r.get('gain_loss', 0)
                console.print(f"  • {r['symbol']}: [green]+{gain_loss*100:.1f}%[/green] (${r.get('current_price', 0):.2f})")

    # Portfolio summary
    console.print("\n")
    console.print(Panel(
        f"[bold]Portfolio Summary[/bold]\n"
        f"Total Positions: {len(recommendations)}\n"
        f"[red]SELL Signals: {len(high_priority)}[/red]\n"
        f"[yellow]Review Needed: {len(medium_priority)}[/yellow]\n"
        f"[green]Hold: {len(low_priority)}[/green]",
        title="Analysis Complete",
        border_style="blue"
    ))

else:
    console.print(f"[red]Error: {response.text}[/red]")
