"""
Lightweight persistent watchlist storage system
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.panel import Panel

console = Console()

class WatchlistManager:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create watchlist_symbols table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists (id) ON DELETE CASCADE,
                    UNIQUE(watchlist_id, symbol)
                )
            """)
            
            # Create default watchlist if it doesn't exist
            cursor.execute("""
                INSERT OR IGNORE INTO watchlists (name, description)
                VALUES ('Default', 'Default watchlist for stock screening')
            """)
            
            conn.commit()
    
    def create_watchlist(self, name: str, description: str = "") -> int:
        """Create a new watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO watchlists (name, description)
                    VALUES (?, ?)
                """, (name, description))
                watchlist_id = cursor.lastrowid
                conn.commit()
                console.print(f"[green]Created watchlist: {name}[/green]")
                return watchlist_id
        except sqlite3.IntegrityError:
            console.print(f"[red]Watchlist '{name}' already exists[/red]")
            return None
    
    def get_watchlists(self) -> List[Dict]:
        """Get all watchlists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM watchlists
                ORDER BY name
            """)
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'created_at': row[3],
                    'updated_at': row[4]
                }
                for row in cursor.fetchall()
            ]
    
    def get_watchlist_by_name(self, name: str) -> Optional[Dict]:
        """Get watchlist by name"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM watchlists
                WHERE name = ?
            """, (name,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'created_at': row[3],
                    'updated_at': row[4]
                }
        return None
    
    def add_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """Add a symbol to a watchlist"""
        watchlist = self.get_watchlist_by_name(watchlist_name)
        if not watchlist:
            console.print(f"[red]Watchlist '{watchlist_name}' not found[/red]")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO watchlist_symbols (watchlist_id, symbol)
                    VALUES (?, ?)
                """, (watchlist['id'], symbol.upper()))
                
                # Update watchlist timestamp
                cursor.execute("""
                    UPDATE watchlists 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist['id'],))
                
                conn.commit()
                console.print(f"[green]Added {symbol} to {watchlist_name}[/green]")
                return True
        except sqlite3.IntegrityError:
            console.print(f"[yellow]{symbol} already exists in {watchlist_name}[/yellow]")
            return False
    
    def remove_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """Remove a symbol from a watchlist"""
        watchlist = self.get_watchlist_by_name(watchlist_name)
        if not watchlist:
            console.print(f"[red]Watchlist '{watchlist_name}' not found[/red]")
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM watchlist_symbols 
                WHERE watchlist_id = ? AND symbol = ?
            """, (watchlist['id'], symbol.upper()))
            
            if cursor.rowcount > 0:
                # Update watchlist timestamp
                cursor.execute("""
                    UPDATE watchlists 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (watchlist['id'],))
                
                conn.commit()
                console.print(f"[green]Removed {symbol} from {watchlist_name}[/green]")
                return True
            else:
                console.print(f"[yellow]{symbol} not found in {watchlist_name}[/yellow]")
                return False
    
    def get_symbols(self, watchlist_name: str) -> List[str]:
        """Get all symbols from a watchlist"""
        watchlist = self.get_watchlist_by_name(watchlist_name)
        if not watchlist:
            console.print(f"[red]Watchlist '{watchlist_name}' not found[/red]")
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM watchlist_symbols
                WHERE watchlist_id = ?
                ORDER BY added_at DESC
            """, (watchlist['id'],))
            return [row[0] for row in cursor.fetchall()]
    
    def clear_watchlist(self, watchlist_name: str) -> bool:
        """Clear all symbols from a watchlist"""
        watchlist = self.get_watchlist_by_name(watchlist_name)
        if not watchlist:
            console.print(f"[red]Watchlist '{watchlist_name}' not found[/red]")
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM watchlist_symbols 
                WHERE watchlist_id = ?
            """, (watchlist['id'],))
            
            # Update watchlist timestamp
            cursor.execute("""
                UPDATE watchlists 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (watchlist['id'],))
            
            conn.commit()
            console.print(f"[green]Cleared watchlist: {watchlist_name}[/green]")
            return True
    
    def import_from_csv(self, watchlist_name: str, csv_content: str) -> Tuple[int, int]:
        """Import symbols from CSV content"""
        import io
        import csv
        
        added_count = 0
        skipped_count = 0
        
        console.print(f"[blue]Starting CSV import for {watchlist_name}[/blue]")
        console.print(f"[blue]CSV content preview: {csv_content[:100]}...[/blue]")
        
        # Try to read CSV content
        try:
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            console.print(f"[blue]Available columns: {csv_reader.fieldnames}[/blue]")
            
            # Look for symbol columns
            symbol_columns = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER', 'Stock', 'stock']
            symbol_column = None
            
            for col in symbol_columns:
                if col in csv_reader.fieldnames:
                    symbol_column = col
                    console.print(f"[green]Found symbol column: {col}[/green]")
                    break
            
            if not symbol_column:
                console.print(f"[red]No symbol column found. Available columns: {csv_reader.fieldnames}[/red]")
                return 0, 0
            
            # Add symbols
            for row in csv_reader:
                symbol = str(row[symbol_column]).strip().upper()
                if symbol and symbol != 'NAN':
                    console.print(f"[blue]Processing symbol: {symbol}[/blue]")
                    if self.add_symbol(watchlist_name, symbol):
                        added_count += 1
                    else:
                        skipped_count += 1
                        
        except Exception as e:
            console.print(f"[red]Error parsing CSV: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 0, 0
        
        console.print(f"[green]Import complete: {added_count} added, {skipped_count} skipped[/green]")
        return added_count, skipped_count
    
    def export_to_csv(self, watchlist_name: str) -> str:
        """Export watchlist to CSV format"""
        symbols = self.get_symbols(watchlist_name)
        
        if not symbols:
            return "Symbol\n"
        
        csv_content = "Symbol\n"
        for symbol in symbols:
            csv_content += f"{symbol}\n"
        
        return csv_content
    
    def delete_watchlist(self, watchlist_name: str) -> bool:
        """Delete a watchlist"""
        if watchlist_name == "Default":
            console.print("[red]Cannot delete the default watchlist[/red]")
            return False
        
        watchlist = self.get_watchlist_by_name(watchlist_name)
        if not watchlist:
            console.print(f"[red]Watchlist '{watchlist_name}' not found[/red]")
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlists WHERE id = ?", (watchlist['id'],))
            conn.commit()
            console.print(f"[green]Deleted watchlist: {watchlist_name}[/green]")
            return True


# Convenience functions for backward compatibility
def get_custom_watchlist(watchlist_name: str = "Default") -> List[str]:
    """Get symbols from custom watchlist (replacement for Wealthsimple)"""
    manager = WatchlistManager()
    symbols = manager.get_symbols(watchlist_name)
    
    if not symbols:
        # Return default stocks if empty
        default_stocks = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "BAC", "DIS"
        ]
        console.print(f"[yellow]No symbols in {watchlist_name}, using default stocks[/yellow]")
        return default_stocks
    
    console.print(f"[green]Found {len(symbols)} symbols in {watchlist_name}[/green]")
    return symbols


if __name__ == "__main__":
    # Test the watchlist manager
    manager = WatchlistManager()
    
    # Test basic operations
    console.print(Panel("[bold blue]Testing Watchlist Manager[/bold blue]", border_style="blue"))
    
    # Add some test symbols
    manager.add_symbol("Default", "AAPL")
    manager.add_symbol("Default", "GOOGL")
    manager.add_symbol("Default", "MSFT")
    
    # Get symbols
    symbols = manager.get_symbols("Default")
    console.print(Panel(f"[bold green]Default watchlist symbols:[/bold green] {symbols}", border_style="green"))
    
    # Test CSV export
    csv_content = manager.export_to_csv("Default")
    console.print(Panel(f"[bold cyan]CSV Export:[/bold cyan]\n{csv_content}", border_style="cyan"))