"""
A module to login to Wealthsimple and perform actions
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()

# Global variables for OTP handling
current_otp = None
otp_requested = False
wealthsimple_client = None


def trigger_otp_request():
    """
    Trigger OTP request by attempting to initialize Wealthsimple client
    """
    global wealthsimple_client, otp_requested
    from wsimple.api import Wsimple
    import os
    from dotenv import load_dotenv

    load_dotenv()

    username = os.getenv("WEALTHSIMPLE_USERNAME")
    password = os.getenv("WEALTHSIMPLE_PASSWORD")

    def get_otp():
        """
        Wait for OTP to be set by the web interface
        """
        import time

        # Wait for OTP to be provided via web interface
        timeout = 300  # 5 minutes timeout
        start_time = time.time()

        while current_otp is None and (time.time() - start_time) < timeout:
            time.sleep(0.5)  # Check every 0.5 seconds

        if current_otp is None:
            raise Exception("OTP timeout - please try again")

        return current_otp

    try:
        # This will trigger the OTP request
        console.print(
            f"[cyan]Attempting to initialize Wsimple with username:[/cyan] [yellow]{username}[/yellow]"
        )

        # Apply monkey patch to fix the 'box' attribute issue
        console.print("[blue]Applying monkey patch for wsimple library bug...[/blue]")

        # Patch the Wsimple class to add the missing box attribute
        def patch_wsimple():
            import types

            # Store original methods
            original_init = Wsimple.__init__
            original_getattr = getattr(Wsimple, "__getattribute__", None)

            def patched_init(self, *args, **kwargs):
                # Call original init
                result = original_init(self, *args, **kwargs)
                # Add missing box attribute
                if not hasattr(self, "box"):
                    self.box = {}
                return result

            def patched_getattribute(self, name):
                if name == "box" and not hasattr(self, "_box"):
                    self._box = {}
                    return self._box
                return object.__getattribute__(self, name)

            # Apply patches
            Wsimple.__init__ = patched_init
            Wsimple.__getattribute__ = patched_getattribute

            return original_init, original_getattr

        # Apply the patches
        original_init, original_getattr = patch_wsimple()

        try:
            wealthsimple_client = Wsimple(username, password, otp_callback=get_otp)
            otp_requested = True
            console.print("[green]Wsimple client initialized successfully[/green]")
            return wealthsimple_client
        finally:
            # Restore original methods
            Wsimple.__init__ = original_init
            if original_getattr:
                Wsimple.__getattribute__ = original_getattr
    except AttributeError as e:
        if "'Wsimple' object has no attribute 'box'" in str(e):
            console.print(f"[red]ERROR: Box attribute error: {e}[/red]")
            console.print(
                "[yellow]WARNING: This might be a version compatibility issue with wsimple library[/yellow]"
            )
            # Try to continue anyway
            otp_requested = False
            raise Exception(
                "Wsimple library compatibility issue. Try updating the library."
            )
        else:
            otp_requested = False
            raise e
    except Exception as e:
        otp_requested = False
        console.print(f"[red]ERROR: Failed to initialize Wsimple: {e}[/red]")
        raise e


def initialize_wealthsimple():
    """
    Initialize wealthsimple client and login - reuse existing client if available
    """
    global wealthsimple_client

    if wealthsimple_client is None:
        wealthsimple_client = trigger_otp_request()

    return wealthsimple_client


def get_wealthsimple_watchlist():
    """
    Gets the tickers in a wealthsimple watchlist
    """
    try:
        wealthsimple_client = initialize_wealthsimple()
        is_operational = wealthsimple_client.is_operational()
        if is_operational:
            console.print("[green]Wealthsimple client is operational[/green]")
        else:
            console.print("[red]ERROR: Wealthsimple client is not operational[/red]")

        if wealthsimple_client.is_operational():
            # Try to get watchlist with error handling for the box attribute bug
            try:
                watchlist = wealthsimple_client.get_watchlist()
                console.print(f"[blue]Watchlist response received[/blue]")

                # Handle different response formats
                if isinstance(watchlist, dict):
                    if "securities" in watchlist:
                        tickers = [
                            ticker["stock"]["symbol"]
                            for ticker in watchlist["securities"]
                        ]
                    elif "results" in watchlist:
                        tickers = [
                            ticker["stock"]["symbol"] for ticker in watchlist["results"]
                        ]
                    else:
                        console.print(
                            f"[yellow]WARNING: Unexpected watchlist format[/yellow]"
                        )
                        return []
                else:
                    console.print(
                        f"[yellow]WARNING: Watchlist is not a dict: {type(watchlist)}[/yellow]"
                    )
                    return []

                if tickers:
                    console.print(
                        f"[green]Found {len(tickers)} tickers in watchlist[/green]"
                    )
                    console.print(f"[blue]Sample ticker: {tickers[0]}[/blue]")
                else:
                    console.print(
                        "[yellow]WARNING: No tickers found in watchlist[/yellow]"
                    )
                return tickers
            except AttributeError as e:
                if "'Wsimple' object has no attribute 'box'" in str(e):
                    console.print(
                        f"[red]ERROR: Known wsimple library bug with 'box' attribute: {e}[/red]"
                    )
                    console.print(
                        "[yellow]WARNING: This is a compatibility issue with wsimple 3.0.1 and Python 3.13[/yellow]"
                    )
                    console.print(
                        "[yellow]Returning empty list to use fallback stocks[/yellow]"
                    )
                    return []
                else:
                    raise e
        else:
            console.print("[red]ERROR: Wealthsimple client not operational[/red]")
            return []
    except Exception as e:
        console.print(f"[red]ERROR: Failed to get watchlist: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []


def add_to_wealthsimple_watchlist(file_name, col="Symbol"):
    """
    Adds tickers to wealthsimple watchlist from a dataframe of tickers
    """
    import pandas as pd

    wealthsimple_client = initialize_wealthsimple()

    local_watchlist = pd.read_csv(file_name)[col].unique().tolist()
    for ticker in local_watchlist:
        try:
            wealthsimple_client.add_watchlist(
                wealthsimple_client.find_securities(ticker)["id"]
            )
        except:
            continue


if __name__ == "__main__":

    watchlist = get_wealthsimple_watchlist()[:3]
    console.print(
        Panel(
            f"[bold green]Sample watchlist:[/bold green] {watchlist}",
            title="Wealthsimple",
            border_style="green",
        )
    )
