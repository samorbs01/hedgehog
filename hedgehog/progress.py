"""Module for displaying real-time progress of analysts."""

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Set

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
# Dark mode colors
DARK_GREEN = "\033[32m"
DARK_RED = "\033[31m"
DARK_YELLOW = "\033[33m"
DARK_CYAN = "\033[36m"

class ProgressTracker:
    """Tracks and displays the progress of analysts in real-time."""

    def __init__(self, dark_mode: bool = False):
        """Initialize the progress tracker.

        Args:
            dark_mode: Whether to use dark mode colors
        """
        self._lock = threading.Lock()
        self._status: Dict[str, Dict[str, str]] = {}  # {analyst: {ticker: status}}
        self._selected_analysts: List[str] = []
        self._selected_model: Optional[str] = None
        self._running = False
        self._completed_analysts: Set[str] = set()
        self._display_thread = None
        self._dark_mode = dark_mode

    @property
    def dark_mode(self) -> bool:
        """Get the current dark mode setting.

        Returns:
            True if dark mode is enabled, False otherwise
        """
        return self._dark_mode

    @dark_mode.setter
    def dark_mode(self, value: bool) -> None:
        """Set the dark mode setting.

        Args:
            value: True to enable dark mode, False to disable
        """
        with self._lock:
            self._dark_mode = value

    def set_analysts(self, analysts: List[str]) -> None:
        """Set the selected analysts.

        Args:
            analysts: List of analyst names
        """
        with self._lock:
            self._selected_analysts = analysts
            self._status = {analyst: {} for analyst in analysts}

    def set_model(self, model: str) -> None:
        """Set the selected model.

        Args:
            model: Model name
        """
        with self._lock:
            self._selected_model = model

    def update_status(self, analyst: str, ticker: str, status: str) -> None:
        """Update the status of an analyst for a specific ticker.

        Args:
            analyst: Analyst name
            ticker: Ticker symbol
            status: Current status message
        """
        with self._lock:
            if analyst not in self._status:
                self._status[analyst] = {}

            self._status[analyst][ticker] = status

            # Mark as completed if the status is "Done"
            if status.lower() == "done":
                self._completed_analysts.add(f"{analyst}:{ticker}")

    def _get_color(self, color_type: str) -> str:
        """Get the appropriate color based on the current mode.

        Args:
            color_type: The type of color to get (green, red, yellow, cyan)

        Returns:
            The ANSI color code
        """
        if self._dark_mode:
            if color_type == "green":
                return DARK_GREEN
            elif color_type == "red":
                return DARK_RED
            elif color_type == "yellow":
                return DARK_YELLOW
            elif color_type == "cyan":
                return DARK_CYAN
        else:
            if color_type == "green":
                return GREEN
            elif color_type == "red":
                return RED
            elif color_type == "yellow":
                return YELLOW
            elif color_type == "cyan":
                return CYAN

        return RESET

    def _render_display(self) -> None:
        """Render the current progress display."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Get color references for current mode
        green = self._get_color("green")
        red = self._get_color("red")
        yellow = self._get_color("yellow")
        cyan = self._get_color("cyan")

        # Print mode indicator
        mode_text = "DARK MODE" if self._dark_mode else "LIGHT MODE"
        mode_color = DARK_CYAN if self._dark_mode else CYAN
        print(f"{mode_color}[{mode_text}]{RESET} (Press 'd' to toggle)")

        # Print model selection
        if self._selected_model:
            model_parts = self._selected_model.split("/")
            if len(model_parts) > 1:
                provider = model_parts[0]
                model_name = model_parts[1]
                print(f"? Select your LLM model: [{green}{provider}{RESET}] {model_name}")
                print("")
                print(f"Selected {cyan}{provider.capitalize()}{RESET} model: {green}{model_name}{RESET}")
                print("")
            else:
                print(f"? Select your LLM model: {green}{self._selected_model}{RESET}")
                print("")
                print(f"Selected model: {green}{self._selected_model}{RESET}")
                print("")

        # Track which tickers we've seen for each analyst
        analyst_tickers = {}

        # First, gather all tickers per analyst
        for analyst in self._selected_analysts:
            analyst_tickers[analyst] = set(self._status.get(analyst, {}).keys())

        # Get all unique tickers
        all_tickers = set()
        for tickers in analyst_tickers.values():
            all_tickers.update(tickers)

        # Print status for each analyst and ticker combo
        for analyst in self._selected_analysts:
            for ticker in all_tickers:
                status = self._status.get(analyst, {}).get(ticker, "")

                if not status:
                    continue

                # Select appropriate status indicator and color
                if status.lower() == "done":
                    indicator = "✓"
                    color = green
                else:
                    indicator = "..."
                    color = yellow

                # Format the analyst name to match screenshot format
                if "Analyst" in analyst:
                    # For "Technical Analyst", "Fundamental Analyst", etc. - just use first word
                    display_name = analyst.split()[0]
                else:
                    # For other names like "Warren Buffett", use as is
                    display_name = analyst

                print(f"{indicator} {display_name:<15} [{cyan}{ticker}{RESET}] {color}{status}{RESET}")

    def start_display(self) -> None:
        """Start the progress display."""
        if self._running:
            return

        self._running = True
        self._display_thread = threading.Thread(target=self._display_loop)
        self._display_thread.daemon = True
        self._display_thread.start()

    def stop_display(self) -> None:
        """Stop the progress display."""
        self._running = False
        if self._display_thread:
            self._display_thread.join(timeout=1.0)

    def toggle_dark_mode(self) -> None:
        """Toggle between dark and light mode."""
        with self._lock:
            self._dark_mode = not self._dark_mode

    def _display_loop(self) -> None:
        """Background thread that continuously updates the display."""
        while self._running:
            with self._lock:
                self._render_display()

            # Check for key presses to toggle dark mode
            if sys.stdin.isatty():  # Only try to read from terminal
                try:
                    if os.name != 'nt':  # Not Windows
                        import select
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            ch = sys.stdin.read(1)
                            if ch == 'd':
                                self.toggle_dark_mode()
                except:
                    pass  # Ignore errors in reading from stdin

            time.sleep(0.1)  # Update 10 times per second

    def is_analysis_complete(self) -> bool:
        """Check if all analysts have completed their analysis.

        Returns:
            True if all analysts are done, False otherwise
        """
        with self._lock:
            total_expected = len(self._selected_analysts) * len(set(ticker for statuses in self._status.values() for ticker in statuses))
            if total_expected == 0:
                return False
            return len(self._completed_analysts) >= total_expected

# Global instance - initialized with light mode by default
progress = ProgressTracker(dark_mode=False)