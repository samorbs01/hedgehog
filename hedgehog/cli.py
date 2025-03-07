"""CLI utilities for interactive selection of options."""

from typing import List, Optional, Dict, Any, Tuple
import sys
import termios
import tty
import os

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Available analyst options
ANALYSTS = [
    "Ben Graham",
    "Warren Buffett",
    "Charlie Munger",
    "Bill Ackman",
    "Cathie Wood",
    "Fundamental Analyst",
    "Technical Analyst",
    "Sentiment Analyst",
    "Valuation Analyst"
]

# Available OpenRouter models
OPENROUTER_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4.5-preview",
    "openai/o3-mini",
    "openai/o3-mini-high",
    "openai/o1",
    "openai/o1-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1-distill-qwen-32b",
]


def _getch():
    """Read a single character from stdin without waiting for enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def select_analysts() -> List[str]:
    """Interactive CLI for selecting analysts using spacebar.

    Returns:
        List of selected analyst names
    """
    # Initialize selected analysts
    selected = [False] * len(ANALYSTS)
    current_pos = 0

    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Display instructions
        print("Select your AI analysts.")
        print("Instructions:")
        print("1. Press Space to select/unselect analysts.")
        print("2. Press 'a' to select/unselect all.")
        print("3. Press Enter when done to run the hedge fund.")
        print("")

        # Display analysts with selection status
        for i, analyst in enumerate(ANALYSTS):
            prefix = "»" if i == current_pos else " "
            checkbox = "○" if not selected[i] else "●"
            print(f"{prefix} {checkbox} {analyst}")

        # Get key press
        key = _getch()

        if key == ' ':  # Spacebar to toggle selection
            selected[current_pos] = not selected[current_pos]
        elif key.lower() == 'a':  # 'a' to toggle all
            all_selected = all(selected)
            selected = [not all_selected] * len(ANALYSTS)
        elif key == '\r' or key == '\n':  # Enter to confirm
            break
        elif key == '\x1b':  # Handle arrow keys (escape sequence)
            next_key = _getch()  # Skip the '['
            if next_key == '[':
                direction = _getch()
                if direction == 'A':  # Up arrow
                    current_pos = (current_pos - 1) % len(ANALYSTS)
                elif direction == 'B':  # Down arrow
                    current_pos = (current_pos + 1) % len(ANALYSTS)
        elif key == 'k':  # Alternative up navigation
            current_pos = (current_pos - 1) % len(ANALYSTS)
        elif key == 'j':  # Alternative down navigation
            current_pos = (current_pos + 1) % len(ANALYSTS)

    # Get the selected analysts names
    selected_analysts = [ANALYSTS[i] for i, is_selected in enumerate(selected) if is_selected]

    # If nothing selected, default to some analysts
    if not selected_analysts:
        selected_analysts = ["Warren Buffett", "Fundamental Analyst", "Technical Analyst"]
        print(f"\nNo analysts selected. Using defaults: {GREEN}{', '.join(selected_analysts)}{RESET}")
    else:
        print(f"\nSelected analysts: {GREEN}{', '.join(selected_analysts)}{RESET}")

    return selected_analysts


def select_model() -> str:
    """Interactive CLI for selecting an OpenRouter model using spacebar.

    Returns:
        Selected model identifier
    """
    # Initialize selection
    current_pos = 0
    selected_idx = 0  # Default to first model

    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Display instructions
        print("Select your OpenRouter model.")
        print("Instructions:")
        print("1. Use Up/Down arrows or j/k keys to navigate.")
        print("2. Press Space or Enter to select a model.")
        print("")

        # Display models
        for i, model in enumerate(OPENROUTER_MODELS):
            prefix = "»" if i == current_pos else " "
            indicator = "●" if i == selected_idx else "○"
            print(f"{prefix} {indicator} {model}")

        # Get key press
        key = _getch()

        if key == ' ':  # Spacebar to select
            selected_idx = current_pos
            break
        elif key == '\r' or key == '\n':  # Enter to confirm selection
            selected_idx = current_pos
            break
        elif key == '\x1b':  # Handle arrow keys (escape sequence)
            next_key = _getch()  # Skip the '['
            if next_key == '[':
                direction = _getch()
                if direction == 'A':  # Up arrow
                    current_pos = (current_pos - 1) % len(OPENROUTER_MODELS)
                elif direction == 'B':  # Down arrow
                    current_pos = (current_pos + 1) % len(OPENROUTER_MODELS)
        elif key == 'k':  # Alternative up navigation
            current_pos = (current_pos - 1) % len(OPENROUTER_MODELS)
        elif key == 'j':  # Alternative down navigation
            current_pos = (current_pos + 1) % len(OPENROUTER_MODELS)

    selected_model = OPENROUTER_MODELS[selected_idx]
    print(f"\nSelected model: {GREEN}{selected_model}{RESET}")
    return selected_model