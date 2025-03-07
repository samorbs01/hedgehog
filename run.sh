#!/bin/bash

# Check if Python is available through the project's virtual environment
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON="venv/bin/python"
else
    # Try system python3
    PYTHON="python3"
fi

# Check if Python is available
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: Python interpreter not found. Please ensure Python is installed."
    exit 1
fi

# Use dark mode by default
DARK_MODE=1

# Check for mode flags
for arg in "$@"; do
    if [ "$arg" == "--light-mode" ]; then
        DARK_MODE=0
        # Remove --light-mode from args
        args=()
        for val in "$@"; do
            if [ "$val" != "--light-mode" ]; then
                args+=("$val")
            fi
        done
        set -- "${args[@]}"
    elif [ "$arg" == "--dark-mode" ]; then
        # Remove redundant --dark-mode flag since it's the default
        args=()
        for val in "$@"; do
            if [ "$val" != "--dark-mode" ]; then
                args+=("$val")
            fi
        done
        set -- "${args[@]}"
    fi
done

# Export dark mode setting as environment variable
export HEDGEHOG_DARK_MODE=$DARK_MODE

# Display mode info
if [ $DARK_MODE -eq 1 ]; then
    echo -e "\033[36m[DARK MODE ENABLED]\033[0m"
else
    echo -e "\033[96m[LIGHT MODE ENABLED]\033[0m"
fi

# Default to analyze AAPL, NVDA, and MSFT with interactive mode if no args provided
if [ $# -eq 0 ]; then
    $PYTHON -m hedgehog.main analyze AAPL NVDA MSFT --interactive
else
    # Execute with the provided arguments
    $PYTHON -m hedgehog.main "$@"
fi