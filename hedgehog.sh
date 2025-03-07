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

# Execute the hedgehog program with all arguments passed to this script
$PYTHON -m hedgehog.main "$@"