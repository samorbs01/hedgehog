"""Logging configuration for the Hedgehog AI Hedge Fund."""

import os
import logfire
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
if not LOGFIRE_TOKEN:
    raise ValueError(
        "LOGFIRE_TOKEN environment variable is not set. "
        "Please set it in your .env file"
    )

STAGE = os.getenv("STAGE")
if not STAGE:
    raise ValueError(
        "STAGE environment variable is not set. "
        "Please set it in your .env file (e.g., STAGE=development)"
    )

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )

# Configure logging
logfire.configure(token=LOGFIRE_TOKEN, environment=STAGE)

# Instrument OpenAI calls for monitoring
logfire.instrument_openai()