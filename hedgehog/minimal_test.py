"""A minimal test file to verify imports are working correctly."""

import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables
load_dotenv()

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )

async def main():
    """Run a simple test of the imports."""
    # Set up the model
    model = OpenAIModel(
        "anthropic/claude-3.5-sonnet",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        ),
    )

    # Create an agent
    agent = Agent(model)

    # Run a simple analysis
    ticker = "AAPL"
    prompt = f"""
    Create a greeting for ticker {ticker}.

    Respond with a friendly greeting and a message about this ticker.
    """

    # Run the agent and get the result
    result = await agent.run(prompt)

    print(f"Response: {result.data}")


if __name__ == "__main__":
    asyncio.run(main())