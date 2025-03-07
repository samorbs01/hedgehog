from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio
import os
from hedgehog.tools.logfire_setup import *


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )


def setup_agent() -> Agent:
    """Create and configure an Agent with OpenAI model using OpenRouter.

    Returns:
        Agent: Configured agent instance using Claude 3.5 Sonnet model via OpenRouter
    """
    model = OpenAIModel(
        "anthropic/claude-3.5-sonnet",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        ),
    )
    return Agent(model)


async def call_agent(prompt: str) -> AgentRunResult:
    """Generate a prediction response for the given prompt.

    Args:
        prompt (str): The input prompt to send to the model

    Returns:
        str: The model's response text
    """
    agent = setup_agent()
    return await agent.run(prompt)


if __name__ == "__main__":
    result = asyncio.run(call_agent("How does pyodide let you run Python in the browser? (short answer please)"))
    print(result.data)