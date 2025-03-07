"""API tools for fetching financial data to power the hedge fund analysis."""

import os
import aiohttp
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
FINDATA_API_KEY = os.getenv("FINDATA_API_KEY")
CHART_IMG_API_KEY = os.getenv("CHART_IMG_API_KEY")


async def fetch_company_data(ticker: str) -> Dict[str, Any]:
    """Fetch comprehensive company data for a given ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict containing company information, financials, and statistics
    """

    async with aiohttp.ClientSession() as session:
        # Basic company info
        company_url = f"https://financialdatasets.ai/api/v1/companies/{ticker}?apikey={FINDATA_API_KEY}"
        async with session.get(company_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch company data for {ticker}: {response.status}")
            company_data = await response.json()

        # Financial statements
        financials_url = f"https://financialdatasets.ai/api/v1/financials/{ticker}?apikey={FINDATA_API_KEY}"
        async with session.get(financials_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch financial data for {ticker}: {response.status}")
            financial_data = await response.json()

        # Combine the data
        result = {
            "company_info": company_data,
            "financials": financial_data
        }

        return result


async def fetch_price_history(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Fetch historical price data for a given ticker.

    Args:
        ticker: Stock ticker symbol
        period: Time period for the data (e.g., '1d', '1m', '1y')

    Returns:
        Dict containing historical price data
    """

    async with aiohttp.ClientSession() as session:
        price_url = f"https://financialdatasets.ai/api/v1/prices/{ticker}?period={period}&apikey={FINDATA_API_KEY}"
        async with session.get(price_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch price history for {ticker}: {response.status}")
            price_data = await response.json()

        return price_data


async def fetch_news_data(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent news articles for a given ticker.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of news articles to retrieve

    Returns:
        List of recent news articles
    """
    # This is a placeholder. In a real implementation, you would call actual
    # news APIs here.

    async with aiohttp.ClientSession() as session:
        news_url = f"https://financialdatasets.ai/api/v1/news/{ticker}?limit={limit}&apikey={FINDATA_API_KEY}"
        async with session.get(news_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch news for {ticker}: {response.status}")
            news_data = await response.json()

        return news_data


async def fetch_peer_companies(ticker: str) -> List[str]:
    """Fetch peer companies for a given ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of peer company ticker symbols
    """

    async with aiohttp.ClientSession() as session:
        peers_url = f"https://financialdatasets.ai/api/v1/peers/{ticker}?apikey={FINDATA_API_KEY}"
        async with session.get(peers_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch peer companies for {ticker}: {response.status}")
            peer_data = await response.json()

        return peer_data.get("peers", [])


async def generate_chart_image(ticker: str, chart_type: str = "candlestick", period: str = "1y") -> str:
    """Generate a chart image URL for a given ticker.

    Args:
        ticker: Stock ticker symbol
        chart_type: Type of chart (e.g., 'candlestick', 'line')
        period: Time period for the chart (e.g., '1d', '1m', '1y')

    Returns:
        URL to the generated chart image
    """

    chart_url = f"https://api.chart-img.com/v1/charts/stock?symbol={ticker}&key={CHART_IMG_API_KEY}&type={chart_type}&period={period}"

    return chart_url