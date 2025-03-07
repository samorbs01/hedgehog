"""Technical analysis agent for evaluating price patterns and indicators."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import agent


class TechnicalIndicators(BaseModel):
    """Key technical indicators for a stock."""

    moving_average_50d: Optional[float] = Field(None, description="50-day moving average")
    moving_average_200d: Optional[float] = Field(None, description="200-day moving average")
    rsi_14: Optional[float] = Field(None, description="14-day Relative Strength Index")
    macd: Optional[float] = Field(None, description="Moving Average Convergence Divergence")
    bollinger_upper: Optional[float] = Field(None, description="Upper Bollinger Band")
    bollinger_lower: Optional[float] = Field(None, description="Lower Bollinger Band")
    volume_avg_30d: Optional[float] = Field(None, description="30-day average volume")


class TechnicalAnalysis(BaseModel):
    """Technical analysis of a stock's price action."""

    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    indicators: TechnicalIndicators = Field(..., description="Technical indicators")
    support_levels: List[float] = Field(..., description="Identified support price levels")
    resistance_levels: List[float] = Field(..., description="Identified resistance price levels")
    patterns: List[str] = Field(..., description="Identified chart patterns")
    signals: List[str] = Field(..., description="Technical signals identified")
    rating: int = Field(..., ge=1, le=10, description="Overall technical rating from 1-10")
    recommendation: str = Field(..., description="Technical recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")


@agent
async def analyze_technicals(ticker: str, price_history: Dict[str, Any]) -> TechnicalAnalysis:
    """Analyze technical indicators and price patterns for a stock.

    Args:
        ticker: The stock ticker symbol
        price_history: Dictionary containing historical price data

    Returns:
        TechnicalAnalysis: A comprehensive technical analysis with recommendation
    """
    # This function will be executed by the AI model, which will analyze the price_history
    # and provide technical analysis
    pass