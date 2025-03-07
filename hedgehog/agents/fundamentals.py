"""Fundamental analysis agent for evaluating company financials."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Schema, agent, model


class FinancialMetrics(BaseModel):
    """Key financial metrics for a company."""

    pe_ratio: Optional[float] = Field(None, description="Price to Earnings ratio")
    pb_ratio: Optional[float] = Field(None, description="Price to Book ratio")
    roe: Optional[float] = Field(None, description="Return on Equity")
    debt_to_equity: Optional[float] = Field(None, description="Debt to Equity ratio")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    profit_margin: Optional[float] = Field(None, description="Profit margin")
    free_cash_flow: Optional[float] = Field(None, description="Free Cash Flow in millions")


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis of a company."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    sector: str = Field(..., description="Industry sector")
    metrics: FinancialMetrics = Field(..., description="Key financial metrics")
    strengths: List[str] = Field(..., description="Fundamental strengths")
    weaknesses: List[str] = Field(..., description="Fundamental weaknesses")
    rating: int = Field(..., ge=1, le=10, description="Overall rating from 1-10")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")


@agent
async def analyze_fundamentals(ticker: str, financial_data: Dict[str, Any]) -> FundamentalAnalysis:
    """Analyze fundamental metrics for a company and provide an investment recommendation.

    Args:
        ticker: The stock ticker symbol
        financial_data: Dictionary containing financial data for the company

    Returns:
        FundamentalAnalysis: A comprehensive fundamental analysis with recommendation
    """
    # This function will be executed by the AI model, which will fill in the analysis
    # based on the financial_data provided
    pass