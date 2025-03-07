"""Valuation analysis agent for calculating intrinsic value of companies."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Schema, agent, model


class DCFAssumptions(BaseModel):
    """Assumptions for Discounted Cash Flow valuation."""

    growth_rate_5yr: float = Field(..., description="5-year growth rate assumption")
    growth_rate_terminal: float = Field(..., description="Terminal growth rate assumption")
    discount_rate: float = Field(..., description="Discount rate (WACC)")
    years_projected: int = Field(..., description="Number of years projected in DCF model")


class ComparableMultiples(BaseModel):
    """Comparable company valuation multiples."""

    avg_pe_ratio: Optional[float] = Field(None, description="Average P/E ratio of peers")
    avg_ps_ratio: Optional[float] = Field(None, description="Average P/S ratio of peers")
    avg_pb_ratio: Optional[float] = Field(None, description="Average P/B ratio of peers")
    avg_ev_ebitda: Optional[float] = Field(None, description="Average EV/EBITDA of peers")


class ValuationAnalysis(BaseModel):
    """Comprehensive valuation analysis of a company."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    current_price: float = Field(..., description="Current stock price")
    dcf_value: float = Field(..., description="DCF valuation per share")
    dcf_assumptions: DCFAssumptions = Field(..., description="DCF model assumptions")
    comparable_value: Optional[float] = Field(None, description="Comparable companies valuation per share")
    comparable_multiples: Optional[ComparableMultiples] = Field(None, description="Peer group multiples")
    intrinsic_value_range: List[float] = Field(..., description="Estimated intrinsic value range [low, high]")
    margin_of_safety: float = Field(..., description="Margin of safety percentage")
    rating: int = Field(..., ge=1, le=10, description="Overall valuation rating from 1-10")
    recommendation: str = Field(..., description="Valuation-based recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")


@agent
async def analyze_valuation(ticker: str, financial_data: Dict[str, Any], peer_data: Optional[List[Dict[str, Any]]] = None) -> ValuationAnalysis:
    """Perform a comprehensive valuation analysis for a company.

    Args:
        ticker: The stock ticker symbol
        financial_data: Dictionary containing financial data for the company
        peer_data: Optional list of dictionaries with peer company financial data

    Returns:
        ValuationAnalysis: A comprehensive valuation analysis with recommendation
    """
    # This function will be executed by the AI model, which will calculate valuation
    # based on the provided financial data
    pass