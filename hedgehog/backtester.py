"""Backtester for simulating hedge fund performance on historical data."""

import asyncio
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from hedgehog.workflow import analyze_company
from hedgehog.tools.api import fetch_price_history


class BacktestParameters(BaseModel):
    """Parameters for running a backtest."""

    tickers: List[str] = Field(..., description="List of tickers to include in the backtest")
    start_date: datetime = Field(..., description="Start date for the backtest")
    end_date: datetime = Field(..., description="End date for the backtest")
    initial_capital: float = Field(..., description="Initial capital for the portfolio")
    max_positions: int = Field(..., description="Maximum number of positions allowed")
    position_size_limit: float = Field(..., description="Maximum position size as percentage")
    rebalance_frequency: int = Field(..., description="Rebalance frequency in days")
    stop_loss_enabled: bool = Field(..., description="Whether to use stop-loss for positions")


class BacktestPosition(BaseModel):
    """An individual position in the backtest."""

    ticker: str = Field(..., description="Stock ticker symbol")
    entry_date: datetime = Field(..., description="Entry date")
    entry_price: float = Field(..., description="Entry price")
    shares: float = Field(..., description="Number of shares")
    cost_basis: float = Field(..., description="Total cost basis")
    stop_loss: Optional[float] = Field(None, description="Stop-loss price")
    target_price: float = Field(..., description="Target price")
    exit_date: Optional[datetime] = Field(None, description="Exit date if position closed")
    exit_price: Optional[float] = Field(None, description="Exit price if position closed")
    is_active: bool = Field(True, description="Whether the position is still active")
    order_type: str = Field(..., description="Type of order (BUY/SELL/HOLD)")
    pnl: Optional[float] = Field(None, description="Realized P&L if position closed")
    pnl_percent: Optional[float] = Field(None, description="Realized P&L percent if position closed")


class BacktestPortfolio(BaseModel):
    """Portfolio state during a backtest."""

    date: datetime = Field(..., description="Current date in the backtest")
    positions: List[BacktestPosition] = Field(default_factory=list, description="Active positions")
    cash: float = Field(..., description="Available cash")
    equity: float = Field(..., description="Total portfolio equity")
    daily_returns: List[float] = Field(default_factory=list, description="Daily return percentages")
    cumulative_returns: List[float] = Field(default_factory=list, description="Cumulative return percentages")

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the portfolio."""
        if not self.daily_returns:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }

        import numpy as np

        returns = np.array(self.daily_returns)
        total_return = (self.equity / self.cash) - 1.0
        annualized_return = ((1 + total_return) ** (252 / len(returns))) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

        # Calculate max drawdown
        cumulative = np.array(self.cumulative_returns)
        max_drawdown = np.min(cumulative / np.maximum.accumulate(cumulative) - 1)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }


class BacktestResult(BaseModel):
    """Results from a completed backtest."""

    params: BacktestParameters = Field(..., description="Backtest parameters")
    final_portfolio: BacktestPortfolio = Field(..., description="Final portfolio state")
    closed_positions: List[BacktestPosition] = Field(..., description="All closed positions")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    portfolio_history: List[Tuple[datetime, float]] = Field(..., description="Portfolio equity history")


async def run_backtest(params: BacktestParameters) -> BacktestResult:
    """Run a backtest with the given parameters.

    Args:
        params: Backtest parameters

    Returns:
        BacktestResult: Results from the completed backtest
    """
    # Set up the model for analysis
    model = OpenAIModel(
        "anthropic/claude-3.5-sonnet",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        ),
    )

    # Initialize portfolio
    portfolio = BacktestPortfolio(
        date=params.start_date,
        cash=params.initial_capital,
        equity=params.initial_capital,
        positions=[],
        daily_returns=[],
        cumulative_returns=[]
    )

    # Keep track of closed positions
    closed_positions = []

    # Portfolio equity history for tracking performance
    portfolio_history = [(params.start_date, params.initial_capital)]

    # Generate date range for the backtest
    current_date = params.start_date
    while current_date <= params.end_date:
        # Skip weekends (simplistic approach)
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Update portfolio date
        portfolio.date = current_date

        # Check if it's time to rebalance
        should_rebalance = (
            (current_date - params.start_date).days % params.rebalance_frequency == 0
            or current_date == params.start_date
        )

        # If it's time to rebalance, analyze tickers and update positions
        if should_rebalance:
            for ticker in params.tickers:
                # Skip if we already have a position for this ticker
                existing_position = next((p for p in portfolio.positions if p.ticker == ticker), None)
                if existing_position:
                    continue

                # Run analysis for the ticker
                try:
                    analysis = await analyze_company(ticker, model)

                    # Get current price
                    price_history = await fetch_price_history(ticker)
                    current_price = price_history.get("latest_price", 0)

                    if current_price <= 0:
                        continue

                    decision = analysis.investment_decision

                    # If decision is to buy and we have cash
                    if (
                        decision.order_type == "BUY"
                        and decision.conviction_level >= 7
                        and len(portfolio.positions) < params.max_positions
                    ):
                        # Calculate position size
                        position_size = min(
                            decision.position_size / 100,
                            params.position_size_limit / 100
                        )
                        position_value = portfolio.cash * position_size

                        # Calculate shares to buy
                        shares = position_value / current_price

                        # Create and add position
                        if shares > 0 and position_value > 0:
                            position = BacktestPosition(
                                ticker=ticker,
                                entry_date=current_date,
                                entry_price=current_price,
                                shares=shares,
                                cost_basis=position_value,
                                stop_loss=decision.stop_loss,
                                target_price=decision.target_price,
                                order_type=decision.order_type
                            )

                            # Update portfolio
                            portfolio.positions.append(position)
                            portfolio.cash -= position_value
                except Exception as e:
                    print(f"Error analyzing {ticker}: {e}")

        # Update positions and check for exits
        updated_positions = []
        for position in portfolio.positions:
            # Fetch latest price data
            try:
                price_history = await fetch_price_history(position.ticker)
                current_price = price_history.get("latest_price", 0)

                if current_price <= 0:
                    updated_positions.append(position)
                    continue

                # Check for stop loss or target price
                hit_stop_loss = (
                    params.stop_loss_enabled
                    and position.stop_loss
                    and current_price <= position.stop_loss
                )
                hit_target = current_price >= position.target_price

                # Exit if stop loss or target hit
                if hit_stop_loss or hit_target:
                    # Update position data
                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.is_active = False

                    # Calculate P&L
                    if position.order_type == "BUY":
                        position.pnl = (current_price - position.entry_price) * position.shares
                        position.pnl_percent = (current_price / position.entry_price) - 1
                    else:  # SELL (short)
                        position.pnl = (position.entry_price - current_price) * position.shares
                        position.pnl_percent = 1 - (current_price / position.entry_price)

                    # Return cash to portfolio
                    portfolio.cash += current_price * position.shares

                    # Add to closed positions
                    closed_positions.append(position)
                else:
                    # Keep position active
                    updated_positions.append(position)
            except Exception as e:
                print(f"Error updating position {position.ticker}: {e}")
                updated_positions.append(position)

        # Update portfolio positions
        portfolio.positions = updated_positions

        # Calculate portfolio equity
        portfolio_value = portfolio.cash
        for position in portfolio.positions:
            try:
                price_history = await fetch_price_history(position.ticker)
                current_price = price_history.get("latest_price", 0)

                if current_price > 0:
                    portfolio_value += position.shares * current_price
            except:
                # If we can't get the price, use the entry price as an approximation
                portfolio_value += position.shares * position.entry_price

        # Update portfolio equity
        previous_equity = portfolio.equity
        portfolio.equity = portfolio_value

        # Calculate return for the day
        daily_return = (portfolio.equity / previous_equity) - 1 if previous_equity > 0 else 0
        portfolio.daily_returns.append(daily_return)

        # Calculate cumulative return
        cumulative_return = (portfolio.equity / params.initial_capital) - 1
        portfolio.cumulative_returns.append(cumulative_return)

        # Add to portfolio history
        portfolio_history.append((current_date, portfolio.equity))

        # Move to next day
        current_date += timedelta(days=1)

    # Calculate final performance metrics
    performance_metrics = portfolio.calculate_metrics()

    # Create and return the backtest result
    return BacktestResult(
        params=params,
        final_portfolio=portfolio,
        closed_positions=closed_positions,
        performance_metrics=performance_metrics,
        portfolio_history=portfolio_history
    )


if __name__ == "__main__":
    # Example backtest parameters
    parameters = BacktestParameters(
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=1000000.0,
        max_positions=10,
        position_size_limit=10.0,  # 10% maximum position size
        rebalance_frequency=30,  # Rebalance monthly
        stop_loss_enabled=True
    )

    # Run the backtest
    result = asyncio.run(run_backtest(parameters))

    # Print results
    print(f"Backtest completed from {parameters.start_date} to {parameters.end_date}")
    print(f"Final portfolio value: ${result.final_portfolio.equity:,.2f}")
    print(f"Total return: {result.performance_metrics['total_return']:.2%}")
    print(f"Annualized return: {result.performance_metrics['annualized_return']:.2%}")
    print(f"Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum drawdown: {result.performance_metrics['max_drawdown']:.2%}")
    print(f"Number of trades: {len(result.closed_positions)}")

    winning_trades = [p for p in result.closed_positions if p.pnl and p.pnl > 0]
    losing_trades = [p for p in result.closed_positions if p.pnl and p.pnl <= 0]

    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(result.closed_positions):.2%})")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(result.closed_positions):.2%})")