import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import threading
import json
from datetime import datetime, timedelta

class Position:
    """Represents a trading position"""
    
    def __init__(self, instrument: str, entry_price: float, quantity: float, 
                entry_time: float, direction: str, stop_loss: float, take_profit: float,
                order_id: str = None):
        self.instrument = instrument
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_id = order_id
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.status = "open"
        self.exit_reason = None
        
    def update_stop_loss(self, new_stop: float):
        """Update stop loss level"""
        self.stop_loss = new_stop
        
    def update_take_profit(self, new_take_profit: float):
        """Update take profit level"""
        self.take_profit = new_take_profit
        
    def close(self, exit_price: float, exit_time: float, reason: str):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "closed"
        self.exit_reason = reason
        
        # Calculate P&L
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            "instrument": self.instrument,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "direction": self.direction,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "order_id": self.order_id,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "status": self.status,
            "exit_reason": self.exit_reason
        }

class PortfolioManager:
    """Manages trading positions and overall portfolio risk"""
    
    def __init__(self, initial_capital: float, risk_per_trade: float = 0.02, 
                max_correlated_risk: float = 0.06, max_drawdown: float = 0.15):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # 2% per trade
        self.max_correlated_risk = max_correlated_risk  # 6% max for correlated instruments
        self.max_drawdown = max_drawdown  # 15% max drawdown
        
        self.positions = []  # All positions (open and closed)
        self.open_positions = {}  # Open positions by instrument
        
        self.peak_capital = initial_capital
        self.daily_pnl = {}
        
        self.correlations = {}  # Correlation matrix for instruments
        self.instrument_groups = {}  # Groups of correlated instruments
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, instrument: str, entry_price: float, 
                               stop_loss: float, direction: str) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            instrument: Trading instrument
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'long' or 'short'
            
        Returns:
            Position size
        """
        # Calculate risk amount in dollars
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate risk per unit
        if direction == "long":
            risk_per_unit = entry_price - stop_loss
        else:  # short
            risk_per_unit = stop_loss - entry_price
            
        # Ensure risk_per_unit is positive
        risk_per_unit = abs(risk_per_unit)
        
        if risk_per_unit <= 0:
            self.logger.warning(f"Invalid risk per unit: {risk_per_unit}")
            return 0
            
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Apply correlation constraints
        correlated_risk = self.calculate_correlated_risk(instrument)
        if correlated_risk + self.risk_per_trade > self.max_correlated_risk:
            # Scale down position to respect correlation limits
            scale_factor = (self.max_correlated_risk - correlated_risk) / self.risk_per_trade
            position_size *= scale_factor
            self.logger.info(f"Position size reduced due to correlation constraints: {scale_factor:.2f}")
        
        return position_size
    
    def calculate_correlated_risk(self, instrument: str) -> float:
        """
        Calculate risk from correlated instruments
        
        Args:
            instrument: Trading instrument
            
        Returns:
            Total risk from correlated positions
        """
        # Get correlation group for instrument
        group = self.get_correlation_group(instrument)
        if not group:
            return 0
            
        # Sum risk for all instruments in the group
        total_risk = 0
        for instr in group:
            if instr in self.open_positions:
                # Calculate current risk as percentage of capital
                position = self.open_positions[instr]
                risk = (position.entry_price - position.stop_loss) * position.quantity
                risk_pct = abs(risk) / self.current_capital
                total_risk += risk_pct
                
        return total_risk
    
    def get_correlation_group(self, instrument: str) -> List[str]:
        """Get group of correlated instruments"""
        # Find the group containing the instrument
        for group_name, instruments in self.instrument_groups.items():
            if instrument in instruments:
                return instruments
        
        # If not found, return empty list
        return []
    
    def update_correlations(self, price_data: Dict[str, pd.DataFrame], lookback_days: int = 90):
        """
        Update correlation matrix for instruments
        
        Args:
            price_data: Dictionary mapping instruments to price dataframes
            lookback_days: Number of days to consider for correlation
        """
        # Extract recent closing prices
        closing_prices = {}
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        for instrument, df in price_data.items():
            if 'Close' in df.columns:
                # Filter for recent data
                recent_data = df[df.index >= cutoff_date]['Close']
                if not recent_data.empty:
                    closing_prices[instrument] = recent_data
        
        # Create DataFrame with all closing prices
        price_df = pd.DataFrame(closing_prices)
        
        # Calculate correlation matrix
        if not price_df.empty and price_df.shape[1] > 1:
            correlation_matrix = price_df.pct_change().corr()
            self.correlations = correlation_matrix.to_dict()
            
            # Update instrument groups
            self.update_instrument_groups()
    
    def update_instrument_groups(self, correlation_threshold: float = 0.7):
        """
        Group instruments based on correlation
        
        Args:
            correlation_threshold: Threshold for considering instruments correlated
        """
        # Reset groups
        self.instrument_groups = {}
        
        # Get list of instruments
        instruments = list(self.correlations.keys())
        
        # Create groups
        grouped = set()
        for i, instr1 in enumerate(instruments):
            if instr1 in grouped:
                continue
                
            # Create new group
            group = [instr1]
            grouped.add(instr1)
            
            # Find correlated instruments
            for instr2 in instruments[i+1:]:
                if instr2 in grouped:
                    continue
                    
                corr = self.correlations.get(instr1, {}).get(instr2, 0)
                if abs(corr) >= correlation_threshold:
                    group.append(instr2)
                    grouped.add(instr2)
            
            # Add group if it has at least one instrument
            if group:
                self.instrument_groups[f"group_{len(self.instrument_groups)+1}"] = group
    
    def can_enter_position(self, instrument: str) -> bool:
        """
        Check if a new position can be entered
        
        Args:
            instrument: Trading instrument
            
        Returns:
            True if new position is allowed, False otherwise
        """
        # Check drawdown limit
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.max_drawdown:
            self.logger.warning(f"Maximum drawdown reached: {current_drawdown:.2%}")
            return False
            
        # Check if already in position for this instrument
        if instrument in self.open_positions:
            self.logger.warning(f"Already in position for {instrument}")
            return False
            
        return True
    
    def open_position(self, instrument: str, entry_price: float, quantity: float,
                     direction: str, stop_loss: float, take_profit: float,
                     order_id: str = None) -> Optional[Position]:
        """
        Open a new position
        
        Args:
            instrument: Trading instrument
            entry_price: Entry price
            quantity: Position size
            direction: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_id: Order ID from broker (optional)
            
        Returns:
            New position or None if not allowed
        """
        # Check if can enter position
        if not self.can_enter_position(instrument):
            return None
            
        # Create new position
        position = Position(
            instrument=instrument,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=time.time(),
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id
        )
        
        # Add to positions lists
        self.positions.append(position)
        self.open_positions[instrument] = position
        
        self.logger.info(f"Opened {direction} position in {instrument}: {quantity} @ {entry_price}")
        
        return position
    
    def close_position(self, instrument: str, exit_price: float, reason: str) -> Optional[Position]:
        """
        Close an open position
        
        Args:
            instrument: Trading instrument
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Closed position or None if not found
        """
        # Check if position exists
        if instrument not in self.open_positions:
            self.logger.warning(f"No open position found for {instrument}")
            return None
            
        # Get position
        position = self.open_positions[instrument]
        
        # Close position
        position.close(exit_price, time.time(), reason)
        
        # Update capital
        self.current_capital += position.pnl
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        # Remove from open positions
        del self.open_positions[instrument]
        
        # Update daily P&L
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += position.pnl
        
        self.logger.info(f"Closed {position.direction} position in {instrument}: {position.pnl:.2f} profit/loss")
        
        return position
    
    def update_position(self, instrument: str, current_price: float) -> bool:
        """
        Update position with current market price (check for stops)
        
        Args:
            instrument: Trading instrument
            current_price: Current market price
            
        Returns:
            True if position was closed, False otherwise
        """
        # Check if position exists
        if instrument not in self.open_positions:
            return False
            
        # Get position
        position = self.open_positions[instrument]
        
        # Check stops
        if position.direction == "long":
            if current_price <= position.stop_loss:
                self.close_position(instrument, position.stop_loss, "stop_loss")
                return True
            elif current_price >= position.take_profit:
                self.close_position(instrument, position.take_profit, "take_profit")
                return True
        else:  # short
            if current_price >= position.stop_loss:
                self.close_position(instrument, position.stop_loss, "stop_loss")
                return True
            elif current_price <= position.take_profit:
                self.close_position(instrument, position.take_profit, "take_profit")
                return True
                
        return False
    
    def update_trailing_stops(self, instrument: str, current_price: float, 
                             trailing_pct: float = 0.02) -> None:
        """
        Update trailing stop for a position
        
        Args:
            instrument: Trading instrument
            current_price: Current market price
            trailing_pct: Trailing stop percentage
        """
        # Check if position exists
        if instrument not in self.open_positions:
            return
            
        # Get position
        position = self.open_positions[instrument]
        
        # Update trailing stop for long positions
        if position.direction == "long":
            # Calculate new stop loss level
            new_stop = current_price * (1 - trailing_pct)
            
            # Only move stop loss up, never down
            if new_stop > position.stop_loss:
                position.update_stop_loss(new_stop)
                self.logger.info(f"Updated trailing stop for {instrument} to {new_stop:.2f}")
                
        # Update trailing stop for short positions
        else:  # short
            # Calculate new stop loss level
            new_stop = current_price * (1 + trailing_pct)
            
            # Only move stop loss down, never up
            if new_stop < position.stop_loss:
                position.update_stop_loss(new_stop)
                self.logger.info(f"Updated trailing stop for {instrument} to {new_stop:.2f}")
    
    def get_portfolio_stats(self) -> Dict:
        """Get portfolio statistics"""
        # Calculate realized P&L
        realized_pnl = sum(pos.pnl for pos in self.positions if pos.status == "closed")
        
        # Calculate unrealized P&L
        unrealized_pnl = 0
        # This would require current market prices for all open positions
        
        # Calculate max drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Calculate win rate
        closed_positions = [pos for pos in self.positions if pos.status == "closed"]
        winning_positions = [pos for pos in closed_positions if pos.pnl > 0]
        
        win_rate = len(winning_positions) / len(closed_positions) if closed_positions else 0
        
        # Calculate average win/loss
        avg_win = np.mean([pos.pnl for pos in winning_positions]) if winning_positions else 0
        avg_loss = np.mean([pos.pnl for pos in closed_positions if pos.pnl < 0]) if closed_positions else 0
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "drawdown": drawdown,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_trades": len(closed_positions),
            "open_positions": len(self.open_positions),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            "daily_pnl": self.daily_pnl
        }
    
    def to_json(self) -> str:
        """Convert portfolio to JSON string"""
        data = {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "peak": self.peak_capital
            },
            "positions": [pos.to_dict() for pos in self.positions],
            "open_positions": {k: v.to_dict() for k, v in self.open_positions.items()},
            "daily_pnl": self.daily_pnl,
            "stats": self.get_portfolio_stats()
        }
        
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_data: str) -> 'PortfolioManager':
        """Create portfolio from JSON string"""
        data = json.loads(json_data)
        
        # Create portfolio manager
        portfolio = cls(
            initial_capital=data["capital"]["initial"]
        )
        
        # Update capital
        portfolio.current_capital = data["capital"]["current"]
        portfolio.peak_capital = data["capital"]["peak"]
        
        # Load positions
        for pos_data in data["positions"]:
            position = Position(
                instrument=pos_data["instrument"],
                entry_price=pos_data["entry_price"],
                quantity=pos_data["quantity"],
                entry_time=pos_data["entry_time"],
                direction=pos_data["direction"],
                stop_loss=pos_data["stop_loss"],
                take_profit=pos_data["take_profit"],
                order_id=pos_data["order_id"]
            )
            
            # Set other attributes
            if pos_data["status"] == "closed":
                position.close(
                    exit_price=pos_data["exit_price"],
                    exit_time=pos_data["exit_time"],
                    reason=pos_data["exit_reason"]
                )
            
            # Add to positions list
            portfolio.positions.append(position)
            
            # Add to open positions if still open
            if position.status == "open":
                portfolio.open_positions[position.instrument] = position
        
        # Load daily P&L
        portfolio.daily_pnl = data["daily_pnl"]
        
        return portfolio