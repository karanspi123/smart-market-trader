import logging
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Local imports
from portfolio_manager import PortfolioManager
from ninja_trader_client import NinjaTraderClient
from data_store import DataStore
from monitoring import MonitoringService

class TradeRequest:
    """Represents a trade request"""
    
    def __init__(self, instrument: str, direction: str, reason: str,
                confidence: float, signal_data: Dict = None):
        self.instrument = instrument
        self.direction = direction  # 'buy' or 'sell'
        self.reason = reason
        self.confidence = confidence
        self.signal_data = signal_data or {}
        self.timestamp = time.time()
        self.id = f"{instrument}_{self.timestamp}"
        
    def to_dict(self) -> Dict:
        """Convert trade request to dictionary"""
        return {
            "id": self.id,
            "instrument": self.instrument,
            "direction": self.direction,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "signal_data": self.signal_data
        }

class TradeResult:
    """Represents the result of a trade execution"""
    
    def __init__(self, request_id: str, success: bool, message: str,
                order_id: Optional[str] = None, position_id: Optional[str] = None,
                details: Dict = None):
        self.request_id = request_id
        self.success = success
        self.message = message
        self.order_id = order_id
        self.position_id = position_id
        self.details = details or {}
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict:
        """Convert trade result to dictionary"""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "message": self.message,
            "order_id": self.order_id,
            "position_id": self.position_id,
            "details": self.details,
            "timestamp": self.timestamp
        }

class TradeExecutor:
    """Handles trade execution with risk management"""
    
    def __init__(self, config: Dict, portfolio_manager: PortfolioManager,
                trader_client: NinjaTraderClient, data_store: DataStore,
                monitoring: MonitoringService):
        self.config = config
        self.portfolio = portfolio_manager
        self.trader = trader_client
        self.data_store = data_store
        self.monitoring = monitoring
        self.logger = logging.getLogger(__name__)
        
        # Configure trading parameters
        self.trading_hours = config.get("trading_hours", {})
        self.trade_min_confidence = config.get("min_confidence", 0.75)
        self.min_time_between_trades = config.get("min_time_between_trades", 300)  # 5 minutes
        self.max_trade_latency = config.get("max_trade_latency", 5.0)  # 5 seconds
        
        # Trade request queue
        self.trade_queue = queue.Queue()
        
        # Trade execution thread
        self.executor_thread = None
        self.running = False
        
        # Trade history
        self.trade_history = []
        self.last_trade_time = {}  # Last trade time by instrument
        
        # Market data cache
        self.market_data = {}
        
    def start(self):
        """Start the trade executor thread"""
        if self.executor_thread and self.executor_thread.is_alive():
            self.logger.warning("Trade executor already running")
            return
            
        self.running = True
        self.executor_thread = threading.Thread(
            target=self._executor_worker,
            daemon=True
        )
        self.executor_thread.start()
        self.logger.info("Trade executor started")
        
    def stop(self):
        """Stop the trade executor thread"""
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5.0)
            self.logger.info("Trade executor stopped")
    
    def _executor_worker(self):
        """Worker thread for processing trade requests"""
        while self.running:
            try:
                # Get trade request from queue (with timeout)
                request = self.trade_queue.get(timeout=1.0)
                
                # Process trade request
                result = self._process_trade_request(request)
                
                # Store result
                self.trade_history.append((request, result))
                
                # Log result
                if result.success:
                    self.logger.info(f"Trade executed: {result.message}")
                else:
                    self.logger.warning(f"Trade failed: {result.message}")
                    
                # Update monitoring
                self.monitoring.log_trade(
                    instrument=request.instrument,
                    direction=request.direction,
                    result=result.success,
                    latency=time.time() - request.timestamp,
                    reason=request.reason
                )
                
                # Mark task as done
                self.trade_queue.task_done()
                
            except queue.Empty:
                # No trade requests
                pass
            except Exception as e:
                self.logger.error(f"Error in trade executor: {e}")
                self.monitoring.log_error(
                    error_type="trade_executor",
                    error_message=str(e)
                )
    
    def _process_trade_request(self, request: TradeRequest) -> TradeResult:
        """
        Process a trade request
        
        Args:
            request: Trade request
            
        Returns:
            Trade result
        """
        # Check trading hours
        if not self._is_trading_allowed():
            return TradeResult(
                request_id=request.id,
                success=False,
                message="Trading not allowed during this time"
            )
            
        # Check confidence
        if request.confidence < self.trade_min_confidence:
            return TradeResult(
                request_id=request.id,
                success=False,
                message=f"Confidence too low: {request.confidence:.2f} < {self.trade_min_confidence:.2f}"
            )
            
        # Check time since last trade
        last_trade_time = self.last_trade_time.get(request.instrument, 0)
        if time.time() - last_trade_time < self.min_time_between_trades:
            return TradeResult(
                request_id=request.id,
                success=False,
                message="Too soon since last trade"
            )
            
        # Check latency
        latency = time.time() - request.timestamp
        if latency > self.max_trade_latency:
            return TradeResult(
                request_id=request.id,
                success=False,
                message=f"Trade request too old: {latency:.2f}s > {self.max_trade_latency:.2f}s"
            )
            
        # Get current market data
        market_data = self._get_market_data(request.instrument)
        if not market_data:
            return TradeResult(
                request_id=request.id,
                success=False,
                message="Failed to get market data"
            )
            
        # Calculate entry, stop loss, and take profit levels
        entry_price, stop_loss, take_profit = self._calculate_trade_levels(
            request.instrument,
            request.direction,
            market_data
        )
        
        # Calculate position size
        direction = "long" if request.direction == "buy" else "short"
        position_size = self.portfolio.calculate_position_size(
            instrument=request.instrument,
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction
        )
        
        if position_size <= 0:
            return TradeResult(
                request_id=request.id,
                success=False,
                message="Position size calculation failed"
            )
            
        # Prepare order
        order_data = {
            "instrument": request.instrument,
            "direction": request.direction,
            "quantity": position_size,
            "price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "order_type": "market"
        }
        
        # Place order
        try:
            order_result = self.trader.place_order(order_data)
            
            if not order_result.get("success", False):
                return TradeResult(
                    request_id=request.id,
                    success=False,
                    message=f"Order placement failed: {order_result.get('error', 'Unknown error')}",
                    details=order_result
                )
                
            # Update last trade time
            self.last_trade_time[request.instrument] = time.time()
            
            # Open position in portfolio
            position = self.portfolio.open_position(
                instrument=request.instrument,
                entry_price=entry_price,
                quantity=position_size,
                direction=direction,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_result.get("order_id")
            )
            
            if not position:
                return TradeResult(
                    request_id=request.id,
                    success=False,
                    message="Failed to open position in portfolio",
                    order_id=order_result.get("order_id")
                )
                
            # Successful trade
            return TradeResult(
                request_id=request.id,
                success=True,
                message=f"Order placed: {position_size} {request.instrument} @ {entry_price}",
                order_id=order_result.get("order_id"),
                position_id=str(position.entry_time),
                details={
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "direction": direction
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            
            return TradeResult(
                request_id=request.id,
                success=False,
                message=f"Order placement error: {str(e)}"
            )
    
    def _is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed based on trading hours"""
        if not self.trading_hours:
            # No restrictions
            return True
            
        now = datetime.now()
        weekday = now.strftime("%A").lower()
        
        # Check if day is in trading days
        if weekday not in self.trading_hours:
            return False
            
        # Check trading hours for this day
        day_hours = self.trading_hours[weekday]
        
        current_time = now.time()
        
        # Check each trading session
        for session in day_hours:
            start_time = datetime.strptime(session["start"], "%H:%M").time()
            end_time = datetime.strptime(session["end"], "%H:%M").time()
            
            if start_time <= current_time <= end_time:
                return True
                
        return False
    
    def _get_market_data(self, instrument: str) -> Optional[Dict]:
        """Get current market data for an instrument"""
        # Check cache
        if instrument in self.market_data:
            # Check if cache is still valid (10 seconds)
            if time.time() - self.market_data[instrument].get("timestamp", 0) < 10:
                return self.market_data[instrument]
                
        try:
            # Get latest data from trader client
            latest_data = self.trader.get_latest_data(instrument)
            
            if latest_data is None:
                return None
                
            # Update cache
            self.market_data[instrument] = {
                "timestamp": time.time(),
                "open": latest_data.get("Open"),
                "high": latest_data.get("High"),
                "low": latest_data.get("Low"),
                "close": latest_data.get("Close"),
                "volume": latest_data.get("Volume"),
                "atr": latest_data.get("ATR")
            }
            
            return self.market_data[instrument]
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
    
    def _calculate_trade_levels(self, instrument: str, direction: str, 
                               market_data: Dict) -> Tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels
        
        Args:
            instrument: Trading instrument
            direction: 'buy' or 'sell'
            market_data: Current market data
            
        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        # Get current price
        current_price = market_data.get("close", 0)
        
        # Get ATR for stop loss calculation
        atr = market_data.get("atr")
        
        # If ATR not available, estimate it as 1% of price
        if atr is None:
            atr = current_price * 0.01
            
        # Calculate entry price (use current price for market orders)
        entry_price = current_price
        
        # Calculate stop loss
        if direction == "buy":
            stop_loss = entry_price - (atr * 2)  # 2 ATR stop loss
        else:  # sell
            stop_loss = entry_price + (atr * 2)
            
        # Calculate take profit (using risk-reward ratio)
        risk_reward_ratio = self.config.get("risk_reward_ratio", 1.5)
        
        if direction == "buy":
            take_profit = entry_price + (atr * 2 * risk_reward_ratio)
        else:  # sell
            take_profit = entry_price - (atr * 2 * risk_reward_ratio)
            
        return entry_price, stop_loss, take_profit
    
    def submit_trade_request(self, request: TradeRequest) -> str:
        """
        Submit a trade request to the queue
        
        Args:
            request: Trade request
            
        Returns:
            Request ID
        """
        # Add to queue
        self.trade_queue.put(request)
        
        # Return request ID
        return request.id
    
    def get_trade_result(self, request_id: str) -> Optional[TradeResult]:
        """
        Get result for a trade request
        
        Args:
            request_id: Trade request ID
            
        Returns:
            Trade result or None if not found
        """
        for request, result in reversed(self.trade_history):
            if request.id == request_id:
                return result
                
        return None
    
    def update_open_positions(self) -> Dict[str, str]:
        """
        Update open positions with latest market data
        
        Returns:
            Dictionary mapping instruments to update status
        """
        results = {}
        
        # Get open positions from portfolio
        open_positions = self.portfolio.open_positions
        
        for instrument, position in open_positions.items():
            try:
                # Get latest market data
                market_data = self._get_market_data(instrument)
                
                if not market_data:
                    results[instrument] = "No market data available"
                    continue
                    
                # Get current price
                current_price = market_data.get("close", 0)
                
                # Update trailing stops if enabled
                if self.config.get("use_trailing_stops", False):
                    # Update trailing stop
                    self.portfolio.update_trailing_stops(
                        instrument, 
                        current_price,
                        trailing_pct=self.config.get("trailing_stop_pct", 0.02)
                    )
                
                # Check if position should be closed
                closed = self.portfolio.update_position(instrument, current_price)
                
                if closed:
                    results[instrument] = "Position closed"
                else:
                    results[instrument] = "Position updated"
                    
            except Exception as e:
                self.logger.error(f"Error updating position for {instrument}: {e}")
                results[instrument] = f"Error: {str(e)}"
                
        return results
    
    def save_state(self) -> bool:
        """
        Save executor state to data store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save trade history
            trade_history_data = []
            for request, result in self.trade_history:
                trade_history_data.append({
                    "request": request.to_dict(),
                    "result": result.to_dict()
                })
                
            # Save last trade times
            last_trade_time_data = {
                k: v for k, v in self.last_trade_time.items()
            }
            
            # Combine data
            state_data = {
                "trade_history": trade_history_data,
                "last_trade_time": last_trade_time_data
            }
            
            # Save to data store
            self.data_store.store_data("trade_executor_state", state_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving trade executor state: {e}")
            return False
    
    def load_state(self) -> bool:
        """
        Load executor state from data store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from data store
            state_data = self.data_store.get_data("trade_executor_state")
            
            if not state_data:
                return False
                
            # Restore last trade times
            self.last_trade_time = state_data.get("last_trade_time", {})
            
            # Restore trade history
            self.trade_history = []
            for entry in state_data.get("trade_history", []):
                request_data = entry.get("request", {})
                result_data = entry.get("result", {})
                
                # Recreate request and result objects
                request = TradeRequest(
                    instrument=request_data.get("instrument", ""),
                    direction=request_data.get("direction", ""),
                    reason=request_data.get("reason", ""),
                    confidence=request_data.get("confidence", 0.0),
                    signal_data=request_data.get("signal_data", {})
                )
                request.timestamp = request_data.get("timestamp", 0)
                request.id = request_data.get("id", "")
                
                result = TradeResult(
                    request_id=result_data.get("request_id", ""),
                    success=result_data.get("success", False),
                    message=result_data.get("message", ""),
                    order_id=result_data.get("order_id"),
                    position_id=result_data.get("position_id"),
                    details=result_data.get("details", {})
                )
                result.timestamp = result_data.get("timestamp", 0)
                
                # Add to trade history
                self.trade_history.append((request, result))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading trade executor state: {e}")
            return False