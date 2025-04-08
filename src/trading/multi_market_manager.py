import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import queue
import json
import os

# Local imports
from model_service import ModelService
from signal_service import SignalService
from data_service import DataService
from backtest_service import BacktestService
from trade_executor import TradeExecutor, TradeRequest
from monitoring import MonitoringService
from data_store import DataStore
from portfolio_manager import PortfolioManager

class MarketInstance:
    """Represents a single market instance with its own models and signals"""
    
    def __init__(self, instrument: str, config: Dict, data_service: DataService,
                model_service: ModelService, signal_service: SignalService,
                backtest_service: BacktestService):
        self.instrument = instrument
        self.config = config
        self.data_service = data_service
        self.model_service = model_service
        self.signal_service = signal_service
        self.backtest_service = backtest_service
        
        self.logger = logging.getLogger(f"market.{instrument}")
        self.active = False
        self.last_prediction_time = 0
        self.last_signal_time = 0
        self.prediction_history = []
        self.signal_history = []
        self.performance_metrics = {}
        
    def start(self) -> bool:
        """Start the market instance"""
        try:
            self.logger.info(f"Starting market instance for {self.instrument}")
            self.active = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to start market instance: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the market instance"""
        try:
            self.logger.info(f"Stopping market instance for {self.instrument}")
            self.active = False
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop market instance: {e}")
            return False
    
    def is_active(self) -> bool:
        """Check if the market instance is active"""
        return self.active
    
    def make_prediction(self, market_data: List[Dict]) -> Optional[Dict]:
        """Make a prediction using the model service"""
        try:
            # Check if we have enough data
            window_size = self.config.get('window_size', 60)
            if len(market_data) < window_size:
                self.logger.warning(f"Not enough data for prediction: {len(market_data)} < {window_size}")
                return None
            
            # Prepare input data
            X = self._prepare_model_input(market_data, window_size)
            
            # Make prediction
            prediction = self.model_service.predict(X)
            
            # Store prediction time
            self.last_prediction_time = time.time()
            
            # Add to history (limit to last 100)
            self.prediction_history.append({
                'timestamp': self.last_prediction_time,
                'prediction': prediction
            })
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            return prediction
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None
    
    def generate_signal(self, prediction: Dict, latest_candle: Dict) -> Optional[Dict]:
        """Generate a trading signal using the signal service"""
        try:
            # Generate signal
            signal = self.signal_service.generate_signal(prediction, latest_candle)
            
            # Store signal time
            self.last_signal_time = time.time()
            
            # Add to history (limit to last 100)
            self.signal_history.append({
                'timestamp': self.last_signal_time,
                'signal': signal
            })
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
            
            return signal
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def _prepare_model_input(self, market_data: List[Dict], window_size: int) -> np.ndarray:
        """Prepare model input from market data"""
        # Extract required features
        features = []
        for candle in market_data[-window_size:]:
            row = [
                candle.get('Open', 0),
                candle.get('High', 0),
                candle.get('Low', 0),
                candle.get('Close', 0),
                candle.get('Volume', 0),
                candle.get('EMA9', 0),
                candle.get('EMA21', 0),
                candle.get('EMA220', 0)
            ]
            features.append(row)
        
        # Create batch with single sequence
        X = np.array([features])
        return X
    
    def update_performance_metrics(self, backtest_results: Dict) -> None:
        """Update performance metrics from backtest results"""
        self.performance_metrics = {
            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
            'sortino_ratio': backtest_results.get('sortino_ratio', 0),
            'total_return': backtest_results.get('total_return', 0),
            'max_drawdown': backtest_results.get('max_drawdown', 0),
            'win_rate': backtest_results.get('win_rate', 0),
            'last_updated': time.time()
        }
    
    def to_dict(self) -> Dict:
        """Convert market instance to dictionary"""
        return {
            'instrument': self.instrument,
            'active': self.active,
            'last_prediction_time': self.last_prediction_time,
            'last_signal_time': self.last_signal_time,
            'performance_metrics': self.performance_metrics
        }

class MultiMarketManager:
    """Manages multiple market instances and coordinates trading across them"""
    
    def __init__(self, config: Dict, data_store: DataStore, monitoring: MonitoringService,
                portfolio_manager: PortfolioManager, trade_executor: TradeExecutor):
        self.config = config
        self.data_store = data_store
        self.monitoring = monitoring
        self.portfolio = portfolio_manager
        self.trade_executor = trade_executor
        
        self.markets = {}
        self.market_data_cache = {}
        self.running = False
        self.worker_thread = None
        self.worker_interval = config.get('worker_interval', 60)  # seconds
        
        self.prediction_queue = queue.Queue()
        self.signal_queue = queue.Queue()
        
        self.logger = logging.getLogger("multi_market_manager")
    
    def add_market(self, instrument: str, market_config: Dict) -> bool:
        """Add a new market instance"""
        try:
            if instrument in self.markets:
                self.logger.warning(f"Market instance for {instrument} already exists")
                return False
            
            # Create service instances for this market
            data_service = DataService(market_config)
            model_service = ModelService(market_config)
            signal_service = SignalService(market_config)
            backtest_service = BacktestService(market_config)
            
            # Create market instance
            market = MarketInstance(
                instrument=instrument,
                config=market_config,
                data_service=data_service,
                model_service=model_service,
                signal_service=signal_service,
                backtest_service=backtest_service
            )
            
            # Add to markets dictionary
            self.markets[instrument] = market
            
            self.logger.info(f"Added market instance for {instrument}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add market instance for {instrument}: {e}")
            return False
    
    def remove_market(self, instrument: str) -> bool:
        """Remove a market instance"""
        try:
            if instrument not in self.markets:
                self.logger.warning(f"Market instance for {instrument} not found")
                return False
            
            # Stop market instance
            self.markets[instrument].stop()
            
            # Remove from markets dictionary
            del self.markets[instrument]
            
            self.logger.info(f"Removed market instance for {instrument}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove market instance for {instrument}: {e}")
            return False
    
    def start(self) -> bool:
        """Start all market instances and worker thread"""
        try:
            self.logger.info("Starting Multi-Market Manager")
            
            # Start each market instance
            for instrument, market in self.markets.items():
                market.start()
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            self.worker_thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Multi-Market Manager: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop all market instances and worker thread"""
        try:
            self.logger.info("Stopping Multi-Market Manager")
            
            # Stop worker thread
            self.running = False
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5.0)
            
            # Stop each market instance
            for instrument, market in self.markets.items():
                market.stop()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Multi-Market Manager: {e}")
            return False
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing all markets"""
        while self.running:
            try:
                # Process each active market
                for instrument, market in self.markets.items():
                    if market.is_active():
                        self._process_market(instrument, market)
                
                # Process prediction queue
                self._process_prediction_queue()
                
                # Process signal queue
                self._process_signal_queue()
                
                # Update portfolio correlations periodically
                if time.time() % 3600 < self.worker_interval:  # Once per hour
                    self._update_market_correlations()
                
                # Save state periodically
                if time.time() % 900 < self.worker_interval:  # Every 15 minutes
                    self.save_state()
                
                # Sleep until next cycle
                time.sleep(self.worker_interval)
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                self.monitoring.log_error(
                    error_type="multi_market_worker",
                    error_message=str(e)
                )
                time.sleep(10)  # Sleep a bit before retrying
    
    def _process_market(self, instrument: str, market: MarketInstance) -> None:
        """Process a single market instance"""
        # Get latest market data
        market_data = self._get_market_data(instrument)
        if not market_data:
            self.logger.warning(f"No market data available for {instrument}")
            return
        
        # Make prediction
        prediction = market.make_prediction(market_data)
        if prediction:
            # Add to prediction queue
            self.prediction_queue.put((instrument, prediction, market_data[-1]))
            
            # Log successful prediction
            self.monitoring.log_prediction(
                instrument=instrument,
                prediction=prediction.get('price_prediction', [0])[0],
                confidence=prediction.get('confidence', [0])[0]
            )
    
    def _process_prediction_queue(self) -> None:
        """Process the prediction queue"""
        try:
            # Process up to 10 predictions at a time
            for _ in range(10):
                if self.prediction_queue.empty():
                    break
                    
                # Get prediction from queue
                instrument, prediction, latest_candle = self.prediction_queue.get(block=False)
                
                # Get market instance
                market = self.markets.get(instrument)
                if not market:
                    continue
                
                # Generate signal
                signal = market.generate_signal(prediction, latest_candle)
                if signal:
                    # Add to signal queue
                    self.signal_queue.put((instrument, prediction, signal, latest_candle))
                    
                    # Log signal
                    self.monitoring.log_signal(
                        instrument=instrument,
                        signal_type=signal.get('signal', 'Hold'),
                        confidence=signal.get('confidence', 0)
                    )
                    
                # Mark task as done
                self.prediction_queue.task_done()
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing prediction queue: {e}")
    
    def _process_signal_queue(self) -> None:
        """Process the signal queue"""
        try:
            # Process up to 10 signals at a time
            for _ in range(10):
                if self.signal_queue.empty():
                    break
                    
                # Get signal from queue
                instrument, prediction, signal, latest_candle = self.signal_queue.get(block=False)
                
                # Skip if signal is 'Hold'
                if signal.get('signal', 'Hold') == 'Hold':
                    self.signal_queue.task_done()
                    continue
                
                # Check if we can trade this instrument
                if not self._can_trade_instrument(instrument):
                    self.signal_queue.task_done()
                    continue
                
                # Create trade request
                if signal.get('signal') == 'Buy':
                    direction = 'buy'
                elif signal.get('signal') == 'Sell':
                    direction = 'sell'
                else:
                    self.signal_queue.task_done()
                    continue
                
                trade_request = TradeRequest(
                    instrument=instrument,
                    direction=direction,
                    reason=signal.get('reason', 'model_signal'),
                    confidence=signal.get('confidence', 0),
                    signal_data={
                        'prediction': prediction,
                        'signal': signal,
                        'candle': latest_candle
                    }
                )
                
                # Submit trade request
                request_id = self.trade_executor.submit_trade_request(trade_request)
                
                self.logger.info(f"Submitted trade request {request_id} for {instrument}")
                
                # Mark task as done
                self.signal_queue.task_done()
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing signal queue: {e}")
    
    def _get_market_data(self, instrument: str) -> Optional[List[Dict]]:
        """Get latest market data for an instrument"""
        try:
            # Try to get from cache
            if instrument in self.market_data_cache:
                # Check if cache is still valid (30 seconds)
                if time.time() - self.market_data_cache[instrument].get('timestamp', 0) < 30:
                    return self.market_data_cache[instrument].get('data', [])
            
            # Get data from data store
            data = self.data_store.get_market_data(instrument)
            if data is None or data.empty:
                self.logger.warning(f"No data found for {instrument}")
                return None
            
            # Convert to list of dictionaries
            candles = data.to_dict('records')
            
            # Update cache
            self.market_data_cache[instrument] = {
                'timestamp': time.time(),
                'data': candles
            }
            
            return candles
        except Exception as e:
            self.logger.error(f"Error getting market data for {instrument}: {e}")
            return None
    
    def _update_market_correlations(self) -> None:
        """Update portfolio correlations based on market data"""
        try:
            # Get price data for all instruments
            price_data = {}
            for instrument in self.markets.keys():
                market_data = self._get_market_data(instrument)
                if market_data:
                    # Create DataFrame from market data
                    df = pd.DataFrame(market_data)
                    if 'Close' in df.columns:
                        price_data[instrument] = df
            
            # Update portfolio correlations
            if price_data:
                self.portfolio.update_correlations(price_data)
                self.logger.info("Updated market correlations")
        except Exception as e:
            self.logger.error(f"Error updating market correlations: {e}")
    
    def _can_trade_instrument(self, instrument: str) -> bool:
        """Check if we can trade this instrument"""
        # Check if market is active
        market = self.markets.get(instrument)
        if not market or not market.is_active():
            return False
        
        # Check if portfolio allows this trade
        return self.portfolio.can_enter_position(instrument)
    
    def get_market_status(self, instrument: Optional[str] = None) -> Dict:
        """Get status of market instances"""
        if instrument:
            # Get status for specific instrument
            market = self.markets.get(instrument)
            if not market:
                return {'error': f'Market instance for {instrument} not found'}
            
            return market.to_dict()
        else:
            # Get status for all markets
            return {
                instrument: market.to_dict()
                for instrument, market in self.markets.items()
            }
    
    def save_state(self) -> bool:
        """Save manager state to disk"""
        try:
            state_data = {
                'markets': {
                    instrument: market.to_dict()
                    for instrument, market in self.markets.items()
                },
                'timestamp': time.time()
            }
            
            # Save to data store
            self.data_store.store_data('multi_market_state', state_data)
            
            self.logger.info("Saved Multi-Market Manager state")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self) -> bool:
        """Load manager state from disk"""
        try:
            # Load from data store
            state_data = self.data_store.get_data('multi_market_state')
            
            if not state_data:
                self.logger.warning("No saved state found")
                return False
            
            # Restore market states
            for instrument, market_data in state_data.get('markets', {}).items():
                if instrument in self.markets:
                    # Restore performance metrics
                    if 'performance_metrics' in market_data:
                        self.markets[instrument].performance_metrics = market_data['performance_metrics']
                    
                    # Restore active state
                    if market_data.get('active', False):
                        self.markets[instrument].start()
                    else:
                        self.markets[instrument].stop()
            
            self.logger.info("Loaded Multi-Market Manager state")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False