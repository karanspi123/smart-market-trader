#!/usr/bin/env python3
"""
Worker service for Smart Market Analyzer - handles background processing tasks
"""

import logging
import os
import signal
import sys
import time
import json
import threading
from typing import Dict, List, Optional, Any
import importlib
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'worker.log'))
    ]
)
logger = logging.getLogger('worker')

# Import local modules
try:
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Import core services
    from data_store import DataStore
    from monitoring import MonitoringService
    from portfolio_manager import PortfolioManager
    from ninja_trader_client import NinjaTraderClient
    from trade_executor import TradeExecutor
    from multi_market_manager import MultiMarketManager
    
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    logger.critical(traceback.format_exc())
    sys.exit(1)

class WorkerService:
    """Main worker service for Smart Market Analyzer"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize worker service"""
        self.running = False
        self.threads = []
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize services
        self.data_store = DataStore(
            redis_url=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        self.monitoring = MonitoringService(
            self.config.get('monitoring', {})
        )
        
        self.portfolio = PortfolioManager(
            initial_capital=self.config.get('portfolio', {}).get('initial_capital', 100000),
            risk_per_trade=self.config.get('portfolio', {}).get('risk_per_trade', 0.02),
            max_correlated_risk=self.config.get('portfolio', {}).get('max_correlated_risk', 0.06),
            max_drawdown=self.config.get('portfolio', {}).get('max_drawdown', 0.15)
        )
        
        # Initialize NinjaTrader client
        ninjatrader_config = self.config.get('data', {}).get('data_sources', {}).get('ninjatrader', {})
        self.trader_client = NinjaTraderClient(
            api_key=ninjatrader_config.get('api_key', ''),
            base_url=ninjatrader_config.get('base_url', 'http://localhost:8000/api')
        )
        
        # Initialize trade executor
        self.trade_executor = TradeExecutor(
            config=self.config.get('trading', {}),
            portfolio_manager=self.portfolio,
            trader_client=self.trader_client,
            data_store=self.data_store,
            monitoring=self.monitoring
        )
        
        # Initialize multi-market manager
        self.market_manager = MultiMarketManager(
            config=self.config,
            data_store=self.data_store,
            monitoring=self.monitoring,
            portfolio_manager=self.portfolio,
            trade_executor=self.trade_executor
        )
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {
                "portfolio": {
                    "initial_capital": 100000,
                    "risk_per_trade": 0.02,
                    "max_correlated_risk": 0.06,
                    "max_drawdown": 0.15
                },
                "trading": {
                    "min_confidence": 0.75,
                    "min_time_between_trades": 300,
                    "use_trailing_stops": True
                },
                "monitoring": {
                    "prometheus": {
                        "enabled": True,
                        "port": 8001
                    },
                    "log_dir": "logs"
                }
            }
    
    def start(self):
        """Start all worker processes"""
        if self.running:
            logger.warning("Worker service is already running")
            return
            
        logger.info("Starting worker service")
        self.running = True
        
        try:
            # Authenticate with NinjaTrader if enabled
            if self.config.get('data', {}).get('data_sources', {}).get('ninjatrader', {}).get('enabled', False):
                auth_result = self.trader_client.authenticate()
                if not auth_result:
                    logger.warning("Failed to authenticate with NinjaTrader API")
                    # Continue anyway, we'll retry later
            
            # Load portfolio state
            portfolio_data = self.data_store.get_data('portfolio_state')
            if portfolio_data:
                self.portfolio = PortfolioManager.from_json(json.dumps(portfolio_data))
                logger.info("Portfolio state loaded")
            
            # Load trade executor state
            self.trade_executor.load_state()
            
            # Initialize instruments
            self._initialize_instruments()
            
            # Start trade executor
            self.trade_executor.start()
            
            # Start market manager
            self.market_manager.start()
            
            # Start position update thread
            self._start_position_update_thread()
            
            # Start state persistence thread
            self._start_state_persistence_thread()
            
            logger.info("Worker service started successfully")
            
            # Run main loop in main thread
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting worker service: {e}")
            logger.error(traceback.format_exc())
            self.running = False
    
    def _initialize_instruments(self):
        """Initialize market instances for configured instruments"""
        instruments = self.config.get('instruments', {})
        for instrument, instrument_config in instruments.items():
            if instrument_config.get('enabled', False):
                logger.info(f"Initializing market instance for {instrument}")
                self.market_manager.add_market(instrument, instrument_config)
    
    def _start_position_update_thread(self):
        """Start thread for updating positions"""
        def update_positions():
            while self.running:
                try:
                    # Update open positions with latest market data
                    results = self.trade_executor.update_open_positions()
                    
                    if results:
                        logger.info(f"Updated positions: {results}")
                    
                    # Sleep for 10 seconds
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error updating positions: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)  # Sleep longer after error
        
        thread = threading.Thread(target=update_positions, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("Position update thread started")
    
    def _start_state_persistence_thread(self):
        """Start thread for persisting state"""
        def persist_state():
            while self.running:
                try:
                    # Save portfolio state
                    portfolio_data = json.loads(self.portfolio.to_json())
                    self.data_store.store_data('portfolio_state', portfolio_data)
                    
                    # Save trade executor state
                    self.trade_executor.save_state()
                    
                    # Save market manager state
                    self.market_manager.save_state()
                    
                    logger.info("State persisted successfully")
                    
                    # Sleep for 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error persisting state: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(60)  # Sleep longer after error
        
        thread = threading.Thread(target=persist_state, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("State persistence thread started")
    
    def _main_loop(self):
        """Main worker loop"""
        health_check_interval = 60  # seconds
        last_health_check = 0
        
        while self.running:
            try:
                # Perform health check periodically
                current_time = time.time()
                if current_time - last_health_check > health_check_interval:
                    self._health_check()
                    last_health_check = current_time
                
                # Process pending tasks
                # (Most processing is done in background threads)
                
                # Sleep to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)  # Sleep after error
    
    def _health_check(self):
        """Perform system health check"""
        try:
            # Check if market manager is running
            if not self.market_manager.running:
                logger.warning("Market manager is not running, restarting...")
                self.market_manager.start()
            
            # Check if trade executor is running
            if not self.trade_executor.running:
                logger.warning("Trade executor is not running, restarting...")
                self.trade_executor.start()
            
            # Check NinjaTrader connection if enabled
            if self.config.get('data', {}).get('data_sources', {}).get('ninjatrader', {}).get('enabled', False):
                instruments = self.trader_client.get_instruments()
                if not instruments:
                    logger.warning("NinjaTrader connection test failed, attempting to reconnect")
                    self.trader_client.authenticate()
            
            # Check Redis connection
            try:
                self.data_store.store_data('health_check', {'timestamp': time.time()})
                health_data = self.data_store.get_data('health_check')
                if not health_data:
                    logger.warning("Redis health check failed")
            except Exception as e:
                logger.warning(f"Redis connection error: {e}")
            
            # Update monitoring metrics
            self.monitoring.track_active_models(len(self.market_manager.markets))
            
            # Log health check success
            logger.info("Health check completed successfully")
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            logger.error(traceback.format_exc())
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.stop()
    
    def stop(self):
        """Stop all worker processes"""
        if not self.running:
            logger.warning("Worker service is not running")
            return
            
        logger.info("Stopping worker service")
        self.running = False
        
        try:
            # Stop market manager
            self.market_manager.stop()
            
            # Stop trade executor
            self.trade_executor.stop()
            
            # Save state before exiting
            logger.info("Saving state before shutdown")
            portfolio_data = json.loads(self.portfolio.to_json())
            self.data_store.store_data('portfolio_state', portfolio_data)
            self.trade_executor.save_state()
            self.market_manager.save_state()
            
            # Wait for threads to finish
            for thread in self.threads:
                thread.join(timeout=5.0)
            
            logger.info("Worker service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping worker service: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Get configuration path from environment or use default
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    
    # Create and start worker service
    worker = WorkerService(config_path)
    
    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        worker.stop()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        worker.stop()
        sys.exit(1)