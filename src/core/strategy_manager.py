import logging
from typing import Dict, List, Optional, Any
import json
import os

class StrategyManager:
    """Manages trading strategies for the Smart Market Analyzer"""
    
    def __init__(self, config_path: str = "config/strategies"):
        """
        Initialize strategy manager
        
        Args:
            config_path: Path to strategy configuration files
        """
        self.config_path = config_path
        self.strategies = {}
        self.active_strategy = None
        self.logger = logging.getLogger(__name__)
        
        # Create strategy config directory if it doesn't exist
        os.makedirs(config_path, exist_ok=True)
        
        # Load available strategies
        self._load_strategies()
    
    def _load_strategies(self):
        """Load strategy configurations from files"""
        try:
            strategy_files = [f for f in os.listdir(self.config_path) if f.endswith('.json')]
            
            for strategy_file in strategy_files:
                strategy_path = os.path.join(self.config_path, strategy_file)
                
                with open(strategy_path, 'r') as f:
                    strategy_config = json.load(f)
                    
                    # Check if strategy has a name
                    if 'name' in strategy_config:
                        self.strategies[strategy_config['name']] = strategy_config
                        self.logger.info(f"Loaded strategy: {strategy_config['name']}")
                    else:
                        self.logger.warning(f"Skipping strategy without name: {strategy_path}")
                        
        except Exception as e:
            self.logger.error(f"Error loading strategies: {e}")
    
    def get_strategy(self, name: str) -> Optional[Dict]:
        """
        Get strategy configuration by name
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy configuration or None if not found
        """
        return self.strategies.get(name)
    
    def add_strategy(self, strategy_config: Dict) -> bool:
        """
        Add a new strategy
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if strategy has a name
            if 'name' not in strategy_config:
                self.logger.error("Strategy must have a name")
                return False
                
            name = strategy_config['name']
            
            # Save strategy to file
            strategy_path = os.path.join(self.config_path, f"{name}.json")
            
            with open(strategy_path, 'w') as f:
                json.dump(strategy_config, f, indent=4)
                
            # Add to strategies dictionary
            self.strategies[name] = strategy_config
            
            self.logger.info(f"Added strategy: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy: {e}")
            return False
    
    def update_strategy(self, name: str, strategy_config: Dict) -> bool:
        """
        Update an existing strategy
        
        Args:
            name: Strategy name
            strategy_config: Updated strategy configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if strategy exists
            if name not in self.strategies:
                self.logger.error(f"Strategy not found: {name}")
                return False
                
            # Ensure name is consistent
            strategy_config['name'] = name
            
            # Save strategy to file
            strategy_path = os.path.join(self.config_path, f"{name}.json")
            
            with open(strategy_path, 'w') as f:
                json.dump(strategy_config, f, indent=4)
                
            # Update strategies dictionary
            self.strategies[name] = strategy_config
            
            self.logger.info(f"Updated strategy: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
            return False
    
    def delete_strategy(self, name: str) -> bool:
        """
        Delete a strategy
        
        Args:
            name: Strategy name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if strategy exists
            if name not in self.strategies:
                self.logger.error(f"Strategy not found: {name}")
                return False
                
            # Remove strategy file
            strategy_path = os.path.join(self.config_path, f"{name}.json")
            
            if os.path.exists(strategy_path):
                os.remove(strategy_path)
                
            # Remove from strategies dictionary
            del self.strategies[name]
            
            # Reset active strategy if it was deleted
            if self.active_strategy == name:
                self.active_strategy = None
                
            self.logger.info(f"Deleted strategy: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting strategy: {e}")
            return False
    
    def set_active_strategy(self, name: str) -> bool:
        """
        Set active strategy
        
        Args:
            name: Strategy name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.strategies:
            self.logger.error(f"Strategy not found: {name}")
            return False
            
        self.active_strategy = name
        self.logger.info(f"Active strategy set to: {name}")
        return True
    
    def get_active_strategy(self) -> Optional[Dict]:
        """
        Get active strategy configuration
        
        Returns:
            Active strategy configuration or None if no active strategy
        """
        if not self.active_strategy:
            return None
            
        return self.strategies.get(self.active_strategy)
    
    def get_all_strategies(self) -> Dict[str, Dict]:
        """
        Get all available strategies
        
        Returns:
            Dictionary mapping strategy names to configurations
        """
        return self.strategies
    
    def compare_strategies(self, backtest_service, test_data: Dict) -> Dict:
        """
        Compare performance of multiple strategies
        
        Args:
            backtest_service: BacktestService instance
            test_data: Test data for backtesting
            
        Returns:
            Dictionary with strategy performance metrics
        """
        results = {}
        
        for name, strategy in self.strategies.items():
            # Update backtest service with strategy configuration
            backtest_service.strategy_config = strategy
            
            # Add strategy indicators
            df_with_indicators = backtest_service.add_strategy_indicators(
                test_data.copy(), 
                strategy.get("type", "custom")
            )
            
            # Generate signals
            signals = backtest_service.generate_strategy_signals(df_with_indicators)
            
            # Run backtest
            backtest_results = backtest_service.simulate_trade(df_with_indicators, signals)
            
            # Store results
            results[name] = {
                "config": strategy,
                "performance": {
                    "total_return": float(backtest_results["total_return"]),
                    "sharpe_ratio": float(backtest_results["sharpe_ratio"]),
                    "sortino_ratio": float(backtest_results["sortino_ratio"]),
                    "max_drawdown": float(backtest_results["max_drawdown"]),
                    "win_rate": float(backtest_results["win_rate"]),
                    "profit_factor": float(backtest_results["profit_factor"]),
                    "num_trades": int(backtest_results["num_trades"])
                }
            }
        
        return results