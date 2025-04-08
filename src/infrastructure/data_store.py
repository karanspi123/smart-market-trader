import redis
import json
import os
import logging
import pandas as pd
from typing import Dict, Optional, Any, List
import time

class DataStore:
    """Data storage service for Smart Market Analyzer"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379/0', 
                data_dir: str = 'data'):
        """
        Initialize data store
        
        Args:
            redis_url: Redis connection URL
            data_dir: Directory for file-based storage
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # Ensure data directories exist
        os.makedirs(os.path.join(data_dir, 'market'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'backtest'), exist_ok=True)
        
        # Initialize Redis connection
        try:
            self.redis = redis.from_url(redis_url)
            self.redis_available = True
            self.logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            self.redis_available = False
            self.logger.warning(f"Failed to connect to Redis: {e}. Using file-based storage only.")
    
    def store_data(self, key: str, data: Any) -> bool:
        """
        Store data with the given key
        
        Args:
            key: Unique identifier for the data
            data: Data to store (must be JSON-serializable)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert data to JSON
            json_data = json.dumps(data)
            
            # Try Redis first if available
            if self.redis_available:
                try:
                    self.redis.set(key, json_data)
                    return True
                except Exception as e:
                    self.logger.warning(f"Failed to store data in Redis: {e}")
            
            # Fallback to file-based storage
            file_path = self._get_file_path(key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(json_data)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store data for key '{key}': {e}")
            return False
    
    def get_data(self, key: str) -> Optional[Any]:
        """
        Get data for the given key
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            Stored data or None if not found
        """
        try:
            # Try Redis first if available
            if self.redis_available:
                try:
                    data = self.redis.get(key)
                    if data:
                        return json.loads(data)
                except Exception as e:
                    self.logger.warning(f"Failed to get data from Redis: {e}")
            
            # Fallback to file-based storage
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.loads(f.read())
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to get data for key '{key}': {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """
        Delete data for the given key
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try Redis first if available
            if self.redis_available:
                try:
                    self.redis.delete(key)
                except Exception as e:
                    self.logger.warning(f"Failed to delete data from Redis: {e}")
            
            # Also delete from file-based storage
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete data for key '{key}': {e}")
            return False
    
    def store_market_data(self, instrument: str, data: pd.DataFrame) -> bool:
        """
        Store market data for an instrument
        
        Args:
            instrument: Instrument symbol
            data: Market data as DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save as CSV file
            market_dir = os.path.join(self.data_dir, 'market')
            os.makedirs(market_dir, exist_ok=True)
            
            file_path = os.path.join(market_dir, f"{instrument.lower()}.csv")
            data.to_csv(file_path)
            
            # Store metadata
            metadata = {
                'instrument': instrument,
                'rows': len(data),
                'columns': list(data.columns),
                'last_updated': time.time()
            }
            
            self.store_data(f"market_data_meta:{instrument}", metadata)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store market data for {instrument}: {e}")
            return False
    
    def get_market_data(self, instrument: str) -> Optional[pd.DataFrame]:
        """
        Get market data for an instrument
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            Market data as DataFrame or None if not found
        """
        try:
            market_dir = os.path.join(self.data_dir, 'market')
            file_path = os.path.join(market_dir, f"{instrument.lower()}.csv")
            
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                
                # Convert timestamp to datetime if present
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                elif 'Time' in data.columns:
                    data['Time'] = pd.to_datetime(data['Time'])
                    data.set_index('Time', inplace=True)
                
                return data
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to get market data for {instrument}: {e}")
            return None
    
    def list_instruments(self) -> List[str]:
        """
        List available instruments
        
        Returns:
            List of instrument symbols
        """
        try:
            market_dir = os.path.join(self.data_dir, 'market')
            if not os.path.exists(market_dir):
                return []
            
            files = os.listdir(market_dir)
            instruments = [os.path.splitext(f)[0].upper() for f in files if f.endswith('.csv')]
            
            return instruments
        except Exception as e:
            self.logger.error(f"Failed to list instruments: {e}")
            return []
    
    def _get_file_path(self, key: str) -> str:
        """
        Get file path for a key
        
        Args:
            key: Data key
            
        Returns:
            File path
        """
        # Sanitize key for use as a filename
        safe_key = key.replace(':', '_').replace('/', '_')
        
        # Determine storage location based on key prefix
        if key.startswith('market_data'):
            return os.path.join(self.data_dir, 'market', f"{safe_key}.json")
        elif key.startswith('model'):
            return os.path.join(self.data_dir, 'models', f"{safe_key}.json")
        elif key.startswith('backtest'):
            return os.path.join(self.data_dir, 'backtest', f"{safe_key}.json")
        else:
            return os.path.join(self.data_dir, f"{safe_key}.json")