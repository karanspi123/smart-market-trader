import requests
import pandas as pd
import logging
from typing import Dict, Optional, List
import threading
import time
import queue
import json
import os
from datetime import datetime, timedelta

class NinjaTraderClient:
    """Client for connecting to NinjaTrader API for real-time market data"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000/api"):
        """
        Initialize NinjaTrader API client
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of NinjaTrader API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": self.api_key})
        self.streaming = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.logger = logging.getLogger(__name__)
        
        # Authentication token
        self.token = None
        self.token_expiry = None
        
        # Connection state
        self.connected = False
        self.last_connection_attempt = 0
        self.connection_retry_interval = 60  # seconds
        
        # Streaming threads by instrument
        self.stream_threads = {}
        
        # Cache for instruments
        self.instruments_cache = None
        self.instruments_cache_time = 0
        self.instruments_cache_ttl = 3600  # 1 hour
        
        # Create data directory if it doesn't exist
        os.makedirs('data/market', exist_ok=True)
    
    def authenticate(self) -> bool:
        """
        Authenticate with NinjaTrader API
        
        Returns:
            True if authentication successful, False otherwise
        """
        # Check if we already have a valid token
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return True
            
        try:
            # Update last connection attempt time
            self.last_connection_attempt = time.time()
            
            # Make authentication request
            response = self.session.post(
                f"{self.base_url}/auth", 
                json={"api_key": self.api_key}
            )
            
            if response.status_code == 200:
                auth_data = response.json()
                self.token = auth_data.get('token')
                
                # Set token expiry (default to 1 day if not provided)
                expiry_seconds = auth_data.get('expires_in', 86400)
                self.token_expiry = datetime.now() + timedelta(seconds=expiry_seconds)
                
                # Update session headers with token
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                
                self.connected = True
                self.logger.info("Successfully authenticated with NinjaTrader API")
                return True
            else:
                self.logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                self.connected = False
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self.connected = False
            return False
    
    def get_instruments(self) -> List[Dict]:
        """
        Get available instruments from NinjaTrader
        
        Returns:
            List of instrument dictionaries
        """
        # Check cache first
        current_time = time.time()
        if (self.instruments_cache is not None and 
            current_time - self.instruments_cache_time < self.instruments_cache_ttl):
            return self.instruments_cache
            
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return []
                
            # Make request
            response = self.session.get(f"{self.base_url}/instruments")
            
            if response.status_code == 200:
                instruments = response.json().get("instruments", [])
                
                # Update cache
                self.instruments_cache = instruments
                self.instruments_cache_time = current_time
                
                return instruments
            else:
                self.logger.error(f"Failed to get instruments: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching instruments: {e}")
            return []
    
    def get_historical_data(self, instrument: str, timeframe: str, 
                           start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical data for an instrument
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (e.g., "1-minute", "5-minute", "1-day")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical data or None if request fails
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return None
                
            # Prepare request parameters
            params = {
                "instrument": instrument,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            }
            
            # Make request
            response = self.session.get(
                f"{self.base_url}/historical", 
                params=params
            )
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                
                if not data:
                    self.logger.warning(f"No historical data returned for {instrument}")
                    return None
                    
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                return df
            else:
                self.logger.error(f"Failed to get historical data: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None
    
    def _stream_data_worker(self, instrument: str, timeframe: str):
        """
        Worker thread for streaming data
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (e.g., "1-minute")
        """
        stream_key = f"{instrument}_{timeframe}"
        
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                self.logger.error(f"Failed to authenticate for streaming {stream_key}")
                return
                
            # Prepare request parameters
            params = {
                "instrument": instrument,
                "timeframe": timeframe
            }
            
            # Open streaming connection
            self.logger.info(f"Starting data stream for {stream_key}")
            
            with self.session.get(
                f"{self.base_url}/stream", 
                params=params, 
                stream=True
            ) as response:
                if response.status_code == 200:
                    # Process streaming data
                    for line in response.iter_lines():
                        # Check if we should stop streaming
                        if stream_key not in self.stream_threads or not self.streaming:
                            self.logger.info(f"Stopping data stream for {stream_key}")
                            break
                            
                        if line:
                            try:
                                # Parse data
                                data = json.loads(line)
                                
                                # Put data in queue
                                if not self.data_queue.full():
                                    self.data_queue.put({
                                        'instrument': instrument,
                                        'timeframe': timeframe,
                                        'data': data,
                                        'timestamp': time.time()
                                    })
                                else:
                                    self.logger.warning("Data queue is full, dropping data")
                                    
                                # Save data to file (for persistence)
                                self._save_market_data(instrument, data)
                                
                            except json.JSONDecodeError:
                                self.logger.error(f"Invalid JSON data: {line}")
                            except Exception as e:
                                self.logger.error(f"Error processing streaming data: {e}")
                else:
                    self.logger.error(f"Streaming failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            self.logger.error(f"Streaming error for {stream_key}: {e}")
            
        finally:
            # Remove thread from dictionary
            if stream_key in self.stream_threads:
                del self.stream_threads[stream_key]
    
    def _save_market_data(self, instrument: str, data: Dict):
        """
        Save market data to file
        
        Args:
            instrument: Instrument symbol
            data: Market data
        """
        try:
            # Create filename with date
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"data/market/{instrument}_{date_str}.jsonl"
            
            # Append data to file
            with open(filename, 'a') as f:
                f.write(json.dumps(data) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
    
    def start_streaming(self, instrument: str, timeframe: str = "1-minute") -> bool:
        """
        Start streaming market data for an instrument
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (e.g., "1-minute")
            
        Returns:
            True if streaming started successfully, False otherwise
        """
        stream_key = f"{instrument}_{timeframe}"
        
        # Check if already streaming
        if stream_key in self.stream_threads and self.stream_threads[stream_key].is_alive():
            self.logger.warning(f"Already streaming data for {stream_key}")
            return True
            
        # Set streaming flag
        self.streaming = True
        
        # Create and start thread
        thread = threading.Thread(
            target=self._stream_data_worker,
            args=(instrument, timeframe),
            daemon=True,
            name=f"stream_{stream_key}"
        )
        thread.start()
        
        # Add to threads dictionary
        self.stream_threads[stream_key] = thread
        
        self.logger.info(f"Started streaming for {stream_key}")
        return True
    
    def stop_streaming(self, instrument: Optional[str] = None, timeframe: str = "1-minute"):
        """
        Stop streaming market data
        
        Args:
            instrument: Instrument symbol (None to stop all)
            timeframe: Timeframe (e.g., "1-minute")
        """
        if instrument:
            # Stop specific instrument stream
            stream_key = f"{instrument}_{timeframe}"
            if stream_key in self.stream_threads:
                self.logger.info(f"Stopping stream for {stream_key}")
                del self.stream_threads[stream_key]
        else:
            # Stop all streams
            self.logger.info("Stopping all data streams")
            self.streaming = False
            self.stream_threads.clear()
    
    def get_latest_data(self, instrument: Optional[str] = None, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get latest data from queue
        
        Args:
            instrument: Filter by instrument (None for any)
            timeout: Queue timeout in seconds
            
        Returns:
            Latest market data or None if queue is empty
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    # Get data from queue with short timeout
                    item = self.data_queue.get(timeout=0.1)
                    
                    # Check if we need to filter by instrument
                    if instrument is None or item.get('instrument') == instrument:
                        return item.get('data')
                    else:
                        # Put item back in queue
                        if not self.data_queue.full():
                            self.data_queue.put(item)
                            
                except queue.Empty:
                    # No data in queue, retry until timeout
                    pass
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return None
    
    def get_latest_data_batch(self, instrument: str, timeframe: str = "1-minute",
                             count: int = 60) -> List[Dict]:
        """
        Get latest batch of data from file
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            count: Number of data points to return
            
        Returns:
            List of market data dictionaries
        """
        try:
            # Create filename with date
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"data/market/{instrument}_{date_str}.jsonl"
            
            if not os.path.exists(filename):
                # Try yesterday's file if today's doesn't exist
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                filename = f"data/market/{instrument}_{yesterday}.jsonl"
                
                if not os.path.exists(filename):
                    self.logger.warning(f"No data file found for {instrument}")
                    return []
            
            # Read last 'count' lines from file
            data = []
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    data.append(json.loads(line.strip()))
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting latest data batch: {e}")
            return []
    
    def place_order(self, order_data: Dict) -> Dict:
        """
        Place an order via NinjaTrader API
        
        Args:
            order_data: Order details
            
        Returns:
            Response data
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.post(
                f"{self.base_url}/orders", 
                json=order_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error placing order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status data
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.get(f"{self.base_url}/orders/{order_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error getting order status: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation result
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.delete(f"{self.base_url}/orders/{order_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error cancelling order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_account_info(self) -> Dict:
        """
        Get account information
        
        Returns:
            Account information
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.get(f"{self.base_url}/account")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error getting account info: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self) -> Dict:
        """
        Get current positions
        
        Returns:
            Dictionary of current positions
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.get(f"{self.base_url}/positions")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error getting positions: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"success": False, "error": str(e)}
    
    def close_position(self, position_id: str) -> Dict:
        """
        Close a position
        
        Args:
            position_id: Position ID
            
        Returns:
            Position closing result
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.post(
                f"{self.base_url}/positions/{position_id}/close"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error closing position: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"success": False, "error": str(e)}
    
    def get_market_data(self, instrument: str) -> Dict:
        """
        Get current market data for an instrument
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            Market data
        """
        try:
            # Ensure we're authenticated
            if not self.connected and not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
                
            # Make request
            response = self.session.get(
                f"{self.base_url}/marketdata/{instrument}"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Error getting market data: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {"success": False, "error": str(e)}
    
    def mock_stream_data(self, instrument: str, timeframe: str = "1-minute",
                        num_candles: int = 100) -> bool:
        """
        Generate mock streaming data for testing
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            num_candles: Number of candles to generate
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Generating mock data for {instrument}")
            
            # Set base values
            base_price = 100.0
            base_volume = 1000
            
            # Generate mock candles
            start_time = datetime.now() - timedelta(minutes=num_candles)
            
            for i in range(num_candles):
                # Calculate time
                candle_time = start_time + timedelta(minutes=i)
                
                # Generate price movement (random walk)
                price_change = (np.random.random() - 0.5) * 0.01 * base_price
                current_price = base_price + price_change
                
                # Generate OHLC
                open_price = current_price
                high_price = open_price * (1 + np.random.random() * 0.005)
                low_price = open_price * (1 - np.random.random() * 0.005)
                close_price = open_price * (1 + (np.random.random() - 0.5) * 0.008)
                
                # Generate volume
                volume = int(base_volume * (0.5 + np.random.random()))
                
                # Create candle data
                candle = {
                    "instrument": instrument,
                    "timestamp": candle_time.isoformat(),
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                    # Add EMAs
                    "EMA9": close_price * (1 + (np.random.random() - 0.5) * 0.001),
                    "EMA21": close_price * (1 + (np.random.random() - 0.5) * 0.002),
                    "EMA220": close_price * (1 + (np.random.random() - 0.5) * 0.005),
                    # Add ATR
                    "ATR": base_price * 0.005
                }
                
                # Put in queue
                if not self.data_queue.full():
                    self.data_queue.put({
                        'instrument': instrument,
                        'timeframe': timeframe,
                        'data': candle,
                        'timestamp': time.time()
                    })
                    
                # Update base price
                base_price = close_price
                
                # Sleep to simulate real-time data
                time.sleep(0.05)
            
            self.logger.info(f"Generated {num_candles} mock candles for {instrument}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating mock data: {e}")
            return False