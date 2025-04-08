import logging
import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import socket

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class MonitoringService:
    """Monitoring service for Smart Market Analyzer"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize monitoring service
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Set up logging directory
        self.log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up metrics file
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
        self.metrics = self._load_metrics()
        
        # Set up prometheus metrics if enabled
        self.prometheus_enabled = self.config.get('prometheus', {}).get('enabled', False) and PROMETHEUS_AVAILABLE
        self.prometheus_metrics = {}
        
        if self.prometheus_enabled:
            try:
                # Set up prometheus metrics
                self._setup_prometheus_metrics()
                
                # Start prometheus server
                prometheus_port = self.config.get('prometheus', {}).get('port', 8001)
                prom.start_http_server(prometheus_port)
                self.logger.info(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus metrics server: {e}")
                self.prometheus_enabled = False
        
        # Start metrics saving thread
        self.running = True
        self.metrics_thread = threading.Thread(target=self._metrics_saver, daemon=True)
        self.metrics_thread.start()
    
    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # Create metrics
        self.prometheus_metrics['predictions'] = prom.Counter(
            'smart_market_predictions_total',
            'Total number of predictions made',
            ['instrument']
        )
        
        self.prometheus_metrics['signals'] = prom.Counter(
            'smart_market_signals_total',
            'Total number of signals generated',
            ['instrument', 'signal_type']
        )
        
        self.prometheus_metrics['trades'] = prom.Counter(
            'smart_market_trades_total',
            'Total number of trades executed',
            ['instrument', 'direction', 'result']
        )
        
        self.prometheus_metrics['errors'] = prom.Counter(
            'smart_market_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        self.prometheus_metrics['active_models'] = prom.Gauge(
            'smart_market_active_models',
            'Number of active models'
        )
        
        self.prometheus_metrics['prediction_latency'] = prom.Histogram(
            'smart_market_prediction_latency_seconds',
            'Prediction latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.prometheus_metrics['signal_latency'] = prom.Histogram(
            'smart_market_signal_latency_seconds',
            'Signal generation latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.prometheus_metrics['trade_latency'] = prom.Histogram(
            'smart_market_trade_latency_seconds',
            'Trade execution latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
    
    def _load_metrics(self) -> Dict:
        """Load metrics from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metrics from {self.metrics_file}: {e}")
        
        # Initialize empty metrics
        return {
            'predictions': {
                'total': 0,
                'by_instrument': {}
            },
            'signals': {
                'total': 0,
                'by_instrument': {},
                'by_type': {
                    'Buy': 0,
                    'Sell': 0,
                    'Hold': 0
                }
            },
            'trades': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'by_instrument': {},
                'by_direction': {
                    'buy': 0,
                    'sell': 0
                }
            },
            'errors': {
                'total': 0,
                'by_type': {}
            },
            'latency': {
                'prediction': [],
                'signal': [],
                'trade': []
            },
            'active_models': 0,
            'system': {
                'start_time': time.time(),
                'hostname': socket.gethostname()
            }
        }
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {self.metrics_file}: {e}")
    
    def _metrics_saver(self):
        """Thread function to periodically save metrics"""
        while self.running:
            try:
                # Save metrics every 60 seconds
                time.sleep(60)
                self._save_metrics()
            except Exception as e:
                self.logger.error(f"Error in metrics saver thread: {e}")
                time.sleep(10)  # Sleep longer after error
    
    def log_prediction(self, instrument: str, prediction: float, confidence: float, latency: float = None):
        """
        Log a prediction
        
        Args:
            instrument: Instrument symbol
            prediction: Predicted value
            confidence: Prediction confidence
            latency: Prediction latency in seconds
        """
        # Update metrics
        self.metrics['predictions']['total'] += 1
        
        if instrument not in self.metrics['predictions']['by_instrument']:
            self.metrics['predictions']['by_instrument'][instrument] = 0
        self.metrics['predictions']['by_instrument'][instrument] += 1
        
        # Log latency if provided
        if latency is not None:
            self.metrics['latency']['prediction'].append(latency)
            # Keep only last 1000 latency measurements
            if len(self.metrics['latency']['prediction']) > 1000:
                self.metrics['latency']['prediction'] = self.metrics['latency']['prediction'][-1000:]
                
            # Update prometheus metrics
            if self.prometheus_enabled:
                self.prometheus_metrics['prediction_latency'].observe(latency)
        
        # Update prometheus metrics
        if self.prometheus_enabled:
            self.prometheus_metrics['predictions'].labels(instrument=instrument).inc()
        
        # Log to file
        self._log_event('prediction', {
            'instrument': instrument,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def log_signal(self, instrument: str, signal_type: str, confidence: float, latency: float = None):
        """
        Log a signal
        
        Args:
            instrument: Instrument symbol
            signal_type: Signal type (Buy, Sell, Hold)
            confidence: Signal confidence
            latency: Signal generation latency in seconds
        """
        # Update metrics
        self.metrics['signals']['total'] += 1
        
        if instrument not in self.metrics['signals']['by_instrument']:
            self.metrics['signals']['by_instrument'][instrument] = 0
        self.metrics['signals']['by_instrument'][instrument] += 1
        
        if signal_type not in self.metrics['signals']['by_type']:
            self.metrics['signals']['by_type'][signal_type] = 0
        self.metrics['signals']['by_type'][signal_type] += 1
        
        # Log latency if provided
        if latency is not None:
            self.metrics['latency']['signal'].append(latency)
            # Keep only last 1000 latency measurements
            if len(self.metrics['latency']['signal']) > 1000:
                self.metrics['latency']['signal'] = self.metrics['latency']['signal'][-1000:]
                
            # Update prometheus metrics
            if self.prometheus_enabled:
                self.prometheus_metrics['signal_latency'].observe(latency)
        
        # Update prometheus metrics
        if self.prometheus_enabled:
            self.prometheus_metrics['signals'].labels(
                instrument=instrument,
                signal_type=signal_type
            ).inc()
        
        # Log to file
        self._log_event('signal', {
            'instrument': instrument,
            'type': signal_type,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def log_trade(self, instrument: str, direction: str, result: bool, latency: float = None, reason: str = None):
        """
        Log a trade
        
        Args:
            instrument: Instrument symbol
            direction: Trade direction (buy, sell)
            result: True if successful, False otherwise
            latency: Trade execution latency in seconds
            reason: Reason for the trade
        """
        # Update metrics
        self.metrics['trades']['total'] += 1
        
        if result:
            self.metrics['trades']['successful'] += 1
        else:
            self.metrics['trades']['failed'] += 1
        
        if instrument not in self.metrics['trades']['by_instrument']:
            self.metrics['trades']['by_instrument'][instrument] = 0
        self.metrics['trades']['by_instrument'][instrument] += 1
        
        if direction not in self.metrics['trades']['by_direction']:
            self.metrics['trades']['by_direction'][direction] = 0
        self.metrics['trades']['by_direction'][direction] += 1
        
        # Log latency if provided
        if latency is not None:
            self.metrics['latency']['trade'].append(latency)
            # Keep only last 1000 latency measurements
            if len(self.metrics['latency']['trade']) > 1000:
                self.metrics['latency']['trade'] = self.metrics['latency']['trade'][-1000:]
                
            # Update prometheus metrics
            if self.prometheus_enabled:
                self.prometheus_metrics['trade_latency'].observe(latency)
        
        # Update prometheus metrics
        if self.prometheus_enabled:
            self.prometheus_metrics['trades'].labels(
                instrument=instrument,
                direction=direction,
                result='success' if result else 'failure'
            ).inc()
        
        # Log to file
        self._log_event('trade', {
            'instrument': instrument,
            'direction': direction,
            'result': 'success' if result else 'failure',
            'reason': reason,
            'timestamp': time.time()
        })
    
    def log_error(self, error_type: str, error_message: str, details: Dict = None):
        """
        Log an error
        
        Args:
            error_type: Type of error
            error_message: Error message
            details: Additional details
        """
        # Update metrics
        self.metrics['errors']['total'] += 1
        
        if error_type not in self.metrics['errors']['by_type']:
            self.metrics['errors']['by_type'][error_type] = 0
        self.metrics['errors']['by_type'][error_type] += 1
        
        # Update prometheus metrics
        if self.prometheus_enabled:
            self.prometheus_metrics['errors'].labels(error_type=error_type).inc()
        
        # Log to file
        self._log_event('error', {
            'type': error_type,
            'message': error_message,
            'details': details,
            'timestamp': time.time()
        })
        
        # Also log to application logger
        self.logger.error(f"{error_type}: {error_message}")
    
    def track_active_models(self, count: int):
        """
        Track number of active models
        
        Args:
            count: Number of active models
        """
        self.metrics['active_models'] = count
        
        # Update prometheus metrics
        if self.prometheus_enabled:
            self.prometheus_metrics['active_models'].set(count)
    
    def _log_event(self, event_type: str, event_data: Dict):
        """
        Log an event to a file
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        try:
            # Create log file path
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(self.log_dir, f"{event_type}_{date_str}.jsonl")
            
            # Append event to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log {event_type} event: {e}")
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics
        
        Returns:
            Current metrics
        """
        return self.metrics
    
    def stop(self):
        """Stop monitoring service"""
        self.running = False
        
        # Wait for metrics thread to finish
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)
        
        # Save metrics one last time
        self._save_metrics()
        
        self.logger.info("Monitoring service stopped")