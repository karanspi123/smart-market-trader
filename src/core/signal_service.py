import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import logging
from typing import Dict, Tuple, List, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalService:
    def __init__(self, config: Dict):
        """
        Initialize signal service with configuration
        
        Args:
            config: Dictionary containing signal configuration
        """
        self.config = config
        self.decision_tree = None
        self.calibrated_classifier = None
        self.signals_history = []
        self.last_atr = None
        self.avg_atr = None
        self.signal_buffer = []
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train decision tree for signal generation
        
        Args:
            X_train: Features including predicted price, price patterns, entropy
            y_train: Historical 1% moves (categorical: -1 for down, 0 for flat, 1 for up)
        """
        # Initialize decision tree with appropriate parameters
        self.decision_tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        
        # Train the decision tree
        self.decision_tree.fit(X_train, y_train)
        logger.info("Decision tree trained for signal generation")
        
        # Calibrate probabilities using Platt scaling
        self.calibrated_classifier = CalibratedClassifierCV(
            base_estimator=self.decision_tree,
            method='sigmoid',
            cv=5
        )
        self.calibrated_classifier.fit(X_train, y_train)
        logger.info("Probability calibration applied to decision tree")
    
    def prepare_features(self, 
                        predicted_price: float, 
                        current_price: float,
                        volume: float,
                        emas: List[float],
                        entropy: float,
                        pattern_label: int) -> np.ndarray:
        """
        Prepare features for decision tree input
        
        Args:
            predicted_price: Price prediction from Transformer
            current_price: Current market price
            volume: Current volume
            emas: List of EMAs [9, 21, 220]
            entropy: Prediction entropy
            pattern_label: CNN pattern recognition label
            
        Returns:
            Feature array for decision tree
        """
        # Calculate price difference as percentage
        price_diff_pct = (predicted_price - current_price) / current_price * 100
        
        # Create feature array
        features = np.array([
            price_diff_pct,       # Predicted price change percentage
            current_price,        # Current price
            volume,               # Volume
            emas[0],              # EMA9
            emas[1],              # EMA21
            emas[2],              # EMA220
            entropy,              # Prediction entropy
            pattern_label,        # Pattern label (0, 1, 2)
            emas[0] - emas[1],    # EMA9 - EMA21 (short term trend)
            emas[1] - emas[2]     # EMA21 - EMA220 (long term trend)
        ]).reshape(1, -1)
        
        return features
    
    def calculate_atr(self, ohlc_data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility filtering
        
        Args:
            ohlc_data: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            ATR value
        """
        high = ohlc_data['High'].values
        low = ohlc_data['Low'].values
        close = ohlc_data['Close'].values
        
        # Handle first row where previous close is not available
        tr1 = np.zeros(len(high))
        tr1[1:] = np.abs(high[1:] - close[:-1])
        
        # True range calculations
        tr2 = np.abs(low - close)
        tr3 = np.abs(high - low)
        
        # True Range is the max of the three calculations
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR using simple moving average
        atr = np.mean(tr[-period:])
        
        return atr
    
    def generate_signal(self, 
                       predicted_price: float, 
                       current_candle: Dict[str, float],
                       historical_data: pd.DataFrame,
                       entropy: float,
                       pattern_label: int) -> Dict[str, Any]:
        """
        Generate trading signal based on predictions and current market data
        
        Args:
            predicted_price: Price prediction from transformer
            current_candle: Current price candle data
            historical_data: Recent historical data for ATR calculation
            entropy: Prediction entropy
            pattern_label: Pattern label from CNN
            
        Returns:
            Dictionary with signal details
        """
        # Start timing for latency measurement
        start_time = time.time()
        
        # Extract current price data
        current_price = current_candle['Close']
        volume = current_candle['Volume']
        
        # Extract EMAs
        emas = [
            current_candle.get('EMA9', 0),
            current_candle.get('EMA21', 0),
            current_candle.get('EMA220', 0)
        ]
        
        # Check entropy threshold
        entropy_threshold = self.config.get('entropy_threshold', 0.75)
        if entropy > entropy_threshold:
            logger.warning(f"High entropy ({entropy:.2f}) exceeds threshold ({entropy_threshold})")
            signal_result = {
                'signal': 'Hold',
                'confidence': 0.0,
                'reason': f"High uncertainty (entropy: {entropy:.2f})",
                'predicted_price': predicted_price,
                'current_price': current_price,
                'entropy': entropy,
                'pattern': pattern_label,
                'latency_ms': 0
            }
            
            # Calculate latency
            end_time = time.time()
            signal_result['latency_ms'] = (end_time - start_time) * 1000
            
            return signal_result
        
        # Calculate ATR for volatility filtering
        if historical_data is not None and len(historical_data) >= 14:
            self.last_atr = self.calculate_atr(historical_data, 14)
            
            # Initialize avg_atr if not set
            if self.avg_atr is None:
                self.avg_atr = self.last_atr
            else:
                # Exponential moving average of ATR
                self.avg_atr = 0.95 * self.avg_atr + 0.05 * self.last_atr
            
            # Volatility filter
            if self.last_atr > 2 * self.avg_atr:
                logger.warning(f"High volatility detected: ATR ({self.last_atr:.4f}) > 2x Avg ATR ({self.avg_atr:.4f})")
                signal_result = {
                    'signal': 'Hold',
                    'confidence': 0.0,
                    'reason': f"High volatility (ATR: {self.last_atr:.4f})",
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'entropy': entropy,
                    'pattern': pattern_label,
                    'latency_ms': 0
                }
                
                # Calculate latency
                end_time = time.time()
                signal_result['latency_ms'] = (end_time - start_time) * 1000
                
                return signal_result
        
        # Prepare features for decision tree
        features = self.prepare_features(
            predicted_price, 
            current_price, 
            volume, 
            emas, 
            entropy, 
            pattern_label
        )
        
        # If decision tree is not trained, use rules-based approach
        if self.decision_tree is None:
            # Calculate price difference as percentage
            price_diff_pct = (predicted_price - current_price) / current_price * 100
            
            # Simple rules for signal generation
            if price_diff_pct > 0.1 and pattern_label == 2:  # Positive pattern
                signal = 'Buy'
                confidence = 0.5 + min(abs(price_diff_pct) / 2, 0.3) + (1 - entropy) * 0.2
                reason = "Predicted price increase and positive pattern"
            elif price_diff_pct < -0.1 and pattern_label == 0:  # Negative pattern
                signal = 'Sell'
                confidence = 0.5 + min(abs(price_diff_pct) / 2, 0.3) + (1 - entropy) * 0.2
                reason = "Predicted price decrease and negative pattern"
            else:
                signal = 'Hold'
                confidence = 0.5
                reason = "No clear signal"
                
            # Adjust confidence based on EMA alignment
            ema_short_above_med = emas[0] > emas[1]
            ema_med_above_long = emas[1] > emas[2]
            
            if signal == 'Buy' and ema_short_above_med and ema_med_above_long:
                confidence = min(confidence + 0.1, 0.95)
                reason += " with EMA confirmation"
            elif signal == 'Sell' and not ema_short_above_med and not ema_med_above_long:
                confidence = min(confidence + 0.1, 0.95)
                reason += " with EMA confirmation"
        else:
            # Use trained decision tree with calibrated probabilities
            probabilities = self.calibrated_classifier.predict_proba(features)[0]
            prediction = self.calibrated_classifier.predict(features)[0]
            
            # Map prediction to signal
            if prediction == 1:  # Up move
                signal = 'Buy'
                confidence = probabilities[2]  # Probability of class 1
                reason = "Decision tree predicts upward move"
            elif prediction == -1:  # Down move
                signal = 'Sell'
                confidence = probabilities[0]  # Probability of class -1
                reason = "Decision tree predicts downward move"
            else:  # Flat
                signal = 'Hold'
                confidence = probabilities[1]  # Probability of class 0
                reason = "Decision tree predicts sideways movement"
        
        # Add signal to buffer for smoothing
        self.signal_buffer.append(signal)
        if len(self.signal_buffer) > 5:  # Keep only the last 5 signals
            self.signal_buffer.pop(0)
        
        # Signal smoothing - require 3 of last 5 signals to agree for a change
        if signal in ['Buy', 'Sell']:
            count = self.signal_buffer.count(signal)
            if count < 3:
                signal = 'Hold'
                confidence = 0.5
                reason = f"Signal ({signal}) not confirmed by buffer ({count}/5 agreement)"
        
        # Build result dictionary
        signal_result = {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'entropy': entropy,
            'pattern': pattern_label,
            'latency_ms': 0
        }
        
        # Calculate latency
        end_time = time.time()
        signal_result['latency_ms'] = (end_time - start_time) * 1000
        
        # Add to signals history
        self.signals_history.append(signal_result)
        
        return signal_result
    
    def generate_historical_training_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for decision tree based on historical price movements
        
        Args:
            historical_data: DataFrame with historical OHLC data
            
        Returns:
            Features and target arrays for training
        """
        window_size = 10  # Use 10 candles for feature extraction
        threshold = 0.01  # 1% move threshold
        
        features = []
        targets = []
        
        # Loop through historical data to create training samples
        for i in range(window_size, len(historical_data) - 1):
            # Current window
            window = historical_data.iloc[i-window_size:i]
            
            # Target: next candle's movement
            next_candle = historical_data.iloc[i+1]
            current_candle = historical_data.iloc[i]
            
            price_move_pct = (next_candle['Close'] - current_candle['Close']) / current_candle['Close']
            
            # Classify movement
            if price_move_pct > threshold:
                target = 1  # Up move
            elif price_move_pct < -threshold:
                target = -1  # Down move
            else:
                target = 0  # Flat
            
            # Extract features
            current_price = current_candle['Close']
            volume = current_candle['Volume']
            
            emas = [
                current_candle.get('EMA9', 0),
                current_candle.get('EMA21', 0),
                current_candle.get('EMA220', 0)
            ]
            
            # Simple trend features
            short_trend = (current_candle.get('EMA9', 0) - window.iloc[0].get('EMA9', 0)) / window.iloc[0].get('EMA9', 0)
            med_trend = (current_candle.get('EMA21', 0) - window.iloc[0].get('EMA21', 0)) / window.iloc[0].get('EMA21', 0)
            
            # Price volatility
            volatility = window['Close'].std() / window['Close'].mean()
            
            # Volume trend
            volume_trend = current_candle['Volume'] / window['Volume'].mean()
            
            # Create feature vector
            feature = [
                current_price,
                volume,
                emas[0],
                emas[1],
                emas[2],
                emas[0] - emas[1],  # EMA9 - EMA21
                emas[1] - emas[2],  # EMA21 - EMA220
                short_trend,
                med_trend,
                volatility,
                volume_trend
            ]
            
            features.append(feature)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated signals
        
        Returns:
            Dictionary with signal statistics
        """
        if not self.signals_history:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "avg_confidence": 0,
                "avg_entropy": 0,
                "avg_latency_ms": 0
            }
        
        # Count signals by type
        total = len(self.signals_history)
        buy_count = sum(1 for s in self.signals_history if s['signal'] == 'Buy')
        sell_count = sum(1 for s in self.signals_history if s['signal'] == 'Sell')
        hold_count = sum(1 for s in self.signals_history if s['signal'] == 'Hold')
        
        # Calculate averages
        avg_confidence = sum(s['confidence'] for s in self.signals_history) / total
        avg_entropy = sum(s['entropy'] for s in self.signals_history) / total
        avg_latency = sum(s['latency_ms'] for s in self.signals_history) / total
        
        return {
            "total_signals": total,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "buy_ratio": buy_count / total if total > 0 else 0,
            "sell_ratio": sell_count / total if total > 0 else 0,
            "hold_ratio": hold_count / total if total > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "avg_latency_ms": avg_latency
        }
    def generate_signal_with_strategy(self, prediction: Dict, latest_candle: Dict, 
                                strategy_indicators: Dict) -> Dict:
    # Get strategy type from config
        strategy_type = self.config.get("strategy", {}).get("type", "model_only")
        
        # First get base signal from model
        base_signal = self.generate_signal(prediction, latest_candle)
        
        # If no strategy overlay, return base signal
        if strategy_type == "model_only":
            return base_signal
        
        # Get confidence threshold
        threshold = self.config.get("confidence_threshold", 0.75)
        
        # Apply strategy overlay
        if strategy_type == "mean_reversion":
            # Get mean reversion indicators
            bb_position = strategy_indicators.get("bb_position", 0.5)
            rsi = strategy_indicators.get("rsi", 50)
            
            # Adjust signal based on mean reversion
            if bb_position < 0.1 and rsi < 30:
                # Strong buy signal from mean reversion
                if base_signal["signal"] == "Buy":
                    # Strengthen buy signal
                    return {
                        "signal": "Buy",
                        "confidence": max(base_signal["confidence"], 0.9),
                        "reason": "model_and_mean_reversion_agree_buy"
                    }
                elif base_signal["signal"] == "Sell" and base_signal["confidence"] < threshold:
                    # Override weak sell signal
                    return {
                        "signal": "Buy",
                        "confidence": 0.8,
                        "reason": "mean_reversion_override_buy"
                    }
                else:
                    # Keep original signal
                    return base_signal
                    
            elif bb_position > 0.9 and rsi > 70:
                # Strong sell signal from mean reversion
                if base_signal["signal"] == "Sell":
                    # Strengthen sell signal
                    return {
                        "signal": "Sell",
                        "confidence": max(base_signal["confidence"], 0.9),
                        "reason": "model_and_mean_reversion_agree_sell"
                    }
                elif base_signal["signal"] == "Buy" and base_signal["confidence"] < threshold:
                    # Override weak buy signal
                    return {
                        "signal": "Sell",
                        "confidence": 0.8,
                        "reason": "mean_reversion_override_sell"
                    }
                else:
                    # Keep original signal
                    return base_signal
            else:
                # No strong mean reversion signal
                return base_signal
        
        # Other strategy types can be implemented similarly
        
        return base_signal
    