import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Any, Optional
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, config: Dict):
        """
        Initialize model service with configuration
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.transformer_model = None
        self.cnn_model = None
        self.combined_model = None
        self.price_history = []
        self.mse_history = []
        self.price_bins = None
    
    def build_transformer_model(self) -> Model:
        """
        Build Transformer model for price prediction
        
        Returns:
            Compiled Transformer model
        """
        window_size = self.config.get('window_size', 60)
        num_features = 9  # Time, OHLCV, EMA9, EMA21, EMA220
        d_model = self.config.get('d_model', 64)
        num_heads = self.config.get('num_heads', 4)
        num_bins = self.config.get('bins', 10)
        
        # Input layer
        inputs = Input(shape=(window_size, num_features))
        
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )(inputs, inputs)
        
        # Add & Norm (residual connection and layer normalization)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn = Dense(d_model, activation='relu', kernel_regularizer=l2(0.01))(attention_output)
        ffn = Dropout(0.2)(ffn)
        
        # Add & Norm
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn)
        
        # Global average pooling to reduce sequence dimension
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        
        # Output layer with bins
        outputs = Dense(num_bins, activation='softmax')(pooled)
        
        # Build and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built Transformer model for price prediction")
        logger.info(f"Model configuration: window_size={window_size}, d_model={d_model}, num_heads={num_heads}")
        
        return model
    
    def build_cnn_model(self) -> Model:
        """
        Build CNN model for pattern recognition
        
        Returns:
            Compiled CNN model
        """
        window_size = self.config.get('window_size', 60)
        num_features = 9  # OHLCV, EMAs
        
        # Input layer
        inputs = Input(shape=(window_size, num_features))
        
        # Reshape for 2D convolution (add channel dimension)
        reshaped = Reshape((window_size, num_features, 1))(inputs)
        
        # First convolutional block
        conv1 = Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            kernel_regularizer=l2(0.01)
        )(reshaped)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Second convolutional block
        conv2 = Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            kernel_regularizer=l2(0.01)
        )(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Flatten and dense layers
        flattened = Flatten()(pool2)
        dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(flattened)
        dropout = Dropout(0.2)(dense)
        
        # Output layer for pattern recognition (3 classes)
        outputs = Dense(3, activation='softmax')(dropout)
        
        # Build and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built CNN model for pattern recognition")
        
        return model
    
    def build_combined_model(self) -> Model:
        """
        Build combined model using transformer and CNN features
        
        Returns:
            Compiled combined model
        """
        window_size = self.config.get('window_size', 60)
        num_features = 9
        
        # Shared input
        inputs = Input(shape=(window_size, num_features))
        
        # Transformer branch
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=16
        )(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        ffn = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(attention_output)
        ffn = Dropout(0.2)(ffn)
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn)
        transformer_features = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        
        # CNN branch
        reshaped = Reshape((window_size, num_features, 1))(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        cnn_features = Flatten()(pool2)
        
        # Combine features
        combined = Concatenate()([transformer_features, cnn_features])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        
        # Output layers
        price_output = Dense(self.config.get('bins', 10), activation='softmax', name='price_prediction')(combined)
        pattern_output = Dense(3, activation='softmax', name='pattern_recognition')(combined)
        
        # Build and compile combined model
        model = Model(inputs=inputs, outputs=[price_output, pattern_output])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'price_prediction': 'categorical_crossentropy',
                'pattern_recognition': 'categorical_crossentropy'
            },
            metrics={
                'price_prediction': ['accuracy'],
                'pattern_recognition': ['accuracy']
            }
        )
        
        logger.info("Built combined model with shared features")
        
        return model
    
    def preprocess_price_targets(self, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess price targets into bins for classification
        
        Args:
            y_data: Array of price targets
            
        Returns:
            One-hot encoded bins and bin edges
        """
        num_bins = self.config.get('bins', 10)
        
        # Find min and max for binning
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Create bins
        bins = np.linspace(y_min, y_max, num_bins + 1)
        
        # Digitize into bin indices
        binned_indices = np.digitize(y_data, bins) - 1
        
        # Clip to ensure valid indices
        binned_indices = np.clip(binned_indices, 0, num_bins - 1)
        
        # One-hot encode
        y_binned = tf.keras.utils.to_categorical(binned_indices, num_classes=num_bins)
        
        return y_binned, bins
    
    def generate_pattern_labels(self, X_data: np.ndarray) -> np.ndarray:
        """
        Generate pattern labels for sequences
        Patterns:
        - 0: Negative (downtrend)
        - 1: Neutral
        - 2: Positive (uptrend)
        
        Args:
            X_data: Sequence data
            
        Returns:
            Pattern labels
        """
        num_sequences = X_data.shape[0]
        pattern_labels = np.zeros(num_sequences)
        
        for i in range(num_sequences):
            sequence = X_data[i]
            
            # Extract close prices from sequence
            close_prices = sequence[:, 3]  # Assuming Close is at index 3
            
            # Use EMAs for trend detection
            # Use EMA9 and EMA21 for trend (if available in the data)
            ema_short = sequence[:, 6]  # EMA9
            ema_long = sequence[:, 7]   # EMA21
            
            # Calculate trend based on EMA crossover
            is_uptrend = ema_short[-1] > ema_long[-1] and ema_short[-5] <= ema_long[-5]
            is_downtrend = ema_short[-1] < ema_long[-1] and ema_short[-5] >= ema_long[-5]
            
            if is_uptrend:
                pattern_labels[i] = 2  # Positive
            elif is_downtrend:
                pattern_labels[i] = 0  # Negative
            else:
                pattern_labels[i] = 1  # Neutral
        
        # Convert to one-hot encoding
        pattern_labels_onehot = tf.keras.utils.to_categorical(pattern_labels, num_classes=3)
        
        return pattern_labels_onehot
    
    def calculate_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate Shannon entropy for prediction probabilities
        
        Args:
            probabilities: Array of prediction probabilities
            
        Returns:
            Entropy values
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1.0)
        
        # Shannon entropy: H = -sum(p_i * log2(p_i))
        entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
        
        # Normalize by max entropy (log2(num_bins))
        num_bins = self.config.get('bins', 10)
        max_entropy = np.log2(num_bins)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def calculate_confidence(self, entropy: np.ndarray) -> np.ndarray:
        """
        Calculate confidence from entropy
        
        Args:
            entropy: Normalized entropy values
            
        Returns:
            Confidence values (1 - normalized_entropy)
        """
        return 1 - entropy
    
    def train_transformer(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Model, Dict]:
        """
        Train Transformer model with time series cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained model and training history
        """
        # Build model if not exists
        if self.transformer_model is None:
            self.transformer_model = self.build_transformer_model()
        
        # Preprocess targets into bins
        y_train_binned, bins = self.preprocess_price_targets(y_train)
        y_val_binned, _ = self.preprocess_price_targets(y_val)
        
        # Store bins for later prediction conversion
        self.price_bins = bins
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.transformer_model.fit(
            X_train, y_train_binned,
            validation_data=(X_val, y_val_binned),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = self.transformer_model.evaluate(X_val, y_val_binned)
        logger.info(f"Transformer validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        return self.transformer_model, history.history
    
    def train_cnn(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[Model, Dict]:
        """
        Train CNN model for pattern recognition
        
        Args:
            X_train: Training features
            X_val: Validation features
            
        Returns:
            Trained model and training history
        """
        # Build model if not exists
        if self.cnn_model is None:
            self.cnn_model = self.build_cnn_model()
        
        # Generate pattern labels
        y_train_patterns = self.generate_pattern_labels(X_train)
        y_val_patterns = self.generate_pattern_labels(X_val)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.cnn_model.fit(
            X_train, y_train_patterns,
            validation_data=(X_val, y_val_patterns),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = self.cnn_model.evaluate(X_val, y_val_patterns)
        logger.info(f"CNN validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        return self.cnn_model, history.history
    
    def train_combined_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Model, Dict]:
        """
        Train combined model with transformer and CNN features
        
        Args:
            X_train: Training features
            y_train: Training targets (price)
            X_val: Validation features
            y_val: Validation targets (price)
            
        Returns:
            Trained model and training history
        """
        # Build model if not exists
        if self.combined_model is None:
            self.combined_model = self.build_combined_model()
        
        # Preprocess price targets into bins
        y_train_binned, bins = self.preprocess_price_targets(y_train)
        y_val_binned, _ = self.preprocess_price_targets(y_val)
        
        # Store bins for later prediction conversion
        self.price_bins = bins
        
        # Generate pattern labels
        y_train_patterns = self.generate_pattern_labels(X_train)
        y_val_patterns = self.generate_pattern_labels(X_val)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.combined_model.fit(
            X_train, 
            {
                'price_prediction': y_train_binned,
                'pattern_recognition': y_train_patterns
            },
            validation_data=(
                X_val, 
                {
                    'price_prediction': y_val_binned,
                    'pattern_recognition': y_val_patterns
                }
            ),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.combined_model, history.history
    
    def predict_prices(self, X_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict prices using the trained model
        
        Args:
            X_data: Input features
            
        Returns:
            Dictionary with predictions, confidence, and entropy
        """
        if self.transformer_model is None and self.combined_model is None:
            raise ValueError("Model not trained yet")
        
        if self.combined_model is not None:
            # Use combined model if available
            predictions = self.combined_model.predict(X_data)
            price_probs = predictions[0]
            pattern_probs = predictions[1]
        else:
            # Use separate models
            price_probs = self.transformer_model.predict(X_data)
            pattern_probs = self.cnn_model.predict(X_data)
        
        # Calculate entropy and confidence
        entropy = self.calculate_entropy(price_probs)
        confidence = self.calculate_confidence(entropy)
        
        # Get predicted bin indices
        pred_bin_indices = np.argmax(price_probs, axis=1)
        
        # Convert bin indices to prices
        if self.price_bins is not None:
            # Use the middle of each bin as the prediction
            bin_midpoints = (self.price_bins[:-1] + self.price_bins[1:]) / 2
            pred_prices = bin_midpoints[pred_bin_indices]
        else:
            pred_prices = pred_bin_indices
        
        # Get pattern predictions
        pattern_indices = np.argmax(pattern_probs, axis=1)
        pattern_labels = ['Negative', 'Neutral', 'Positive']
        pred_patterns = np.array([pattern_labels[i] for i in pattern_indices])
        
        # Track MSE for drift detection
        if len(self.price_history) > 0:
            # Compare new predictions with actual prices
            actual_prices = X_data[:, -1, 3]  # Last timestep, Close price
            mse = mean_squared_error(actual_prices, pred_prices)
            self.mse_history.append(mse)
            
            # Check for drift
            if len(self.mse_history) > 5:
                current_mse = np.mean(self.mse_history[-5:])
                baseline_mse = np.mean(self.mse_history[:5])
                
                if current_mse > baseline_mse * 1.2:  # 20% increase in MSE
                    logger.warning("Detected model drift. Consider retraining.")
        
        # Store predictions for drift detection
        self.price_history.append(pred_prices)
        
        return {
            'price_predictions': pred_prices,
            'confidence': confidence,
            'entropy': entropy,
            'pattern_predictions': pred_patterns,
            'price_probabilities': price_probs,
            'pattern_probabilities': pattern_probs
        }
    
    def save_models(self, path: str) -> None:
        """
        Save trained models
        
        Args:
            path: Directory path to save models
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        if self.transformer_model:
            self.transformer_model.save(os.path.join(path, 'transformer_model'))
            
        if self.cnn_model:
            self.cnn_model.save(os.path.join(path, 'cnn_model'))
            
        if self.combined_model:
            self.combined_model.save(os.path.join(path, 'combined_model'))
            
        # Save price bins for conversion
        if self.price_bins is not None:
            np.save(os.path.join(path, 'price_bins.npy'), self.price_bins)
            
        # Save config
        with open(os.path.join(path, 'model_config.json'), 'w') as f:
            json.dump(self.config, f)
            
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str) -> None:
        """
        Load trained models
        
        Args:
            path: Directory path to load models from
        """
        # Load Transformer model
        transformer_path = os.path.join(path, 'transformer_model')
        if os.path.exists(transformer_path):
            self.transformer_model = tf.keras.models.load_model(transformer_path)
            logger.info("Loaded Transformer model")
            
        # Load CNN model
        cnn_path = os.path.join(path, 'cnn_model')
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            logger.info("Loaded CNN model")
            
        # Load combined model
        combined_path = os.path.join(path, 'combined_model')
        if os.path.exists(combined_path):
            self.combined_model = tf.keras.models.load_model(combined_path)
            logger.info("Loaded combined model")
            
        # Load price bins
        bins_path = os.path.join(path, 'price_bins.npy')
        if os.path.exists(bins_path):
            self.price_bins = np.load(bins_path)
            
        # Load config
        config_path = os.path.join(path, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        logger.info(f"Models loaded from {path}")
    
    def check_drift(self, new_data: np.ndarray, actual_prices: np.ndarray) -> bool:
        """
        Check for model drift
        
        Args:
            new_data: New input data
            actual_prices: Actual price values
            
        Returns:
            Boolean indicating whether drift is detected
        """
        # Predict prices
        predictions = self.predict_prices(new_data)
        pred_prices = predictions['price_predictions']
        
        # Calculate MSE
        mse = mean_squared_error(actual_prices, pred_prices)
        self.mse_history.append(mse)
        
        # Check for drift (20% increase in MSE)
        if len(self.mse_history) > 10:
            current_mse = np.mean(self.mse_history[-5:])
            baseline_mse = np.mean(self.mse_history[:5])
            
            if current_mse > baseline_mse * 1.2:
                logger.warning(f"Drift detected: baseline MSE={baseline_mse:.4f}, current MSE={current_mse:.4f}")
                return True
                
        return False