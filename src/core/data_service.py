import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import datetime
import logging
from typing import Dict, Tuple, Optional, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataService:
    def __init__(self, config: Dict):
        """
        Initialize DataService with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file and validate
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame or None if validation fails
        """
        try:
            # Try to load the specified CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded data from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {csv_path}: {e}")
            
            # Fallback to another source
            fallback_path = self._get_fallback_path(csv_path)
            if fallback_path:
                try:
                    df = pd.read_csv(fallback_path)
                    logger.info(f"Successfully loaded fallback data from {fallback_path}")
                    return df
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback data: {fallback_e}")
            
            return None
    
    def _get_fallback_path(self, original_path: str) -> Optional[str]:
        """Get fallback data path if original fails"""
        # In a real system, this would attempt to fetch from Yahoo Finance
        # For this implementation, we'll look for a file with "_fallback" suffix
        base_path = os.path.splitext(original_path)[0]
        fallback_path = f"{base_path}_fallback.csv"
        
        if os.path.exists(fallback_path):
            return fallback_path
        return None
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data:
        - Convert Time to datetime index
        - Validate data
        - Handle missing values and outliers
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")
        
        # Convert Time to datetime index
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected: {missing_values}")
            # Fill missing values with linear interpolation
            df = df.interpolate(method='linear')
        
        # Check for outliers
        df = self._handle_outliers(df)
        
        # Filter by liquidity
        if 'Volume' in df.columns:
            df = df[df['Volume'] >= 100]
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data"""
        # Check for invalid OHLC relationships (High < Low)
        invalid_ohlc = (df['High'] < df['Low']).sum()
        if invalid_ohlc > 0:
            logger.warning(f"Found {invalid_ohlc} instances where High < Low")
            # Fix by swapping values
            invalid_idx = df['High'] < df['Low']
            df.loc[invalid_idx, ['High', 'Low']] = df.loc[invalid_idx, ['Low', 'High']].values
        
        # Cap extreme values at 3 standard deviations
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                mean, std = df[col].mean(), df[col].std()
                lower_bound, upper_bound = mean - 3*std, mean + 3*std
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    logger.warning(f"Capping {outliers} outliers in {col}")
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all numeric features to 0-1 range"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_normalized = df.copy()
        df_normalized[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df_normalized
    
    def resample_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample data to multiple timeframes based on config
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            Dictionary with resampled DataFrames
        """
        sample_rate = self.config.get('sample_rate', '1-minute')
        resampled = {'1-minute': df}
        
        if sample_rate == '1-minute':
            # No resampling needed
            return resampled
        
        # Define resampling rules
        resample_rules = {
            '5-minute': '5min',
            '15-minute': '15min',
            '1-hour': '1H'
        }
        
        # Resample OHLC data
        for timeframe, rule in resample_rules.items():
            if timeframe == sample_rate or sample_rate == 'multi':
                resampled[timeframe] = df.resample(rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                
                # Handle EMAs
                ema_cols = [col for col in df.columns if col.startswith('EMA')]
                if ema_cols:
                    for ema_col in ema_cols:
                        resampled[timeframe][ema_col] = df[ema_col].resample(rule).last()
        
        return resampled
    
    def create_regime_aware_split(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """
        Create train/val/test splits that are aware of market regimes
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with split DataFrames and regime info
        """
        # Extract splits from config
        train_pct = self.config.get('split', {}).get('train', 0.7)
        val_pct = self.config.get('split', {}).get('val', 0.15)
        test_pct = self.config.get('split', {}).get('test', 0.15)
        
        # Calculate returns for regime clustering
        df['returns'] = df['Close'].pct_change()
        
        # Use a rolling window to calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Create features for regime clustering
        cluster_features = df[['returns', 'volatility']].dropna()
        
        # K-means clustering to identify regimes
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_features_scaled = MinMaxScaler().fit_transform(cluster_features)
        df.loc[cluster_features.index, 'regime'] = kmeans.fit_predict(cluster_features_scaled)
        
        # Fill any missing regime values with forward fill
        df['regime'] = df['regime'].ffill().fillna(0).astype(int)
        
        # Get regimes and their counts
        regimes = df['regime'].unique()
        regime_counts = {regime: (df['regime'] == regime).sum() for regime in regimes}
        logger.info(f"Identified regimes with counts: {regime_counts}")
        
        # First do a chronological split
        n = len(df)
        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Check regime representation
        train_regimes = set(train_df['regime'].unique())
        val_regimes = set(val_df['regime'].unique())
        test_regimes = set(test_df['regime'].unique())
        
        all_regimes = set(regimes)
        
        # If any split is missing a regime, adjust splits
        if (train_regimes != all_regimes or 
            val_regimes != all_regimes or 
            test_regimes != all_regimes):
            
            logger.warning("Regimes not represented in all splits, adjusting...")
            
            # Ensure representation by adding samples from each regime
            for regime in all_regimes:
                regime_indices = df[df['regime'] == regime].index
                if regime not in train_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(100, len(regime_indices)))
                    train_df = pd.concat([train_df, df.loc[sample_idx]])
                
                if regime not in val_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(50, len(regime_indices)))
                    val_df = pd.concat([val_df, df.loc[sample_idx]])
                    
                if regime not in test_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(50, len(regime_indices)))
                    test_df = pd.concat([test_df, df.loc[sample_idx]])
        
        # Remove auxiliary columns used for regime detection
        for split_df in [train_df, val_df, test_df]:
            if 'returns' in split_df.columns:
                split_df.drop(columns=['returns'], inplace=True)
            if 'volatility' in split_df.columns:
                split_df.drop(columns=['volatility'], inplace=True)
        
        return {
            'train': train_df.sort_index(),
            'val': val_df.sort_index(),
            'test': test_df.sort_index()
        }, {
            'regime_counts': regime_counts,
            'regimes_in_train': list(train_df['regime'].unique()),
            'regimes_in_val': list(val_df['regime'].unique()),
            'regimes_in_test': list(test_df['regime'].unique())
        }
    
    def prepare_model_inputs(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Prepare model inputs with sequences and targets
        
        Args:
            splits: Dictionary with split DataFrames
            
        Returns:
            Dictionary with X and y data for each split
        """
        window_size = self.config.get('window_size', 60)
        feature_cols = [col for col in splits['train'].columns 
                        if col not in ['regime']]
        
        result = {}
        
        for split_name, split_df in splits.items():
            X_sequences = []
            y_price = []
            
            # Remove regime column for input features if present
            X_data = split_df[feature_cols]
            
            # Create sequences
            for i in range(len(X_data) - window_size):
                X_sequences.append(X_data.iloc[i:i+window_size].values)
                # Target is the next close price
                y_price.append(X_data.iloc[i+window_size]['Close'])
            
            if X_sequences:
                result[split_name] = {
                    'X': np.array(X_sequences),
                    'y': np.array(y_price)
                }
            else:
                logger.warning(f"No sequences could be created for {split_name} split")
                result[split_name] = {
                    'X': np.array([]),
                    'y': np.array([])
                }
                
        return result
    
    def process_data(self, csv_path: str) -> Dict:
        """
        Main method to process data end-to-end
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary with processed data
        """
        # Load data
        df = self.load_data(csv_path)
        if df is None:
            raise ValueError("Failed to load data")
        
        # Preprocess
        df = self.preprocess(df)
        
        # Normalize
        df_normalized = self.normalize_features(df)
        
        # Resample if needed
        resampled_data = self.resample_data(df_normalized)
        
        # Create splits for each timeframe
        result = {}
        for timeframe, timeframe_df in resampled_data.items():
            logger.info(f"Processing {timeframe} data")
            
            # Create regime-aware splits
            splits, regime_info = self.create_regime_aware_split(timeframe_df)
            
            # Prepare inputs for model
            model_inputs = self.prepare_model_inputs(splits)
            
            result[timeframe] = {
                'splits': splits,
                'model_inputs': model_inputs,
                'regime_info': regime_info,
                'feature_scaler': self.scaler
            }
        
        return result
    
    def generate_mock_data(self, sample_row: str, num_rows: int = 100) -> pd.DataFrame:
        """
        Generate mock data based on a sample row
        
        Args:
            sample_row: Sample row string
            num_rows: Number of rows to generate
            
        Returns:
            DataFrame with mock data
        """
        # Parse sample row
        columns = ["Time", "Open", "High", "Low", "Close", "Volume", "EMA9", "EMA21", "EMA220"]
        sample_values = sample_row.split(',')
        
        # Create base dataframe
        base_time = pd.to_datetime(sample_values[0] + " " + sample_values[1])
        times = [base_time + datetime.timedelta(minutes=i) for i in range(num_rows)]
        
        # Initialize with sample values
        data = {
            "Time": times,
            "Open": float(sample_values[2]),
            "High": float(sample_values[3]),
            "Low": float(sample_values[4]),
            "Close": float(sample_values[5]),
            "Volume": int(sample_values[6]),
            "EMA9": float(sample_values[7]),
            "EMA21": float(sample_values[8]),
            "EMA220": float(sample_values[9]) if len(sample_values) > 9 else float(sample_values[8])
        }
        
        df = pd.DataFrame(data)
        
        # Generate random walk for prices
        volatility = data["Close"] * 0.001  # 0.1% volatility
        
        for i in range(1, num_rows):
            prev_close = df.loc[i-1, "Close"]
            # Random price change with mean 0
            price_change = np.random.normal(0, volatility)
            
            # Update OHLC with random walk
            df.loc[i, "Close"] = prev_close + price_change
            df.loc[i, "Open"] = df.loc[i-1, "Close"] + np.random.normal(0, volatility*0.5)
            df.loc[i, "High"] = max(df.loc[i, "Open"], df.loc[i, "Close"]) + abs(np.random.normal(0, volatility*0.3))
            df.loc[i, "Low"] = min(df.loc[i, "Open"], df.loc[i, "Close"]) - abs(np.random.normal(0, volatility*0.3))
            
            # Generate realistic volume
            df.loc[i, "Volume"] = int(abs(np.random.normal(data["Volume"], data["Volume"]*0.3)))
            
            # Update EMAs
            alpha9 = 2 / (9 + 1)
            alpha21 = 2 / (21 + 1)
            alpha220 = 2 / (220 + 1)
            
            df.loc[i, "EMA9"] = df.loc[i-1, "EMA9"] * (1 - alpha9) + df.loc[i, "Close"] * alpha9
            df.loc[i, "EMA21"] = df.loc[i-1, "EMA21"] * (1 - alpha21) + df.loc[i, "Close"] * alpha21
            df.loc[i, "EMA220"] = df.loc[i-1, "EMA220"] * (1 - alpha220) + df.loc[i, "Close"] * alpha220
        
        return df