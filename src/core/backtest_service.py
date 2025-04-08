import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestService:
    def __init__(self, config: Dict):
        """
        Initialize backtest service with configuration
        
        Args:
            config: Dictionary containing backtest configuration
        """
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.results = {}
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR calculation
        atr = tr.rolling(period).mean()
        
        return atr
    
    def calculate_stop_loss(self, entry_price: float, entry_type: str, candle: pd.Series) -> float:
        """
        Calculate stop loss price based on strategy configuration
        
        Args:
            entry_price: Entry price
            entry_type: 'long' or 'short'
            candle: Current candle data
            
        Returns:
            Stop loss price
        """
        stop_config = self.strategy_config.get('stop_loss', {})
        stop_type = stop_config.get('type', 'ATR')
        
        if stop_type == 'ATR':
            # ATR-based stop loss
            multiplier = stop_config.get('multiplier', 2)
            atr_value = candle.get('ATR', 0)
            
            if entry_type == 'long':
                return entry_price - (atr_value * multiplier)
            else:  # short
                return entry_price + (atr_value * multiplier)
                
        elif stop_type == 'Fixed':
            # Fixed percentage stop loss
            percentage = stop_config.get('percentage', 1.0) / 100
            
            if entry_type == 'long':
                return entry_price * (1 - percentage)
            else:  # short
                return entry_price * (1 + percentage)
                
        elif stop_type == 'Trailing':
            # Trailing stop (will be updated during simulation)
            percentage = stop_config.get('percentage', 1.0) / 100
            
            if entry_type == 'long':
                return entry_price * (1 - percentage)
            else:  # short
                return entry_price * (1 + percentage)
        
        # Default fallback
        return entry_price * 0.95 if entry_type == 'long' else entry_price * 1.05
    
    def calculate_take_profit(self, entry_price: float, entry_type: str, candle: pd.Series) -> float:
        """
        Calculate take profit price based on strategy configuration
        
        Args:
            entry_price: Entry price
            entry_type: 'long' or 'short'
            candle: Current candle data
            
        Returns:
            Take profit price
        """
        profit_config = self.strategy_config.get('profit_take', {})
        profit_type = profit_config.get('type', 'Fixed')
        
        if profit_type == 'Fibonacci':
            # Fibonacci-based profit target
            level = profit_config.get('level', 0.618)
            atr_value = candle.get('ATR', entry_price * 0.01)  # Default to 1% if ATR not available
            
            if entry_type == 'long':
                return entry_price + (atr_value * level * 3)  # Scaling factor for reasonable targets
            else:  # short
                return entry_price - (atr_value * level * 3)
                
        elif profit_type == 'Fixed':
            # Fixed percentage profit target
            percentage = profit_config.get('percentage', 2.0) / 100
            
            if entry_type == 'long':
                return entry_price * (1 + percentage)
            else:  # short
                return entry_price * (1 - percentage)
                
        elif profit_type == 'ATR':
            # ATR-based profit target
            multiplier = profit_config.get('multiplier', 3)
            atr_value = candle.get('ATR', 0)
            
            if entry_type == 'long':
                return entry_price + (atr_value * multiplier)
            else:  # short
                return entry_price - (atr_value * multiplier)
        
        # Default fallback
        return entry_price * 1.02 if entry_type == 'long' else entry_price * 0.98
    
    def simulate_trade(self, df: pd.DataFrame, signals: List[Dict], 
                       initial_capital: float = 10000.0) -> Dict:
        """
        Simulate trades based on signals
        
        Args:
            df: DataFrame with market data
            signals: List of signal dictionaries
            initial_capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        # Ensure we have ATR for stop loss calculation
        if 'ATR' not in df.columns:
            atr_period = self.strategy_config.get('stop_loss', {}).get('period', 14)
            df['ATR'] = self.calculate_atr(df, period=atr_period)
        
        # Configuration parameters
        slippage = self.config.get('slippage', 0.001)  # 0.1% slippage
        trade_cost = self.config.get('trade_cost', 2.0)  # $2 per trade
        max_position_size = self.config.get('max_position', 0.02)  # 2% of capital
        max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5% daily loss limit
        
        # Simulation variables
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        daily_pnl = {}
        current_day = None
        daily_loss = 0
        
        # Statistics
        num_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        max_drawdown = 0
        peak_capital = initial_capital
        equity_curve = []
        
        # Add latency simulation - delay signal execution by 1 candle
        delayed_signals = [None] + signals[:-1]
        
        # Run simulation
        for i, candle in df.iterrows():
            if i >= len(delayed_signals):
                break
                
            # Get signal (with simulated latency)
            signal = delayed_signals[i]
            
            # Track daily P&L
            candle_day = pd.to_datetime(candle.name).date() if hasattr(candle.name, 'date') else None
            if current_day is None:
                current_day = candle_day
            elif candle_day != current_day:
                # Reset daily loss on new day
                daily_pnl[current_day] = daily_loss
                daily_loss = 0
                current_day = candle_day
            
            # Check for stop loss or take profit if in position
            if position != 0:
                # Simulate slippage for execution
                execution_slippage = np.random.normal(0, slippage * candle['Close'])
                
                if position == 1:  # Long position
                    # Check for stop loss
                    if candle['Low'] <= stop_loss:
                        # Stop loss hit (with slippage)
                        exit_price = max(stop_loss * (1 - slippage), candle['Low'])
                        position_profit = (exit_price - entry_price) * position_size - trade_cost
                        
                        # Update statistics
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': candle.name,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit': position_profit,
                            'type': 'long',
                            'exit_reason': 'stop_loss'
                        })
                        
                        # Update capital
                        capital += position_profit
                        daily_loss += position_profit if position_profit < 0 else 0
                        
                        # Reset position
                        position = 0
                        
                        # Update statistics
                        num_trades += 1
                        if position_profit > 0:
                            winning_trades += 1
                            total_profit += position_profit
                        else:
                            losing_trades += 1
                            total_loss += abs(position_profit)
                    
                    # Check for take profit
                    elif candle['High'] >= take_profit:
                        # Take profit hit (with slippage)
                        exit_price = min(take_profit * (1 + slippage), candle['High'])
                        position_profit = (exit_price - entry_price) * position_size - trade_cost
                        
                        # Update statistics
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': candle.name,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit': position_profit,
                            'type': 'long',
                            'exit_reason': 'take_profit'
                        })
                        
                        # Update capital
                        capital += position_profit
                        
                        # Reset position
                        position = 0
                        
                        # Update statistics
                        num_trades += 1
                        if position_profit > 0:
                            winning_trades += 1
                            total_profit += position_profit
                        else:
                            losing_trades += 1
                            total_loss += abs(position_profit)
                
                elif position == -1:  # Short position
                    # Check for stop loss
                    if candle['High'] >= stop_loss:
                        # Stop loss hit (with slippage)
                        exit_price = min(stop_loss * (1 + slippage), candle['High'])
                        position_profit = (entry_price - exit_price) * position_size - trade_cost
                        
                        # Update statistics
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': candle.name,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit': position_profit,
                            'type': 'short',
                            'exit_reason': 'stop_loss'
                        })
                        
                        # Update capital
                        capital += position_profit
                        daily_loss += position_profit if position_profit < 0 else 0
                        
                        # Reset position
                        position = 0
                        
                        # Update statistics
                        num_trades += 1
                        if position_profit > 0:
                            winning_trades += 1
                            total_profit += position_profit
                        else:
                            losing_trades += 1
                            total_loss += abs(position_profit)
                    
                    # Check for take profit
                    elif candle['Low'] <= take_profit:
                        # Take profit hit (with slippage)
                        exit_price = max(take_profit * (1 - slippage), candle['Low'])
                        position_profit = (entry_price - exit_price) * position_size - trade_cost
                        
                        # Update statistics
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': candle.name,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit': position_profit,
                            'type': 'short',
                            'exit_reason': 'take_profit'
                        })
                        
                        # Update capital
                        capital += position_profit
                        
                        # Reset position
                        position = 0
                        
                        # Update statistics
                        num_trades += 1
                        if position_profit > 0:
                            winning_trades += 1
                            total_profit += position_profit
                        else:
                            losing_trades += 1
                            total_loss += abs(position_profit)
            
            # Check for explicit exit signal if in position
            if position != 0 and signal is not None:
                if (position == 1 and signal['signal'] == 'Sell') or (position == -1 and signal['signal'] == 'Buy'):
                    # Exit position (with slippage)
                    exit_price = candle['Close'] * (1 - slippage if position == 1 else 1 + slippage)
                    position_profit = (exit_price - entry_price) * position_size - trade_cost if position == 1 else (entry_price - exit_price) * position_size - trade_cost
                    
                    # Update statistics
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': candle.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position_size,
                        'profit': position_profit,
                        'type': 'long' if position == 1 else 'short',
                        'exit_reason': 'signal'
                    })
                    
                    # Update capital
                    capital += position_profit
                    daily_loss += position_profit if position_profit < 0 else 0
                    
                    # Reset position
                    position = 0
                    
                    # Update statistics
                    num_trades += 1
                    if position_profit > 0:
                        winning_trades += 1
                        total_profit += position_profit
                    else:
                        losing_trades += 1
                        total_loss += abs(position_profit)
            
            # Check for entry signal if not in position
            if position == 0 and signal is not None:
                # Check daily loss limit
                if abs(daily_loss) >= initial_capital * max_daily_loss:
                    # Skip trading for the day due to loss limit
                    continue
                
                if signal['signal'] == 'Buy':
                    # Enter long position (with slippage)
                    entry_price = candle['Close'] * (1 + slippage)
                    
                    # Calculate position size (2% of capital)
                    position_size = capital * max_position_size / entry_price
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(entry_price, 'long', candle)
                    take_profit = self.calculate_take_profit(entry_price, 'long', candle)
                    
                    # Set position
                    position = 1
                    entry_date = candle.name
                    
                elif signal['signal'] == 'Sell':
                    # Enter short position (with slippage)
                    entry_price = candle['Close'] * (1 - slippage)
                    
                    # Calculate position size (2% of capital)
                    position_size = capital * max_position_size / entry_price
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(entry_price, 'short', candle)
                    take_profit = self.calculate_take_profit(entry_price, 'short', candle)
                    
                    # Set position
                    position = -1
                    entry_date = candle.name
            
            # Update equity curve
            equity_curve.append(capital)
            
            # Update max drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        # Handle open position at end of simulation
        if position != 0:
            # Close position at last price
            last_candle = df.iloc[-1]
            exit_price = last_candle['Close'] * (1 - slippage if position == 1 else 1 + slippage)
            position_profit = (exit_price - entry_price) * position_size - trade_cost if position == 1 else (entry_price - exit_price) * position_size - trade_cost
            
            # Update statistics
            trades.append({
                'entry_date': entry_date,
                'exit_date': last_candle.name,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'profit': position_profit,
                'type': 'long' if position == 1 else 'short',
                'exit_reason': 'end_of_simulation'
            })
            
            # Update capital
            capital += position_profit
            
            # Update statistics
            num_trades += 1
            if position_profit > 0:
                winning_trades += 1
                total_profit += position_profit
            else:
                losing_trades += 1
                total_loss += abs(position_profit)
        
        # Calculate performance metrics
        if num_trades > 0:
            win_rate = winning_trades / num_trades
            avg_win = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate Sharpe Ratio (annualized)
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Calculate Sortino Ratio (annualized)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        
        # Compile results
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        self.results = results
        return results
    
    def walk_forward_optimization(self, df: pd.DataFrame, signal_generator, parameter_ranges: Dict) -> Dict:
        """
        Perform walk-forward optimization
        
        Args:
            df: DataFrame with market data
            signal_generator: Function to generate signals
            parameter_ranges: Dictionary with parameter ranges to test
            
        Returns:
            Dictionary with optimization results
        """
        # Define time windows (6-month windows)
        total_days = (df.index[-1] - df.index[0]).days
        window_size = 180  # 6 months in days
        step_size = 30    # 1 month step
        
        window_results = []
        best_params = None
        best_performance = -float('inf')
        
        # Split the data into windows
        for start_idx in range(0, len(df), step_size):
            end_idx = start_idx + window_size
            if end_idx >= len(df):
                break
                
            train_df = df.iloc[start_idx:end_idx]
            test_df = df.iloc[end_idx:min(end_idx + step_size, len(df))]
            
            if len(test_df) < 5:  # Skip if test window is too small
                continue
            
            logger.info(f"Optimizing window {start_idx//step_size + 1}: {train_df.index[0]} to {train_df.index[-1]}")
            
            # Grid search parameters
            best_window_params = None
            best_window_performance = -float('inf')
            
            # TODO: Implement grid search over parameter_ranges
            # For this example, we'll just use default parameters
            best_window_params = {key: values[0] for key, values in parameter_ranges.items()}
            
            # Generate signals with best parameters
            signals = signal_generator(test_df, best_window_params)
            
            # Backtest on test data
            self.config.update(best_window_params)
            results = self.simulate_trade(test_df, signals)
            
            # Store window results
            window_results.append({
                'start_date': train_df.index[0],
                'end_date': test_df.index[-1],
                'parameters': best_window_params,
                'performance': results['total_return']
            })
            
            # Update best overall parameters
            if results['total_return'] > best_performance:
                best_performance = results['total_return']
                best_params = best_window_params
        
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'window_results': window_results
        }
    
    def stress_test(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """
        Perform stress testing with increased volatility
        
        Args:
            df: DataFrame with market data
            signals: List of signal dictionaries
            
        Returns:
            Dictionary with stress test results
        """
        # Create a copy of data with doubled volatility
        stress_df = df.copy()
        
        # Calculate daily returns
        stress_df['returns'] = stress_df['Close'].pct_change()
        
        # Double the volatility by scaling returns
        stress_df['stress_returns'] = stress_df['returns'] * 2
        
        # Recalculate prices with increased volatility
        base_price = stress_df['Close'].iloc[0]
        stress_prices = [base_price]
        
        for i in range(1, len(stress_df)):
            stress_price = stress_prices[-1] * (1 + stress_df['stress_returns'].iloc[i])
            stress_prices.append(stress_price)
        
        stress_df['Close'] = stress_prices
        
        # Adjust High, Low, and Open proportionally
        for i in range(1, len(stress_df)):
            orig_range = df['High'].iloc[i] - df['Low'].iloc[i]
            orig_close = df['Close'].iloc[i]
            
            # Calculate new range proportionally
            new_range = orig_range * 2
            
            # Recalculate High and Low around new Close
            stress_df.loc[stress_df.index[i], 'High'] = stress_df['Close'].iloc[i] + new_range/2
            stress_df.loc[stress_df.index[i], 'Low'] = stress_df['Close'].iloc[i] - new_range/2
            
            # Adjust Open proportionally to Close
            orig_open_pct = (df['Open'].iloc[i] - orig_close) / orig_close
            stress_df.loc[stress_df.index[i], 'Open'] = stress_df['Close'].iloc[i] * (1 + orig_open_pct)
        
        # Remove temporary columns
        stress_df = stress_df.drop(['returns', 'stress_returns'], axis=1)
        
        # Recalculate ATR
        stress_df['ATR'] = self.calculate_atr(stress_df)
        
        # Run backtest with stressed data
        stress_results = self.simulate_trade(stress_df, signals)
        
        return {
            'normal_results': self.results,
            'stress_results': stress_results,
            'performance_change': {
                'total_return': stress_results['total_return'] - self.results['total_return'],
                'max_drawdown': stress_results['max_drawdown'] - self.results['max_drawdown'],
                'sharpe_ratio': stress_results['sharpe_ratio'] - self.results['sharpe_ratio']
            }
        }
    
    def visualize_results(self) -> Dict:
        """
        Generate visualizations for backtest results
        
        Returns:
            Dictionary with plot data
        """
        if not self.results:
            logger.error("No backtest results to visualize")
            return {}
        
        # Prepare visualization data
        equity_curve = self.results['equity_curve']
        trades = self.results['trades']
        
        # Calculate drawdown curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        
        # Prepare trade markers
        trade_entries = []
        trade_exits = []
        trade_profits = []
        
        for trade in trades:
            trade_entries.append(trade['entry_date'])
            trade_exits.append(trade['exit_date'])
            trade_profits.append(trade['profit'])
        
        return {
            'equity_curve': {
                'x': list(range(len(equity_curve))),
                'y': equity_curve
            },
            'drawdown': {
                'x': list(range(len(drawdown))),
                'y': drawdown
            },
            'trades': {
                'entries': trade_entries,
                'exits': trade_exits,
                'profits': trade_profits
            },
            'metrics': {
                'total_return': self.results['total_return'],
                'sharpe_ratio': self.results['sharpe_ratio'],
                'sortino_ratio': self.results['sortino_ratio'],
                'max_drawdown': self.results['max_drawdown'],
                'win_rate': self.results['win_rate']
            }
        }
    def add_strategy_indicators(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        if strategy_type == "mean_reversion":
            # Bollinger Bands
            lookback = self.strategy_config.get("lookback", 20)
            std_mult = self.strategy_config.get("std_mult", 2.0)
            
            df['MA'] = df['Close'].rolling(window=lookback).mean()
            df['STD'] = df['Close'].rolling(window=lookback).std()
            df['Upper'] = df['MA'] + (std_mult * df['STD'])
            df['Lower'] = df['MA'] - (std_mult * df['STD'])
            df['BB_Position'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=lookback).mean()
            avg_loss = loss.rolling(window=lookback).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            df['RSI'] = 100 - (100 / (1 + rs))
            
        elif strategy_type == "trend_following":
            # Moving Average Crossover
            fast_period = self.strategy_config.get("fast_period", 9)
            slow_period = self.strategy_config.get("slow_period", 21)
            
            df['Fast_MA'] = df['Close'].rolling(window=fast_period).mean()
            df['Slow_MA'] = df['Close'].rolling(window=slow_period).mean()
            df['MA_Cross'] = ((df['Fast_MA'] > df['Slow_MA']) & 
                            (df['Fast_MA'].shift(1) <= df['Slow_MA'].shift(1))).astype(int)
            
            # ADX for trend strength
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
        return df

    def generate_strategy_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate signals based on strategy type
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            List of signal dictionaries
        """
        strategy_type = self.strategy_config.get("type", "custom")
        signals = []
        
        if strategy_type == "mean_reversion":
            # Mean reversion signals
            for i, row in df.iterrows():
                if pd.notna(row['BB_Position']) and pd.notna(row['RSI']):
                    if row['BB_Position'] < 0.05 and row['RSI'] < 30:
                        signals.append({
                            'signal': 'Buy',
                            'confidence': 0.8,
                            'reason': 'mean_reversion_oversold'
                        })
                    elif row['BB_Position'] > 0.95 and row['RSI'] > 70:
                        signals.append({
                            'signal': 'Sell',
                            'confidence': 0.8,
                            'reason': 'mean_reversion_overbought'
                        })
                    else:
                        signals.append({
                            'signal': 'Hold',
                            'confidence': 0.5,
                            'reason': 'no_signal'
                        })
                else:
                    signals.append({
                        'signal': 'Hold',
                        'confidence': 0.5,
                        'reason': 'missing_data'
                    })
        
        elif strategy_type == "trend_following":
            # Trend following signals
            for i, row in df.iterrows():
                if pd.notna(row['Fast_MA']) and pd.notna(row['Slow_MA']):
                    if row['MA_Cross'] == 1:
                        signals.append({
                            'signal': 'Buy',
                            'confidence': 0.8,
                            'reason': 'ma_crossover_bullish'
                        })
                    elif row['Fast_MA'] < row['Slow_MA'] and row['Fast_MA'].shift(1) > row['Slow_MA'].shift(1):
                        signals.append({
                            'signal': 'Sell',
                            'confidence': 0.8,
                            'reason': 'ma_crossover_bearish'
                        })
                    else:
                        signals.append({
                            'signal': 'Hold',
                            'confidence': 0.5,
                            'reason': 'no_signal'
                        })
                else:
                    signals.append({
                        'signal': 'Hold',
                        'confidence': 0.5,
                        'reason': 'missing_data'
                    })
        
        return signals

    def optimize_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Optimize strategy parameters
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with optimal parameters
        """
        strategy_type = self.strategy_config.get("type", "custom")
        
        if strategy_type == "mean_reversion":
            # Optimize mean reversion parameters
            lookback_range = range(10, 31, 5)  # 10, 15, 20, 25, 30
            std_range = [1.5, 2.0, 2.5, 3.0]
            
            best_sharpe = 0
            best_params = {}
            
            for lookback in lookback_range:
                for std_mult in std_range:
                    # Update config
                    test_config = self.strategy_config.copy()
                    test_config.update({
                        "lookback": lookback,
                        "std_mult": std_mult
                    })
                    
                    # Create temporary backtest service with this config
                    temp_service = BacktestService(test_config)
                    
                    # Add indicators
                    df_with_indicators = temp_service.add_strategy_indicators(df.copy(), strategy_type)
                    
                    # Generate signals
                    signals = temp_service.generate_strategy_signals(df_with_indicators)
                    
                    # Run backtest
                    results = temp_service.simulate_trade(df_with_indicators, signals)
                    
                    # Check if better
                    if results['sharpe_ratio'] > best_sharpe:
                        best_sharpe = results['sharpe_ratio']
                        best_params = {
                            "lookback": lookback,
                            "std_mult": std_mult,
                            "performance": {
                                "sharpe_ratio": results['sharpe_ratio'],
                                "total_return": results['total_return'],
                                "max_drawdown": results['max_drawdown']
                            }
                        }
            
            return best_params