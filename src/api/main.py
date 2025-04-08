from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Any
import logging
import json
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Import services
from data_service import DataService
from model_service import ModelService
from signal_service import SignalService
from backtest_service import BacktestService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Smart Market Analyzer", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class DataConfig(BaseModel):
    csv_path: str
    instrument: str = "NQ"
    split: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
    sample_rate: str = "1-minute"
    
    @validator('split')
    def validate_split(cls, split):
        if sum(split.values()) != 1.0:
            raise ValueError("Split values must sum to 1.0")
        if split.get('train', 0) < 0.5:
            raise ValueError("Training split must be at least 0.5")
        return split

class ModelConfig(BaseModel):
    price_model: str = "Transformer"
    pattern_model: str = "CNN"
    timeframe: str = "1-minute"
    window_size: int = 60
    bins: int = 10
    num_heads: int = 4
    d_model: int = 64
    entropy_threshold: float = 0.75

class BacktestConfig(BaseModel):
    strategy: Dict = {
        "stop_loss": {"type": "ATR", "multiplier": 2, "period": 14},
        "profit_take": {"type": "Fibonacci", "level": 0.618},
        "entry": {"condition": "signal == 'Buy'"},
        "exit": {"condition": "signal == 'Sell'"}
    }
    slippage: float = 0.001
    trade_cost: float = 2.0
    max_position: float = 0.02
    max_daily_loss: float = 0.05

class PredictRequest(BaseModel):
    data: List[Dict]
    window_size: int = 60

class SignalRequest(BaseModel):
    prediction: Dict
    latest_candle: Dict

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    backtest: Optional[BacktestConfig] = None

# Global state
app_state = {
    "config": None,
    "data_service": None,
    "model_service": None,
    "signal_service": None,
    "backtest_service": None,
    "processed_data": None,
    "training_state": "not_started",
    "trained_models": None
}

# Helper to get services
def get_data_service():
    if app_state["data_service"] is None and app_state["config"] is not None:
        app_state["data_service"] = DataService(app_state["config"]["data"].dict())
    return app_state["data_service"]

def get_model_service():
    if app_state["model_service"] is None and app_state["config"] is not None:
        app_state["model_service"] = ModelService(app_state["config"]["model"].dict())
    return app_state["model_service"]

def get_signal_service():
    if app_state["signal_service"] is None and app_state["config"] is not None:
        combined_config = {**app_state["config"]["model"].dict()}
        app_state["signal_service"] = SignalService(combined_config)
    return app_state["signal_service"]

def get_backtest_service():
    if app_state["backtest_service"] is None and app_state["config"] is not None:
        backtest_config = app_state["config"].get("backtest", BacktestConfig()).dict()
        app_state["backtest_service"] = BacktestService(backtest_config)
    return app_state["backtest_service"]

# Background task for training models
async def train_models_task():
    try:
        app_state["training_state"] = "in_progress"
        logger.info("Starting model training")
        
        # Get processed data
        processed_data = app_state["processed_data"]
        timeframe = app_state["config"]["model"].timeframe
        timeframe_data = processed_data[timeframe]
        
        # Get model inputs
        model_inputs = timeframe_data["model_inputs"]
        X_train, y_train = model_inputs["train"]["X"], model_inputs["train"]["y"]
        X_val, y_val = model_inputs["val"]["X"], model_inputs["val"]["y"]
        
        # Train models
        model_service = get_model_service()
        training_results = model_service.train_models(X_train, y_train, X_val, y_val)
        
        # Store trained models
        app_state["trained_models"] = training_results
        app_state["training_state"] = "completed"
        logger.info("Model training completed successfully")
        
        # Train signal service decision tree
        signal_service = get_signal_service()
        
        # Prepare data for decision tree
        train_df = timeframe_data["splits"]["train"]
        
        # Generate predictions for training data
        train_predictions = []
        batch_size = 100
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_predictions = model_service.predict(batch_X)
            train_predictions.extend([batch_predictions for _ in range(len(batch_X))])
        
        # Prepare decision tree features and labels
        X_decision_tree, y_decision_tree = signal_service.prepare_training_data(
            train_df, train_predictions[:len(train_df)-60]  # Adjust for window size
        )
        
        # Train decision tree
        signal_service.train_decision_tree(X_decision_tree, y_decision_tree)
        logger.info("Signal service decision tree trained successfully")
        
    except Exception as e:
        app_state["training_state"] = "failed"
        logger.error(f"Model training failed: {e}")
        raise

# Routes
@app.post("/config")
async def set_config(config: Config):
    """Set the configuration for all services"""
    app_state["config"] = config
    logger.info(f"Configuration set: {config.dict()}")
    return {"message": "Configuration set successfully"}

@app.post("/data")
async def process_data(background_tasks: BackgroundTasks, data_service: DataService = Depends(get_data_service)):
    """Process data from CSV file"""
    try:
        config = app_state["config"]
        if config is None:
            raise HTTPException(status_code=400, detail="Configuration not set")
        
        csv_path = config.data.csv_path
        
        # Check if the file exists
        if not os.path.exists(csv_path):
            # Generate mock data if file doesn't exist
            sample_row = "2012-12-30,17:21:00,4323.5,4323.75,4323.25,4323.5,19,4323.66503824493,4322.29011793893,4318.87143055653"
            logger.info(f"CSV file not found, generating mock data with sample: {sample_row}")
            
            df = data_service.generate_mock_data(sample_row, num_rows=1000)
            
            # Save mock data to CSV
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=True)
            logger.info(f"Generated mock data saved to {csv_path}")
        
        # Process data
        logger.info(f"Processing data from {csv_path}")
        processed_data = data_service.process_data(csv_path)
        app_state["processed_data"] = processed_data
        
        # Start training models in background
        background_tasks.add_task(train_models_task)
        
        return {
            "message": "Data processed successfully, model training started",
            "timeframes": list(processed_data.keys()),
            "samples": {tf: len(data["splits"]["train"]) for tf, data in processed_data.items()}
        }
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-status")
async def get_training_status():
    """Get the status of model training"""
    return {
        "status": app_state["training_state"],
        "models": [
            {"name": "Transformer", "type": "price prediction"},
            {"name": "CNN", "type": "pattern recognition"}
        ] if app_state["trained_models"] is not None else []
    }

@app.post("/predict")
async def predict(request: PredictRequest, model_service: ModelService = Depends(get_model_service)):
    """Generate predictions for input data"""
    try:
        if app_state["trained_models"] is None:
            raise HTTPException(status_code=400, detail="Models not trained yet")
        
        # Convert input data to numpy array
        input_data = []
        for candle in request.data:
            row = [
                candle.get("Open", 0),
                candle.get("High", 0),
                candle.get("Low", 0),
                candle.get("Close", 0),
                candle.get("Volume", 0),
                candle.get("EMA9", 0),
                candle.get("EMA21", 0),
                candle.get("EMA220", 0)
            ]
            input_data.append(row)
        
        # Create sequences of window_size
        window_size = request.window_size
        sequences = []
        for i in range(len(input_data) - window_size + 1):
            sequences.append(input_data[i:i+window_size])
        
        if not sequences:
            raise HTTPException(status_code=400, detail="Not enough data for prediction")
        
        # Convert to numpy array
        X = np.array(sequences)
        
        # Add latency simulation
        time.sleep(0.1)  # 100ms
        
        # Make predictions
        predictions = model_service.predict(X)
        
        return {
            "predictions": [
                {
                    "index": i,
                    "price": float(predictions["price_prediction"][i]),
                    "pattern": predictions["pattern_labels"][i],
                    "entropy": float(predictions["entropy"][i]),
                    "confidence": float(predictions["confidence"][i])
                }
                for i in range(len(predictions["price_prediction"]))
            ]
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signal")
async def generate_signal(request: SignalRequest, signal_service: SignalService = Depends(get_signal_service)):
    """Generate trading signal from prediction"""
    try:
        prediction = request.prediction
        latest_candle = request.latest_candle
        
        # Add latency simulation
        time.sleep(0.1)  # 100ms
        
        # Update volatility filter
        signal_service.update_volatility_filter(latest_candle)
        
        # Generate signal
        signal = signal_service.generate_signal(prediction, latest_candle)
        
        return {
            "signal": signal["signal"],
            "confidence": float(signal["confidence"]),
            "metadata": {
                "pattern": signal.get("pattern", "Unknown"),
                "price_prediction": float(signal.get("price_prediction", 0)),
                "entropy": float(signal.get("entropy", 0)),
                "reason": signal.get("reason", "")
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def run_backtest(
    backtest_service: BacktestService = Depends(get_backtest_service),
    model_service: ModelService = Depends(get_model_service),
    signal_service: SignalService = Depends(get_signal_service)
):
    """Run backtest with trained models"""
    try:
        if app_state["trained_models"] is None:
            raise HTTPException(status_code=400, detail="Models not trained yet")
        
        if app_state["processed_data"] is None:
            raise HTTPException(status_code=400, detail="No processed data available")
        
        # Get test data
        timeframe = app_state["config"]["model"].timeframe
        timeframe_data = app_state["processed_data"][timeframe]
        test_df = timeframe_data["splits"]["test"]
        model_inputs = timeframe_data["model_inputs"]
        X_test = model_inputs["test"]["X"]
        
        # Make predictions in batches
        logger.info(f"Running predictions on {len(X_test)} test samples")
        predictions = []
        batch_size = 100
        for i in range(0, len(X_test), batch_size):
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(X_test) + batch_size - 1)//batch_size}")
            batch_X = X_test[i:i+batch_size]
            batch_predictions = model_service.predict(batch_X)
            predictions.extend([
                {
                    "price_prediction": batch_predictions["price_prediction"][j],
                    "pattern": batch_predictions["pattern"][j],
                    "pattern_labels": [batch_predictions["pattern_labels"][j]],
                    "entropy": [batch_predictions["entropy"][j]],
                    "confidence": [batch_predictions["confidence"][j]]
                }
                for j in range(len(batch_X))
            ])
        
        # Generate signals
        logger.info("Generating signals")
        signals = []
        for i, pred in enumerate(predictions):
            if i >= len(test_df) - 60:  # Adjust for window size
                break
            
            candle = test_df.iloc[i].to_dict()
            signal = signal_service.generate_signal(pred, candle)
            signals.append(signal)
        
        # Run backtest
        logger.info("Running backtest simulation")
        backtest_results = backtest_service.simulate_trade(test_df.iloc[60:], signals)
        
        # Run stress test
        logger.info("Running stress test")
        stress_results = backtest_service.stress_test(test_df.iloc[60:], signals)
        
        # Generate visualizations
        visualization_data = backtest_service.visualize_results()
        
        return {
            "results": {
                "total_return": float(backtest_results["total_return"]),
                "sharpe_ratio": float(backtest_results["sharpe_ratio"]),
                "sortino_ratio": float(backtest_results["sortino_ratio"]),
                "max_drawdown": float(backtest_results["max_drawdown"]),
                "win_rate": float(backtest_results["win_rate"]),
                "profit_factor": float(backtest_results["profit_factor"]),
                "num_trades": int(backtest_results["num_trades"])
            },
            "stress_test": {
                "normal_return": float(backtest_results["total_return"]),
                "stress_return": float(stress_results["stress_results"]["total_return"]),
                "return_change": float(stress_results["performance_change"]["total_return"]),
                "drawdown_change": float(stress_results["performance_change"]["max_drawdown"])
            },
            "visualization": visualization_data
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    """Root endpoint with API info"""
    return {
        "name": "Smart Market Analyzer API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/config", "method": "POST", "description": "Set configuration"},
            {"path": "/data", "method": "POST", "description": "Process data from CSV"},
            {"path": "/training-status", "method": "GET", "description": "Get model training status"},
            {"path": "/predict", "method": "POST", "description": "Generate predictions"},
            {"path": "/signal", "method": "POST", "description": "Generate trading signals"},
            {"path": "/backtest", "method": "POST", "description": "Run backtest simulation"}
        ]
    }

@app.post("/strategy/select")
async def select_strategy(
    strategy_config: Dict,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Select and configure trading strategy"""
    try:
        # Update strategy configuration
        app_state["config"]["strategy"] = strategy_config
        
        # Initialize backtest service with updated config
        backtest_service = BacktestService(app_state["config"]["strategy"])
        
        return {
            "message": "Strategy updated successfully",
            "config": strategy_config
        }
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy/compare")
async def compare_strategies(
    strategies: List[Dict],
    backtest_service: BacktestService = Depends(get_backtest_service),
    data_service: DataService = Depends(get_data_service)
):
    """Compare multiple trading strategies"""
    try:
        # Get test data
        timeframe = app_state["config"]["model"].timeframe
        timeframe_data = app_state["processed_data"][timeframe]
        test_df = timeframe_data["splits"]["test"]
        
        results = {}
        for i, strategy in enumerate(strategies):
            # Update backtest config with strategy
            backtest_service.strategy_config = strategy
            
            # Run backtest
            backtest_results = backtest_service.simulate_trade(test_df.iloc[60:], strategies_signals[i])
            
            # Store results
            results[f"strategy_{i+1}"] = {
                "name": strategy.get("name", f"Strategy {i+1}"),
                "config": strategy,
                "performance": {
                    "total_return": float(backtest_results["total_return"]),
                    "sharpe_ratio": float(backtest_results["sharpe_ratio"]),
                    "sortino_ratio": float(backtest_results["sortino_ratio"]),
                    "max_drawdown": float(backtest_results["max_drawdown"]),
                    "win_rate": float(backtest_results["win_rate"])
                }
            }
        
        return results
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/strategies")
async def get_strategies(strategy_manager: StrategyManager = Depends(get_strategy_manager)):
    """Get all available strategies"""
    return strategy_manager.get_all_strategies()

@app.post("/strategies")
async def add_strategy(strategy_config: Dict, strategy_manager: StrategyManager = Depends(get_strategy_manager)):
    """Add a new strategy"""
    success = strategy_manager.add_strategy(strategy_config)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add strategy")
    return {"message": "Strategy added successfully", "name": strategy_config.get("name")}

@app.put("/strategies/{name}")
async def update_strategy(name: str, strategy_config: Dict, strategy_manager: StrategyManager = Depends(get_strategy_manager)):
    """Update an existing strategy"""
    success = strategy_manager.update_strategy(name, strategy_config)
    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {name}")
    return {"message": "Strategy updated successfully", "name": name}

@app.delete("/strategies/{name}")
async def delete_strategy(name: str, strategy_manager: StrategyManager = Depends(get_strategy_manager)):
    """Delete a strategy"""
    success = strategy_manager.delete_strategy(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {name}")
    return {"message": "Strategy deleted successfully", "name": name}

@app.post("/strategies/{name}/activate")
async def activate_strategy(name: str, strategy_manager: StrategyManager = Depends(get_strategy_manager)):
    """Set active strategy"""
    success = strategy_manager.set_active_strategy(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {name}")
    return {"message": "Active strategy set successfully", "name": name}

@app.get("/strategies/compare")
async def compare_strategies(
    strategy_manager: StrategyManager = Depends(get_strategy_manager),
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """Compare all strategies"""
    # Get test data
    timeframe = app_state["config"]["model"].timeframe
    timeframe_data = app_state["processed_data"][timeframe]
    test_df = timeframe_data["splits"]["test"]
    
    # Compare strategies
    results = strategy_manager.compare_strategies(backtest_service, test_df)
    
    return results    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)