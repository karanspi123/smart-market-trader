#!/bin/bash
# Startup script for Smart Market Analyzer

# Check if running with Docker Compose
if [ "$1" == "docker" ]; then
    echo "Starting with Docker Compose..."
    cd smart-market-analyzer
    docker-compose -f deployment/docker-compose.yml up
    exit $?
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH=./smart-market-analyzer:$PYTHONPATH

# Ensure data directories exist
mkdir -p smart-market-analyzer/data/market
mkdir -p smart-market-analyzer/data/models
mkdir -p smart-market-analyzer/data/backtest
mkdir -p smart-market-analyzer/logs

# Start either API server or worker based on argument
if [ "$1" == "api" ] || [ "$1" == "" ]; then
    echo "Starting API server..."
    cd smart-market-analyzer
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
elif [ "$1" == "worker" ]; then
    echo "Starting worker service..."
    cd smart-market-analyzer
    python worker.py
elif [ "$1" == "both" ]; then
    echo "Starting both API server and worker service..."
    cd smart-market-analyzer
    # Start API server in background
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    
    # Start worker
    python worker.py &
    WORKER_PID=$!
    
    # Trap SIGINT and SIGTERM to kill both processes
    trap "kill $API_PID $WORKER_PID; exit" SIGINT SIGTERM
    
    # Wait for any process to finish
    wait
else
    echo "Invalid argument: $1"
    echo "Usage: $0 [api|worker|both|docker]"
    exit 1
fi
