#!/bin/bash

# Railway startup script for Stock Price Predictor
echo "Starting Stock Price Predictor on Railway..."

# Create necessary directories
mkdir -p models logs data

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src:${PWD}"

# Set Streamlit configuration
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start the Streamlit app
echo "Starting Streamlit on port $PORT..."
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
