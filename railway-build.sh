#!/bin/bash

# Railway build script for Stock Price Predictor
echo "🚀 Starting Railway build for Stock Price Predictor..."

# Update pip to latest version
echo "📦 Updating pip..."
python -m pip install --upgrade pip

# Install dependencies with specific constraints
echo "🔧 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs data

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:./src"

echo "✅ Railway build completed successfully!"
