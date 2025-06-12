# Deployment script for Render.com
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models logs data

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:./src"

echo "Build completed successfully!"
