#!/bin/bash
# Run script for Stock Market Prediction System

echo "============================================"
echo "  Starting Stock Market Prediction System"
echo "============================================"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
    source .venv/bin/activate
fi

# Run Streamlit app
echo "Starting Streamlit server..."
echo ""
echo "Access the application at: http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
