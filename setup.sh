#!/bin/bash
# Setup script for Stock Market Prediction System

echo "============================================"
echo "  Stock Market Prediction System Setup"
echo "============================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo ""
echo "Installing required packages..."
echo ""

# Install packages
pip install --upgrade pip
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install plotly>=5.18.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install Bharat-sm-data
pip install ta>=0.10.0
pip install requests>=2.31.0
pip install beautifulsoup4>=4.12.0

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "To run the application, use:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py --server.port 8501"
echo ""
