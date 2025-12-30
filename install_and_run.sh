#!/bin/bash
# One-liner installation and run script
# Copy and paste this into your terminal

cd /workspaces/stock_market_prediction && \
source .venv/bin/activate && \
pip install -q streamlit pandas numpy scikit-learn plotly matplotlib seaborn Bharat-sm-data ta requests beautifulsoup4 && \
echo "Installation complete! Starting the application..." && \
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
