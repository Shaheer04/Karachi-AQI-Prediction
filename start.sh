#!/bin/bash

# Start Streamlit in the foreground on the port provided by Railway ($PORT)
echo "Starting Streamlit App..."
streamlit run src/frontend/app.py --server.port $PORT --server.address 0.0.0.0
