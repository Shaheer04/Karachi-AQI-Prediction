#!/bin/bash

# Start FastAPI in the background on port 8000
echo "Starting FastAPI Backend..."
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground on the port provided by Railway ($PORT)
echo "Starting Streamlit Frontend..."
streamlit run src/frontend/app.py --server.port $PORT --server.address 0.0.0.0
