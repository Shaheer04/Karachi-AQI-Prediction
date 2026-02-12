# Use Python 3.11 slim image for a smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for LightGBM, Prophet, and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose ports (Streamlit default 8501, FastAPI 8000)
# Railway will assign a PORT env var, which start.sh uses for Streamlit
EXPOSE 8501 8000

# Set entrypoint
CMD ["./start.sh"]
