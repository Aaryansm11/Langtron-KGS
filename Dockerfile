FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (added more for your packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies with better error handling
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "=== DEPENDENCY CONFLICTS DETECTED ===" && \
     echo "Trying alternative installation method..." && \
     pip install --no-cache-dir --force-reinstall -r requirements.txt)

# Download spaCy model (with fallback)
RUN python -m spacy download en_core_web_trf || \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories for logs and temp files
RUN mkdir -p /app/logs /app/temp

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Create startup script for development (single container)
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting FastAPI server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &\n\
API_PID=$!\n\
echo "Waiting for API to start..."\n\
sleep 10\n\
echo "Starting Streamlit UI..."\n\
streamlit run streamlit_ui/visualize.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &\n\
UI_PID=$!\n\
echo "Both services started. API PID: $API_PID, UI PID: $UI_PID"\n\
trap "kill $API_PID $UI_PID 2>/dev/null || true" EXIT\n\
wait\n\
' > start.sh && chmod +x start.sh

# Default command
CMD ["./start.sh"]