FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_trf

# Copy application code
COPY . .

# Create directories for logs and temp files
RUN mkdir -p /app/logs /app/temp

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create startup script for development (single container)
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &\n\
API_PID=$!\n\
echo "Waiting for API to start..."\n\
sleep 10\n\
echo "Starting Streamlit UI..."\n\
streamlit run streamlit_ui/visualize.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &\n\
UI_PID=$!\n\
echo "Both services started. API PID: $API_PID, UI PID: $UI_PID"\n\
wait\n\
' > start.sh && chmod +x start.sh

# Default command
CMD ["./start.sh"]