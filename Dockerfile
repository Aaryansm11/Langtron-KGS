FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_trf

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &\n\
streamlit run streamlit_ui/visualize.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
