# Langtron Knowledge Graph Service

A modular microservice designed to convert unstructured enterprise documents into a structured knowledge graph using NER (Named Entity Recognition) and Relation Extraction (RE), with visual exploration and REST API access.

## 🚀 Features

- Document ingestion (PDF, DOCX, TXT)
- Named Entity Recognition using spaCy
- BERT-based Relation Extraction
- Neo4j Graph Database integration
- FastAPI REST API
- Interactive graph visualization with Streamlit
- Wikidata entity linking (optional)
- Comprehensive logging and monitoring

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langtron-kg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_trf
   ```

4. Install and start Neo4j:
   - Download and install [Neo4j Desktop](https://neo4j.com/download/)
   - Create a new database with password 'password' (or update config)
   - Start the database

## 🏃‍♂️ Running the Service

1. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit UI:
   ```bash
   streamlit run streamlit_ui/visualize.py
   ```

3. Access the services:
   - API documentation: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

## 🐳 Docker Deployment

Build and run using Docker:

```bash
docker build -t langtron-kg .
docker run -p 8000:8000 -p 8501:8501 langtron-kg
```

## 📚 API Documentation

### Endpoints

- `POST /upload`: Upload and process a document
- `GET /entities/{doc_id}`: Get entities from a document
- `GET /relations/{doc_id}`: Get relations from a document
- `GET /graph/{doc_id}`: Get complete knowledge graph

## 🧪 Testing

Run tests using pytest:

```bash
pytest
```

## 📊 Monitoring

The service includes comprehensive logging using Python's logging module with CloudWatch integration for production deployments.

## 🔒 Environment Variables

- `NEO4J_URI`: Neo4j database URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `API_HOST`: FastAPI host
- `API_PORT`: FastAPI port
- `STREAMLIT_HOST`: Streamlit host
- `STREAMLIT_PORT`: Streamlit port
- `LOG_LEVEL`: Logging level

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
