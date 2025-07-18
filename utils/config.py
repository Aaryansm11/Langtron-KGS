"""
Enhanced configuration management with validation and environment support
"""
import os
import logging
from pathlib import Path
from typing import List, Set
from pydantic import BaseSettings, validator
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with validation and type safety"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Streamlit Configuration
    STREAMLIT_HOST: str = "0.0.0.0"
    STREAMLIT_PORT: int = 8501
    
    # Model Configuration
    SPACY_MODEL: str = "en_core_web_trf"
    BIOBERT_MODEL: str = "dmis-lab/biobert-v1.1"
    RELATION_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Processing Configuration
    MAX_ENTITIES_PER_DOC: int = 10000
    MAX_RELATIONS_PER_DOC: int = 50000
    BATCH_SIZE: int = 32
    
    # Confidence Thresholds
    NER_CONFIDENCE: float = 0.7
    RE_CONFIDENCE: float = 0.8
    SIMILARITY_THRESHOLD: float = 0.85
    
    # Entity Types (Medical/Clinical focus)
    ENTITY_TYPES: Set[str] = {
        "DISEASE", "SYMPTOM", "DRUG", "CHEMICAL", "GENE", "PROTEIN",
        "ANATOMY", "PROCEDURE", "TEST", "DEVICE", "ORGANISM", "CELL_TYPE",
        "TISSUE", "CLINICAL_EVENT", "LAB_VALUE", "DOSAGE", "FREQUENCY"
    }
    
    # Relation Types
    RELATION_TYPES: List[str] = [
        "TREATS", "CAUSES", "PREVENTS", "DIAGNOSES", "ASSOCIATED_WITH",
        "ADMINISTERED_TO", "INTERACTS_WITH", "PART_OF", "SYMPTOM_OF",
        "LOCATED_IN", "AFFECTS", "PRODUCES", "REGULATES", "INHIBITS",
        "ACTIVATES", "BINDS_TO", "METABOLIZES", "CONTRAINDICATES"
    ]
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".docx", ".txt", ".rtf", ".doc"}
    
    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Redis Configuration (for caching and job queue)
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    @validator("TEMP_DIR", "UPLOAD_DIR", pre=True)
    def create_directories(cls, v):
        """Ensure directories exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("NER_CONFIDENCE", "RE_CONFIDENCE", "SIMILARITY_THRESHOLD")
    def validate_confidence(cls, v):
        """Ensure confidence values are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence values must be between 0 and 1")
        return v
    
    @validator("MAX_UPLOAD_SIZE")
    def validate_upload_size(cls, v):
        """Ensure reasonable upload size limits"""
        if v > 1024 * 1024 * 1024:  # 1GB
            raise ValueError("Upload size limit too large")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.value),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.BASE_DIR / "logs" / "app.log")
        ]
    )
    
    # Create logs directory
    (settings.BASE_DIR / "logs").mkdir(exist_ok=True)
    
    return logging.getLogger(__name__)


# Global settings instance
settings = Settings()
logger = setup_logging(settings)