# config.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD", "password")

# API / UI
API_HOST  = os.getenv("API_HOST", "0.0.0.0")
API_PORT  = int(os.getenv("API_PORT", 8000))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

# *** Clinical models ***
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_sci_lg")       # scispaCy large
BERT_RELATION_MODEL = os.getenv(
    "BERT_RELATION_MODEL",
    "dmis-lab/biobert-v1.1-finetuned-re"                       # BioBERT-RE
)

# *** Comprehensive entity & relation lists ***
ENTITY_TYPES = set(os.getenv("ENTITY_TYPES", ""
    "ANAT,DISEASE,SYMPTOM,SIGN,DRUG,CHEMICAL,GENE,PROTEIN,"
    "TEST,PROCEDURE,CLINICAL_EVENT,LAB_VALUE").split(","))

RELATION_TYPES = os.getenv("RELATION_TYPES", ""
    "TREATS,CAUSES,PREVENTS,DIAGNOSES,ASSOCIATED_WITH,"
    "ADMINISTERED_TO,INTERACTS_WITH,PART_OF,SYMPTOM_OF").split(",")

# Confidence thresholds
NER_CONFIDENCE        = float(os.getenv("NER_CONFIDENCE", 0.60))
RE_CONFIDENCE         = float(os.getenv("RE_CONFIDENCE", 0.70))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")