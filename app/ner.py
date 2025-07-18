# ner.py
import spacy, logging
from utils.parser import DocumentParser
from config import SPACY_MODEL, ENTITY_TYPES, NER_CONFIDENCE

logger = logging.getLogger(__name__)

class NERPipeline:
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            spacy.cli.download(SPACY_MODEL)
            self.nlp = spacy.load(SPACY_MODEL)
        self.parser = DocumentParser()

    def extract_entities(self, file_path):
        text = self.parser.parse_document(file_path)
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            if ent.label_.upper() in ENTITY_TYPES and ent._.confidence >= NER_CONFIDENCE:
                entities.append({
                    "id": len(entities),
                    "text": ent.text,
                    "type": ent.label_.upper(),
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities