# relation_extractor.py
import torch, logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import BERT_RELATION_MODEL, RELATION_TYPES, RE_CONFIDENCE

logger = logging.getLogger(__name__)

class RelationExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_RELATION_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_RELATION_MODEL)
        self.model.eval()
        self.relation_types = RELATION_TYPES

    def extract_relations(self, entities):
        relations = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                rel = self._classify(e1, e2)
                if rel:
                    relations.append({
                        "source_id": e1["id"],
                        "target_id": e2["id"],
                        "type": rel,
                    })
        return relations

    @torch.no_grad()
    def _classify(self, e1, e2):
        text = f"{e1['text']} [SEP] {e2['text']}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        best, score = probs.max(-1)
        if score.item() >= RE_CONFIDENCE:
            return self.relation_types[best.item()]
        return None