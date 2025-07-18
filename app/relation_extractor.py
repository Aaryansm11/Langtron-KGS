"""
Enhanced relation extractor using BERT models and rule-based approaches
"""
import torch
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
from itertools import combinations
import re
from concurrent.futures import ThreadPoolExecutor
import time

from config import settings
from ner_pipeline import Entity

logger = logging.getLogger(__name__)


@dataclass
class Relation:
    """Represents an extracted relation between two entities"""
    id: str
    source_entity: Entity
    target_entity: Entity
    relation_type: str
    confidence: float
    context: str
    extraction_method: str
    metadata: Dict
    sentence: str
    distance: int  # Token distance between entities


class RelationExtractor:
    """Enhanced relation extractor with multiple approaches"""
    
    def __init__(self):
        self.confidence_threshold = settings.RE_CONFIDENCE
        self.max_relations = settings.MAX_RELATIONS_PER_DOC
        self.relation_types = settings.RELATION_TYPES
        
        # Initialize models
        self._init_models()
        
        # Rule-based patterns
        self._init_patterns()
        
        # Medical relation mappings
        self._init_medical_mappings()
    
    def _init_models(self):
        """Initialize BERT models for relation extraction"""
        try:
            model_name = settings.RELATION_MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            # Initialize relation classification pipeline
            self.relation_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"Loaded relation extraction model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load relation model: {e}")
            self.model = None
            self.tokenizer = None
            self.relation_pipeline = None
    
    def _init_patterns(self):
        """Initialize rule-based patterns for relation extraction"""
        self.patterns = {
            'TREATS': [
                r'{drug}.*(?:treats?|treatment|therapy).*{disease}',
                r'{drug}.*(?:for|against).*{disease}',
                r'(?:treat|treating|treated).*{disease}.*(?:with|using).*{drug}',
                r'{drug}.*(?:indicated|prescribed).*(?:for|in).*{disease}'
            ],
            'CAUSES': [
                r'{cause}.*(?:causes?|leads? to|results? in).*{effect}',
                r'{cause}.*(?:associated with|linked to).*{effect}',
                r'(?:due to|caused by|resulting from).*{cause}.*{effect}',
                r'{effect}.*(?:secondary to|due to).*{cause}'
            ],
            'PREVENTS': [
                r'{drug}.*(?:prevents?|prevention).*{disease}',
                r'{drug}.*(?:prophylaxis|prophylactic).*{disease}',
                r'(?:prevent|preventing).*{disease}.*(?:with|using).*{drug}'
            ],
            'DIAGNOSES': [
                r'{test}.*(?:diagnoses?|diagnostic).*{disease}',
                r'{test}.*(?:shows?|reveals?|indicates?).*{disease}',
                r'(?:diagnosed|detected).*{disease}.*(?:by|using|with).*{test}'
            ],
            'SYMPTOM_OF': [
                r'{symptom}.*(?:symptom|sign).*(?:of|in).*{disease}',
                r'{disease}.*(?:presents?|characterized).*{symptom}',
                r'(?:experiencing|showing).*{symptom}.*{disease}'
            ],
            'LOCATED_IN': [
                r'{structure}.*(?:located|found|situated).*(?:in|at).*{location}',
                r'{location}.*(?:contains?|includes?).*{structure}',
                r'{structure}.*(?:of|in).*(?:the|a).*{location}'
            ],
            'INTERACTS_WITH': [
                r'{drug1}.*(?:interacts?|interaction).*(?:with|and).*{drug2}',
                r'{drug1}.*(?:combined|used).*(?:with|and).*{drug2}',
                r'(?:interaction|combination).*(?:between|of).*{drug1}.*(?:and|with).*{drug2}'
            ],
            'ADMINISTERED_TO': [
                r'{drug}.*(?:administered|given|prescribed).*(?:to|for).*{patient}',
                r'{patient}.*(?:received|taking|prescribed).*{drug}',
                r'(?:prescribe|give|administer).*{drug}.*(?:to|for).*{patient}'
            ]
        }
    
    def _init_medical_mappings(self):
        """Initialize medical concept mappings"""
        self.entity_type_mappings = {
            ('DRUG', 'DISEASE'): ['TREATS', 'CAUSES', 'PREVENTS'],
            ('DISEASE', 'SYMPTOM'): ['SYMPTOM_OF', 'CAUSES'],
            ('TEST', 'DISEASE'): ['DIAGNOSES', 'ASSOCIATED_WITH'],
            ('GENE', 'DISEASE'): ['CAUSES', 'ASSOCIATED_WITH'],
            ('ANATOMY', 'DISEASE'): ['LOCATED_IN', 'AFFECTED_BY'],
            ('DRUG', 'DRUG'): ['INTERACTS_WITH'],
            ('PROCEDURE', 'DISEASE'): ['TREATS', 'DIAGNOSES'],
            ('CHEMICAL', 'DISEASE'): ['CAUSES', 'TREATS'],
            ('PROTEIN', 'GENE'): ['PRODUCED_BY', 'REGULATES']
        }
    
    def extract_relations(self, entities: List[Entity], text: str) -> List[Relation]:
        """Extract relations between entities"""
        start_time = time.time()
        
        try:
            # Filter entities by distance and compatibility
            entity_pairs = self._get_entity_pairs(entities)
            
            # Extract relations using multiple methods
            bert_relations = self._extract_with_bert(entity_pairs, text)
            rule_relations = self._extract_with_rules(entity_pairs, text)
            
            # Combine and deduplicate relations
            all_relations = bert_relations + rule_relations
            combined_relations = self._combine_relations(all_relations)
            
            # Post-process and filter
            filtered_relations = self._filter_relations(combined_relations)
            
            processing_time = time.time() - start_time
            logger.info(f"Extracted {len(filtered_relations)} relations in {processing_time:.2f}s")
            
            return filtered_relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    def _get_entity_pairs(self, entities: List[Entity]) -> List[Tuple[Entity, Entity]]:
        """Get valid entity pairs for relation extraction"""
        pairs = []
        max_distance = 500  # Maximum character distance
        
        for e1, e2 in combinations(entities, 2):
            # Calculate distance
            distance = abs(e1.start_char - e2.start_char)
            
            # Skip if too far apart
            if distance > max_distance:
                continue
            
            # Check if entity types are compatible
            if self._are_entities_compatible(e1, e2):
                pairs.append((e1, e2))
        
        return pairs
    
    def _are_entities_compatible(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities can have a relation"""
        # Don't relate identical entities
        if e1.normalized_text == e2.normalized_text:
            return False
        
        # Check if entity type pair is in our mappings
        pair_types = (e1.label, e2.label)
        reverse_pair = (e2.label, e1.label)
        
        return pair_types in self.entity_type_mappings or reverse_pair in self.entity_type_mappings
    
    def _extract_with_bert(self, entity_pairs: List[Tuple[Entity, Entity]], text: str) -> List[Relation]:
        """Extract relations using BERT model"""
        if not self.model:
            return []
        
        relations = []
        
        for e1, e2 in entity_pairs:
            try:
                # Get sentence containing both entities
                sentence = self._get_sentence_with_entities(e1, e2, text)
                
                # Create input for BERT
                input_text = f"{e1.text} [SEP] {e2.text} [SEP] {sentence}"
                
                # Get prediction
                with torch.no_grad():
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get best prediction
                    max_prob, predicted_idx = torch.max(probabilities, dim=-1)
                    confidence = max_prob.item()
                    
                    if confidence >= self.confidence_threshold:
                        relation_type = self.relation_types[predicted_idx.item()]
                        
                        relation = Relation(
                            id=f"rel_{e1.id}_{e2.id}_{relation_type}",
                            source_entity=e1,
                            target_entity=e2,
                            relation_type=relation_type,
                            confidence=confidence,
                            context=sentence,
                            extraction_method="BERT",
                            sentence=sentence,
                            distance=abs(e1.start_char - e2.start_char),
                            metadata={
                                'model_confidence': confidence,
                                'all_probabilities': probabilities.tolist()[0]
                            }
                        )
                        relations.append(relation)
                        
            except Exception as e:
                logger.warning(f"BERT extraction failed for {e1.text}-{e2.text}: {e}")
                continue
        
        return relations
    
    def _extract_with_rules(self, entity_pairs: List[Tuple[Entity, Entity]], text: str) -> List[Relation]:
        """Extract relations using rule-based patterns"""
        relations = []
        
        for e1, e2 in entity_pairs:
            # Get possible relation types for this entity pair
            possible_relations = self._get_possible_relations(e1, e2)
            
            for relation_type in possible_relations:
                if relation_type in self.patterns:
                    # Check if pattern matches
                    sentence = self._get_sentence_with_entities(e1, e2, text)
                    confidence = self._check_pattern_match(e1, e2, relation_type, sentence)
                    
                    if confidence >= self.confidence_threshold:
                        relation = Relation(
                            id=f"rel_{e1.id}_{e2.id}_{relation_type}",
                            source_entity=e1,
                            target_entity=e2,
                            relation_type=relation_type,
                            confidence=confidence,
                            context=sentence,
                            extraction_method="RULES",
                            sentence=sentence,
                            distance=abs(e1.start_char - e2.start_char),
                            metadata={
                                'pattern_matched': True,
                                'rule_confidence': confidence
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _get_possible_relations(self, e1: Entity, e2: Entity) -> List[str]:
        """Get possible relation types between two entities"""
        pair_types = (e1.label, e2.label)
        reverse_pair = (e2.label, e1.label)
        
        relations = []
        if pair_types in self.entity_type_mappings:
            relations.extend(self.entity_type_mappings[pair_types])
        if reverse_pair in self.entity_type_mappings:
            relations.extend(self.entity_type_mappings[reverse_pair])
        
        return list(set(relations))
    
    def _check_pattern_match(self, e1: Entity, e2: Entity, relation_type: str, sentence: str) -> float:
        """Check if a pattern matches and return confidence"""
        patterns = self.patterns.get(relation_type, [])
        sentence_lower = sentence.lower()
        
        for pattern in patterns:
            # Create pattern with entity placeholders
            pattern_variants = [
                pattern.format(
                    drug=re.escape(e1.normalized_text),
                    disease=re.escape(e2.normalized_text),
                    cause=re.escape(e1.normalized_text),
                    effect=re.escape(e2.normalized_text),
                    test=re.escape(e1.normalized_text),
                    symptom=re.escape(e1.normalized_text),
                    structure=re.escape(e1.normalized_text),
                    location=re.escape(e2.normalized_text),
                    drug1=re.escape(e1.normalized_text),
                    drug2=re.escape(e2.normalized_text),
                    patient=re.escape(e2.normalized_text)
                ),
                pattern.format(
                    drug=re.escape(e2.normalized_text),
                    disease=re.escape(e1.normalized_text),
                    cause=re.escape(e2.normalized_text),
                    effect=re.escape(e1.normalized_text),
                    test=re.escape(e2.normalized_text),
                    symptom=re.escape(e2.normalized_text),
                    structure=re.escape(e2.normalized_text),
                    location=re.escape(e1.normalized_text),
                    drug1=re.escape(e2.normalized_text),
                    drug2=re.escape(e1.normalized_text),
                    patient=re.escape(e1.normalized_text)
                )
            ]
            
            for variant in pattern_variants:
                if re.search(variant, sentence_lower):
                    # Calculate confidence based on pattern specificity
                    confidence = 0.8  # Base confidence for rule matches
                    
                    # Boost confidence for exact matches
                    if variant.count('.*') < 2:  # More specific pattern
                        confidence += 0.1
                    
                    # Boost confidence for medical keywords
                    medical_keywords = ['patient', 'treatment', 'diagnosis', 'symptom']
                    keyword_count = sum(1 for kw in medical_keywords if kw in sentence_lower)
                    confidence += keyword_count * 0.02
                    
                    return min(confidence, 1.0)
        
        return 0.0
    
    def _get_sentence_with_entities(self, e1: Entity, e2: Entity, text: str) -> str:
        """Get the sentence containing both entities"""
        # Find sentence boundaries
        start_pos = min(e1.start_char, e2.start_char)
        end_pos = max(e1.end_char, e2.end_char)
        
        # Extend to sentence boundaries
        sentence_start = max(0, start_pos - 200)
        sentence_end = min(len(text), end_pos + 200)
        
        # Find actual sentence boundaries
        for i in range(sentence_start, start_pos):
            if text[i] in '.!?':
                sentence_start = i + 1
                break
        
        for i in range(end_pos, sentence_end):
            if text[i] in '.!?':
                sentence_end = i + 1
                break
        
        return text[sentence_start:sentence_end].strip()
    
    def _combine_relations(self, relations: List[Relation]) -> List[Relation]:
        """Combine relations from different methods and remove duplicates"""
        # Group by entity pair and relation type
        relation_groups = {}
        
        for relation in relations:
            key = (
                relation.source_entity.id,
                relation.target_entity.id,
                relation.relation_type
            )
            
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(relation)
        
        # Keep the relation with highest confidence from each group
        combined_relations = []
        for group in relation_groups.values():
            best_relation = max(group, key=lambda r: r.confidence)
            
            # If we have both BERT and rule-based predictions, average confidence
            if len(group) > 1:
                avg_confidence = sum(r.confidence for r in group) / len(group)
                best_relation.confidence = avg_confidence
                best_relation.extraction_method = "HYBRID"
            
            combined_relations.append(best_relation)
        
        return combined_relations
    
    def _filter_relations(self, relations: List[Relation]) -> List[Relation]:
        """Filter relations based on confidence and limits"""
        # Sort by confidence (descending)
        sorted_relations = sorted(relations, key=lambda r: r.confidence, reverse=True)
        
        # Apply confidence threshold
        filtered = [r for r in sorted_relations if r.confidence >= self.confidence_threshold]
        
        # Apply relation limit
        if len(filtered) > self.max_relations:
            filtered = filtered[:self.max_relations]
            logger.warning(f"Truncated relations to {self.max_relations} due to limit")
        
        return filtered
    
    async def extract_relations_async(self, entities: List[Entity], text: str) -> List[Relation]:
        """Asynchronous relation extraction for large entity sets"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Split entity pairs into chunks
            entity_pairs = self._get_entity_pairs(entities)
            chunk_size = max(1, len(entity_pairs) // 4)
            chunks = [entity_pairs[i:i + chunk_size] for i in range(0, len(entity_pairs), chunk_size)]
            
            # Process chunks in parallel
            tasks = []
            for chunk in chunks:
                task = loop.run_in_executor(executor, self._extract_with_bert, chunk, text)
                tasks.append(task)
            
            # Wait for all tasks to complete
            chunk_results = await asyncio.gather(*tasks)
            
            # Combine results
            all_relations = []
            for relations in chunk_results:
                all_relations.extend(relations)
            
            # Add rule-based relations
            rule_relations = self._extract_with_rules(entity_pairs, text)
            all_relations.extend(rule_relations)
            
            # Post-process combined results
            combined_relations = self._combine_relations(all_relations)
            filtered_relations = self._filter_relations(combined_relations)
            
            return filtered_relations
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict:
        """Get statistics about extracted relations"""
        stats = {
            'total_relations': len(relations),
            'relation_types': {},
            'extraction_methods': {},
            'confidence_distribution': {
                'high': 0,    # > 0.9
                'medium': 0,  # 0.7-0.9
                'low': 0      # < 0.7
            },
            'average_confidence': 0.0,
            'entity_type_pairs': {}
        }
        
        if not relations:
            return stats
        
        # Count by relation type
        for relation in relations:
            stats['relation_types'][relation.relation_type] = stats['relation_types'].get(relation.relation_type, 0) + 1
        
        # Count by extraction method
        for relation in relations:
            stats['extraction_methods'][relation.extraction_method] = stats['extraction_methods'].get(relation.extraction_method, 0) + 1
        
        # Confidence distribution
        confidences = [r.confidence for r in relations]
        stats['average_confidence'] = sum(confidences) / len(confidences)
        
        for conf in confidences:
            if conf > 0.9:
                stats['confidence_distribution']['high'] += 1
            elif conf > 0.7:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        # Entity type pairs
        for relation in relations:
            pair = (relation.source_entity.label, relation.target_entity.label)
            stats['entity_type_pairs'][f"{pair[0]}-{pair[1]}"] = stats['entity_type_pairs'].get(f"{pair[0]}-{pair[1]}", 0) + 1
        
        return stats