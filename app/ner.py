"""
Enhanced NER pipeline with medical models, confidence scoring, and entity linking
"""
import spacy
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict
import time

from config import settings
from document_parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Enhanced entity representation"""
    id: str
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    context: str
    normalized_text: str
    metadata: Dict
    section: Optional[str] = None
    wikidata_id: Optional[str] = None
    umls_cui: Optional[str] = None


class ModelManager:
    """Manages multiple NLP models for different domains"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'general': {
                'model_name': 'en_core_web_trf',
                'priority': 1
            },
            'biomedical': {
                'model_name': 'en_core_sci_lg',
                'priority': 2
            },
            'clinical': {
                'model_name': 'en_core_sci_md',
                'priority': 3
            }
        }
        
    def load_model(self, model_type: str) -> spacy.Language:
        """Load and cache spaCy model"""
        if model_type not in self.models:
            config = self.model_configs.get(model_type, self.model_configs['general'])
            model_name = config['model_name']
            
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded {model_name} model")
            except OSError:
                logger.warning(f"Model {model_name} not found, downloading...")
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
            
            # Add custom components
            self._add_custom_components(nlp)
            self.models[model_type] = nlp
        
        return self.models[model_type]
    
    def _add_custom_components(self, nlp: spacy.Language):
        """Add custom pipeline components"""
        # Add confidence scoring if not present
        if not nlp.has_pipe("confidence"):
            @spacy.Language.component("confidence")
            def confidence_component(doc):
                for ent in doc.ents:
                    # Calculate confidence based on various factors
                    confidence = self._calculate_confidence(ent, doc)
                    ent._.confidence = confidence
                return doc
            
            # Add confidence extension
            if not spacy.tokens.Span.has_extension("confidence"):
                spacy.tokens.Span.set_extension("confidence", default=0.0)
            
            nlp.add_pipe("confidence", last=True)
    
    def _calculate_confidence(self, ent: spacy.tokens.Span, doc: spacy.tokens.Doc) -> float:
        """Calculate entity confidence score"""
        # Base confidence from model
        base_confidence = 0.7  # Default
        
        # Factors that increase confidence
        factors = []
        
        # Length factor (longer entities often more reliable)
        length_factor = min(len(ent.text) / 20, 1.0)
        factors.append(length_factor * 0.2)
        
        # Case factor (proper case increases confidence)
        if ent.text.istitle():
            factors.append(0.1)
        
        # Context factor (surrounded by relevant words)
        context_words = [token.text.lower() for token in doc[max(0, ent.start-3):ent.end+3]]
        medical_context = ['patient', 'diagnosis', 'treatment', 'symptom', 'drug', 'disease']
        context_score = sum(1 for word in context_words if word in medical_context) / len(medical_context)
        factors.append(context_score * 0.2)
        
        # Position factor (entities in certain sections more reliable)
        factors.append(0.1)  # Placeholder
        
        return min(base_confidence + sum(factors), 1.0)


class NERPipeline:
    """Enhanced NER pipeline with multiple models and post-processing"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.entity_types = settings.ENTITY_TYPES
        self.confidence_threshold = settings.NER_CONFIDENCE
        self.max_entities = settings.MAX_ENTITIES_PER_DOC
        
        # Medical entity mappings
        self.medical_mappings = {
            'PERSON': 'PATIENT',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'PRODUCT': 'DRUG',
            'SUBSTANCE': 'CHEMICAL',
            'DISEASE': 'DISEASE',
            'CHEMICAL': 'CHEMICAL',
            'GENE_OR_GENOME': 'GENE',
            'CELL_TYPE': 'CELL_TYPE',
            'TISSUE': 'TISSUE',
            'ORGAN': 'ANATOMY'
        }
        
        # Load primary models
        self.primary_model = self.model_manager.load_model('biomedical')
        self.fallback_model = self.model_manager.load_model('general')
    
    def extract_entities(self, document: ParsedDocument) -> List[Entity]:
        """Extract entities from parsed document"""
        start_time = time.time()
        
        try:
            # Extract entities from full text
            entities = self._extract_from_text(document.text)
            
            # Extract entities from sections
            section_entities = self._extract_from_sections(document.sections)
            
            # Combine and deduplicate
            all_entities = self._combine_entities(entities, section_entities)
            
            # Post-process entities
            processed_entities = self._post_process_entities(all_entities, document)
            
            # Filter by confidence and limits
            filtered_entities = self._filter_entities(processed_entities)
            
            processing_time = time.time() - start_time
            logger.info(f"Extracted {len(filtered_entities)} entities in {processing_time:.2f}s")
            
            return filtered_entities
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            raise
    
    def _extract_from_text(self, text: str) -> List[Entity]:
        """Extract entities from full text"""
        entities = []
        
        # Process with primary model
        doc = self.primary_model(text)
        
        for ent in doc.ents:
            if self._is_valid_entity(ent):
                entity = self._create_entity(ent, doc)
                entities.append(entity)
        
        return entities
    
    def _extract_from_sections(self, sections: List[Tuple[str, str]]) -> List[Entity]:
        """Extract entities from individual sections"""
        entities = []
        
        for section_name, section_text in sections:
            doc = self.primary_model(section_text)
            
            for ent in doc.ents:
                if self._is_valid_entity(ent):
                    entity = self._create_entity(ent, doc, section_name)
                    entities.append(entity)
        
        return entities
    
    def _is_valid_entity(self, ent: spacy.tokens.Span) -> bool:
        """Check if entity meets validation criteria"""
        # Check label
        mapped_label = self.medical_mappings.get(ent.label_, ent.label_)
        if mapped_label not in self.entity_types:
            return False
        
        # Check confidence
        confidence = getattr(ent._, 'confidence', 0.7)
        if confidence < self.confidence_threshold:
            return False
        
        # Check length
        if len(ent.text.strip()) < 2:
            return False
        
        # Check if it's mostly punctuation
        if len(ent.text.strip()) == sum(1 for c in ent.text if c in '.,;:!?'):
            return False
        
        return True
    
    def _create_entity(self, ent: spacy.tokens.Span, doc: spacy.tokens.Doc, section: Optional[str] = None) -> Entity:
        """Create Entity object from spaCy span"""
        # Get context (surrounding words)
        context_start = max(0, ent.start - 5)
        context_end = min(len(doc), ent.end + 5)
        context = doc[context_start:context_end].text
        
        # Normalize text
        normalized_text = self._normalize_entity_text(ent.text)
        
        # Get confidence
        confidence = getattr(ent._, 'confidence', 0.7)
        
        # Map label
        mapped_label = self.medical_mappings.get(ent.label_, ent.label_)
        
        return Entity(
            id=f"ent_{len(context)}_{hash(ent.text)}",
            text=ent.text,
            label=mapped_label,
            start_char=ent.start_char,
            end_char=ent.end_char,
            confidence=confidence,
            context=context,
            normalized_text=normalized_text,
            section=section,
            metadata={
                'original_label': ent.label_,
                'pos_tags': [token.pos_ for token in ent],
                'dependency_labels': [token.dep_ for token in ent]
            }
        )
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for better matching"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for diseases
        text = text.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
        
        # Remove trailing punctuation
        text = text.rstrip('.,;:!?')
        
        return text.strip()
    
    def _combine_entities(self, entities1: List[Entity], entities2: List[Entity]) -> List[Entity]:
        """Combine entities from different sources and remove duplicates"""
        all_entities = entities1 + entities2
        
        # Group by normalized text and position
        entity_groups = defaultdict(list)
        for entity in all_entities:
            key = (entity.normalized_text, entity.start_char, entity.end_char)
            entity_groups[key].append(entity)
        
        # Keep the entity with highest confidence from each group
        combined_entities = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda e: e.confidence)
            combined_entities.append(best_entity)
        
        return combined_entities
    
    def _post_process_entities(self, entities: List[Entity], document: ParsedDocument) -> List[Entity]:
        """Post-process entities for better accuracy"""
        processed_entities = []
        
        for entity in entities:
            # Skip very short entities
            if len(entity.text.strip()) < 2:
                continue
            
            # Skip entities that are mostly numbers (unless they're lab values)
            if entity.label not in ['LAB_VALUE', 'DOSAGE'] and entity.text.replace('.', '').replace('-', '').isdigit():
                continue
            
            # Enhance with additional metadata
            entity.metadata.update({
                'document_hash': document.file_hash,
                'extraction_time': time.time()
            })
            
            # Try to link to medical ontologies (placeholder)
            entity.umls_cui = self._link_to_umls(entity)
            
            processed_entities.append(entity)
        
        return processed_entities
    
    def _link_to_umls(self, entity: Entity) -> Optional[str]:
        """Link entity to UMLS (Unified Medical Language System)"""
        # Placeholder for UMLS linking
        # In a real implementation, this would query UMLS REST API
        return None
    
    def _filter_entities(self, entities: List[Entity]) -> List[Entity]:
        """Filter entities based on confidence and limits"""
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        # Apply confidence threshold
        filtered = [e for e in sorted_entities if e.confidence >= self.confidence_threshold]
        
        # Apply entity limit
        if len(filtered) > self.max_entities:
            filtered = filtered[:self.max_entities]
            logger.warning(f"Truncated entities to {self.max_entities} due to limit")
        
        return filtered
    
    async def extract_entities_async(self, document: ParsedDocument) -> List[Entity]:
        """Asynchronous entity extraction for large documents"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Split document into chunks for parallel processing
            chunks = self._split_text_into_chunks(document.text)
            
            # Process chunks in parallel
            tasks = []
            for chunk in chunks:
                task = loop.run_in_executor(executor, self._extract_from_text, chunk)
                tasks.append(task)
            
            # Wait for all tasks to complete
            chunk_results = await asyncio.gather(*tasks)
            
            # Combine results
            all_entities = []
            for entities in chunk_results:
                all_entities.extend(entities)
            
            # Post-process combined results
            processed_entities = self._post_process_entities(all_entities, document)
            filtered_entities = self._filter_entities(processed_entities)
            
            return filtered_entities
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into overlapping chunks for parallel processing"""
        chunks = []
        overlap = 500  # Character overlap between chunks
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def batch_extract_entities(self, documents: List[ParsedDocument]) -> List[List[Entity]]:
        """Extract entities from multiple documents in batch"""
        results = []
        
        for doc in documents:
            try:
                entities = self.extract_entities(doc)
                results.append(entities)
            except Exception as e:
                logger.error(f"Failed to extract entities from document {doc.file_hash}: {e}")
                results.append([])
        
        return results
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict:
        """Get statistics about extracted entities"""
        stats = {
            'total_entities': len(entities),
            'entity_types': {},
            'confidence_distribution': {
                'high': 0,    # > 0.9
                'medium': 0,  # 0.7-0.9
                'low': 0      # < 0.7
            },
            'sections': {},
            'average_confidence': 0.0
        }
        
        if not entities:
            return stats
        
        # Count by type
        for entity in entities:
            stats['entity_types'][entity.label] = stats['entity_types'].get(entity.label, 0) + 1
        
        # Confidence distribution
        confidences = [e.confidence for e in entities]
        stats['average_confidence'] = sum(confidences) / len(confidences)
        
        for conf in confidences:
            if conf > 0.9:
                stats['confidence_distribution']['high'] += 1
            elif conf > 0.7:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        # Section distribution
        for entity in entities:
            if entity.section:
                stats['sections'][entity.section] = stats['sections'].get(entity.section, 0) + 1
        
        return stats