"""
Enhanced document parser with better error handling, security, and format support
"""
import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pypdf
import python_docx2txt
import mammoth
import textract
from bs4 import BeautifulSoup
import magic
import hashlib

from config import settings

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    RTF = "rtf"
    HTML = "html"
    XML = "xml"


@dataclass
class ParsedDocument:
    """Structured representation of a parsed document"""
    text: str
    metadata: Dict
    sections: List[Tuple[str, str]]
    word_count: int
    char_count: int
    file_hash: str
    file_size: int
    document_type: DocumentType
    processing_time: float
    errors: List[str]


class DocumentParser:
    """Enhanced document parser with security, error handling, and format support"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_UPLOAD_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.temp_dir = settings.TEMP_DIR
        
        # Clinical section patterns
        self.clinical_sections = {
            'chief_complaint': r'(?:chief complaint|cc|presenting complaint)',
            'history': r'(?:history of present illness|hpi|history)',
            'past_medical_history': r'(?:past medical history|pmh|medical history)',
            'medications': r'(?:medications?|drugs?|prescriptions?)',
            'allergies': r'(?:allergies?|adverse reactions?)',
            'social_history': r'(?:social history|social)',
            'family_history': r'(?:family history|fh)',
            'review_of_systems': r'(?:review of systems|ros)',
            'physical_exam': r'(?:physical exam|physical examination|pe)',
            'assessment': r'(?:assessment|impression|diagnosis)',
            'plan': r'(?:plan|treatment|management)',
            'labs': r'(?:labs?|laboratory|lab results)',
            'imaging': r'(?:imaging|radiology|x-ray|ct|mri)',
            'procedures': r'(?:procedures?|operations?|surgery)'
        }
    
    def parse_document(self, file_path: Union[str, Path]) -> ParsedDocument:
        """Parse document with comprehensive error handling"""
        start_time = time.time()
        file_path = Path(file_path)
        errors = []
        
        try:
            # Security checks
            self._validate_file(file_path)
            
            # Detect file type
            doc_type = self._detect_document_type(file_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Parse based on type
            text, metadata = self._parse_by_type(file_path, doc_type)
            
            # Post-process text
            text = self._clean_text(text)
            
            # Extract sections
            sections = self._extract_sections(text)
            
            # Calculate metrics
            word_count = len(text.split())
            char_count = len(text)
            file_size = file_path.stat().st_size
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully parsed {file_path.name} ({doc_type}) in {processing_time:.2f}s")
            
            return ParsedDocument(
                text=text,
                metadata=metadata,
                sections=sections,
                word_count=word_count,
                char_count=char_count,
                file_hash=file_hash,
                file_size=file_size,
                document_type=doc_type,
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            errors.append(str(e))
            raise
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate file security and constraints"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {file_path.stat().st_size} bytes")
        
        if file_path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Check for malicious content using python-magic
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            if mime_type.startswith('application/x-executable'):
                raise ValueError("Executable files not allowed")
        except Exception as e:
            logger.warning(f"MIME type detection failed: {e}")
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type reliably"""
        suffix = file_path.suffix.lower()
        
        # Primary detection by extension
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.doc': DocumentType.DOC,
            '.txt': DocumentType.TXT,
            '.rtf': DocumentType.RTF,
            '.html': DocumentType.HTML,
            '.xml': DocumentType.XML
        }
        
        if suffix in type_mapping:
            return type_mapping[suffix]
        
        # Fallback to magic number detection
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            if 'pdf' in mime_type:
                return DocumentType.PDF
            elif 'word' in mime_type or 'officedocument' in mime_type:
                return DocumentType.DOCX
            elif 'text' in mime_type:
                return DocumentType.TXT
        except:
            pass
        
        # Default fallback
        return DocumentType.TXT
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _parse_by_type(self, file_path: Path, doc_type: DocumentType) -> Tuple[str, Dict]:
        """Parse document based on detected type"""
        metadata = {}
        
        if doc_type == DocumentType.PDF:
            return self._parse_pdf(file_path, metadata)
        elif doc_type == DocumentType.DOCX:
            return self._parse_docx(file_path, metadata)
        elif doc_type == DocumentType.DOC:
            return self._parse_doc(file_path, metadata)
        elif doc_type == DocumentType.TXT:
            return self._parse_txt(file_path, metadata)
        elif doc_type == DocumentType.RTF:
            return self._parse_rtf(file_path, metadata)
        else:
            # Fallback to textract
            return self._parse_with_textract(file_path, metadata)
    
    def _parse_pdf(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Parse PDF with multiple fallback methods"""
        try:
            # Try pypdf first (faster)
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Extract metadata
                if reader.metadata:
                    metadata.update({
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'pages': len(reader.pages)
                    })
                
                return text, metadata
                
        except Exception as e:
            logger.warning(f"pypdf failed, trying textract: {e}")
            # Fallback to textract
            return self._parse_with_textract(file_path, metadata)
    
    def _parse_docx(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Parse DOCX with metadata extraction"""
        try:
            # Extract text
            text = python_docx2txt.process(str(file_path))
            
            # Try to extract more detailed content with mammoth
            try:
                with open(file_path, "rb") as docx_file:
                    result = mammoth.extract_raw_text(docx_file)
                    if result.value:
                        text = result.value
                    if result.messages:
                        metadata['conversion_messages'] = [msg.message for msg in result.messages]
            except Exception as e:
                logger.warning(f"Mammoth extraction failed: {e}")
            
            return text, metadata
            
        except Exception as e:
            logger.warning(f"DOCX parsing failed, trying textract: {e}")
            return self._parse_with_textract(file_path, metadata)
    
    def _parse_doc(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Parse legacy DOC format"""
        return self._parse_with_textract(file_path, metadata)
    
    def _parse_txt(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Parse plain text with encoding detection"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                metadata['encoding'] = 'utf-8'
                return text, metadata
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                        metadata['encoding'] = encoding
                        return text, metadata
                except UnicodeDecodeError:
                    continue
        
        raise ValueError("Could not decode text file with any supported encoding")
    
    def _parse_rtf(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Parse RTF format"""
        return self._parse_with_textract(file_path, metadata)
    
    def _parse_with_textract(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Fallback parser using textract"""
        try:
            text = textract.process(str(file_path)).decode('utf-8')
            metadata['parser'] = 'textract'
            return text, metadata
        except Exception as e:
            logger.error(f"Textract parsing failed: {e}")
            raise ValueError(f"Could not parse document: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve medical notation
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\{\}\/\%\+\=\<\>\@\#\$\&\*]', '', text)
        
        # Normalize line endings
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract clinical sections from text"""
        sections = []
        text_lower = text.lower()
        
        # Find section boundaries
        section_positions = []
        for section_name, pattern in self.clinical_sections.items():
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                section_positions.append((match.start(), section_name, match.group()))
        
        # Sort by position
        section_positions.sort()
        
        # Extract text for each section
        for i, (start_pos, section_name, match_text) in enumerate(section_positions):
            # Find end position (next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            if section_text:
                sections.append((section_name, section_text))
        
        return sections

    def batch_parse(self, file_paths: List[Path]) -> List[ParsedDocument]:
        """Parse multiple documents in batch"""
        results = []
        for file_path in file_paths:
            try:
                result = self.parse_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                # Add error document
                results.append(ParsedDocument(
                    text="",
                    metadata={"error": str(e)},
                    sections=[],
                    word_count=0,
                    char_count=0,
                    file_hash="",
                    file_size=0,
                    document_type=DocumentType.TXT,
                    processing_time=0.0,
                    errors=[str(e)]
                ))
        
        return results
