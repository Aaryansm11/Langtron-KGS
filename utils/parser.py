# parser.py
import os, re, textract, logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentParser:
    def parse_document(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return textract.process(str(file_path), method='pdfminer').decode()
        elif suffix == ".docx":
            return textract.process(str(file_path)).decode()
        elif suffix in {".txt", ".rtf"}:
            return textract.process(str(file_path)).decode()
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    @staticmethod
    def segment_sections(text: str):
        """Return (section_name, paragraph) tuples for later filtering."""
        pattern = re.compile(r'^(Chief Complaint|History|Assessment|Plan|Medications|Diagnosis|Impression)', re.I | re.M)
        lines = text.splitlines()
        sections = []
        current, buffer = None, []
        for line in lines:
            m = pattern.match(line)
            if m:
                if buffer:
                    sections.append((current, "\n".join(buffer)))
                current, buffer = m.group(0), [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append((current, "\n".join(buffer)))
        return sections