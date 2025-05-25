"""
Document processing for APU knowledge base.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger("CustomRAG")

class APUKnowledgeBaseParser:
    """Parser for APU Knowledge Base content."""
    
    @staticmethod
    def parse_apu_kb(text: str, source: str = "Unknown", filename: str = "Unknown") -> List[Document]:
        """
        Parse APU Knowledge Base text into documents.
        
        Args:
            text: Raw text content
            source: Source of the text
            filename: Filename of the source
            
        Returns:
            List of Document objects
        """
        # Split text into pages based on markdown-style headers
        page_pattern = r'---\s*PAGE:\s*(.*?)\s*---'
        pages = re.split(page_pattern, text)
        
        # First element is empty if text starts with a page marker
        if pages and not pages[0].strip():
            pages.pop(0)
        
        documents = []
        
        # Process pages in pairs (title, content)
        for i in range(0, len(pages), 2):
            if i + 1 >= len(pages):
                break
                
            title = pages[i].strip()
            content = pages[i + 1].strip()
            
            # Skip empty pages
            if not title or not content:
                continue
            
            # Create metadata
            metadata = {
                "source": source,
                "filename": filename,
                "page_title": title,
                "content_type": "apu_kb_page",
                "is_faq": title.endswith('?'),  # Mark as FAQ if title ends with question mark
                "page_number": i // 2 + 1
            }
            
            # Special metadata for medical insurance related pages
            if any(term in title.lower() for term in ["medical", "insurance", "health", "card"]):
                metadata["is_medical_insurance"] = True
                metadata["priority_topic"] = "medical_insurance"
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            documents.append(doc)
        
        logger.info(f"Parsed {len(documents)} pages from APU KB file: {filename}")
        return documents

class APUKnowledgeBaseLoader:
    """Loader for APU Knowledge Base files."""
    
    @staticmethod
    def load(file_path: str) -> List[Document]:
        """
        Load an APU Knowledge Base file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Get filename
            filename = os.path.basename(file_path)
            
            # Parse the text
            return APUKnowledgeBaseParser.parse_apu_kb(text, source=file_path, filename=filename)
            
        except Exception as e:
            logger.error(f"Error loading APU KB file {file_path}: {e}")
            return []