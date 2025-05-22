"""
Parsers and loaders for APU Knowledge Base format.
"""

import os
import re
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger("CustomRAG")

class APUKnowledgeBaseParser:
    """Parser for APU Knowledge Base format."""
    
    PAGE_PATTERN = r"--- PAGE: (.*?) ---\s*(.*?)(?=--- PAGE:|$)"
    RELATED_PAGES_PATTERN = r"Related Pages –\s*(.*?)$"
    
    @classmethod
    def parse_knowledge_base(cls, content: str) -> List[Dict[str, Any]]:
        """
        Parse the APU Knowledge Base content into structured pages.
        
        Args:
            content: Full content of the knowledge base file
            
        Returns:
            List of dictionaries representing each page with metadata
        """
        # Find all pages using regex pattern
        pages = []
        for match in re.finditer(cls.PAGE_PATTERN, content, re.DOTALL):
            title = match.group(1).strip()
            content = match.group(2).strip()
            
            # Skip empty pages
            if not content:
                continue
            
            # Extract related pages if present
            related_pages = []
            related_match = re.search(cls.RELATED_PAGES_PATTERN, content, re.MULTILINE | re.DOTALL)
            if related_match:
                related_text = related_match.group(1).strip()
                # Process "label in ( ... )" format or regular links
                if "label in" in related_text:
                    # Just store as is, can be processed later if needed
                    related_pages = [related_text]
                else:
                    # Split by newlines and strip each line
                    related_pages = [line.strip() for line in related_text.split('\n') if line.strip()]
                    
                # Clean content by removing the related pages section
                content = content.replace(related_match.group(0), "").strip()
            
            # Create a structured page object
            page = {
                "title": title,
                "content": content,
                "related_pages": related_pages,
                "is_faq": cls._is_faq_page(title, content)
            }
            
            pages.append(page)
        
        logger.info(f"Parsed {len(pages)} pages from APU Knowledge Base")
        return pages
    
    @staticmethod
    def _is_faq_page(title: str, content: str) -> bool:
        """
        Determine if a page is an FAQ type page.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            Boolean indicating if the page is an FAQ
        """
        # Check if the title is a question
        has_question_mark = "?" in title
        
        # Check for question words in title
        question_words = ["how", "what", "where", "when", "why", "who", "can", "do", "is", "are", "will"]
        starts_with_question_word = any(title.lower().startswith(word) for word in question_words)
        
        # Either has a question mark or starts with a question word
        return has_question_mark or starts_with_question_word

class APUKnowledgeBaseLoader(BaseLoader):
    """Specialized loader for APU Knowledge Base files."""
    
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """
        Load the APU Knowledge Base file into Document objects.
        
        Returns:
            List of Document objects
        """
        try:
            # Read the file content
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the knowledge base
            pages = APUKnowledgeBaseParser.parse_knowledge_base(content)
            
            # Convert pages to LangChain Document objects
            documents = []
            for page in pages:
                # Create metadata dictionary
                metadata = {
                    "source": self.file_path,
                    "filename": os.path.basename(self.file_path),
                    "page_title": page["title"],
                    "is_faq": page["is_faq"],
                    "related_pages": page["related_pages"],
                    "content_type": "apu_kb_page"
                }
                
                # Tags for improved searchability
                metadata["tags"] = self._extract_tags(page["title"], page["content"])
                
                # Create Document object
                doc = Document(
                    page_content=page["content"],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading APU Knowledge Base file {self.file_path}: {str(e)}")
            return []
    
    @staticmethod
    def _extract_tags(title: str, content: str) -> List[str]:
        """
        Extract relevant tags from the page title and content.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            List of tag strings
        """
        tags = []
        
        tags.append(title.lower())
        
        # Extract possible key terms from title
        title_words = title.lower().split()
        for word in title_words:
            if len(word) > 3 and word not in ["what", "when", "where", "how", "does", "will"]:
                tags.append(word)
        
        # Extract key acronyms from content
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        for acronym in acronyms:
            tags.append(acronym)
        
        # Extract course codes (e.g., APU1F2103CS)
        course_codes = re.findall(r'\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]{2,}\b', content)
        for code in course_codes:
            tags.append(code)
        
        return list(set(tags))  # Remove duplicates