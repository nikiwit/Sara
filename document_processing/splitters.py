"""
Text splitters for document processing.
"""

import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger("Sara")

class APUKnowledgeBaseTextSplitter:
    """Custom text splitter for APU Knowledge Base content."""
    
    def __init__(self, chunk_size=500, chunk_overlap=150, **kwargs):
        # Accept and ignore any additional kwargs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split APU KB documents into chunks."""
        result_docs = []
        total_docs = len(documents)
        
        logger.info(f"APUKnowledgeBaseTextSplitter processing {total_docs} documents")
        
        for i, doc in enumerate(documents):
            # Log progress periodically 
            if i % 20 == 0:
                logger.info(f"Processing APU KB document {i+1}/{total_docs}")
                
            # Most APU KB documents are small FAQs - if they're under chunk size, keep them intact
            if len(doc.page_content) <= self.chunk_size:
                result_docs.append(doc)
                continue
                
            # For longer documents, add the document as is without chunking to avoid processing issues
            # This preserves the FAQ structure better for the APU KB
            result_docs.append(doc)
        
        logger.info(f"APUKnowledgeBaseTextSplitter finished processing {len(result_docs)} documents")
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # If the text is short enough, don't split
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split on newlines first to preserve question/answer structure
        chunks = []
        sections = text.split("\n\n")
        
        current_chunk = ""
        for section in sections:
            # If adding this section would exceed chunk size and current chunk is not empty,
            # save current chunk and start a new one
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            # Otherwise, add section to current chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Adding the last chunk if it's not empty for generation
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If any chunks are still too large, split them further
        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # Use a simple character-based split as fallback
                result.extend(self._basic_split(chunk))
            else:
                result.append(chunk)
        
        return result
    
    def _basic_split(self, text: str) -> List[str]:
        """Basic character-based split for oversized chunks."""
        result = []
        # Split with overlap
        start = 0
        while start < len(text):
            # Find a good breakpoint (end of sentence if possible)
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # Try to find a sentence boundary
                last_period = text.rfind('. ', start, end)
                if last_period > start:
                    end = last_period + 1  # Include the period
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return result