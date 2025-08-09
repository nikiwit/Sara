"""
Text splitters for document processing.
"""

import logging
import re
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
        """Split APU KB documents into contextual chunks with headers (NotebookLM approach)."""
        result_docs = []
        total_docs = len(documents)
        
        logger.info(f"APUKnowledgeBaseTextSplitter processing {total_docs} documents")
        
        for i, doc in enumerate(documents):
            # Log progress periodically 
            if i % 20 == 0:
                logger.info(f"Processing APU KB document {i+1}/{total_docs}")
            
            # Create contextual header for better semantic understanding
            page_title = doc.metadata.get('page_title', 'Unknown Topic')
            filename = doc.metadata.get('filename', 'Unknown Source')
            
            # Extract space name from filename (e.g., apu_ITSM_kb.txt -> ITSM)
            space_name = 'APU'
            if '_' in filename:
                space_name = filename.replace('apu_', '').replace('_kb.txt', '').upper()
            
            # Create NotebookLM-style contextual header
            context_header = f"[{space_name} Knowledge Base - {page_title}]"
            
            # Add contextual header to content for better embeddings
            contextual_content = f"{context_header}\n\n{doc.page_content}"
            
            # Update metadata with context information
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata["context_header"] = context_header
            enhanced_metadata["space_name"] = space_name
            enhanced_metadata["enhanced_for_retrieval"] = True
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=contextual_content,
                metadata=enhanced_metadata
            )
            
            # Most APU KB documents are small FAQs - if they're under chunk size, keep them intact
            if len(contextual_content) <= self.chunk_size:
                result_docs.append(enhanced_doc)
                continue
            
            # For longer documents, create semantic chunks while preserving context
            chunks = self._create_semantic_chunks(doc.page_content, context_header)
            
            for chunk_idx, chunk_content in enumerate(chunks):
                chunk_metadata = enhanced_metadata.copy()
                chunk_metadata["chunk_id"] = chunk_idx
                chunk_metadata["total_chunks"] = len(chunks)
                
                # Each chunk gets the contextual header
                chunk_with_context = f"{context_header}\n\n{chunk_content}"
                
                chunk_doc = Document(
                    page_content=chunk_with_context,
                    metadata=chunk_metadata
                )
                result_docs.append(chunk_doc)
        
        logger.info(f"APUKnowledgeBaseTextSplitter finished processing {len(result_docs)} documents")
        return result_docs
    
    def _create_semantic_chunks(self, content: str, context_header: str) -> List[str]:
        """Create semantic chunks that preserve meaning and structure."""
        # Strategy 1: Split on procedural steps (Step 1, Step 2, etc.)
        step_pattern = r'(Step \d+|^\d+\.|\n\d+\.)'
        if re.search(step_pattern, content, re.MULTILINE):
            return self._split_by_steps(content)
        
        # Strategy 2: Split on FAQ sections or major headings
        if '---' in content or re.search(r'^[A-Z][A-Za-z\s]+:$', content, re.MULTILINE):
            return self._split_by_sections(content)
        
        # Strategy 3: Fallback to paragraph-based splitting
        return self._split_by_paragraphs(content)
    
    def _split_by_steps(self, content: str) -> List[str]:
        """Split content by numbered steps while preserving context."""
        step_pattern = r'(Step \d+|^\d+\.)'
        parts = re.split(step_pattern, content, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            if re.match(step_pattern, part.strip()):
                if current_chunk.strip() and len(current_chunk) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
                
            # Prevent overly large chunks
            if len(current_chunk) > self.chunk_size * 1.5:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by natural sections."""
        # Split on section boundaries
        sections = re.split(r'\n(?=\w+:|\d+\.|[A-Z][A-Za-z\s]+:)', content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk + section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk = current_chunk + "\n" + section if current_chunk else section
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs as fallback."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
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