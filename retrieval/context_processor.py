"""
Context processing for retrieved documents.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from config import config
from sara_types import QueryType, DocumentRelevance

logger = logging.getLogger("Sara")

class ContextProcessor:
    """Processes retrieved documents into a coherent context for the LLM."""
    
    def __init__(self):
        """Initialize the context processor."""
        self.max_context_size = config.MAX_CONTEXT_SIZE
        self.use_compression = config.USE_CONTEXT_COMPRESSION
    
    def process_context(self, documents: List[Document], query_analysis: Dict[str, Any]) -> str:
        """Process documents with monitoring for context size."""
        if not documents:
            return "No relevant information found in the APU knowledge base."
        
        # Get query elements
        query_type = query_analysis["query_type"]
        keywords = query_analysis["keywords"]
        
        # Score and prioritize documents
        scored_docs = self._score_documents(documents, keywords, query_type)
        
        # Select documents to include based on priority and size constraints
        selected_docs = self._select_documents(scored_docs)
        
        # Format the selected documents
        formatted_context = self._format_documents(selected_docs, query_analysis)
        
        # Monitoring
        context_size = len(formatted_context)
        context_ratio = context_size / self.max_context_size
        
        if context_ratio > 0.95:
            logger.warning(f"Context size critical: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        elif context_ratio > 0.85:
            logger.info(f"Context size approaching limit: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        
        return formatted_context
    
    def _score_documents(self, documents: List[Document], keywords: List[str], query_type: QueryType) -> List[Tuple[Document, float, DocumentRelevance]]:
        """
        Score documents based on relevance to the query, with APU KB optimizations.
        
        Args:
            documents: List of documents
            keywords: List of query keywords
            query_type: Type of query
            
        Returns:
            List of tuples (document, score, relevance_level)
        """
        scored_docs = []
        
        for doc in documents:
            # Start with base score (if available in metadata)
            base_score = doc.metadata.get('score', 0.5)
            
            # Check if this is an APU KB page
            is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
            is_faq = doc.metadata.get('is_faq', False)
            
            # Adjust score based on keyword matches
            keyword_score = 0
            content_lower = doc.page_content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # Count occurrences
                    count = content_lower.count(keyword.lower())
                    keyword_score += min(count / 10.0, 0.5)  # Cap at 0.5
            
            # Boost for APU KB pages
            if is_apu_kb:
                # Higher score for FAQ pages
                if is_faq:
                    base_score += 0.2
                
                # Check title match for keywords
                title = doc.metadata.get('page_title', '').lower()
                if any(keyword in title for keyword in keywords):
                    base_score += 0.3
                
                # Check tags for keyword matches
                if 'tags' in doc.metadata:
                    for tag in doc.metadata.get('tags', []):
                        if any(keyword in tag for keyword in keywords):
                            base_score += 0.1
            
            # Consider document length (prefer medium-sized chunks)
            length = len(doc.page_content)
            length_score = 0
            if 200 <= length <= 1000:
                length_score = 0.2  # Prefer medium chunks
            elif length > 1000:
                length_score = 0.1  # Long chunks are okay
            else:
                length_score = 0  # Short chunks less preferred
            
            # Adjust score based on query type and document content
            type_score = 0
            
            if query_type == QueryType.ACADEMIC:
                # For academic queries, prefer documents with relevant terms
                academic_terms = ["course", "module", "exam", "grade", "assessment", "lecture", "assignment"]
                if any(term in content_lower for term in academic_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.ADMINISTRATIVE:
                # For administrative queries, prefer documents with process terms
                admin_terms = ["form", "application", "procedure", "process", "submit", "request", "office"]
                if any(term in content_lower for term in admin_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.FINANCIAL:
                # For financial queries, prefer documents with financial terms
                financial_terms = ["fee", "payment", "scholarship", "loan", "refund", "invoice", "charge"]
                if any(term in content_lower for term in financial_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.FACTUAL:
                # For factual queries, prefer documents with data, numbers, definitions
                if re.search(r'\b(?:defined?|mean|refer|is a|are a|definition)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\d+', content_lower):
                    type_score += 0.2
                    
            elif query_type == QueryType.PROCEDURAL:
                # For procedural queries, prefer step-by-step content
                if re.search(r'\b(?:step|procedure|process|how to|guide|instruction)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\b(?:first|second|third|next|then|finally)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.CONCEPTUAL:
                # For conceptual queries, prefer explanations
                if re.search(r'\b(?:concept|theory|explanation|principle|understand|because)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.COMPARATIVE:
                # For comparative queries, prefer content with comparisons
                if re.search(r'\b(?:compare|contrast|versus|vs|difference|similarity|advantage|disadvantage)\b', content_lower):
                    type_score += 0.4
            
            # Combine scores
            combined_score = (base_score * 0.4) + (keyword_score * 0.3) + (length_score * 0.1) + (type_score * 0.2)
            
            # Determine relevance level
            relevance = DocumentRelevance.MEDIUM  # Default
            if combined_score > 0.7:
                relevance = DocumentRelevance.HIGH
            elif combined_score < 0.3:
                relevance = DocumentRelevance.LOW
            
            scored_docs.append((doc, combined_score, relevance))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _select_documents(self, scored_docs: List[Tuple[Document, float, DocumentRelevance]]) -> List[Tuple[Document, DocumentRelevance]]:
        """Select documents with stricter prioritization and size management."""
        selected_docs = []
        current_size = 0
        max_size = self.max_context_size
        
        # Group documents by relevance and sort by score within each group
        high_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.HIGH]
        high_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        medium_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.MEDIUM]
        medium_docs.sort(key=lambda x: x[1], reverse=True)
        
        low_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.LOW]
        
        # Process high relevance documents first with strict size management
        for doc, _, relevance in high_docs:
            doc_size = len(doc.page_content)
            
            # For very large documents, always compress
            if doc_size > max_size * 0.5:  # If document takes more than 50% of context
                if self.use_compression:
                    compressed_content = self._compress_document(doc.page_content)
                    compressed_size = len(compressed_content)
                    
                    # Only add if it fits in remaining space
                    if current_size + compressed_size <= max_size:
                        compressed_doc = Document(
                            page_content=compressed_content,
                            metadata=doc.metadata
                        )
                        selected_docs.append((compressed_doc, relevance))
                        current_size += compressed_size
                continue
            
            if current_size + doc_size <= max_size:
                selected_docs.append((doc, relevance))
                current_size += doc_size
        
        remaining_docs = medium_docs + low_docs
        for doc, _, relevance in remaining_docs:
            # Stop if we've reached 90% capacity
            if current_size >= max_size * 0.9:
                break
                
            doc_size = len(doc.page_content)
            if current_size + doc_size <= max_size:
                selected_docs.append((doc, relevance))
                current_size += doc_size
        
        logger.info(f"Selected {len(selected_docs)} documents for context (size: {current_size}/{max_size}) - {current_size/max_size:.1%} capacity")
        return selected_docs
    
    def _compress_document(self, content: str) -> str:
        """Enhanced document compression that preserves key information."""
        # For short content, don't compress
        if len(content) <= self.max_context_size / 2:
            return content
        
        # First pass: Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', content).strip()
        
        # Second pass: Identify and extract key information sections
        # For FAQ content, preserve question and direct answer
        if re.search(r'\?', compressed[:100]):  # Likely a question
            # Try to keep question and first paragraph of answer
            question_end = compressed.find('?', 0, 100) + 1
            first_para_end = compressed.find('\n\n', question_end)
            
            if question_end > 0 and first_para_end > question_end:
                key_content = compressed[:first_para_end].strip()
                if len(key_content) <= self.max_context_size / 2:
                    return key_content
        
        # For other content, apply progressive compression
        compressed = self._remove_filler_phrases(compressed)
        compressed = self._deduplicate_sentences(compressed)
        
        # If still too long, extract key sentences
        if len(compressed) > self.max_context_size / 2:
            compressed = self._extract_key_sentences(compressed, self.max_context_size / 2)
        
        return compressed

    def _remove_filler_phrases(self, text):
        """Remove common filler phrases."""
        filler_patterns = [
            (r'in order to', 'to'),
            (r'due to the fact that', 'because'),
            (r'in spite of the fact that', 'although'),
            (r'as a matter of fact', ''),
            (r'for the most part', 'mostly'),
            (r'for all intents and purposes', ''),
            (r'with regard to', 'regarding'),
            (r'in the event that', 'if'),
            (r'in the process of', 'while'),
        ]
        
        for pattern, replacement in filler_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _deduplicate_sentences(self, text):
        """Remove duplicate or near-duplicate sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        unique_sentences = []
        fingerprints = set()
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            if len(words) > 3:
                fingerprint = ' '.join(sorted(words[:5]))
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    unique_sentences.append(sentence)
            else:
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)

    def _extract_key_sentences(self, text, max_length):
        """Extract key sentences based on importance markers."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on importance markers
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Sentences at beginning and end are often important
            if i < 3:
                score += 3
            elif i >= len(sentences) - 3:
                score += 2
                
            # Sentences with numbers often contain key facts
            if re.search(r'\d', sentence):
                score += 2
                
            # Sentences with key transitional phrases
            if re.search(r'\b(?:however|therefore|thus|in conclusion|importantly|note that|remember)\b', 
                        sentence, re.IGNORECASE):
                score += 2
            
            if re.search(r'\b(?:question|answer|ask|query|refer|check|contact)\b', sentence, re.IGNORECASE):
                score += 1
                
            scored_sentences.append((score, sentence))
        
        # Sort by score and select until we reach max length
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        result = []
        current_length = 0
        
        for _, sentence in scored_sentences:
            if current_length + len(sentence) + 1 <= max_length:
                result.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        # Reorder sentences to maintain original flow
        original_order = sorted([(sentences.index(s), s) for s in result])
        return ' '.join(s for _, s in original_order)
    
    def _format_documents(self, selected_docs: List[Tuple[Document, DocumentRelevance]], query_analysis: Dict[str, Any]) -> str:
        """Format selected documents with overhead estimation to stay within limits."""
        if not selected_docs:
            return "No relevant information found in the APU knowledge base."
                
        # Estimate formatting overhead per document
        estimated_overhead_per_doc = 60  # Characters for headers, labels, etc.
        estimated_doc_separator = 20     # Extra characters between docs
        
        # Calculate total overhead
        total_overhead = len(selected_docs) * estimated_overhead_per_doc + (len(selected_docs) - 1) * estimated_doc_separator
        
        # Calculate available space for actual content
        available_content_space = self.max_context_size - total_overhead
        
        # Get query details for contextual formatting
        query_type = query_analysis["query_type"]
        
        # Group documents by relevance
        high_docs = []
        medium_docs = []
        low_docs = []
        unique_titles = set()
        
        for doc, relevance in selected_docs:
            if relevance == DocumentRelevance.HIGH:
                high_docs.append(doc)
            elif relevance == DocumentRelevance.MEDIUM:
                medium_docs.append(doc)
            else:
                low_docs.append(doc)
            
            # Track unique page titles for APU KB pages
            if doc.metadata.get('content_type') == 'apu_kb_page':
                title = doc.metadata.get('page_title', 'Untitled')
                unique_titles.add(title)

        # Calculate current content size (without formatting)
        current_content_size = sum(len(doc.page_content) for doc in high_docs + medium_docs + low_docs)
        
        # If content will exceed available space, reduce documents
        if current_content_size > available_content_space:
            # Prioritize keeping high relevance docs
            content_to_remove = current_content_size - available_content_space
            
            # Remove low relevance docs first
            while low_docs and content_to_remove > 0:
                doc = low_docs.pop()
                content_to_remove -= len(doc.page_content)
            
            # Then medium relevance if still needed
            while medium_docs and content_to_remove > 0:
                doc = medium_docs.pop()
                content_to_remove -= len(doc.page_content)
                
            # In extreme cases, remove lower-scored high relevance docs
            if content_to_remove > 0:
                # Sort high docs by length (remove longest first to minimize info loss)
                high_docs.sort(key=lambda doc: len(doc.page_content), reverse=True)
                
                while high_docs and content_to_remove > 0:
                    doc = high_docs.pop()
                    content_to_remove -= len(doc.page_content)

        # Begin formatting
        formatted_docs = []
        
        doc_count = len(high_docs) + len(medium_docs) + len(low_docs)
        title_count = len(unique_titles)
        
        summary_header = f"Found {doc_count} relevant sections"
        if title_count > 0:
            summary_header += f" from {title_count} topics in the APU knowledge base"
        
        if query_type == QueryType.ACADEMIC:
            summary_header = f"Found {doc_count} relevant sections about academic procedures at APU"
        elif query_type == QueryType.ADMINISTRATIVE:
            summary_header = f"Found {doc_count} relevant sections about administrative processes at APU"
        elif query_type == QueryType.FINANCIAL:
            summary_header = f"Found {doc_count} relevant sections about financial matters at APU"
        
        formatted_docs.append(f"{summary_header}\n")
        
        # Format the documents
        formatted_content = self._format_doc_group(high_docs, "HIGHLY RELEVANT INFORMATION", formatted_docs, 0)
        formatted_content = self._format_doc_group(medium_docs, "ADDITIONAL RELEVANT INFORMATION", formatted_docs, len(high_docs))
        formatted_content = self._format_doc_group(low_docs, "SUPPLEMENTARY INFORMATION", formatted_docs, len(high_docs) + len(medium_docs))
        
        final_context = "\n".join(formatted_docs)
        
        # Log the actual final size
        context_size = len(final_context)
        context_ratio = context_size / self.max_context_size
        
        if context_ratio > 0.95:
            logger.warning(f"Context size critical: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        elif context_ratio > 0.85:
            logger.info(f"Context size approaching limit: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        
        return final_context

    def _format_doc_group(self, docs: List[Document], section_title: str, formatted_docs: List[str], start_index: int) -> List[str]:
        """Helper to format a group of documents with consistent style."""
        if not docs:
            return formatted_docs
        
        formatted_docs.append(f"--- {section_title} ---")
        
        for i, doc in enumerate(docs):
            is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
            is_faq = doc.metadata.get('is_faq', False)
            
            if is_apu_kb:
                title = doc.metadata.get('page_title', 'Untitled')
                if is_faq:
                    # Format as question-answer (more compact)
                    formatted_text = f"Q: {title}\nA: {doc.page_content}\n"
                else:
                    # Format as topic (more compact)
                    formatted_text = f"Topic: {title}\n{doc.page_content}\n"
                
                # Related pages if available (more compact)
                related_pages = doc.metadata.get('related_pages', [])
                if related_pages and not any("label in" in page for page in related_pages):
                    if len(related_pages) > 2:
                        formatted_text += f"Related: {', '.join(related_pages[:2])}\n"
                    else:
                        formatted_text += f"Related: {', '.join(related_pages)}\n"
            else:
                # Standard document format (more compact)
                source = doc.metadata.get('source', 'Unknown source')
                filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
                
                formatted_text = f"Doc {i+1+start_index} ({filename}): {doc.page_content}\n"
            
            formatted_docs.append(formatted_text)
        
        return formatted_docs