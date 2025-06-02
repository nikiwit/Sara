"""
Context processing for retrieved documents into coherent LLM input.
Handles document scoring, selection, compression, and formatting for optimal context utilization.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from config import Config
from apurag_types import QueryType, DocumentRelevance

logger = logging.getLogger("CustomRAG")

class ContextProcessor:
    """Processes retrieved documents into coherent context for the LLM with intelligent optimization."""
    
    def __init__(self):
        """Initialize the context processor with enhanced configuration and performance tracking."""
        self.max_context_size = Config.MAX_CONTEXT_SIZE
        self.use_compression = Config.USE_CONTEXT_COMPRESSION
        
        # Enhanced configuration with intelligent defaults
        self.compression_ratio = getattr(Config, 'CONTEXT_COMPRESSION_RATIO', 0.7)
        self.priority_boost = getattr(Config, 'CONTEXT_PRIORITY_BOOST', 1.5)
        self.target_utilization = getattr(Config, 'CONTEXT_TARGET_UTILIZATION', 0.80)  # Target 80% for optimal performance
        self.max_safe_utilization = 0.85  # 85% maximum safe utilization to prevent truncation
        
        # Performance tracking for optimization and monitoring
        self._processing_stats = {
            "total_processed": 0,
            "avg_context_utilization": 0.0,
            "compression_saved": 0
        }
        
        logger.info(f"Context processor initialized - Target: {self.target_utilization:.0%}, Max: {self.max_safe_utilization:.0%}")
    
    def process_context(self, documents: List[Document], query_analysis: Dict[str, Any]) -> str:
        """Process documents into optimized context with comprehensive monitoring and smart optimization."""
        if not documents:
            return "No relevant information found in the APU knowledge base."
        
        self._processing_stats["total_processed"] += 1
        
        # Extract query analysis components
        query_type = query_analysis["query_type"]
        keywords = query_analysis["keywords"]
        original_query = query_analysis.get("original_query", "")
        
        # Enhanced document scoring with multiple relevance factors
        scored_docs = self._score_documents_optimized(documents, keywords, query_type, original_query)
        
        # Smart document selection targeting optimal context utilization
        selected_docs = self._select_documents_optimized(scored_docs)
        
        # Efficient document formatting with minimal overhead
        formatted_context = self._format_documents_optimized(selected_docs, query_analysis)
        
        # Enhanced monitoring with actionable insights for optimization
        context_size = len(formatted_context)
        utilization = context_size / self.max_context_size
        
        # Update running average for trend analysis
        old_avg = self._processing_stats["avg_context_utilization"]
        total = self._processing_stats["total_processed"]
        self._processing_stats["avg_context_utilization"] = (old_avg * (total - 1) + utilization) / total
        
        # Smart logging based on utilization thresholds
        if utilization > 0.90:
            logger.warning(f"Context size critical: {context_size}/{self.max_context_size} ({utilization:.1%}) - Consider optimization")
        elif utilization > self.max_safe_utilization:
            logger.info(f"Context size high: {context_size}/{self.max_context_size} ({utilization:.1%}) - Within safe range")
        elif utilization < 0.50:
            logger.info(f"Context size low: {context_size}/{self.max_context_size} ({utilization:.1%}) - Could add more content")
        else:
            logger.debug(f"Context size optimal: {context_size}/{self.max_context_size} ({utilization:.1%})")
        
        return formatted_context
    
    def _score_documents_optimized(self, documents: List[Document], keywords: List[str], 
                                 query_type: QueryType, original_query: str = "") -> List[Tuple[Document, float, DocumentRelevance]]:
        """Enhanced document scoring with comprehensive relevance analysis and improved accuracy."""
        scored_docs = []
        query_lower = original_query.lower()
        
        for doc in documents:
            # Initialize with base score from vector similarity
            base_score = doc.metadata.get('score', 0.5)
            
            # Document metadata analysis for enhanced scoring
            is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
            is_faq = doc.metadata.get('is_faq', False)
            content_lower = doc.page_content.lower()
            
            # Multi-factor scoring with weighted contributions
            keyword_score = self._calculate_keyword_score(content_lower, keywords, doc.metadata)
            apu_score = self._calculate_apu_score(doc, keywords, is_apu_kb, is_faq)
            length_score = self._calculate_length_score(len(doc.page_content))
            type_score = self._calculate_type_score(content_lower, query_type)
            phrase_score = self._calculate_phrase_score(content_lower, query_lower, original_query)
            
            # Weighted combination optimized for best relevance detection
            combined_score = (
                base_score * 0.25 +      # Vector similarity baseline
                keyword_score * 0.25 +   # Keyword relevance
                apu_score * 0.15 +       # APU-specific content boost
                length_score * 0.05 +    # Content length optimization
                type_score * 0.15 +      # Query type alignment
                phrase_score * 0.15      # Exact phrase matching bonus
            )
            
            # Apply priority boost for high-quality sources
            if is_apu_kb and (is_faq or doc.metadata.get('priority_topic')):
                combined_score *= self.priority_boost
            
            # Determine relevance level with refined thresholds for better recall
            if combined_score > 0.75:
                relevance = DocumentRelevance.HIGH
            elif combined_score > 0.45:  # Lowered from 0.5 for better recall
                relevance = DocumentRelevance.MEDIUM
            else:
                relevance = DocumentRelevance.LOW
            
            scored_docs.append((doc, combined_score, relevance))
        
        # Sort by score in descending order for optimal selection
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _calculate_keyword_score(self, content_lower: str, keywords: List[str], metadata: Dict) -> float:
        """Enhanced keyword scoring with context awareness and position weighting."""
        if not keywords:
            return 0.0
        
        score = 0.0
        title = metadata.get('page_title', '').lower()
        
        for keyword in keywords:
            # Count keyword occurrences in content
            content_count = content_lower.count(keyword.lower())
            if content_count > 0:
                # Logarithmic scaling to prevent single keyword dominance
                score += min(0.3, 0.1 + 0.05 * (min(content_count, 4)))
                
                # Title match bonus for increased relevance
                if title and keyword.lower() in title:
                    score += 0.2
                
                # Position bonus for early keyword appearance
                first_pos = content_lower.find(keyword.lower())
                if first_pos >= 0 and first_pos < 200:  # First 200 characters
                    score += 0.1
        
        # Normalize by number of keywords to prevent bias toward longer keyword lists
        return min(1.0, score / len(keywords))
    
    def _calculate_apu_score(self, doc: Document, keywords: List[str], is_apu_kb: bool, is_faq: bool) -> float:
        """Enhanced APU-specific scoring for institutional content prioritization."""
        score = 0.0
        
        if is_apu_kb:
            score += 0.3  # Base APU knowledge base bonus
            
            if is_faq:
                score += 0.2  # FAQ content bonus
            
            # Priority topic bonus for important content
            priority_topic = doc.metadata.get('priority_topic')
            if priority_topic:
                score += 0.3
                
            # Related pages bonus indicating comprehensive coverage
            related_pages = doc.metadata.get('related_pages', [])
            if related_pages:
                score += min(0.1, len(related_pages) * 0.02)
            
            # Tags matching for topic relevance
            tags = doc.metadata.get('tags', [])
            if tags:
                for tag in tags:
                    if any(keyword.lower() in tag.lower() for keyword in keywords):
                        score += 0.1
                        break
        
        return min(1.0, score)
    
    def _calculate_length_score(self, length: int) -> float:
        """Smart length scoring that considers content density and readability."""
        if length < 100:
            return 0.0  # Too short to be useful
        elif length < 300:
            return 0.3  # Short but acceptable content
        elif length < 800:
            return 1.0  # Optimal range for most queries
        elif length < 1500:
            return 0.8  # Good content but longer
        elif length < 2500:
            return 0.6  # Long but still manageable
        else:
            return 0.4  # Very long, might need compression
    
    def _calculate_type_score(self, content_lower: str, query_type: QueryType) -> float:
        """Enhanced query type matching for better context relevance."""
        score = 0.0
        
        if query_type == QueryType.ACADEMIC:
            academic_terms = ["course", "module", "exam", "grade", "assessment", "lecture", "assignment"]
            if any(term in content_lower for term in academic_terms):
                score += 0.3
                
        elif query_type == QueryType.ADMINISTRATIVE:
            admin_terms = ["form", "application", "procedure", "process", "submit", "request", "office"]
            if any(term in content_lower for term in admin_terms):
                score += 0.3
                
        elif query_type == QueryType.FINANCIAL:
            financial_terms = ["fee", "payment", "scholarship", "loan", "refund", "invoice", "charge"]
            if any(term in content_lower for term in financial_terms):
                score += 0.3
                
        elif query_type == QueryType.PROCEDURAL:
            if re.search(r'\b(?:step|procedure|process|how to|guide|instruction)\b', content_lower):
                score += 0.3
            if re.search(r'\b(?:first|second|third|next|then|finally)\b', content_lower):
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_phrase_score(self, content_lower: str, query_lower: str, original_query: str) -> float:
        """Calculate bonus score for exact phrase matches to improve relevance."""
        if len(original_query) < 10:  # Skip for very short queries
            return 0.0
        
        # Exact query match provides highest phrase score
        if query_lower in content_lower:
            return 1.0
        
        # Partial phrase matches for multi-word queries
        words = query_lower.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if phrase in content_lower:
                    return 0.5
        
        return 0.0
    
    def _select_documents_optimized(self, scored_docs: List[Tuple[Document, float, DocumentRelevance]]) -> List[Tuple[Document, DocumentRelevance]]:
        """Optimized document selection targeting efficient context usage with intelligent compression."""
        if not scored_docs:
            return []
        
        selected_docs = []
        current_size = 0
        
        # Calculate target and maximum sizes for intelligent space management
        target_size = int(self.max_context_size * self.target_utilization)
        max_size = int(self.max_context_size * self.max_safe_utilization)
        
        # Estimate formatting overhead more accurately for better planning
        estimated_overhead_per_doc = 45  # Reduced from 60 for efficiency
        estimated_base_overhead = 100   # Headers, summaries, etc.
        
        # Group documents by relevance for prioritized selection
        high_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.HIGH]
        medium_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.MEDIUM]
        low_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.LOW]
        
        # Process high relevance documents first with intelligent sizing
        for doc, score, relevance in high_docs:
            doc_size = len(doc.page_content)
            overhead = estimated_overhead_per_doc
            total_doc_size = doc_size + overhead
            
            # Smart compression decision for oversized content
            if current_size + total_doc_size > target_size:
                if self.use_compression and doc_size > 300:
                    compressed_content = self._compress_document_optimized(doc.page_content, target_size - current_size - overhead)
                    if compressed_content and len(compressed_content) > 100:  # Ensure meaningful content remains
                        compressed_doc = Document(page_content=compressed_content, metadata=doc.metadata)
                        selected_docs.append((compressed_doc, relevance))
                        current_size += len(compressed_content) + overhead
                        self._processing_stats["compression_saved"] += doc_size - len(compressed_content)
                        continue
                
                # Check if we can still fit within maximum safe limits
                if current_size + total_doc_size <= max_size:
                    selected_docs.append((doc, relevance))
                    current_size += total_doc_size
                else:
                    break  # Cannot fit this document
            else:
                selected_docs.append((doc, relevance))
                current_size += total_doc_size
        
        # Add medium relevance documents if space allows
        remaining_space = target_size - current_size
        for doc, score, relevance in medium_docs:
            if remaining_space <= 0:
                break
            
            doc_size = len(doc.page_content)
            total_doc_size = doc_size + estimated_overhead_per_doc
            
            if total_doc_size <= remaining_space:
                selected_docs.append((doc, relevance))
                current_size += total_doc_size
                remaining_space -= total_doc_size
        
        # Add low relevance documents only if significant space remains
        remaining_space = max_size - current_size
        if remaining_space > self.max_context_size * 0.15:  # Only if >15% space left
            for doc, score, relevance in low_docs[:1]:  # Maximum 1 low relevance document
                doc_size = len(doc.page_content)
                total_doc_size = doc_size + estimated_overhead_per_doc
                
                if total_doc_size <= remaining_space:
                    selected_docs.append((doc, relevance))
                    current_size += total_doc_size
                    break
        
        utilization = current_size / self.max_context_size
        logger.info(f"Selected {len(selected_docs)} documents (estimated size: {current_size}/{self.max_context_size} = {utilization:.1%})")
        
        return selected_docs
    
    def _compress_document_optimized(self, content: str, target_length: int) -> str:
        """Enhanced document compression that preserves key information using multi-stage approach."""
        if len(content) <= target_length:
            return content
        
        # Multi-stage compression for optimal information preservation
        compressed = content
        
        # Stage 1: Clean up excessive whitespace and formatting
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)  # Reduce multiple newlines
        
        if len(compressed) <= target_length:
            return compressed
        
        # Stage 2: Remove filler phrases that don't add meaning
        compressed = self._remove_filler_phrases_optimized(compressed)
        
        if len(compressed) <= target_length:
            return compressed
        
        # Stage 3: Intelligent sentence extraction preserving key information
        compressed = self._extract_key_sentences_optimized(compressed, target_length)
        
        return compressed
    
    def _remove_filler_phrases_optimized(self, text: str) -> str:
        """Enhanced filler phrase removal to reduce verbosity while preserving meaning."""
        # Comprehensive filler patterns with smart replacements
        filler_replacements = [
            (r'\bin order to\b', 'to'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bin spite of the fact that\b', 'although'),
            (r'\bas a matter of fact\b', ''),
            (r'\bfor the most part\b', 'mostly'),
            (r'\bat this point in time\b', 'now'),
            (r'\bwith regard to\b', 'regarding'),
            (r'\bin the event that\b', 'if'),
            (r'\bit should be noted that\b', ''),
            (r'\bit is important to note\b', ''),
            (r'\bplease note that\b', ''),
            (r'\bas mentioned earlier\b', ''),
        ]
        
        for pattern, replacement in filler_replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up resulting double spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_key_sentences_optimized(self, text: str, target_length: int) -> str:
        """Enhanced key sentence extraction with sophisticated scoring for optimal information retention."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return text[:target_length]
        
        # Score sentences using multiple criteria for importance
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Position scoring (beginning and end sentences often contain key information)
            if i == 0:
                score += 5  # First sentence often contains main topic
            elif i < 3:
                score += 3  # Early sentences establish context
            elif i >= len(sentences) - 2:
                score += 2  # Last sentences may contain conclusions
            
            # Content-based scoring for information density
            if re.search(r'\d', sentence):
                score += 2  # Numbers often contain key facts
            
            if re.search(r'\b(?:important|note|remember|must|should|required|mandatory)\b', sentence_lower):
                score += 3  # Emphasis words indicate importance
            
            if re.search(r'\b(?:question|answer|ask|help|contact|information)\b', sentence_lower):
                score += 2  # Interactive/helpful content
            
            if re.search(r'\b(?:however|therefore|thus|because|since|although)\b', sentence_lower):
                score += 1  # Logical connectors
            
            # Length penalty for very long or very short sentences
            if len(sentence) > 200:
                score -= 1
            elif len(sentence) < 20:
                score -= 2  # Very short sentences often incomplete
            
            scored_sentences.append((score, i, sentence))
        
        # Sort by score with positional preference for natural flow
        scored_sentences.sort(key=lambda x: (x[0], -abs(x[1] - len(sentences)//2)), reverse=True)
        
        # Select sentences until target length is reached
        selected_sentences = []
        current_length = 0
        selected_indices = []
        
        for score, idx, sentence in scored_sentences:
            sentence_length = len(sentence) + 1  # +1 for space
            if current_length + sentence_length <= target_length:
                selected_sentences.append((idx, sentence))
                selected_indices.append(idx)
                current_length += sentence_length
            elif current_length < target_length * 0.8:  # If well under target
                # Try to fit a truncated version
                remaining = target_length - current_length - 3  # -3 for "..."
                if remaining > 50:  # Only if meaningful content can fit
                    truncated = sentence[:remaining] + "..."
                    selected_sentences.append((idx, truncated))
                    break
        
        # Sort selected sentences by original order to maintain logical flow
        selected_sentences.sort(key=lambda x: x[0])
        
        result = ' '.join(sentence for _, sentence in selected_sentences)
        return result if result else text[:target_length]
    
    def _format_documents_optimized(self, selected_docs: List[Tuple[Document, DocumentRelevance]], 
                                  query_analysis: Dict[str, Any]) -> str:
        """Optimized document formatting with reduced overhead and improved organization."""
        if not selected_docs:
            return "No relevant information found in the APU knowledge base."
        
        # Efficient formatting with minimal overhead
        formatted_parts = []
        query_type = query_analysis["query_type"]
        
        # Context-aware header based on query type
        doc_count = len(selected_docs)
        
        if query_type == QueryType.ACADEMIC:
            header = f"Academic Information ({doc_count} sources):"
        elif query_type == QueryType.ADMINISTRATIVE:
            header = f"Administrative Information ({doc_count} sources):"
        elif query_type == QueryType.FINANCIAL:
            header = f"Financial Information ({doc_count} sources):"
        else:
            header = f"Relevant Information ({doc_count} sources):"
        
        formatted_parts.append(header)
        
        # Group documents by relevance for logical organization
        high_docs = [doc for doc, rel in selected_docs if rel == DocumentRelevance.HIGH]
        medium_docs = [doc for doc, rel in selected_docs if rel == DocumentRelevance.MEDIUM]
        low_docs = [doc for doc, rel in selected_docs if rel == DocumentRelevance.LOW]
        
        # Format high relevance documents first (most important content)
        if high_docs:
            for i, doc in enumerate(high_docs):
                formatted_parts.append(self._format_single_document(doc, i + 1, "Priority"))
        
        # Format medium relevance documents (supporting content)
        if medium_docs:
            start_idx = len(high_docs) + 1
            for i, doc in enumerate(medium_docs):
                formatted_parts.append(self._format_single_document(doc, start_idx + i, "Additional"))
        
        # Format low relevance documents if any (supplementary content)
        if low_docs:
            start_idx = len(high_docs) + len(medium_docs) + 1
            for i, doc in enumerate(low_docs):
                formatted_parts.append(self._format_single_document(doc, start_idx + i, "Supplementary"))
        
        return "\n\n".join(formatted_parts)
    
    def _format_single_document(self, doc: Document, index: int, category: str) -> str:
        """Format a single document with efficient structure and clear categorization."""
        is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
        
        if is_apu_kb:
            title = doc.metadata.get('page_title', f'Document {index}')
            if doc.metadata.get('is_faq', False):
                return f"[{category}] Q: {title}\nA: {doc.page_content}"
            else:
                return f"[{category}] {title}:\n{doc.page_content}"
        else:
            filename = doc.metadata.get('filename', 'Unknown')
            return f"[{category}] Source {index} ({filename}):\n{doc.page_content}"
    
    def get_processing_stats(self) -> Dict:
        """Retrieve processing statistics for performance monitoring and optimization."""
        return self._processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics for fresh monitoring periods."""
        self._processing_stats = {
            "total_processed": 0,
            "avg_context_utilization": 0.0,
            "compression_saved": 0
        }