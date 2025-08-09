"""
Advanced reranking system for improved RAG accuracy using cross-encoder models.
Based on 2025 best practices for production RAG systems.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import numpy as np

logger = logging.getLogger("Sara")

class AdvancedReranker:
    """
    Advanced reranking system that combines multiple scoring methods
    to improve retrieval accuracy, inspired by NotebookLM's approach.
    """
    
    def __init__(self):
        """Initialize the reranker with multiple scoring strategies."""
        self.initialized = False
        self.cross_encoder = None
        self._init_models()
    
    def _init_models(self):
        """Initialize reranking models (lazy loading for performance)."""
        try:
            # Try to import sentence-transformers for cross-encoder reranking
            from sentence_transformers import CrossEncoder
            
            # Use a lightweight but effective cross-encoder model
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.cross_encoder = CrossEncoder(model_name)
            self.initialized = True
            logger.info(f"Initialized cross-encoder reranker with model: {model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers not available for cross-encoder reranking")
            self.initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.initialized = False
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents using multiple scoring methods for better accuracy.
        
        Args:
            query: The search query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents in order of relevance
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents  # No need to rerank if we have fewer documents
        
        # Combine multiple scoring methods
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_composite_score(query, doc)
            scored_docs.append((doc, score))
        
        # Sort by composite score (higher is better)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents
        reranked = [doc for doc, score in scored_docs[:top_k]]
        
        logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked)}")
        return reranked
    
    def _calculate_composite_score(self, query: str, document: Document) -> float:
        """Calculate composite relevance score using multiple methods."""
        scores = []
        weights = []
        
        # 1. Cross-encoder score (most important if available)
        if self.initialized and self.cross_encoder:
            try:
                cross_score = self.cross_encoder.predict([[query, document.page_content]])[0]
                scores.append(float(cross_score))
                weights.append(0.6)  # High weight for cross-encoder
            except Exception as e:
                logger.debug(f"Cross-encoder scoring failed: {e}")
        
        # 2. Keyword overlap score
        keyword_score = self._calculate_keyword_overlap(query, document.page_content)
        scores.append(keyword_score)
        weights.append(0.2)
        
        # 3. Context relevance score (APU-specific)
        context_score = self._calculate_context_relevance(query, document)
        scores.append(context_score)
        weights.append(0.15)
        
        # 4. Document quality score
        quality_score = self._calculate_document_quality(document)
        scores.append(quality_score)
        weights.append(0.05)
        
        # Calculate weighted average
        if scores and weights:
            composite_score = np.average(scores, weights=weights)
        else:
            composite_score = 0.0
        
        return float(composite_score)
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap score between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_context_relevance(self, query: str, document: Document) -> float:
        """Calculate context-specific relevance for APU documents."""
        score = 0.0
        query_lower = query.lower()
        content_lower = document.page_content.lower()
        
        # APU-specific relevance signals
        relevance_patterns = {
            # Login/Authentication queries
            'login': ['sign in', 'authenticate', 'apkey', 'password', 'access'],
            'apspace': ['student portal', 'apspace', 'dashboard'],
            'trouble': ['unable', 'cannot', 'problem', 'issue', 'error'],
            
            # Academic queries
            'exam': ['examination', 'docket', 'schedule', 'result'],
            'fee': ['payment', 'tuition', 'cost', 'outstanding'],
            'visa': ['student pass', 'immigration', 'permit'],
            
            # Service queries
            'library': ['book', 'resource', 'database', 'research'],
            'room': ['classroom', 'discussion', 'booking', 'reservation'],
        }
        
        # Check for pattern matches
        for key_concept, related_terms in relevance_patterns.items():
            if key_concept in query_lower:
                for term in related_terms:
                    if term in content_lower:
                        score += 0.1
        
        # Boost score for documents from relevant space
        space_name = document.metadata.get('space_name', '').lower()
        context_header = document.metadata.get('context_header', '').lower()
        
        # Space-specific boosts
        space_boosts = {
            'itsm': ['login', 'apspace', 'apkey', 'password', 'technical'],
            'lib': ['library', 'book', 'database', 'research'],
            'visa': ['visa', 'student pass', 'immigration'],
            'bur': ['fee', 'payment', 'tuition', 'cost'],
            'aa': ['academic', 'exam', 'result', 'docket'],
            'lno': ['room', 'residence', 'accommodation']
        }
        
        for space, boost_terms in space_boosts.items():
            if space in space_name:
                for term in boost_terms:
                    if term in query_lower:
                        score += 0.2
                        break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_document_quality(self, document: Document) -> float:
        """Calculate document quality score based on content characteristics."""
        content = document.page_content
        score = 0.5  # Base score
        
        # Prefer documents with structured content
        if 'Step' in content or re.search(r'\d+\.', content):
            score += 0.2
        
        # Prefer documents with URLs (more informative)
        if 'http' in content:
            score += 0.1
        
        # Prefer FAQ-style content
        if document.metadata.get('is_faq', False):
            score += 0.15
        
        # Prefer documents with good length (not too short, not too long)
        content_length = len(content.split())
        if 50 <= content_length <= 300:
            score += 0.1
        elif content_length < 20:
            score -= 0.2
        
        return min(score, 1.0)  # Cap at 1.0


class LightweightReranker:
    """Lightweight reranker for systems without sentence-transformers."""
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Simple reranking based on keyword matching and metadata."""
        if not documents or len(documents) <= top_k:
            return documents
        
        scored_docs = []
        for doc in documents:
            score = self._simple_relevance_score(query, doc)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
    
    def _simple_relevance_score(self, query: str, document: Document) -> float:
        """Simple relevance scoring without external dependencies."""
        query_words = query.lower().split()
        content_words = document.page_content.lower().split()
        
        # Keyword overlap
        matches = sum(1 for word in query_words if word in content_words)
        base_score = matches / len(query_words) if query_words else 0
        
        # Boost for FAQ documents
        if document.metadata.get('is_faq', False):
            base_score += 0.2
        
        # Boost for documents with structured content
        if 'Step' in document.page_content:
            base_score += 0.1
        
        return base_score