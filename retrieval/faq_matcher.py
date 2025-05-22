"""
FAQ matching for direct question-answer retrieval.
"""

import logging
from typing import Dict, Any
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

logger = logging.getLogger("CustomRAG")

class FAQMatcher:
    """Specialized matcher for FAQ content in the APU knowledge base."""
    
    def __init__(self, vector_store):
        """Initialize the FAQ matcher."""
        self.vector_store = vector_store
    
    def match_faq(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to find a direct FAQ match for the query.
        
        Args:
            query_analysis: Query analysis or dictionary with original_query
            
        Returns:
            Dictionary with match result or None if no good match found
        """
        # Extract query from analysis or get original_query
        if isinstance(query_analysis, dict):
            if "original_query" in query_analysis:
                query = query_analysis["original_query"]
            else:
                return None
        else:
            return None
        
        # Check if query is a question (ends with ?)
        is_question = query.strip().endswith('?')
        
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or not all_docs.get('documents'):
                return None
                
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            # Track best match
            best_match = None
            best_score = 0
            
            # Process each document
            for i, doc_text in enumerate(documents):
                if i >= len(metadatas):
                    continue
                    
                metadata = metadatas[i]
                
                # Only consider APU KB pages that are FAQs
                if metadata.get('content_type') != 'apu_kb_page' or not metadata.get('is_faq', False):
                    continue
                
                # Get the page title (which should be the question)
                title = metadata.get('page_title', '')
                
                # Skip if no title
                if not title:
                    continue
                
                # Calculate similarity between query and title
                similarity = self._calculate_faq_similarity(query, title)
                
                # Boost for question-to-question matching
                if is_question and title.strip().endswith('?'):
                    similarity *= 1.2
                
                # Also check content for exact matches to the query
                content_match = 0
                if query.lower() in doc_text.lower():
                    content_match = 0.3
                
                # Combined score
                score = similarity + content_match
                
                # Update best match if better
                if score > best_score and score > 0.5:  # Threshold for considering a match
                    best_score = score
                    best_match = {
                        "document": Document(
                            page_content=doc_text,
                            metadata=metadata
                        ),
                        "match_score": score
                    }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in FAQ matching: {e}")
            return None
    
    def _calculate_faq_similarity(self, query: str, title: str) -> float:
        """
        Calculate the similarity between a query and an FAQ title with enhanced financial question matching.
        
        Args:
            query: User query
            title: FAQ title
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize
        query = query.lower().strip().rstrip('?')
        title = title.lower().strip().rstrip('?')
        
        # Check for exact match
        if query == title:
            return 1.0
        
        # Check for financial keywords match
        financial_keywords = [
            "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
            "receipt", "outstanding", "due", "overdue", "installment",
            "scholarship", "loan", "charge", "refund", "deposit"
        ]
        
        query_has_financial = any(kw in query for kw in financial_keywords)
        title_has_financial = any(kw in title for kw in financial_keywords)
        
        # Apply finance-specific boosting
        finance_boost = 0
        if query_has_financial and title_has_financial:
            finance_boost = 0.3
        
        # Check for substring match
        if query in title or title in query:
            # Calculate relative length ratio for boosting based on how close in length they are
            length_ratio = min(len(query), len(title)) / max(len(query), len(title))
            substring_score = 0.8 * length_ratio
            return substring_score + finance_boost
        
        # Tokenize
        query_tokens = set(word_tokenize(query))
        title_tokens = set(word_tokenize(title))
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        query_tokens = {token for token in query_tokens if token not in stop_words}
        title_tokens = {token for token in title_tokens if token not in stop_words}
        
        # Calculate Jaccard similarity
        intersection = len(query_tokens.intersection(title_tokens))
        union = len(query_tokens.union(title_tokens))
        
        jaccard = intersection / union if union > 0 else 0
        
        # Check for overlapping n-grams (phrases)
        ngram_match = self._check_ngram_overlap(query, title)
        
        # Combine scores (weighted)
        return (jaccard * 0.6) + (ngram_match * 0.2) + finance_boost
    
    def _check_ngram_overlap(self, text1: str, text2: str) -> float:
        """
        Check for overlapping n-grams between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score between 0 and 1
        """
        # Tokenize
        tokens1 = word_tokenize(text1)
        tokens2 = word_tokenize(text2)
        
        # Generate n-grams (bigrams and trigrams)
        bigrams1 = set(ngrams(tokens1, 2)) if len(tokens1) >= 2 else set()
        bigrams2 = set(ngrams(tokens2, 2)) if len(tokens2) >= 2 else set()
        
        trigrams1 = set(ngrams(tokens1, 3)) if len(tokens1) >= 3 else set()
        trigrams2 = set(ngrams(tokens2, 3)) if len(tokens2) >= 3 else set()
        
        # Calculate overlap
        bigram_overlap = len(bigrams1.intersection(bigrams2)) / max(1, min(len(bigrams1), len(bigrams2)))
        trigram_overlap = len(trigrams1.intersection(trigrams2)) / max(1, min(len(trigrams1), len(trigrams2)))
        
        # Combine (trigrams are stronger indicators of similarity)
        combined_overlap = (bigram_overlap * 0.4) + (trigram_overlap * 0.6)
        
        return combined_overlap