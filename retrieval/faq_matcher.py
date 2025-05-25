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
                
                # Consider all APU KB pages, not just those marked as FAQs
                # This is a key change to improve recall
                if metadata.get('content_type') != 'apu_kb_page':
                    continue
                
                # Get the page title
                title = metadata.get('page_title', '')
                
                # Skip if no title
                if not title:
                    continue
                
                # Calculate similarity between query and title
                similarity = self._calculate_faq_similarity(query, title)
                
                # Boost for question-to-question matching
                if is_question and title.strip().endswith('?'):
                    similarity *= 1.2
                
                # Also check content for keyword matches to the query
                content_match = self._calculate_content_match(query, doc_text)
                
                # Combined score
                score = similarity + content_match
                
                # Update best match if better
                # Lowered threshold from 0.5 to 0.4 to improve recall
                if score > best_score and score > 0.4:
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
    
    def _calculate_content_match(self, query: str, content: str) -> float:
        """
        Calculate content match score based on keyword presence.
        
        Args:
            query: User query
            content: Document content
            
        Returns:
            Content match score between 0 and 0.5
        """
        # Normalize
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Check for exact phrase match
        if query_lower in content_lower:
            return 0.5
        
        # Extract keywords from query
        query_tokens = word_tokenize(query_lower)
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in query_tokens if token not in stop_words and len(token) > 2]
        
        # Count keyword matches
        matches = 0
        for keyword in keywords:
            if keyword in content_lower:
                matches += 1
        
        # Calculate score based on proportion of keywords found
        if not keywords:
            return 0
        
        return min(0.4, (matches / len(keywords)) * 0.4)
    
    def _calculate_faq_similarity(self, query: str, title: str) -> float:
        """
        Calculate the similarity between a query and an FAQ title with enhanced matching.
        
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
        
        # Check for domain-specific keywords match
        domain_keywords = [
            # Financial terms
            "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
            "receipt", "outstanding", "due", "overdue", "installment",
            "scholarship", "loan", "charge", "refund", "deposit",
            # Medical/insurance terms
            "medical", "insurance", "health", "card", "coverage", "claim",
            "collect", "pickup", "pick up", "pick-up", "counter", "office",
            # Administrative terms
            "visa", "passport", "document", "form", "application", "submit",
            "register", "registration", "enroll", "enrollment"
        ]
        
        query_has_domain_term = any(kw in query for kw in domain_keywords)
        title_has_domain_term = any(kw in title for kw in domain_keywords)
        
        # Apply domain-specific boosting
        domain_boost = 0
        if query_has_domain_term and title_has_domain_term:
            domain_boost = 0.3
        
        # Check for substring match (more lenient)
        if query in title or title in query:
            # Calculate relative length ratio for boosting based on how close in length they are
            length_ratio = min(len(query), len(title)) / max(len(query), len(title))
            substring_score = 0.8 * length_ratio
            return substring_score + domain_boost
        
        # Check for partial matches (new)
        query_parts = query.split()
        title_parts = title.split()
        
        # Check if most words in the query appear in the title
        common_words = set(query_parts).intersection(set(title_parts))
        if len(common_words) >= len(query_parts) * 0.7:  # 70% of query words appear in title
            partial_score = 0.7 * (len(common_words) / len(query_parts))
            return partial_score + domain_boost
        
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
        return (jaccard * 0.6) + (ngram_match * 0.2) + domain_boost
    
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