"""
FAQ matching for direct question-answer retrieval with enhanced keyword detection and ChromaDB compatibility.
Provides intelligent matching of user queries to FAQ content in the APU knowledge base.
"""

import logging
import time
import re
from typing import Dict, Any, Union, Set
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

logger = logging.getLogger("CustomRAG")

class FAQMatcher:
    """Specialized matcher for FAQ content in the APU knowledge base with enhanced similarity algorithms."""
    
    def __init__(self, vector_store):
        """Initialize the FAQ matcher with performance caching and optimized data structures."""
        self.vector_store = vector_store
        
        # Performance optimization caches
        self._compiled_patterns = {}
        self._stopwords_cache = None
        self._domain_keywords_cache = None
        
        # Initialize cached data for better performance
        self._init_caches()
    
    def _init_caches(self):
        """Initialize cached data structures for improved performance during matching operations."""
        try:
            # Cache NLTK stopwords for faster access
            self._stopwords_cache = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data not available
            self._stopwords_cache = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
            }
        
        # Cache domain-specific keywords for APU-related queries
        self._domain_keywords_cache = {
            # Financial and payment related terms
            'financial': {
                "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
                "receipt", "outstanding", "due", "overdue", "installment",
                "scholarship", "loan", "charge", "refund", "deposit", "tuition"
            },
            # Medical and insurance related terms
            'medical': {
                "medical", "insurance", "health", "card", "coverage", "claim",
                "collect", "pickup", "pick up", "pick-up", "counter", "office"
            },
            # Administrative and procedural terms
            'administrative': {
                "visa", "passport", "document", "form", "application", "submit",
                "register", "registration", "enroll", "enrollment", "docket",
                "change", "switch", "transfer", "modify", "update"
            },
            # Academic and course related terms
            'academic': {
                "course", "subject", "class", "exam", "examination", "test",
                "grade", "result", "transcript", "certificate", "attendance",
                "tutorial", "group", "timetable", "schedule", "lecturer"
            }
        }
    
    def match_faq(self, query_analysis: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        """
        Find the best FAQ match for a user query using comprehensive similarity analysis.
        
        Args:
            query_analysis: Query analysis dictionary containing original_query and metadata
            
        Returns:
            Dictionary with match result including document, score, and reason, or None if no match found
        """
        start_time = time.time()
        
        # Extract and validate query from analysis
        query = self._extract_query_from_analysis(query_analysis)
        if not query:
            return None
        
        # Pre-filter queries that are unlikely to have FAQ matches
        if not self._has_faq_keywords_inclusive(query):
            logger.debug(f"Query '{query}' lacks FAQ keywords, skipping FAQ matching")
            return None
        
        # Determine if query is formatted as a question
        is_question = query.strip().endswith('?')
        
        try:
            # Retrieve all documents from vector store with error handling
            all_docs = self._get_vector_store_documents()
            if not all_docs:
                return None
                
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            if not documents or not metadatas:
                logger.warning("No documents or metadata found in vector store")
                return None
            
            # Track best match across all documents
            best_match = None
            best_score = 0
            
            # Process each document for potential FAQ matching
            for i, doc_text in enumerate(documents):
                if i >= len(metadatas):
                    continue
                    
                metadata = metadatas[i]
                
                # Filter to only valid FAQ documents
                if not self._is_valid_faq_document(metadata):
                    continue
                
                # Extract page title for matching
                title = metadata.get('page_title', '').strip()
                if not title:
                    continue
                
                # Calculate comprehensive similarity score
                similarity_score = self._calculate_comprehensive_similarity(
                    query, title, doc_text, is_question
                )
                
                # Update best match using dynamic threshold
                threshold = self._get_dynamic_threshold(query, title)
                if similarity_score > best_score and similarity_score > threshold:
                    best_score = similarity_score
                    best_match = {
                        "document": Document(
                            page_content=doc_text,
                            metadata=metadata
                        ),
                        "match_score": similarity_score,
                        "match_reason": f"FAQ match - Title: {title[:50]}..."
                    }
            
            # Log performance metrics for optimization
            elapsed = time.time() - start_time
            if elapsed > 0.5:  # Log if taking more than 500ms
                logger.info(f"FAQ matching took {elapsed:.2f}s for query: {query[:30]}...")
            
            if best_match:
                logger.info(f"Found FAQ match with score {best_score:.3f} for: {query[:30]}...")
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in FAQ matching for query '{query}': {e}")
            return None
    
    def _extract_query_from_analysis(self, query_analysis: Dict[str, Any]) -> Union[str, None]:
        """Extract and validate query string from analysis dictionary with comprehensive error handling."""
        if not isinstance(query_analysis, dict):
            logger.warning("query_analysis is not a dictionary")
            return None
        
        if "original_query" not in query_analysis:
            logger.warning("query_analysis missing 'original_query'")
            return None
        
        query = query_analysis["original_query"]
        if not query or not isinstance(query, str) or len(query.strip()) < 2:
            logger.warning("Invalid or too short query")
            return None
        
        return query.strip()
    
    def _has_faq_keywords_inclusive(self, query: str) -> bool:
        """Comprehensive FAQ keyword detection using both specific terms and question patterns."""
        query_lower = query.lower()
        
        # Comprehensive FAQ keywords including common question patterns
        faq_keywords = {
            # Question words and interrogatives
            "how", "what", "where", "when", "why", "who", "which", "can", "could",
            "should", "would", "will", "do", "does", "did", "is", "are", "am",
            
            # Action and procedure words
            "collect", "get", "obtain", "apply", "register", "pay", "payment",
            "change", "switch", "transfer", "modify", "update", "request", "submit",
            "need", "want", "require", "allow", "permit", "eligible",
            
            # APU-specific institutional terms
            "fee", "insurance", "medical", "card", "exam", "docket", "visa",
            "form", "document", "procedure", "process", "requirement",
            "tutorial", "group", "class", "course", "subject", "timetable",
            "schedule", "lecturer", "attendance", "result", "grade",
            
            # General academic and administrative terms
            "student", "program", "degree", "certificate", "transcript",
            "registration", "enrollment", "application", "admission"
        }
        
        # Question patterns that indicate FAQ-like queries
        question_patterns = [
            r'\bcan\s+i\b',
            r'\bhow\s+(?:do|can)\s+i\b',
            r'\bwhere\s+(?:do|can)\s+i\b',
            r'\bwhat\s+(?:is|are)\b',
            r'\bis\s+it\s+possible\b',
            r'\bam\s+i\s+(?:able|allowed)\b'
        ]
        
        # Check for FAQ keywords in query
        query_words = set(query_lower.split())
        has_keywords = bool(query_words.intersection(faq_keywords))
        
        # Check for question patterns using regex
        has_question_pattern = any(re.search(pattern, query_lower) for pattern in question_patterns)
        
        # Accept query if either keywords or question patterns match
        result = has_keywords or has_question_pattern
        
        if not result:
            logger.debug(f"Query '{query}' rejected by FAQ filter - no keywords or question patterns")
        else:
            logger.debug(f"Query '{query}' accepted by FAQ filter")
        
        return result
    
    def _get_vector_store_documents(self) -> Union[Dict, None]:
        """Retrieve documents from vector store with comprehensive error handling."""
        try:
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                logger.warning("No documents found in vector store")
                return None
            return all_docs
        except Exception as e:
            logger.error(f"Error accessing vector store: {e}")
            return None
    
    def _is_valid_faq_document(self, metadata: Dict) -> bool:
        """Determine if document is valid for FAQ matching based on metadata flags."""
        # Accept APU knowledge base pages which include FAQ content
        if metadata.get('content_type') == 'apu_kb_page':
            return True
        
        # Accept explicitly marked FAQ documents
        if metadata.get('is_faq', False):
            return True
        
        return False
    
    def _get_dynamic_threshold(self, query: str, title: str) -> float:
        """Calculate dynamic similarity threshold based on query and title characteristics."""
        base_threshold = 0.3  # Lowered for better recall
        
        # Lower threshold for longer, more specific queries
        if len(query.split()) > 5:
            base_threshold -= 0.1
        
        # Lower threshold if query and title have similar lengths
        length_ratio = min(len(query), len(title)) / max(len(query), len(title), 1)
        if length_ratio > 0.7:
            base_threshold -= 0.05
        
        # Lower threshold for common FAQ question patterns
        if any(pattern in query.lower() for pattern in ['can i', 'how do i', 'where do i']):
            base_threshold -= 0.05
        
        # Ensure threshold remains within reasonable bounds
        return max(0.2, min(0.5, base_threshold))
    
    def _calculate_comprehensive_similarity(self, query: str, title: str, content: str, is_question: bool) -> float:
        """
        Calculate comprehensive similarity score combining multiple matching factors and strategies.
        """
        # Title similarity receives highest weight as it's most indicative
        title_score = self._calculate_enhanced_title_similarity(query, title)
        
        # Content matching provides additional context validation
        content_score = self._calculate_enhanced_content_match(query, content)
        
        # Domain-specific boosting for APU-related topics
        domain_boost = self._calculate_domain_boost(query, title, content)
        
        # Question format matching boost
        question_boost = 0.1 if (is_question and title.strip().endswith('?')) else 0
        
        # Special boost for tutorial and group related queries
        tutorial_boost = self._calculate_tutorial_boost(query, title, content)
        
        # Combine scores with optimized weights for best performance
        final_score = (
            title_score * 0.55 +      # Title matching is most important
            content_score * 0.25 +    # Content provides supporting context
            domain_boost * 0.1 +      # Domain relevance adds precision
            question_boost * 0.05 +   # Question format alignment
            tutorial_boost * 0.05     # Tutorial/group specific enhancement
        )
        
        return min(final_score, 1.0)  # Ensure score doesn't exceed maximum
    
    def _calculate_tutorial_boost(self, query: str, title: str, content: str) -> float:
        """Calculate special boost for tutorial and group related queries which are common at APU."""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Tutorial and group specific terminology
        tutorial_terms = ['tutorial', 'group', 'class', 'timetable', 'schedule', 'change', 'switch']
        
        query_has_tutorial = any(term in query_lower for term in tutorial_terms)
        title_has_tutorial = any(term in title_lower for term in tutorial_terms)
        content_has_tutorial = any(term in content_lower for term in tutorial_terms)
        
        if query_has_tutorial and (title_has_tutorial or content_has_tutorial):
            return 0.15
        
        return 0.0
    
    def _calculate_enhanced_title_similarity(self, query: str, title: str) -> float:
        """Enhanced title similarity calculation using multiple matching strategies for maximum accuracy."""
        # Normalize text by converting to lowercase and removing trailing question marks
        query_norm = query.lower().strip().rstrip('?')
        title_norm = title.lower().strip().rstrip('?')
        
        # Exact match receives highest score
        if query_norm == title_norm:
            return 1.0
        
        # Substring matching with length ratio consideration
        if query_norm in title_norm or title_norm in query_norm:
            length_ratio = min(len(query_norm), len(title_norm)) / max(len(query_norm), len(title_norm))
            return 0.8 * length_ratio
        
        # Enhanced word-level matching analysis
        query_words = self._tokenize_and_clean(query_norm)
        title_words = self._tokenize_and_clean(title_norm)
        
        if not query_words or not title_words:
            return 0.0
        
        # Calculate word overlap ratio for semantic similarity
        common_words = query_words.intersection(title_words)
        word_overlap_ratio = len(common_words) / len(query_words)
        
        # Boost for good word overlap with lowered threshold for better recall
        if word_overlap_ratio >= 0.6:  # 60% of query words found in title
            return 0.7 * word_overlap_ratio
        
        # Jaccard similarity for set-based comparison
        jaccard = len(common_words) / len(query_words.union(title_words))
        
        # N-gram similarity for sequence-based comparison
        ngram_score = self._calculate_ngram_similarity(query_norm, title_norm)
        
        # Combine Jaccard and n-gram scores with weighted average
        return (jaccard * 0.7) + (ngram_score * 0.3)
    
    def _calculate_enhanced_content_match(self, query: str, content: str) -> float:
        """Enhanced content matching with weighted keyword analysis for improved relevance scoring."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Exact phrase match receives highest content score
        if query_lower in content_lower:
            return 0.5
        
        # Extract and weight keywords based on importance
        query_keywords = self._extract_weighted_keywords(query_lower)
        if not query_keywords:
            return 0.0
        
        # Calculate weighted keyword matches in content
        total_weight = sum(query_keywords.values())
        matched_weight = 0
        
        for keyword, weight in query_keywords.items():
            if keyword in content_lower:
                matched_weight += weight
        
        # Normalize score by total weight
        keyword_score = matched_weight / total_weight if total_weight > 0 else 0
        
        return min(0.4, keyword_score * 0.4)  # Cap content score contribution
    
    def _extract_weighted_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords from text with importance weights based on domain relevance."""
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Filter tokens and assign importance weights
        weighted_keywords = {}
        
        for token in tokens:
            if len(token) <= 2 or token in self._stopwords_cache:
                continue
            
            # Assign base weight
            weight = 1.0
            
            # Higher weight for domain-specific terms
            if self._is_domain_keyword(token):
                weight = 2.0
            
            # Higher weight for action words commonly used in queries
            if token in {'collect', 'get', 'obtain', 'apply', 'register', 'pay', 'change', 'switch'}:
                weight = 1.5
            
            weighted_keywords[token] = weight
        
        return weighted_keywords
    
    def _is_domain_keyword(self, word: str) -> bool:
        """Check if word is a domain-specific keyword relevant to APU operations."""
        for domain_set in self._domain_keywords_cache.values():
            if word in domain_set:
                return True
        return False
    
    def _calculate_domain_boost(self, query: str, title: str, content: str) -> float:
        """Calculate domain-specific boosting score for queries matching APU operational domains."""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        boost_score = 0.0
        
        # Check alignment across each domain category
        for domain, keywords in self._domain_keywords_cache.items():
            query_has_domain = any(kw in query_lower for kw in keywords)
            title_has_domain = any(kw in title_lower for kw in keywords)
            content_has_domain = any(kw in content_lower for kw in keywords)
            
            if query_has_domain and (title_has_domain or content_has_domain):
                boost_score += 0.15  # Boost for domain alignment
        
        return min(boost_score, 0.3)  # Cap total domain boost
    
    def _tokenize_and_clean(self, text: str) -> Set[str]:
        """Tokenize text and remove stopwords for cleaner similarity comparison."""
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        return {
            token.lower() for token in tokens 
            if len(token) > 2 and token.lower() not in self._stopwords_cache
        }
    
    def _calculate_ngram_similarity(self, text1: str, text2: str) -> float:
        """Calculate n-gram similarity using bigram overlap for sequence-based matching."""
        try:
            tokens1 = word_tokenize(text1)
            tokens2 = word_tokenize(text2)
        except:
            tokens1 = text1.split()
            tokens2 = text2.split()
        
        if len(tokens1) < 2 or len(tokens2) < 2:
            return 0.0
        
        # Generate bigrams for sequence comparison
        bigrams1 = set(ngrams(tokens1, 2))
        bigrams2 = set(ngrams(tokens2, 2))
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        # Calculate bigram overlap ratio
        overlap = len(bigrams1.intersection(bigrams2))
        total = min(len(bigrams1), len(bigrams2))
        
        return overlap / total if total > 0 else 0.0
    
    # Legacy methods maintained for backward compatibility
    def _calculate_faq_similarity(self, query: str, title: str) -> float:
        """Legacy method - redirects to enhanced version for backward compatibility."""
        return self._calculate_enhanced_title_similarity(query, title)
    
    def _calculate_content_match(self, query: str, content: str) -> float:
        """Legacy method - redirects to enhanced version for backward compatibility."""
        return self._calculate_enhanced_content_match(query, content)
    
    def _check_ngram_overlap(self, text1: str, text2: str) -> float:
        """Legacy method - redirects to enhanced version for backward compatibility."""
        return self._calculate_ngram_similarity(text1, text2)