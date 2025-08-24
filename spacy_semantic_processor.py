"""
spaCy-based semantic processor following 2025 best practices.
Simple, reliable, and production-ready solution.
"""

import logging
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import warnings
import spacy
from spacy.tokens import Doc, Token
import re
import time
from config import config

# Suppress spaCy similarity warnings for small models without word vectors
warnings.filterwarnings("ignore", message=".*has no word vectors loaded.*", category=UserWarning)

logger = logging.getLogger("Sara")

class SpacySemanticProcessor:
    """
    Clean spaCy-based semantic processor for query understanding and expansion.
    Based on 2025 best practices for production RAG systems.
    """
    
    def __init__(self):
        """Initialize spaCy model and semantic resources."""
        try:
            # Load spaCy model
            model_name = getattr(config, 'SEMANTIC_MODEL', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            
            # Initialize domain-specific vocabulary
            self._setup_domain_vocabulary()
            
            # Setup grammar correction patterns
            self._setup_grammar_patterns()
            
            # Initialize monitoring
            self.error_count = 0
            self.max_errors = getattr(config, 'SEMANTIC_ERROR_THRESHOLD', 5)
            self.fallback_enabled = True
            self.stats = {
                'queries_processed': 0,
                'errors_encountered': 0,
                'average_processing_time': 0.0,
                'last_error': None
            }
            
            self.initialized = True
            logger.info("spaCy semantic processor initialized with monitoring")
            
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize spaCy processor: {e}")
            self.initialized = False
    
    def _setup_domain_vocabulary(self):
        """Setup APU-specific domain vocabulary for semantic matching."""
        self.domain_clusters = {
            # Time/Schedule related
            'temporal': {
                'core': ['hours', 'time', 'schedule', 'open', 'close', 'operates', 'available', 'operation'],
                'expansions': ['operating hours', 'operation time', 'working hours', 'business hours', 'opening hours', 'closing hours', 'schedule', 'timing', 'availability'],
                'patterns': ['when', 'what time', 'schedule', 'hours', 'open', 'close', 'opens', 'closes']
            },
            
            # Authentication/Login related  
            'authentication': {
                'core': ['password', 'login', 'apspace', 'apkey', 'access', 'signin'],
                'expansions': ['student portal', 'authentication', 'credentials', 'account access', 'sign in'],
                'patterns': ['forgot', 'reset', 'recover', 'cannot', 'unable']
            },
            
            # Academic related
            'academic': {
                'core': ['course', 'module', 'class', 'subject', 'program', 'degree'],
                'expansions': ['academic program', 'study', 'curriculum', 'education'],
                'patterns': ['enroll', 'register', 'study', 'learn']
            },
            
            # Administrative related
            'administrative': {
                'core': ['application', 'form', 'procedure', 'process', 'registration'],
                'expansions': ['admin process', 'office procedure', 'documentation'],
                'patterns': ['apply', 'submit', 'request', 'process']
            },
            
            # Financial related
            'financial': {
                'core': ['fee', 'payment', 'cost', 'tuition', 'scholarship'],
                'expansions': ['financial aid', 'payment process', 'fee structure'],
                'patterns': ['pay', 'transfer', 'fund', 'charge']
            }
        }
    
    def _setup_grammar_patterns(self):
        """Setup common grammar correction patterns."""
        self.grammar_corrections = {
            # Time-related corrections for "closes"
            r'\bwhen\s+does\s+(.*?)\s+closes\b': r'when does \1 close',
            r'\bwhen\s+do\s+(.*?)\s+closes\b': r'when do \1 close',
            r'\bwhen\s+is\s+(.*?)\s+closes\b': r'when does \1 close',
            r'\b(library|office|center)\s+closes\b': r'\1 close',
            
            # Time-related corrections for "open" (fix asymmetry)
            r'\bwhen\s+(the\s+)?(library|office|center)\s+open\b': r'when does the \2 open',
            r'\bwhat\s+time\s+(.*?)\s+open\b': r'what time does \1 open',
            r'\bwhen\s+is\s+(.*?)\s+open\b': r'when does \1 open',
            
            # Login-related corrections
            r'\bcannot\s+login\b': r'cannot log in',
            r'\bunable\s+login\b': r'unable to log in',
            r'\bforgot\s+(my\s+)?apspace\s+password\b': r'forgot APSpace password',
        }
    
    def correct_grammar(self, query: str) -> str:
        """Apply grammar corrections to query."""
        if not self.initialized:
            return query
            
        corrected = query
        for pattern, replacement in self.grammar_corrections.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def semantic_query_expansion(self, query: str, max_expansions: int = None) -> List[str]:
        """
        Expand query using spaCy's semantic understanding with monitoring.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of semantically expanded queries
        """
        if not self.initialized:
            return [query]
        
        # Check if we've exceeded error threshold
        if self.error_count >= self.max_errors:
            logger.warning(f"spaCy processor disabled due to {self.error_count} errors")
            return [query]
        
        # Use config setting if not specified
        if max_expansions is None:
            max_expansions = getattr(config, 'SEMANTIC_EXPANSION_LIMIT', 5)
        
        start_time = time.time()
        
        try:
            # Start with corrected query
            corrected_query = self.correct_grammar(query)
            expanded_queries = [corrected_query]
            
            # Process with spaCy
            doc = self.nlp(corrected_query)
            
            # Domain-based expansion
            domain_expansions = self._get_domain_expansions(doc)
            expanded_queries.extend(domain_expansions)
            
            # Lemma-based expansion
            lemma_expansions = self._get_lemma_expansions(doc)
            expanded_queries.extend(lemma_expansions)
            
            # Entity-based expansion
            entity_expansions = self._get_entity_expansions(doc)
            expanded_queries.extend(entity_expansions)
            
            # Remove duplicates and limit
            unique_expansions = []
            seen = set()
            for exp in expanded_queries:
                if exp.lower() not in seen:
                    unique_expansions.append(exp)
                    seen.add(exp.lower())
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return unique_expansions[:max_expansions]
            
        except Exception as e:
            self.error_count += 1
            self.stats['errors_encountered'] += 1
            self.stats['last_error'] = str(e)
            logger.error(f"spaCy expansion error {self.error_count}/{self.max_errors}: {e}")
            return [query]
    
    def _get_domain_expansions(self, doc: Doc) -> List[str]:
        """Generate domain-specific expansions."""
        expansions = []
        query_text = doc.text.lower()
        
        for domain, vocab in self.domain_clusters.items():
            # Check if query matches this domain
            core_matches = sum(1 for term in vocab['core'] if term in query_text)
            pattern_matches = sum(1 for pattern in vocab['patterns'] if pattern in query_text)
            
            if core_matches > 0 or pattern_matches > 0:
                # Add domain-specific expansions
                for expansion in vocab['expansions'][:2]:  # Limit to avoid explosion
                    if expansion not in query_text:
                        expansions.append(f"{query_text} {expansion}")
                        expansions.append(f"{expansion} {query_text}")
                
                # Special handling for temporal queries - ensure open/close symmetry (prioritize these)
                if domain == 'temporal':
                    if 'close' in query_text and 'open' not in query_text:
                        # For close queries, also generate open-related expansions (INSERT AT BEGINNING)
                        expansions.insert(0, query_text.replace('close', 'open'))
                        expansions.insert(1, f"{query_text} open hours")
                    elif 'open' in query_text and 'close' not in query_text:
                        # For open queries, also generate close-related expansions (INSERT AT BEGINNING)
                        expansions.insert(0, query_text.replace('open', 'close'))
                        expansions.insert(1, f"{query_text} close hours")
        
        return expansions
    
    def _get_lemma_expansions(self, doc: Doc) -> List[str]:
        """Generate lemma-based expansions for better matching."""
        expansions = []
        
        # Create lemmatized version
        lemmas = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.lemma_ != token.text:
                lemmas.append(token.lemma_)
            else:
                lemmas.append(token.text)
        
        lemmatized = ' '.join(lemmas)
        if lemmatized != doc.text:
            expansions.append(lemmatized)
        
        return expansions
    
    def _get_entity_expansions(self, doc: Doc) -> List[str]:
        """Generate expansions based on named entities."""
        expansions = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'FACILITY', 'GPE']:
                # Add variations for organizational entities
                if 'library' in ent.text.lower():
                    expansions.append(doc.text.replace(ent.text, 'learning center'))
                    expansions.append(doc.text.replace(ent.text, 'resource center'))
        
        return expansions
    
    def calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate semantic similarity between two queries using spaCy.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.initialized:
            return 0.0
        
        try:
            doc1 = self.nlp(query1)
            doc2 = self.nlp(query2)
            
            # Use spaCy's built-in similarity
            similarity = doc1.similarity(doc2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def extract_intent_and_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract query intent and entities using spaCy.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with intent and entities
        """
        if not self.initialized:
            return {'intent': 'unknown', 'entities': [], 'domain': 'general'}
        
        doc = self.nlp(query)
        
        # Extract entities
        entities = [
            {'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
            for ent in doc.ents
        ]
        
        # Classify intent based on linguistic patterns
        intent = self._classify_intent(doc)
        
        # Determine domain
        domain = self._classify_domain(doc)
        
        return {
            'intent': intent,
            'entities': entities,
            'domain': domain,
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'dependencies': [(token.text, token.dep_, token.head.text) for token in doc]
        }
    
    def _classify_intent(self, doc: Doc) -> str:
        """Classify query intent based on linguistic patterns."""
        query_text = doc.text.lower()
        
        # Question word mapping
        question_intents = {
            'what': 'factual',
            'when': 'temporal',
            'where': 'locational', 
            'how': 'procedural',
            'why': 'explanatory',
            'who': 'entity_query',
            'which': 'comparative'
        }
        
        # Check for question words
        for token in doc:
            if token.text.lower() in question_intents:
                return question_intents[token.text.lower()]
        
        # Check for action verbs (procedural intent)
        action_patterns = ['reset', 'change', 'update', 'submit', 'apply', 'request']
        if any(pattern in query_text for pattern in action_patterns):
            return 'procedural'
        
        # Check for problem/issue patterns
        problem_patterns = ['forgot', 'cannot', 'unable', 'problem', 'issue', 'error']
        if any(pattern in query_text for pattern in problem_patterns):
            return 'troubleshooting'
        
        return 'informational'
    
    def _classify_domain(self, doc: Doc) -> str:
        """Classify query domain based on content."""
        query_text = doc.text.lower()
        
        domain_scores = {}
        for domain, vocab in self.domain_clusters.items():
            score = 0
            
            # Score based on core terms
            for term in vocab['core']:
                if term in query_text:
                    score += 2
            
            # Score based on patterns
            for pattern in vocab['patterns']:
                if pattern in query_text:
                    score += 1
            
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def rank_documents_by_similarity(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Rank documents by semantic similarity to query.
        
        Args:
            query: Search query
            documents: List of document texts
            
        Returns:
            List of (document, similarity_score) tuples sorted by similarity
        """
        if not self.initialized or not documents:
            return [(doc, 0.0) for doc in documents]
        
        try:
            query_doc = self.nlp(query)
            similarities = []
            
            for doc_text in documents:
                doc_doc = self.nlp(doc_text[:1000])  # Limit for performance
                similarity = query_doc.similarity(doc_doc)
                similarities.append((doc_text, float(similarity)))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
            
        except Exception as e:
            logger.warning(f"Document ranking failed: {e}")
            return [(doc, 0.0) for doc in documents]
    
    @lru_cache(maxsize=100)
    def get_word_synonyms(self, word: str) -> List[str]:
        """
        Get semantic synonyms for a word using spaCy's word vectors.
        
        Args:
            word: Input word
            
        Returns:
            List of semantically similar words
        """
        if not self.initialized:
            return []
        
        try:
            # This is a simplified version - in production you'd use 
            # a more comprehensive synonym database or word2vec model
            word_doc = self.nlp(word)
            
            # For now, return domain-specific synonyms
            for domain, vocab in self.domain_clusters.items():
                if word.lower() in vocab['core']:
                    return vocab['expansions'][:3]
            
            return []
            
        except Exception as e:
            logger.warning(f"Synonym extraction failed for '{word}': {e}")
            return []
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.stats['queries_processed'] += 1
        
        # Update average processing time
        total_queries = self.stats['queries_processed']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics and health information."""
        return {
            'initialized': self.initialized,
            'error_count': self.error_count,
            'max_errors': self.max_errors,
            'fallback_enabled': self.fallback_enabled,
            'stats': self.stats.copy(),
            'model_loaded': hasattr(self, 'nlp') and self.nlp is not None,
            'domain_clusters': len(self.domain_clusters) if hasattr(self, 'domain_clusters') else 0,
            'grammar_patterns': len(self.grammar_corrections) if hasattr(self, 'grammar_corrections') else 0
        }
    
    def reset_error_count(self):
        """Reset error count for recovery."""
        self.error_count = 0
        self.stats['last_error'] = None
        logger.info("spaCy processor error count reset")
    
    def is_healthy(self) -> bool:
        """Check if processor is healthy and operational."""
        return (
            self.initialized and 
            self.error_count < self.max_errors and
            hasattr(self, 'nlp') and 
            self.nlp is not None
        )