"""
Advanced topic extraction module using spaCy NLP for robust topic identification.
Implements best practices for word boundary detection and semantic understanding.
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import spacy
from spacy.tokens import Doc
from config import config

logger = logging.getLogger("Sara")


class SemanticTopicExtractor:
    """
    Advanced topic extractor using spaCy NLP for accurate topic identification.
    
    Features:
    - Proper word boundary detection using NLP tokenization
    - Semantic similarity for topic matching
    - Entity recognition for university-specific terms
    - Confidence scoring for topic relevance
    """
    
    def __init__(self):
        """Initialize the semantic topic extractor."""
        try:
            # Load spaCy model - use configured model with fallbacks
            primary_model = config.SEMANTIC_MODEL
            fallback_models = ['en_core_web_md', 'en_core_web_sm']
            model_names = [primary_model] + [m for m in fallback_models if m != primary_model]
            self.nlp = None
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except IOError:
                    continue
                    
            if self.nlp is None:
                logger.warning("No spaCy model found. Topic extraction will use fallback method.")
                self.initialized = False
                return
                
            # Load topic definitions dynamically
            self.topic_definitions = self._load_topic_definitions()
            
            # Pre-compute keyword vectors for semantic similarity
            self._precompute_topic_vectors()
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize SemanticTopicExtractor: {e}")
            self.initialized = False
    
    def _load_topic_definitions(self) -> Dict:
        """Load topic definitions from externalized configuration."""
        try:
            from nlp_config_loader import config_loader
            
            config_topics = config_loader.get_topic_definitions()
            definitions = {}
            
            for topic, topic_config in config_topics.items():
                definitions[topic] = {
                    'keywords': topic_config.get('keywords', []),
                    'entities': topic_config.get('entities', ['ORG']),
                    'semantic_threshold': topic_config.get('confidence_threshold', config.TOPIC_EXTRACTOR_SEMANTIC_THRESHOLD),
                    'pos_tags': ['NOUN', 'VERB']
                }
            
            logger.info(f"Loaded {len(definitions)} topic definitions from configuration")
            return definitions
            
        except Exception as e:
            logger.error(f"Failed to load topic configuration: {e}")
            return self._get_fallback_definitions()
    
    def _get_fallback_definitions(self) -> Dict:
        """Minimal fallback topic definitions."""
        return {
            'general': {
                'keywords': ['help', 'support', 'assistance'],
                'entities': ['ORG'],
                'semantic_threshold': 0.5,
                'pos_tags': ['NOUN']
            }
        }
    
    def _precompute_topic_vectors(self):
        """Pre-compute spaCy vectors for topic keywords for efficient similarity matching."""
        self.topic_vectors = {}
        
        if not self.nlp.vocab.vectors_length:
            logger.info("spaCy model has no word vectors. Using token-based matching only.")
            return
            
        for topic, definition in self.topic_definitions.items():
            vectors = []
            for keyword in definition['keywords']:
                doc = self.nlp(keyword)
                if doc.has_vector and doc.vector_norm > 0:
                    vectors.append(doc.vector)
            
            if vectors:
                # Average the vectors for the topic
                import numpy as np
                self.topic_vectors[topic] = np.mean(vectors, axis=0)
    
    def extract_topics(self, query: str, confidence_threshold: float = None) -> List[Tuple[str, float]]:
        """
        Extract topics from query using advanced NLP techniques.
        
        Args:
            query: User query text
            confidence_threshold: Minimum confidence for topic inclusion
            
        Returns:
            List of (topic, confidence_score) tuples sorted by confidence
        """
        if not self.initialized:
            return self._fallback_extraction(query)
        
        if confidence_threshold is None:
            confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        try:
            # Process query with spaCy
            doc = self.nlp(query)
            topic_scores = {}
            
            # Method 1: Token-based matching with proper word boundaries
            for topic, definition in self.topic_definitions.items():
                score = self._calculate_token_score(doc, definition)
                if score > 0:
                    topic_scores[topic] = max(topic_scores.get(topic, 0), score)
            
            # Method 2: Semantic similarity (if vectors available)
            if hasattr(self, 'topic_vectors') and self.topic_vectors and doc.has_vector:
                semantic_scores = self._calculate_semantic_scores(doc)
                for topic, score in semantic_scores.items():
                    topic_scores[topic] = max(topic_scores.get(topic, 0), score)
            
            # Method 3: Named Entity Recognition boost
            entity_scores = self._calculate_entity_scores(doc)
            for topic, score in entity_scores.items():
                topic_scores[topic] = max(topic_scores.get(topic, 0), score * config.TOPIC_EXTRACTOR_ENTITY_BOOST)
            
            # Filter by confidence threshold and sort
            filtered_topics = [
                (topic, score) for topic, score in topic_scores.items()
                if score >= confidence_threshold
            ]
            
            return sorted(filtered_topics, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in semantic topic extraction: {e}")
            return self._fallback_extraction(query)
    
    def _calculate_token_score(self, doc: Doc, definition: Dict) -> float:
        """Calculate topic score based on token matching with proper word boundaries."""
        total_score = 0.0
        query_tokens = {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct}
        
        for keyword in definition['keywords']:
            keyword_doc = self.nlp(keyword)
            keyword_lemma = keyword_doc[0].lemma_.lower()
            
            if keyword_lemma in query_tokens:
                total_score += config.TOPIC_EXTRACTOR_EXACT_MATCH_SCORE
                continue
                
            for token in doc:
                if (not token.is_stop and not token.is_punct and 
                    keyword_lemma in token.lemma_.lower() and 
                    len(keyword_lemma) > config.TOPIC_EXTRACTOR_MIN_KEYWORD_LENGTH):
                    total_score += config.TOPIC_EXTRACTOR_PARTIAL_MATCH_SCORE
                    break
        
        return total_score / len(definition['keywords']) if definition['keywords'] else 0.0
    
    def _calculate_semantic_scores(self, doc: Doc) -> Dict[str, float]:
        """Calculate topic scores using semantic similarity."""
        scores = {}
        
        for topic, topic_vector in self.topic_vectors.items():
            if doc.has_vector and doc.vector_norm > 0:
                import numpy as np
                
                # Calculate cosine similarity
                similarity = np.dot(doc.vector, topic_vector) / (
                    np.linalg.norm(doc.vector) * np.linalg.norm(topic_vector)
                )
                
                # Apply topic-specific threshold
                threshold = self.topic_definitions[topic]['semantic_threshold']
                if similarity >= threshold:
                    scores[topic] = float(similarity)
        
        return scores
    
    def _calculate_entity_scores(self, doc: Doc) -> Dict[str, float]:
        """Calculate topic scores based on named entities."""
        scores = {}
        entities = {ent.label_ for ent in doc.ents}
        
        for topic, definition in self.topic_definitions.items():
            entity_overlap = entities.intersection(set(definition['entities']))
            if entity_overlap:
                scores[topic] = len(entity_overlap) / len(definition['entities'])
        
        return scores
    
    def _fallback_extraction(self, query: str) -> List[Tuple[str, float]]:
        """Fallback method using regex word boundaries when spaCy is unavailable."""
        query_lower = query.lower()
        topic_scores = []
        
        for topic, definition in self.topic_definitions.items():
            score = 0.0
            for keyword in definition['keywords']:
                # Use word boundary regex to avoid false positives
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    score += 1.0
            
            if score > 0:
                normalized_score = score / len(definition['keywords'])
                topic_scores.append((topic, normalized_score))
        
        return sorted(topic_scores, key=lambda x: x[1], reverse=True)
    
    @lru_cache(maxsize=128)
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a specific topic (cached for performance)."""
        return self.topic_definitions.get(topic, {}).get('keywords', [])
    
    def is_healthy(self) -> bool:
        """Check if the topic extractor is functioning properly."""
        return self.initialized and self.nlp is not None
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the topic extractor."""
        return {
            'initialized': self.initialized,
            'model_loaded': self.nlp is not None,
            'has_vectors': (self.nlp.vocab.vectors_length > 0) if self.nlp else False,
            'topic_count': len(self.topic_definitions),
            'vector_topics': len(getattr(self, 'topic_vectors', {}))
        }