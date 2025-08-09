"""
Query decomposition for complex questions to improve retrieval accuracy.
Based on 2025 RAG best practices for handling multi-faceted queries.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

logger = logging.getLogger("Sara")

class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries for better retrieval.
    Follows 2025 best practices for production RAG systems.
    """
    
    def __init__(self):
        """Initialize the query decomposer."""
        self.conjunction_patterns = [
            r'\band\b', r'\bor\b', r'\bbut\b', r'\bhowever\b',
            r'\bwhile\b', r'\balso\b', r'\badditionally\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\bbesides\b'
        ]
        
        # APU-specific multi-topic patterns
        self.multi_topic_patterns = [
            # Academic + Financial
            (r'(?=.*(?:exam|grade|result))(?=.*(?:fee|payment|cost))', ['academic', 'financial']),
            # Academic + Administrative  
            (r'(?=.*(?:exam|course|assignment))(?=.*(?:visa|form|document))', ['academic', 'administrative']),
            # Multiple processes in one query
            (r'(?=.*(?:apply|application))(?=.*(?:register|registration))', ['application_process', 'registration_process']),
            # Login + specific service
            (r'(?=.*(?:login|sign in|access))(?=.*(?:apspace|library|system))', ['login_issues', 'service_access']),
        ]
    
    def should_decompose(self, query: str, query_analysis: Dict[str, Any]) -> bool:
        """
        Determine if a query should be decomposed.
        
        Args:
            query: The original query string
            query_analysis: Analysis from InputProcessor
            
        Returns:
            True if query should be decomposed
        """
        # Don't decompose simple queries
        if len(query.split()) < 8:
            return False
        
        # Check for conjunctions
        query_lower = query.lower()
        for pattern in self.conjunction_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for multi-topic patterns
        for pattern, _ in self.multi_topic_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for multiple question marks or question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        question_count = sum(1 for word in question_words if word in query_lower)
        if question_count >= 2:
            return True
        
        # Check for procedural complexity (multiple steps implied)
        step_indicators = ['first', 'then', 'next', 'after', 'before', 'step', 'process']
        step_count = sum(1 for indicator in step_indicators if indicator in query_lower)
        if step_count >= 2:
            return True
        
        return False
    
    def decompose_query(self, query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: The original query string
            query_analysis: Analysis from InputProcessor
            
        Returns:
            List of sub-query dictionaries with metadata
        """
        logger.info(f"Decomposing complex query: {query}")
        
        # Try different decomposition strategies
        sub_queries = []
        
        # Strategy 1: Split on conjunctions
        conjunction_splits = self._split_on_conjunctions(query)
        if len(conjunction_splits) > 1:
            for i, sub_query in enumerate(conjunction_splits):
                sub_queries.append({
                    'query': sub_query.strip(),
                    'type': 'conjunction_split',
                    'priority': 1.0 - (i * 0.1),  # First parts get higher priority
                    'original_position': i
                })
        
        # Strategy 2: Extract topic-specific questions
        topic_queries = self._extract_topic_queries(query)
        for topic_query in topic_queries:
            sub_queries.append(topic_query)
        
        # Strategy 3: Extract procedural steps
        step_queries = self._extract_procedural_steps(query)
        for step_query in step_queries:
            sub_queries.append(step_query)
        
        # If no decomposition worked, return the original query
        if not sub_queries:
            sub_queries = [{
                'query': query,
                'type': 'original',
                'priority': 1.0,
                'original_position': 0
            }]
        
        # Remove duplicates and very short queries
        sub_queries = self._deduplicate_and_filter(sub_queries)
        
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        return sub_queries
    
    def _split_on_conjunctions(self, query: str) -> List[str]:
        """Split query on conjunctions while preserving context."""
        # Split on major conjunctions but keep context
        split_patterns = [r'\band\b', r'\bbut\b', r'\bhowever\b', r'\bwhile\b']
        
        current_parts = [query]
        
        for pattern in split_patterns:
            new_parts = []
            for part in current_parts:
                splits = re.split(f'({pattern})', part, flags=re.IGNORECASE)
                if len(splits) > 1:
                    # Combine splits intelligently
                    for i in range(0, len(splits), 2):  # Skip the separators
                        if i < len(splits):
                            new_parts.append(splits[i])
                else:
                    new_parts.append(part)
            current_parts = new_parts
        
        # Filter out very short parts
        return [part.strip() for part in current_parts if len(part.strip()) > 10]
    
    def _extract_topic_queries(self, query: str) -> List[Dict[str, Any]]:
        """Extract topic-specific sub-queries."""
        sub_queries = []
        query_lower = query.lower()
        
        # APU-specific topic extraction
        topics = {
            'fees': ['fee', 'payment', 'cost', 'charge', 'outstanding', 'tuition'],
            'exam': ['exam', 'docket', 'test', 'assessment', 'result', 'grade'],
            'visa': ['visa', 'student pass', 'immigration', 'permit'],
            'login': ['login', 'sign in', 'access', 'apkey', 'password'],
            'library': ['library', 'book', 'database', 'research'],
            'accommodation': ['accommodation', 'housing', 'room', 'residence'],
            'registration': ['register', 'registration', 'enroll', 'enrollment'],
            'application': ['apply', 'application', 'form', 'submit']
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # If multiple topics detected, create focused sub-queries
        if len(detected_topics) > 1:
            for i, topic in enumerate(detected_topics):
                # Create a focused question for this topic
                topic_keywords = topics[topic]
                relevant_keywords = [kw for kw in topic_keywords if kw in query_lower]
                
                if relevant_keywords:
                    # Extract the part of the query related to this topic
                    topic_query = self._extract_topic_context(query, relevant_keywords)
                    if topic_query and len(topic_query) > 15:
                        sub_queries.append({
                            'query': topic_query,
                            'type': 'topic_focused',
                            'topic': topic,
                            'priority': 0.9 - (i * 0.1),
                            'original_position': i
                        })
        
        return sub_queries
    
    def _extract_topic_context(self, query: str, keywords: List[str]) -> str:
        """Extract the context around topic keywords."""
        sentences = re.split(r'[.!?]+', query)
        
        # Find sentences containing the keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        
        # Fallback: extract phrases around keywords
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                start = max(0, query_lower.find(keyword) - 50)
                end = min(len(query), query_lower.find(keyword) + len(keyword) + 50)
                context = query[start:end].strip()
                if len(context) > 15:
                    return context
        
        return ""
    
    def _extract_procedural_steps(self, query: str) -> List[Dict[str, Any]]:
        """Extract procedural steps from the query."""
        sub_queries = []
        
        # Look for step indicators
        step_patterns = [
            r'first[,\s]+(.+?)(?=\bthen\b|\bafter\b|\bnext\b|$)',
            r'then[,\s]+(.+?)(?=\bafter\b|\bnext\b|\bfinally\b|$)',
            r'next[,\s]+(.+?)(?=\bafter\b|\bthen\b|\bfinally\b|$)',
            r'after[,\s]+(.+?)(?=\bthen\b|\bnext\b|\bfinally\b|$)',
            r'finally[,\s]+(.+?)(?=\.$|$)'
        ]
        
        for i, pattern in enumerate(step_patterns):
            matches = re.findall(pattern, query, re.IGNORECASE | re.DOTALL)
            for match in matches:
                step_query = match.strip()
                if len(step_query) > 10:
                    sub_queries.append({
                        'query': step_query,
                        'type': 'procedural_step',
                        'step_number': i + 1,
                        'priority': 0.8 - (i * 0.05),
                        'original_position': i
                    })
        
        return sub_queries
    
    def _deduplicate_and_filter(self, sub_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and filter out low-quality sub-queries."""
        # Remove very short queries
        filtered = [sq for sq in sub_queries if len(sq['query'].strip()) > 10]
        
        # Remove near-duplicates using simple similarity
        deduplicated = []
        for sq in filtered:
            is_duplicate = False
            for existing in deduplicated:
                if self._are_similar(sq['query'], existing['query']):
                    # Keep the higher priority one
                    if sq.get('priority', 0) > existing.get('priority', 0):
                        deduplicated.remove(existing)
                        deduplicated.append(sq)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(sq)
        
        # Sort by priority
        deduplicated.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Limit to top 4 sub-queries to avoid overwhelming the system
        return deduplicated[:4]
    
    def _are_similar(self, query1: str, query2: str, threshold: float = 0.7) -> bool:
        """Check if two queries are similar using simple word overlap."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union
        return jaccard_similarity >= threshold
    
    def merge_results(self, sub_query_results: List[Tuple[Dict[str, Any], List[Document]]]) -> List[Document]:
        """
        Merge results from multiple sub-queries intelligently.
        
        Args:
            sub_query_results: List of (sub_query_info, documents) tuples
            
        Returns:
            Merged and ranked documents
        """
        if not sub_query_results:
            return []
        
        # If only one sub-query, return its results
        if len(sub_query_results) == 1:
            return sub_query_results[0][1]
        
        # Merge results with weighted scoring
        doc_scores = {}
        doc_map = {}
        
        for sub_query_info, documents in sub_query_results:
            priority = sub_query_info.get('priority', 0.5)
            
            for rank, doc in enumerate(documents):
                doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                doc_map[doc_id] = doc
                
                # Score based on rank and sub-query priority
                rank_score = 1.0 / (rank + 1)  # Higher rank = higher score
                weighted_score = rank_score * priority
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += weighted_score
                else:
                    doc_scores[doc_id] = weighted_score
        
        # Sort by combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top documents
        merged_docs = []
        for doc_id, score in sorted_docs:
            doc = doc_map[doc_id]
            doc.metadata['decomposition_score'] = score
            merged_docs.append(doc)
        
        logger.info(f"Merged results from {len(sub_query_results)} sub-queries into {len(merged_docs)} documents")
        return merged_docs