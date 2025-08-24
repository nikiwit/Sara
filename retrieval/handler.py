"""
Main retrieval handler for processing queries and retrieving documents.
"""

import os
import re
import math
import time
import types
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Union, Iterator, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np

from config import config
from sara_types import QueryType, RetrievalStrategy
from .system_info import SystemInformation
from .faq_matcher import FAQMatcher
from .reranker import AdvancedReranker, LightweightReranker
# from query_processing.decomposer import QueryDecomposer

logger = logging.getLogger("Sara")

class RetrievalHandler:
    """Handles document retrieval using multiple strategies."""
    
    def __init__(self, vector_store, embeddings, memory, context_processor):
        """Initialize the retrieval handler."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory = memory
        self.context_processor = context_processor
        
        # Create retrievers for different strategies
        self.retrievers = self._create_retrievers()
        
        # Initialize FAQ matcher for direct FAQ matching
        self.faq_matcher = FAQMatcher(vector_store)
        
        # Initialize reranker for improved document ranking
        try:
            self.reranker = AdvancedReranker()
            logger.info("Initialized AdvancedReranker with cross-encoder model")
        except Exception as e:
            logger.warning(f"Failed to initialize AdvancedReranker, falling back to LightweightReranker: {e}")
            self.reranker = LightweightReranker()
        
        # Initialize response cache
        from response.cache import ResponseCache
        self.response_cache = ResponseCache(ttl=config.CACHE_TTL if hasattr(config, 'CACHE_TTL') else 3600)
        logger.info("Initialized response cache")
        
        # Add spaCy processor for semantic ranking (Phase 4: Configuration-driven)
        self.use_semantic_ranking = config.USE_ENHANCED_SEMANTICS
        
        if self.use_semantic_ranking:
            try:
                from spacy_semantic_processor import SpacySemanticProcessor
                self.spacy_processor = SpacySemanticProcessor()
                if self.spacy_processor.initialized:
                    logger.info("spaCy semantic ranking enabled")
                else:
                    self.use_semantic_ranking = False
                    logger.warning("spaCy semantic ranking failed to initialize")
            except Exception as e:
                logger.warning(f"spaCy semantic ranking not available: {e}")
                self.spacy_processor = None
                self.use_semantic_ranking = False
        else:
            logger.info("Semantic ranking disabled by configuration")
            self.spacy_processor = None
        
        # Initialize query decomposer for complex queries
        # self.query_decomposer = QueryDecomposer()
    
    def _create_retrievers(self) -> Dict[RetrievalStrategy, Any]:
        """Create retrievers for different strategies."""
        retrievers = {}
        
        # Semantic search retriever with higher retrieval count for better matching
        retrievers[RetrievalStrategy.SEMANTIC] = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K * 2}  # Double the search space
        )
        
        # MMR retriever for diversity
        retrievers[RetrievalStrategy.MMR] = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.RETRIEVER_K * 4,  # Increased fetch for better diversity
                "lambda_mult": 0.3  # Lower lambda for more diversity
            }
        )
        
        # For keyword and hybrid search, we'll implement custom methods
        
        return retrievers
    
    def _validate_response(self, response: str, query: str) -> str:
        """
        Validate response quality and provide fallback if needed.
        Based on 2025 best practices for conversational AI response validation.
        
        Args:
            response: Generated response
            query: Original query
            
        Returns:
            Validated response or fallback response
        """
        if not response or response.strip() == "":
            logger.warning(f"Empty response generated for query: {query}")
            return self._get_empty_response_fallback(query)
        
        # Check for very short responses that might indicate generation failure
        if len(response.strip()) < 10:
            logger.warning(f"Suspiciously short response for query: {query}")
            return self._get_short_response_fallback(query, response)
        
        # Check for error responses from LLM
        error_indicators = [
            "error:", "failed to", "could not", "unable to generate",
            "connection error", "timeout", "invalid response"
        ]
        
        response_lower = response.lower()
        if any(indicator in response_lower for indicator in error_indicators):
            logger.warning(f"Error response detected for query: {query}")
            return self._get_error_response_fallback(query)
        
        # Response passes validation
        return response
    
    def _get_empty_response_fallback(self, query: str) -> str:
        """Fallback for empty responses."""
        return (
            "I apologize, but I couldn't generate a proper response to your question. "
            "Could you please try rephrasing your question? I'm here to help with APU-related information "
            "such as academic procedures, administrative services, fees, and student support."
        )
    
    def _get_short_response_fallback(self, query: str, original_response: str) -> str:
        """Fallback for suspiciously short responses."""
        return (
            f"I started to respond with '{original_response}' but I think I can provide more helpful information. "
            "Could you please provide more details about what you're looking for? I'm here to help with "
            "APU services, procedures, and policies."
        )
    
    def _get_error_response_fallback(self, query: str) -> str:
        """Fallback for error responses."""
        return (
            "I encountered an issue while processing your question. Please try asking again, "
            "or feel free to rephrase your question. I'm here to help with APU information including "
            "academic programs, administrative procedures, student services, and campus facilities."
        )

    def _stream_text_response(self, text: str):
        """Helper method to stream text word by word with consistent delay."""
        words = text.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(config.STREAM_DELAY)
    
    def process_query(self, query_analysis: Dict[str, Any], stream=False) -> Union[str, Iterator[str]]:
        """
        Process a query using the appropriate retrieval strategy.
        
        Args:
            query_analysis: The analysis from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Response string or iterator of response tokens if stream=True
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        expanded_queries = query_analysis.get("expanded_queries", [original_query])
        
        # Handle query_type safely whether it's enum or string
        query_type_str = query_type.value if hasattr(query_type, 'value') else str(query_type)
        logger.info(f"Processing {query_type_str} query: {original_query}")
        
        # Validate input sanity first
        if not self._validate_input_sanity(original_query):
            logger.info(f"Detected nonsensical query: {original_query}")
            nonsense_response = (
                "I didn't quite understand your question. Could you please rephrase it? "
                "I'm here to help with APU-related information like academic procedures, "
                "student services, fees, and campus facilities."
            )
            
            # Update memory 
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(nonsense_response)
            
            # For nonsense queries, always return non-streaming response for better UX
            # Streaming a "please rephrase" message is not critical
            return nonsense_response
        
        # Check cache for non-streaming requests first
        if not stream:
            cached_response = self.response_cache.get(original_query)
            if cached_response:
                logger.info("Returning cached response")
                # Update memory with cached response
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(cached_response)
                return cached_response
        
        
        # Handle identity questions with SystemInformation
        if query_type == QueryType.IDENTITY:
            response = SystemInformation.get_response_for_identity_query(original_query)
                
            # Update memory and cache
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
            # Cache identity responses
            if self.response_cache.should_cache(original_query, response):
                self.response_cache.set(original_query, response, {'type': 'identity'})
            
            # If streaming, return character by character iterator instead of full response
            if stream:
                # Return an iterator that yields one character at a time with consistent delay
                return self._stream_text_response(response)
            else:
                return response
        
        # First try direct FAQ matching for a quick answer
        faq_match_result = self.faq_matcher.match_faq(query_analysis)
        
        # Optimized threshold to 0.65 for bge-large-en-v1.5 model performance
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.65:
            # Additional validation: check topical relevance
            if self._validate_faq_relevance(original_query, faq_match_result):
                # High confidence direct FAQ match
                logger.info(f"Found direct FAQ match with score {faq_match_result['match_score']}")
                
                # Create a simplified context with just the matched FAQ
                context = self._format_faq_match(faq_match_result)
                
                # Generate response using the matched FAQ
                input_dict = {
                    "question": original_query,
                    "context": context,
                    "chat_history": self.memory.chat_memory.messages,
                    "is_faq_match": True,
                    "match_score": faq_match_result["match_score"]
                }
                
                # Generate response through LLM with streaming if requested
                # Import here to avoid circular imports
                from response.generator import RAGSystem
                response = RAGSystem.stream_ollama_response(
                    self._create_prompt(input_dict), 
                    config.LLM_MODEL_NAME,
                    stream_output=stream
                )
                
                # Post-process response to clean formatting and add source URLs
                if not stream:
                    response = self._post_process_response(response, [faq_match_result["document"]])
                    # Validate response quality
                    response = self._validate_response(response, original_query)
                    self.memory.chat_memory.add_user_message(original_query)
                    self.memory.chat_memory.add_ai_message(response)
                    # Cache FAQ responses (high priority for caching)
                    if self.response_cache.should_cache(original_query, response):
                        self.response_cache.set(original_query, response, {
                            'type': 'faq_match',
                            'match_score': faq_match_result["match_score"]
                        })
                    return response
                else:
                    # For streaming responses, return a streaming generator
                    return self._stream_with_postprocess(response, [faq_match_result["document"]], original_query)
        
        # If no good FAQ match, proceed with standard retrieval
        # Standard retrieval (query decomposition temporarily disabled)
        retrieval_strategy = self._select_retrieval_strategy(query_type)
        logger.info(f"Selected retrieval strategy: {retrieval_strategy.value}")
        context_docs = self._retrieve_documents(expanded_queries, retrieval_strategy)
        
        # If no documents found, try a fallback strategy
        if not context_docs and retrieval_strategy != RetrievalStrategy.HYBRID:
            logger.info("No documents found, trying hybrid fallback strategy")
            context_docs = self._retrieve_documents(expanded_queries, RetrievalStrategy.HYBRID)
        
        # If still no relevant documents, return a "no information" response
        if not context_docs:
            no_info_response = self._get_boundary_response(original_query)
            
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(no_info_response)
            
            # If streaming, return as character stream for consistency
            if stream:
                return self._stream_text_response(no_info_response)
            else:
                return no_info_response
        
        confidence_score = self._calculate_confidence_score(context_docs, original_query)
        logger.info(f"Confidence score: {confidence_score:.3f}")
        
        # If confidence is too low, return boundary response
        if confidence_score < config.CONFIDENCE_THRESHOLD:
            boundary_response = self._get_boundary_response(original_query)
            
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(boundary_response)
            
            # If streaming, return as character stream for consistency
            if stream:
                return self._stream_text_response(boundary_response)
            else:
                return boundary_response
        
        # Process the retrieved documents
        context = self.context_processor.process_context(context_docs, query_analysis)
        
        # Check for hallucination risk
        is_risky, risk_reason = self._check_hallucination_risk(original_query, context)
        
        if is_risky:
            logger.warning(f"Hallucination risk detected: {risk_reason}")
            
            # For high risk queries, add a caution to the context
            hallucination_warning = (
                f"CAUTION: This query has elements that may not be fully addressed in the retrieved "
                f"context ({risk_reason}). Only provide information directly stated in the context, "
                f"and be clear about information gaps.\n\n"
            )
            
            context = hallucination_warning + context
        
        # Generate response using RAG system
        input_dict = {
            "question": original_query,
            "context": context,
            "chat_history": self.memory.chat_memory.messages,
            "is_faq_match": False,
            "confidence_score": confidence_score
        }
        
        # Generate response through LLM with streaming if requested
        # Import here to avoid circular imports
        from response.generator import RAGSystem
        response = RAGSystem.stream_ollama_response(
            self._create_prompt(input_dict), 
            config.LLM_MODEL_NAME,
            stream_output=stream
        )
        
        # Post-process response to clean formatting and add source URLs
        if not stream:
            response = self._post_process_response(response, context_docs)
            # Validate response quality
            response = self._validate_response(response, original_query)
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
            # Cache successful RAG responses
            if self.response_cache.should_cache(original_query, response):
                self.response_cache.set(original_query, response, {
                    'type': 'rag_response',
                    'confidence_score': confidence_score
                })
            return response
        else:
            # For streaming responses, return a streaming generator
            return self._stream_with_postprocess(response, context_docs, original_query)
    
    
    def _intelligent_document_filtering(self, query: str, retrieved_docs: List[Document]) -> List[Document]:
        """
        Apply semantic document filtering using spaCy for general excellent performance.
        
        Args:
            query: Original query
            retrieved_docs: Documents retrieved from vector search
            
        Returns:
            Semantically ranked documents
        """
        if not retrieved_docs:
            return retrieved_docs
        
        # Use spaCy semantic re-ranking for general excellent performance
        if self.use_semantic_ranking and self.spacy_processor:
            try:
                # Extract document texts (limit for performance)
                doc_texts = [doc.page_content[:500] for doc in retrieved_docs]
                
                # Rank by semantic similarity using spaCy
                ranked_results = self.spacy_processor.rank_documents_by_similarity(
                    query, doc_texts
                )
                
                # Reorder documents based on semantic similarity
                semantic_scores = {text: score for text, score in ranked_results}
                filtered_docs = sorted(
                    retrieved_docs,
                    key=lambda doc: semantic_scores.get(doc.page_content[:500], 0), 
                    reverse=True
                )
                
                logger.info("Applied spaCy semantic ranking for general document relevance")
                return filtered_docs
                
            except Exception as e:
                logger.warning(f"spaCy semantic ranking failed: {e}")
        
        # Fallback: return documents as-is if semantic ranking unavailable
        return retrieved_docs
    
    
    def _post_process_response(self, response: str, docs: List[Document]) -> str:
        """
        Post-process LLM response to clean formatting and add source URLs.
        
        Args:
            response: The raw LLM response
            docs: The source documents used for the response
            
        Returns:
            Cleaned response with proper source URLs
        """
        if not response:
            return response
        
        logger.debug(f"Post-processing response of length {len(response)}")
        original_response = response
        
        import re
        
        # Clean up the response by removing unwanted elements
        response = re.sub(r'\[.*?Knowledge Base[^\]]*\]', '', response)
        response = re.sub(r'Related Pages.*', '', response, flags=re.IGNORECASE | re.DOTALL)
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        response = response.strip()
        
        source_urls = self._extract_source_attribution(response, docs)
        if source_urls and self._response_contains_useful_info(response):
            response += f"\n\nSource: {source_urls[0]}"
        
        if response != original_response:
            logger.debug("Response was modified during post-processing")
        else:
            logger.debug("Response unchanged during post-processing")
        
        return response
    
    def _find_most_relevant_document_for_response(self, response: str, docs: List[Document]) -> Document:
        """
        Find the document most relevant to the response content for better source attribution.
        
        Args:
            response: The generated response text
            docs: List of source documents
            
        Returns:
            Most relevant document or None if no good match
        """
        if not docs:
            return None
            
        response_lower = response.lower()
        
        # Extract key terms from the response
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        try:
            # Clean and tokenize response
            response_text = re.sub(r'[^\w\s]', ' ', response_lower)
            response_words = word_tokenize(response_text)
            stop_words = set(stopwords.words('english'))
            response_keywords = [word for word in response_words 
                               if word not in stop_words and len(word) > 2]
            
            best_doc = None
            best_score = 0
            
            for doc in docs:
                doc_content_lower = doc.page_content.lower()
                
                # Calculate keyword overlap score
                doc_words = word_tokenize(re.sub(r'[^\w\s]', ' ', doc_content_lower))
                doc_keywords = set([word for word in doc_words 
                                  if word not in stop_words and len(word) > 2])
                
                # Calculate relevance score based on keyword overlap
                if response_keywords:
                    common_keywords = len(set(response_keywords) & doc_keywords)
                    relevance_score = common_keywords / len(response_keywords)
                    
                    # Strong boost for title matches (primary topic indicator)
                    page_title = doc.metadata.get('page_title', '').lower()
                    title_keywords = word_tokenize(re.sub(r'[^\w\s]', ' ', page_title))
                    title_keyword_matches = len(set(response_keywords) & set(title_keywords))
                    if title_keyword_matches > 0:
                        # Higher weight for title matches - these indicate primary topic
                        title_relevance = (title_keyword_matches / len(response_keywords))
                        relevance_score += 0.6 * title_relevance
                        
                    # Boost for exact phrase matches in content
                    response_phrases = [response_lower[i:i+20] for i in range(0, len(response_lower)-20, 10)]
                    content_phrase_boost = 0
                    for phrase in response_phrases:
                        if phrase.strip() and phrase in doc_content_lower:
                            content_phrase_boost = 0.2
                            break
                    relevance_score += content_phrase_boost
                    
                    # General approach: Documents with high title relevance should be preferred
                    # over documents with only content matches (no hardcoded rules)
                    # The title match boost already handles this naturally by giving more weight
                    # to documents whose primary topic (title) aligns with the response
                    
                    if relevance_score > best_score:
                        best_score = relevance_score
                        best_doc = doc
                        
            logger.debug(f"Found most relevant document with score {best_score:.3f}")
            return best_doc if best_score > 0.1 else None
            
        except Exception as e:
            logger.debug(f"Error finding most relevant document: {e}")
            return None
    
    def _response_uses_knowledge_base_info(self, response: str, docs: List[Document] = None) -> bool:
        """
        Determine if the response actually uses information from the knowledge base
        using semantic similarity analysis instead of hardcoded patterns.
        
        Args:
            response: The generated response text
            docs: Source documents used for context (optional)
            
        Returns:
            True if response contains actual knowledge base information, False otherwise
        """
        response_lower = response.lower().strip()
        
        # Quick check: empty or very short responses likely don't use KB info
        if len(response_lower) < 20:
            return False
            
        # Check for explicit "no information" patterns (minimal, common phrases only)
        no_info_indicators = [
            "i don't have information",
            "no information available", 
            "information is not available",
            "i don't know"
        ]
        
        for indicator in no_info_indicators:
            if indicator in response_lower:
                return False
        
        # Special case: If we have FAQ documents, assume they're being used for knowledge
        # FAQ matches are specifically triggered for relevant questions, so responses should get sources
        if docs and any(doc.metadata.get('is_faq', False) for doc in docs):
            # For FAQ matches, if the response is substantive (not a "no info" response), 
            # assume it uses the knowledge base information
            return len(response_lower) > 50
        
        # Modern approach: If we have source documents, use semantic analysis
        if docs:
            return self._analyze_response_knowledge_usage(response, docs)
        
        # Fallback: Simple heuristics for substantive content
        # Responses with specific data, procedures, or detailed information likely use KB
        return (
            len(response_lower) > 50 and  # Substantial length
            (
                # Contains structured information indicators
                any(indicator in response_lower for indicator in [':', '-', '•', 'step', 'http']) or
                # Contains specific data patterns (minimal detection)
                len([word for word in response_lower.split() if word.isdigit()]) > 0 or
                # Contains URL or structured content
                'http' in response_lower or
                len(response_lower.split('\n')) > 2  # Multi-line suggests detailed info
            )
        )
    
    def _analyze_response_knowledge_usage(self, response: str, docs: List[Document]) -> bool:
        """
        Analyze if response uses knowledge base information through semantic similarity.
        
        Args:
            response: Generated response text
            docs: Source documents
            
        Returns:
            True if response shows high semantic overlap with source documents
        """
        if not docs or not response.strip():
            return False
            
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Tokenize and clean response
            response_words = set(word_tokenize(response.lower()))
            stop_words = set(stopwords.words('english'))
            response_keywords = {word for word in response_words 
                               if word not in stop_words and len(word) > 2 and word.isalnum()}
            
            if len(response_keywords) < 3:  # Too few keywords to analyze
                return False
                
            # Analyze overlap with source documents
            max_overlap = 0
            for doc in docs:
                doc_words = set(word_tokenize(doc.page_content.lower()))
                doc_keywords = {word for word in doc_words 
                              if word not in stop_words and len(word) > 2 and word.isalnum()}
                
                if doc_keywords:
                    overlap_ratio = len(response_keywords & doc_keywords) / len(response_keywords)
                    max_overlap = max(max_overlap, overlap_ratio)
            
            # If response shares >20% keywords with source docs, likely uses KB info
            # Lower threshold for better source attribution
            return max_overlap > 0.2
            
        except Exception as e:
            logger.debug(f"Error in semantic analysis: {e}")
            # Fallback to simple length-based heuristic
            return len(response.strip()) > 50
    
    def _response_contains_useful_info(self, response: str) -> bool:
        """
        Check if the response contains useful information that warrants a source citation.
        
        Args:
            response: The generated response
            
        Returns:
            True if response contains useful info, False for "I don't know" type responses
        """
        response_lower = response.lower()
        
        # Check for "no information" indicators
        no_info_phrases = [
            "i don't have information",
            "i don't have detailed information", 
            "i don't have specific information",
            "i cannot find",
            "i'm unable to find",
            "i don't have access to",
            "i'm not able to",
            "the information is not available",
            "i don't currently have",
            "i don't know",
            "sorry, i don't have",
            "i cannot provide",
            "i'm unable to provide",
            "no information available",
            "information not found"
        ]
        
        # If response contains any "no info" phrases, don't add source
        for phrase in no_info_phrases:
            if phrase in response_lower:
                return False
        
        # Check if response is too short to be useful (likely an error or "no info")
        if len(response.strip()) < 20:
            return False
            
        return True
    
    def _extract_source_attribution(self, response: str, docs: List[Any]) -> List[str]:
        """
        Extract source URLs from documents.
        
        This implements multiple attribution strategies for robust source detection:
        1. Semantic similarity-based matching
        2. Key phrase overlap detection  
        3. Document ranking-based fallback
        4. Content density analysis
        
        Args:
            response: Generated response text
            docs: List of source documents used for generation
            
        Returns:
            List of source URLs ordered by relevance (primary source first)
        """
        if not docs:
            return []
        
        source_candidates = []
        response_lower = response.lower()
        
        for doc in docs:
            main_url = doc.metadata.get('main_url', '')
            if not main_url:
                continue
                
            # Strategy 1: Content-based similarity scoring
            content_score = self._calculate_content_similarity(response, doc.page_content)
            
            # Strategy 2: Key phrase overlap (improved beyond simple word matching)  
            phrase_score = self._calculate_phrase_overlap(response, doc.page_content)
            
            # Strategy 3: Document metadata scoring (title, section relevance)
            metadata_score = self._calculate_metadata_relevance(response, doc.metadata)
            
            # Combined weighted score
            total_score = (
                content_score * 0.4 +      # Content similarity (most important)
                phrase_score * 0.4 +       # Phrase overlap (most important)  
                metadata_score * 0.2       # Metadata relevance (supporting)
            )
            
            if total_score > 0.1:  # Minimum threshold for attribution
                source_candidates.append({
                    'url': main_url,
                    'score': total_score,
                    'doc': doc
                })
        
        # Fallback strategy: If no good matches found, use document order (retrieval ranking)
        if not source_candidates:
            for doc in docs[:2]:  # Take top 2 retrieved documents
                main_url = doc.metadata.get('main_url', '')
                if main_url:
                    source_candidates.append({
                        'url': main_url,
                        'score': 0.5,  # Medium confidence fallback score
                        'doc': doc
                    })
        
        # Sort by score and return URLs
        source_candidates.sort(key=lambda x: x['score'], reverse=True)
        attributed_urls = [candidate['url'] for candidate in source_candidates]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in attributed_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls[:3]  # Return top 3 sources maximum
    
    def _calculate_content_similarity(self, response: str, doc_content: str) -> float:
        """
        Calculate semantic content similarity between response and document.
        
        Uses improved matching beyond simple word overlap:
        - N-gram matching (2-4 word phrases)
        - Normalized scoring by content length
        - Handles paraphrasing better than word-only matching
        """
        try:
            # Tokenize both texts
            response_tokens = response.lower().split()
            doc_tokens = doc_content.lower().split()
            
            if not response_tokens or not doc_tokens:
                return 0.0
            
            # Generate n-grams for better semantic matching
            def get_ngrams(tokens, n):
                return set(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
            
            # Calculate overlaps for different n-gram sizes
            bigram_response = get_ngrams(response_tokens, 2)
            bigram_doc = get_ngrams(doc_tokens, 2)
            bigram_overlap = len(bigram_response.intersection(bigram_doc))
            
            trigram_response = get_ngrams(response_tokens, 3)  
            trigram_doc = get_ngrams(doc_tokens, 3)
            trigram_overlap = len(trigram_response.intersection(trigram_doc))
            
            # Single word overlap (with less weight)
            word_response = set(response_tokens)
            word_doc = set(doc_tokens)
            word_overlap = len(word_response.intersection(word_doc))
            
            # Weighted score (higher n-grams get more weight)
            total_overlap = trigram_overlap * 3 + bigram_overlap * 2 + word_overlap * 1
            
            # Normalize by response length to avoid bias toward long documents
            max_possible_score = len(response_tokens) * 2  # Reasonable normalization factor
            similarity_score = min(total_overlap / max_possible_score, 1.0)
            
            return similarity_score
            
        except Exception as e:
            logger.warning(f"Content similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_phrase_overlap(self, response: str, doc_content: str) -> float:
        """
        Calculate overlap of key phrases between response and document.
        
        Focuses on meaningful phrases rather than common words.
        """
        try:
            # Extract key phrases (filter out common words)
            import re
            
            # Common words to ignore in phrase matching
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
            
            # Extract meaningful phrases (2-5 words, avoiding common words)
            def extract_key_phrases(text):
                # Split into sentences and then phrases
                sentences = re.split(r'[.!?]', text.lower())
                phrases = set()
                
                for sentence in sentences:
                    words = sentence.split()
                    # Extract 2-5 word phrases that contain at least one non-common word
                    for i in range(len(words)):
                        for length in [2, 3, 4, 5]:
                            if i + length <= len(words):
                                phrase = ' '.join(words[i:i+length])
                                phrase_words = phrase.split()
                                # Include phrase if it has at least one meaningful word
                                if any(word not in common_words and len(word) > 3 for word in phrase_words):
                                    phrases.add(phrase)
                
                return phrases
            
            response_phrases = extract_key_phrases(response)
            doc_phrases = extract_key_phrases(doc_content)
            
            if not response_phrases:
                return 0.0
            
            # Calculate overlap
            overlap_count = len(response_phrases.intersection(doc_phrases))
            phrase_score = overlap_count / len(response_phrases)
            
            return min(phrase_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Phrase overlap calculation failed: {e}")
            return 0.0
    
    def _calculate_metadata_relevance(self, response: str, metadata: dict) -> float:
        """
        Calculate relevance based on document metadata (title, section, etc.).
        
        Gives bonus scores for documents whose metadata aligns with response content.
        """
        try:
            score = 0.0
            response_lower = response.lower()
            
            # Check title relevance
            title = metadata.get('page_title', '').lower()
            if title:
                title_words = set(title.split())
                response_words = set(response_lower.split())
                title_overlap = len(title_words.intersection(response_words))
                if title_overlap > 0:
                    score += 0.3  # Bonus for title relevance
            
            # Check section relevance  
            section = metadata.get('section_title', '').lower()
            if section:
                section_words = set(section.split())
                response_words = set(response_lower.split())
                section_overlap = len(section_words.intersection(response_words))
                if section_overlap > 0:
                    score += 0.2  # Bonus for section relevance
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Metadata relevance calculation failed: {e}")
            return 0.0
    
    def _format_faq_match(self, match_result: Dict[str, Any]) -> str:
        """Format a direct FAQ match into a context for the LLM."""
        doc = match_result["document"]
        score = match_result["match_score"]
        
        # Get metadata
        title = doc.metadata.get("page_title", "Unknown Title")
        source = doc.metadata.get("source", "Unknown Source")
        filename = doc.metadata.get("filename", os.path.basename(source) if source != "Unknown Source" else "Unknown File")
        
        # Try to get complete FAQ content by finding related chunks
        complete_content = self._get_complete_faq_content(doc, title)
        
        # Clean the document content before formatting
        import re
        clean_content = complete_content
        # Remove bracketed headers
        clean_content = re.sub(r'\[.*?Knowledge Base[^\]]*\]', '', clean_content)
        # Remove "Related Pages" sections
        clean_content = re.sub(r'Related Pages.*', '', clean_content, flags=re.IGNORECASE | re.DOTALL)
        # Clean up whitespace
        clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content).strip()
        
        # Format the context with clear but not overly intimidating instructions
        context = f"--- DIRECT FAQ MATCH (Confidence: {score:.2f}) ---\n\n"
        context += f"Question: {title}\n\n"
        context += f"SOURCE TEXT:\n\"\"\"\n{clean_content}\n\"\"\"\n\n"
        
        # Add valid email addresses that exist
        emails_in_text = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_content)
        if emails_in_text:
            context += "Email addresses mentioned in the text:\n"
            for email in emails_in_text:
                context += f"- {email}\n"
            context += "\nRemember: Only use email addresses that are explicitly provided in the source text.\n"
        
        # Add explicit note about contact options if "or" appears
        if " or " in clean_content.lower():
            context += "\nNote: When multiple contact options are mentioned with 'or', these are separate alternatives.\n"
        
        # Add related pages if available
        related_pages = doc.metadata.get("related_pages", [])
        if related_pages:
            context += "\nRelated information can be found in:\n"
            for page in related_pages[:3]:
                context += f"- {page}\n"
        
        return context
    
    def _get_complete_faq_content(self, current_doc: Document, title: str) -> str:
        """
        Get complete FAQ content by finding and combining all chunks with the same page_title.
        
        Args:
            current_doc: The current document (one chunk)
            title: The FAQ title to search for
            
        Returns:
            Complete FAQ content reconstructed from all chunks
        """
        try:
            # Get all documents from the vector store
            all_docs = self.vector_store.get()
            
            if not all_docs or not all_docs.get('documents'):
                logger.warning("No documents found in vector store for FAQ reconstruction")
                return current_doc.page_content
            
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            # Find all chunks belonging to the same FAQ page
            matching_chunks = []
            
            for i, doc_text in enumerate(documents):
                if i >= len(metadatas):
                    continue
                
                metadata = metadatas[i]
                page_title = metadata.get('page_title', '')
                
                # Check if this chunk belongs to the same FAQ
                if page_title == title and metadata.get('content_type') == 'apu_kb_page':
                    chunk_id = metadata.get('chunk_id', 0)
                    matching_chunks.append({
                        'content': doc_text,
                        'chunk_id': chunk_id,
                        'metadata': metadata
                    })
            
            # If only one chunk found, return the current content
            if len(matching_chunks) <= 1:
                logger.debug(f"Only one chunk found for FAQ '{title}'")
                return current_doc.page_content
            
            logger.info(f"Found {len(matching_chunks)} chunks for FAQ '{title}', reconstructing complete content")
            
            # Sort chunks by chunk_id to maintain proper order
            matching_chunks.sort(key=lambda x: x['chunk_id'])
            
            # Combine all chunks while removing duplicate headers
            complete_content = ""
            seen_headers = set()
            
            for chunk in matching_chunks:
                content = chunk['content']
                
                # Remove context headers from individual chunks
                import re
                # Remove bracketed headers like [ITSM Knowledge Base - ...]
                content = re.sub(r'\[.*?Knowledge Base[^\]]*\]\s*\n*', '', content)
                
                # Split content into lines for processing
                lines = content.split('\n')
                processed_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line is a duplicate header
                    line_lower = line.lower()
                    is_header = (
                        line_lower == title.lower() or
                        line_lower.startswith(title.lower()[:20]) or
                        line_lower in seen_headers
                    )
                    
                    if is_header and line_lower in seen_headers:
                        continue  # Skip duplicate header
                    
                    if is_header:
                        seen_headers.add(line_lower)
                    
                    processed_lines.append(line)
                
                # Add processed content to complete content
                if processed_lines:
                    chunk_text = '\n'.join(processed_lines)
                    if complete_content:
                        complete_content += '\n\n' + chunk_text
                    else:
                        complete_content = chunk_text
            
            # Clean up the final content
            complete_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', complete_content).strip()
            
            logger.debug(f"Reconstructed complete FAQ content ({len(complete_content)} chars) from {len(matching_chunks)} chunks")
            return complete_content
            
        except Exception as e:
            logger.error(f"Error reconstructing complete FAQ content for '{title}': {e}")
            return current_doc.page_content
    
    def _select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select the appropriate retrieval strategy based on query type.
        """
        if config.RETRIEVER_SEARCH_TYPE != "auto":
            # Use the configured strategy if not set to auto
            return RetrievalStrategy(config.RETRIEVER_SEARCH_TYPE)
        
        # Select strategy based on query type
        if query_type in [QueryType.FACTUAL, QueryType.ACADEMIC]:
            return RetrievalStrategy.HYBRID  # Precise for factual/academic questions
        
        elif query_type in [QueryType.PROCEDURAL, QueryType.ADMINISTRATIVE]:
            return RetrievalStrategy.HYBRID  # Procedural and admin processes need semantic and keyword
        
        elif query_type == QueryType.FINANCIAL:
            return RetrievalStrategy.HYBRID  # Financial questions need precision
        
        elif query_type == QueryType.COMPARATIVE:
            return RetrievalStrategy.MMR  # Diverse results for comparison
        
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalStrategy.MMR  # Diverse results for exploration
        
        else:
            return RetrievalStrategy.HYBRID  # Default to hybrid
    
    def _retrieve_documents(self, queries: List[str], strategy: RetrievalStrategy) -> List[Document]:
        """
        Retrieve documents using the specified strategy.
        
        Args:
            queries: List of query strings (original and expanded)
            strategy: Retrieval strategy to use
            
        Returns:
            List of retrieved documents
        """
        all_docs = []
        seen_ids = set()  # Track document IDs to avoid duplicates
        
        # Process each query (original + expanded)
        for query in queries:
            try:
                if strategy == RetrievalStrategy.SEMANTIC or strategy == RetrievalStrategy.MMR:
                    # Use the appropriate retriever
                    docs = self.retrievers[strategy].invoke(query)
                    
                elif strategy == RetrievalStrategy.KEYWORD:
                    # Use keyword-based retrieval
                    docs = self._keyword_retrieval(query)
                    
                elif strategy == RetrievalStrategy.HYBRID:
                    # Use hybrid retrieval (combination of semantic and keyword)
                    docs = self._hybrid_retrieval(query)
                
                # Apply intelligent filtering to the retrieved documents
                docs = self._intelligent_document_filtering(query, docs)
                
                for doc in docs:
                    # Create a unique ID based on content and source
                    doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Apply advanced reranking using cross-encoder model
        all_docs = self.reranker.rerank_documents(queries[0], all_docs, top_k=config.RETRIEVER_K)
        
        logger.info(f"Retrieved and reranked {len(all_docs)} documents")
        return all_docs
    
    def _keyword_retrieval(self, query: str) -> List[Document]:
        """
        Perform keyword-based retrieval, enhanced for APU knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        ids = all_docs.get('ids', [])
        
        # Extract keywords from the query
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Domain-specific keywords for better matching
        domain_keywords = ["medical", "insurance", "collect", "card", "visa", "counter"]
        for kw in domain_keywords:
            if kw in query.lower() and kw not in keywords:
                keywords.append(kw)
        
        # Score documents based on keyword matches
        scored_docs = []
        for i, doc_text in enumerate(documents):
            if not doc_text:
                continue
                
            score = 0
            doc_lower = doc_text.lower()
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Check if this is an APU KB page
            is_apu_kb = metadata.get('content_type') == 'apu_kb_page'
            page_title = metadata.get('page_title', '')
            
            
            # Count exact keyword matches
            for keyword in keywords:
                if keyword in doc_lower:
                    score += 1
                    # Boost score if keyword appears in title
                    if page_title and keyword in page_title.lower():
                        score += 0.5
            
            # Normalize score based on number of keywords
            if keywords:
                score = score / len(keywords)
                
                # Boost APU KB pages
                if is_apu_kb:
                    score *= 1.2
                
                # Add to scored docs if score is above threshold
                if score > 0.2:  # Lowered threshold for better recall
                    scored_docs.append((score, i))
        
        # Sort by score (descending)
        scored_docs.sort(reverse=True)
        
        # Convert to Document objects
        result_docs = []
        for score, i in scored_docs[:config.RETRIEVER_K]:
            if i < len(documents) and i < len(metadatas):
                result_docs.append(
                    Document(
                        page_content=documents[i],
                        metadata=metadatas[i]
                    )
                )
        
        return result_docs
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Perform advanced hybrid retrieval using BM25 + semantic fusion.
        
        Args:
            query: Query string
            
        Returns:
            List of documents with fused scores
        """
        # Get semantic search results with scores
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get BM25 scores for all documents
        bm25_docs_with_scores = self._bm25_retrieval(query)
        
        # Perform score fusion using Reciprocal Rank Fusion (RRF)
        fused_docs = self._reciprocal_rank_fusion(
            semantic_docs, bm25_docs_with_scores, query
        )
        
        return fused_docs
    
    def _bm25_retrieval(self, query: str) -> List[Tuple[Document, float]]:
        """
        Perform BM25 retrieval with scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (Document, BM25_score) tuples
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        
        # Preprocess query
        query_tokens = self._tokenize_and_clean(query)
        if not query_tokens:
            return []
        
        # Build document corpus for BM25
        corpus = []
        doc_objects = []
        
        for i, doc_text in enumerate(documents):
            if not doc_text or i >= len(metadatas):
                continue
                
            doc_tokens = self._tokenize_and_clean(doc_text)
            if doc_tokens:  # Only include non-empty documents
                corpus.append(doc_tokens)
                doc_objects.append(Document(
                    page_content=doc_text,
                    metadata=metadatas[i]
                ))
        
        if not corpus:
            return []
        
        # Calculate BM25 scores
        scores = self._calculate_bm25_scores(query_tokens, corpus)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(doc_objects, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K documents with scores
        return doc_scores[:config.RETRIEVER_K * 2]  # Get more for fusion
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text for BM25 processing."""
        try:
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            
            # Filter tokens: remove stop words, keep alphanumeric with length > 2
            cleaned_tokens = [
                token for token in tokens 
                if token.isalnum() and len(token) > 2 and token not in stop_words
            ]
            
            return cleaned_tokens
        except Exception:
            # Fallback to simple split if NLTK fails
            return [word.lower() for word in text.split() if len(word) > 2]
    
    def _calculate_bm25_scores(self, query_tokens: List[str], corpus: List[List[str]]) -> List[float]:
        """
        Calculate BM25 scores for query against corpus.
        
        Args:
            query_tokens: Tokenized query
            corpus: List of tokenized documents
            
        Returns:
            List of BM25 scores
        """
        # BM25 parameters (standard values)
        k1 = 1.5  # Term frequency saturation point
        b = 0.75  # Length normalization parameter
        
        # Calculate document frequencies
        N = len(corpus)  # Total number of documents
        doc_freqs = Counter()
        doc_lengths = []
        
        # Count term frequencies and document lengths
        for doc in corpus:
            doc_length = len(doc)
            doc_lengths.append(doc_length)
            
            # Count unique terms in this document
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        
        # Calculate average document length
        avg_doc_length = sum(doc_lengths) / N if N > 0 else 0
        
        # Calculate BM25 score for each document
        scores = []
        
        for i, doc in enumerate(corpus):
            score = 0.0
            doc_length = doc_lengths[i]
            
            # Count term frequencies in this document
            term_freqs = Counter(doc)
            
            for term in query_tokens:
                if term not in doc_freqs:
                    continue  # Term not in any document
                
                # Term frequency in current document
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue
                
                # Document frequency (number of documents containing term)
                df = doc_freqs[term]
                
                # Inverse document frequency
                idf = math.log((N - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                score += idf * (numerator / denominator)
            
            scores.append(max(0.0, score))  # Ensure non-negative scores
        
        return scores
    
    def _reciprocal_rank_fusion(self, semantic_docs: List[Document], 
                               bm25_docs: List[Tuple[Document, float]], 
                               query: str) -> List[Document]:
        """
        Fuse semantic and BM25 results using Reciprocal Rank Fusion.
        
        Args:
            semantic_docs: Documents from semantic search
            bm25_docs: Documents with BM25 scores
            query: Original query
            
        Returns:
            Fused and ranked documents
        """
        # Create document ID mapping
        doc_map = {}
        fused_scores = defaultdict(float)
        
        # RRF constant (typically 60)
        k = 60
        
        # Process semantic results (rank-based scoring)
        for rank, doc in enumerate(semantic_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            doc_map[doc_id] = doc
            
            # RRF score: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[doc_id] += rrf_score * 0.6  # Weight for semantic
        
        # Process BM25 results
        bm25_docs_sorted = sorted(bm25_docs, key=lambda x: x[1], reverse=True)
        
        for rank, (doc, bm25_score) in enumerate(bm25_docs_sorted):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            doc_map[doc_id] = doc
            
            # Normalized BM25 score + RRF
            max_bm25 = bm25_docs_sorted[0][1] if bm25_docs_sorted else 1.0
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[doc_id] += (rrf_score + normalized_bm25) * 0.4  # Weight for BM25
        
        # Sort by fused scores
        sorted_docs = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top documents
        result_docs = []
        for doc_id, score in sorted_docs[:config.RETRIEVER_K]:
            doc = doc_map[doc_id]
            doc.metadata['fusion_score'] = score
            result_docs.append(doc)
        
        logger.info(f"Fused {len(semantic_docs)} semantic + {len(bm25_docs)} BM25 results into {len(result_docs)} documents")
        return result_docs
    
    def _retrieve_with_decomposition(self, query: str, query_analysis: Dict[str, Any]) -> List[Document]:
        """
        Retrieve documents using query decomposition for complex queries.
        
        Args:
            query: Original complex query
            query_analysis: Query analysis from InputProcessor
            
        Returns:
            List of documents from merged sub-query results
        """
        # Decompose the query
        sub_queries = self.query_decomposer.decompose_query(query, query_analysis)
        logger.info(f"Query decomposed into {len(sub_queries)} sub-queries")
        
        # Retrieve documents for each sub-query
        sub_query_results = []
        
        for sub_query_info in sub_queries:
            sub_query_text = sub_query_info['query']
            sub_query_type = query_analysis.get('query_type', QueryType.UNKNOWN)
            
            logger.debug(f"Processing sub-query ({sub_query_info['type']}): {sub_query_text}")
            
            # Select appropriate strategy for this sub-query
            retrieval_strategy = self._select_retrieval_strategy(sub_query_type)
            
            # Retrieve documents for this sub-query
            try:
                sub_docs = self._retrieve_documents([sub_query_text], retrieval_strategy)
                if sub_docs:
                    sub_query_results.append((sub_query_info, sub_docs))
                    logger.debug(f"Retrieved {len(sub_docs)} documents for sub-query")
            except Exception as e:
                logger.warning(f"Failed to retrieve documents for sub-query '{sub_query_text}': {e}")
        
        # Merge results from all sub-queries
        if sub_query_results:
            merged_docs = self.query_decomposer.merge_results(sub_query_results)
            logger.info(f"Merged decomposition results: {len(merged_docs)} documents")
            return merged_docs
        else:
            logger.warning("No results from query decomposition, falling back to standard retrieval")
            # Fallback to standard retrieval
            retrieval_strategy = self._select_retrieval_strategy(query_analysis.get('query_type', QueryType.UNKNOWN))
            return self._retrieve_documents([query], retrieval_strategy)
    
    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            docs: List of documents
            query: Original query string
            
        Returns:
            Reranked list of documents
        """
        if not docs:
            return []
        
        # Extract keywords from query
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Domain-specific keywords for better matching
        domain_keywords = ["medical", "insurance", "collect", "card", "visa", "counter"]
        for kw in domain_keywords:
            if kw in query.lower() and kw not in keywords:
                keywords.append(kw)
        
        # Score documents
        scored_docs = []
        for doc in docs:
            base_score = doc.metadata.get('retrieval_score', 0.5)
            
            # Additional scoring factors
            content_lower = doc.page_content.lower()
            
            # Check for exact phrase match
            phrase_match = 0
            if query.lower() in content_lower:
                phrase_match = 0.3
            
            # Check for keyword matches
            keyword_match = 0
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_match += 0.1
            keyword_match = min(0.3, keyword_match)  # Cap at 0.3
            
            # Check if document is an FAQ
            faq_boost = 0
            if doc.metadata.get('is_faq', False):
                faq_boost = 0.1
                
            # Check if document is an APU KB page
            apu_kb_boost = 0
            if doc.metadata.get('content_type') == 'apu_kb_page':
                apu_kb_boost = 0.2
            
            
            # Calculate final score
            final_score = base_score + phrase_match + keyword_match + faq_boost + apu_kb_boost
            
            # Add to scored docs
            scored_docs.append((final_score, doc))
        
        # Sort by score (descending)
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Return reranked docs
        return [doc for _, doc in scored_docs]
    
    def _assess_document_relevance(self, docs: List[Document], query: str) -> float:
        """
        Assess the overall relevance of retrieved documents to the query.
        
        Args:
            docs: List of retrieved documents
            query: Original query string
            
        Returns:
            Relevance score between 0 and 1
        """
        if not docs:
            return 0
        
        # Extract keywords from query
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Domain-specific keywords for better matching
        domain_keywords = ["medical", "insurance", "collect", "card", "visa", "counter"]
        for kw in domain_keywords:
            if kw in query.lower() and kw not in keywords:
                keywords.append(kw)
        
        # Check for exact phrase matches
        exact_match_score = 0
        for doc in docs:
            if query.lower() in doc.page_content.lower():
                exact_match_score = 1.0
                break
        
        # Check for keyword coverage
        keyword_coverage = 0
        if keywords:
            matched_keywords = set()
            for doc in docs:
                doc_lower = doc.page_content.lower()
                for keyword in keywords:
                    if keyword in doc_lower:
                        matched_keywords.add(keyword)
            
            keyword_coverage = len(matched_keywords) / len(keywords)
        
        # Check for FAQ matches
        faq_match_score = 0
        for doc in docs:
            if doc.metadata.get('is_faq', False):
                faq_match_score = 0.5
                break
        
        # Check for APU KB matches
        apu_kb_match_score = 0
        for doc in docs:
            if doc.metadata.get('content_type') == 'apu_kb_page':
                apu_kb_match_score = 0.5
                break
        
        # Calculate overall relevance
        relevance = max(
            exact_match_score,
            keyword_coverage * 0.7,
            faq_match_score,
            apu_kb_match_score
        )
        
        return relevance
    
    def _check_hallucination_risk(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Check if the query has a high risk of hallucination given the context.
        
        Args:
            query: Original query string
            context: Retrieved context
            
        Returns:
            Tuple of (is_risky, risk_reason)
        """
        # Check for specific entities in query that might not be in context
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Check for named entities that might be hallucinated
        named_entities = [
            "professor", "dr.", "doctor", "mr.", "mrs.", "ms.", "dean", "director",
            "department", "office", "building", "hall", "room", "center"
        ]
        
        for entity in named_entities:
            if entity in query_lower and entity not in context_lower:
                return True, f"Named entity '{entity}' in query not found in context"
        
        # Check for specific dates or times
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\d{1,2} [a-z]+ \d{2,4}'    # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            query_dates = re.findall(pattern, query_lower)
            if query_dates and not any(date in context_lower for date in query_dates):
                return True, "Specific dates in query not found in context"
        
        # Check for specific numbers that might be hallucinated
        number_patterns = [
            r'\$\d+',  # Dollar amounts
            r'\d+ dollars',  # Dollar amounts in words
            r'\d+ percent',  # Percentages
            r'\d+%'  # Percentages
        ]
        
        for pattern in number_patterns:
            query_numbers = re.findall(pattern, query_lower)
            if query_numbers and not any(num in context_lower for num in query_numbers):
                return True, "Specific numbers in query not found in context"
        
        return False, ""
    
    def _calculate_confidence_score(self, docs: List[Document], query: str) -> float:
        """
        Calculate confidence score for retrieved documents.
        
        Args:
            docs: List of retrieved documents
            query: Original query string
            
        Returns:
            Confidence score between 0 and 1
        """
        if not docs:
            return 0.0
        
        # Factor 1: Document relevance (existing method)
        relevance_score = self._assess_document_relevance(docs, query)
        
        # Factor 2: Average document length (longer docs often more informative)
        avg_doc_length = sum(len(doc.page_content) for doc in docs) / len(docs)
        length_score = min(1.0, avg_doc_length / 500)  # Normalize to 500 chars
        
        # Factor 3: Metadata quality (APU KB pages are more reliable)
        metadata_score = 0
        for doc in docs:
            if doc.metadata.get('content_type') == 'apu_kb_page':
                metadata_score += 0.3
            if doc.metadata.get('is_faq', False):
                metadata_score += 0.2
            if doc.metadata.get('page_title'):
                metadata_score += 0.1
        metadata_score = min(1.0, metadata_score / len(docs))
        
        # Factor 4: Query specificity (specific queries need higher confidence)
        query_tokens = query.lower().split()
        specificity_penalty = 0
        if len(query_tokens) > 8:  # Long, specific queries
            specificity_penalty = 0.1
        
        # Weighted combination
        confidence = (
            relevance_score * 0.5 +      # Primary factor
            length_score * 0.2 +         # Document quality
            metadata_score * 0.3         # Source reliability
        ) - specificity_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _get_boundary_response(self, query: str) -> str:
        """
        Generate appropriate boundary response based on query type.
        
        Args:
            query: Original query string
            
        Returns:
            Appropriate boundary response
        """
        query_lower = query.lower()
        
        # Categorize out-of-scope queries
        if any(word in query_lower for word in ['program', 'course', 'degree', 'admission', 'requirement']):
            return (
                "I don't have detailed information about academic programs and admission requirements. "
                "For the most current information about programs, courses, and admission criteria, "
                "please contact the Admissions Office at admissions@apu.edu.my or visit the APU website."
            )
        
        elif any(word in query_lower for word in ['gpa', 'grade', 'result', 'transcript', 'record']):
            return (
                "I cannot access personal academic records. To check your grades, GPA, or academic records, "
                "please log in to APSpace or contact the Registrar's Office at registrar@apu.edu.my."
            )
        
        elif any(word in query_lower for word in ['weather', 'temperature', 'climate', 'rain', 'sunny']):
            return (
                "I'm designed to help with APU-specific information rather than weather updates. "
                "For weather information, please check your local weather app or website. "
                "Is there anything about APU services or procedures I can help you with instead?"
            )
        
        elif any(word in query_lower for word in ['time', 'current time', 'what time', 'clock']):
            return (
                "I don't have access to current time information. Please check your device's clock. "
                "However, I can help you with APU schedules, timetables, and office hours. "
                "What APU-related timing information do you need?"
            )
        
        elif any(word in query_lower for word in ['news', 'current events', 'politics', 'sports']):
            return (
                "I focus on APU-related information rather than general news. "
                "For current events, please check news websites or apps. "
                "I'd be happy to help with APU announcements, procedures, or services instead!"
            )
        
        elif any(word in query_lower for word in ['staff', 'faculty', 'professor', 'teacher', 'dean']):
            return (
                "I don't have detailed information about specific staff members or faculty. "
                "For faculty contact information and office hours, please check the APU website "
                "or contact the relevant department directly."
            )
        
        
        elif any(word in query_lower for word in ['accommodation', 'hostel', 'housing', 'dormitory']):
            return (
                "I don't have detailed information about accommodation options. "
                "For information about student housing and accommodation, please contact "
                "Student Services at student.services@apu.edu.my."
            )
        
        else:
            return (
                "I don't have specific information about that topic in my knowledge base. "
                "For accurate and detailed information, I recommend contacting the appropriate "
                "APU department directly. You can also visit the APU website or call the main office "
                "for general inquiries."
            )

    def _create_prompt(self, input_dict: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM based on the input dictionary.
        
        Args:
            input_dict: Dictionary with prompt inputs
            
        Returns:
            Formatted prompt string
        """
        question = input_dict["question"]
        context = input_dict["context"]
        is_faq_match = input_dict.get("is_faq_match", False)
        confidence_score = input_dict.get("confidence_score", 0.5)
        
        if is_faq_match:
            prompt = f"""You are Sara, an AI assistant for APU (Asia Pacific University). Answer this student's question directly and naturally.

Question: {question}

Available Information:
{context}

Provide a clear, direct answer. Be concise and short for simple questions and comprehensive for complex ones. Include full relevant details like locations, steps, contact information, and requirements where available and only if relevant to the question. Write naturally without mentioning sources or information basis.

Answer:"""

        elif confidence_score < 0.25:  # Low confidence - likely partial match
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). The user asked a question, but the available information may only be partially related.

    User's Question: {question}

    Available Information:
    {context}

    Instructions:
    1. First, check if the available information directly answers the user's question.
    2. If YES - provide a direct, helpful answer.
    3. If NO - but there is some related information, respond using this EXACT format:

    "I found some related information, but I'd like to help you find exactly what you're looking for.

    Could you please specify which of these you're interested in:

    1) [First related topic you found in the information]
    2) [Second related topic if available]
    3) [Third related topic if available]

    Or feel free to ask your question in a different way, and I'll do my best to help!"

    4. If the available information is completely unrelated, say: "I don't have specific information about that topic. Could you provide more details about what you're looking for?"

    IMPORTANT RULES:
    - Never mention "documents", "context", "FAQ", "questions", or other technical terms
    - Extract actual topic names from the information (e.g., "Collecting examination dockets", "Exam schedule information")
    - Keep topics specific and user-friendly
    - Maximum 4 options
    - Always be helpful and encouraging

    Answer:"""

        else:
            # Regular prompt for good relevance with enhanced URL preservation
            prompt = f"""You are Sara, an AI assistant for APU (Asia Pacific University). Answer the question using the information available.

Question: {question}

Available Information:
{context}

Instructions:
1. Answer the question directly and concisely using the available information.
2. **CRITICAL**: Preserve ALL URLs and links exactly as they appear (e.g., https://cas.apiit.edu.my/cas/login).
3. For step-by-step procedures, format them clearly with numbers or bullet points.
4. Include ONLY information that directly answers the user's question. Do NOT add supplementary information (like contact details) unless specifically requested.
5. **ABSOLUTELY FORBIDDEN**: Do NOT assume the user's personal circumstances from the source material. If the source mentions "you are currently doing your internship" - this is an EXAMPLE scenario, NOT about this specific user.
6. **ABSOLUTELY FORBIDDEN**: Do NOT start responses with phrases like "I understand you are..." or "I see that you..." about situations not mentioned by the user.
7. **REQUIRED**: When the source describes specific situations (internships, attendance issues, etc.), present them as conditional options: "If you are doing an internship...", "For students who...", "In cases where..."
8. NEVER personalize generic information (e.g., don't say "your attendance is 73%" - say "if attendance is below 80%").
9. NEVER assume specific personal details about the student (attendance, fees, grades, etc.).
10. Provide general guidance that covers different scenarios without assuming which applies to the user.
11. **UX CRITICAL**: If the information doesn't fully answer the question, say "I don't have information about [specific topic]" - NEVER mention "provided information", "documents", "sections", "knowledge base", or "context".
12. **UX CRITICAL**: NEVER mention internal system details. Speak naturally as if you're a knowledgeable assistant, not a system processing documents.
13. Use a helpful and professional tone appropriate for a university assistant.

Answer:"""

        return prompt
    
    def _validate_faq_relevance(self, query: str, faq_match_result: Dict[str, Any]) -> bool:
        """
        Validate that the FAQ match is actually relevant to the user's query.
        Prevents hallucinations from loose semantic matching.
        
        Args:
            query: Original user query
            faq_match_result: FAQ match result with document and score
            
        Returns:
            True if FAQ is relevant, False otherwise
        """
        query_lower = query.lower()
        faq_doc = faq_match_result.get("document")
        
        if not faq_doc:
            return False
            
        faq_content = faq_doc.page_content.lower()
        faq_metadata = faq_doc.metadata
        faq_url = faq_metadata.get('source', '').lower()
        
        # Extract key terms from query (remove stop words)
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import string
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract meaningful terms from query
        query_tokens = word_tokenize(query_lower)
        query_keywords = [word for word in query_tokens 
                         if word not in stop_words and word not in string.punctuation and len(word) > 2]
        
        # For very short queries or nonsensical queries, be extra strict
        if len(query_keywords) == 0 or query.strip() in ['', '?', '??', '???', '????', '?????', '??????']:
            logger.info(f"FAQ validation failed: nonsensical query '{query}'")
            return False
        
        # Check for keyword overlap between query and FAQ
        keyword_overlap = 0
        for keyword in query_keywords:
            if keyword in faq_content or keyword in faq_url:
                keyword_overlap += 1
        
        overlap_ratio = keyword_overlap / len(query_keywords) if query_keywords else 0
        
        # Specific validations for common mismatch patterns
        
        # 1. PhD/Academic program queries shouldn't match library employment
        if any(word in query_lower for word in ['phd', 'doctorate', 'degree', 'program', 'admission', 'apply']):
            if any(word in faq_content for word in ['library', 'work', 'vacancy', 'employment', 'job']):
                logger.info(f"FAQ validation failed: academic query matched library employment FAQ")
                return False
        
        # 2. Library employment queries shouldn't match academic programs
        if any(word in query_lower for word in ['work', 'job', 'employment', 'vacancy', 'hire']):
            if 'library' not in faq_content and any(word in faq_content for word in ['degree', 'program', 'course']):
                logger.info(f"FAQ validation failed: employment query matched academic FAQ")
                return False
        
        # 3. Generic or nonsensical queries shouldn't match specific FAQs
        if query.strip() in ['?', '??', '???', '????', '?????', '??????'] or len(query.strip()) < 3:
            logger.info(f"FAQ validation failed: nonsensical query '{query}' shouldn't match specific FAQ")
            return False
        
        # 4. Time-related queries need to actually be about time/schedules
        if any(word in query_lower for word in ['time', 'when', 'hour', 'schedule']):
            if not any(word in faq_content for word in ['time', 'hour', 'schedule', 'open', 'close', 'operation']):
                logger.info(f"FAQ validation failed: time query matched non-time FAQ")
                return False
        
        # 5. Require minimum keyword overlap (reduced to 25% for bge-large compatibility)
        if overlap_ratio < 0.25:
            logger.info(f"FAQ validation failed: insufficient keyword overlap {overlap_ratio:.2f} for query '{query}'")
            return False
        
        logger.info(f"FAQ validation passed: {overlap_ratio:.2f} keyword overlap for query '{query}'")
        return True
    
    def _validate_input_sanity(self, query: str) -> bool:
        """
        Validate that the input query is meaningful and not nonsensical.
        
        Args:
            query: User's query
            
        Returns:
            True if query is meaningful, False if nonsensical
        """
        query_stripped = query.strip()
        
        # Empty or whitespace only
        if not query_stripped:
            return False
        
        # Only punctuation
        import string
        if all(c in string.punctuation + string.whitespace for c in query_stripped):
            return False
        
        # Single character (except valid ones like 'a')
        if len(query_stripped) == 1 and query_stripped.lower() not in ['a', 'i']:
            return False
        
        # All question marks
        if query_stripped.strip('?').strip() == '':
            return False
        
        # Repeated nonsense
        words = query_stripped.split()
        if len(words) > 1 and len(set(words)) == 1:  # All words are the same
            return False
        
        return True
    
    def _stream_with_postprocess(self, response, docs, original_query):
        """
        Separate method to handle streaming with post-processing.
        This keeps yield statements out of the main process_query method.
        """
        # For streaming responses, collect full response and then post-process
        full_response = ""
        for token in response:
            full_response += token
            yield token
        
        # Post-process the complete response and yield the source URLs
        processed_response = self._post_process_response(full_response, docs)
        if processed_response != full_response:
            # Extract just the added source URLs
            source_part = processed_response[len(full_response):].strip()
            if source_part:
                yield "\n\n" + source_part
        
        # Update memory with the processed response
        self.memory.chat_memory.add_user_message(original_query)
        self.memory.chat_memory.add_ai_message(processed_response)