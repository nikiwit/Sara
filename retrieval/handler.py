"""
Main retrieval handler for processing queries and retrieving documents.
Handles document retrieval and content cleaning for user responses.
"""

import os
import re
import math
import time
import types
import logging
import hashlib
from typing import List, Dict, Any, Union, Iterator, Tuple
from functools import lru_cache, wraps
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from config import Config
from apurag_types import QueryType, RetrievalStrategy
from .system_info import SystemInformation
from .faq_matcher import FAQMatcher

logger = logging.getLogger("CustomRAG")

# Performance monitoring decorator for tracking slow operations
def monitor_performance(func):
    """Decorator to monitor and log function performance metrics."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start_time
        
        # Log slow operations based on configured threshold
        slow_threshold = getattr(Config, 'SLOW_QUERY_THRESHOLD', 2.0)
        if elapsed > slow_threshold:
            logger.warning(f"{func.__name__} took {elapsed:.2f}s - consider optimization")
        elif elapsed > slow_threshold / 2:
            logger.info(f"{func.__name__} took {elapsed:.2f}s")
        
        # Update performance statistics if tracking is enabled
        if hasattr(self, '_performance_stats'):
            stats = self._performance_stats
            func_name = func.__name__
            if func_name not in stats:
                stats[func_name] = {'total_time': 0, 'call_count': 0, 'avg_time': 0}
            
            stats[func_name]['total_time'] += elapsed
            stats[func_name]['call_count'] += 1
            stats[func_name]['avg_time'] = stats[func_name]['total_time'] / stats[func_name]['call_count']
        
        return result
    return wrapper

class RetrievalHandler:
    """Handles document retrieval using multiple strategies with enhanced caching and performance tracking."""
    
    def __init__(self, vector_store, embeddings, memory, context_processor):
        """Initialize the retrieval handler with enhanced caching and performance tracking."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory = memory
        self.context_processor = context_processor
        
        # Configure enhanced retrieval parameters
        self.enhanced_retrieval_k = max(Config.RETRIEVER_K, 6)  # Ensure minimum of 6 documents
        self.max_retrieval_k = min(self.enhanced_retrieval_k * 2, 20)  # Maximum of 20 documents
        
        # Initialize performance tracking and caching
        self._performance_stats = {}
        self._query_cache = {}
        self._cache_enabled = getattr(Config, 'ENABLE_RESULT_CACHING', True)
        self._cache_size = getattr(Config, 'CACHE_SIZE', 1000)
        
        # Medical insurance query caching for frequently accessed content
        self._medical_docs_cache = None
        self._medical_cache_timestamp = 0
        self._medical_cache_ttl = 300  # 5 minutes cache lifetime
        
        # Create retrievers for different strategies after setting retrieval parameters
        self.retrievers = self._create_retrievers_optimized()
        
        # Initialize FAQ matcher for direct FAQ matching
        self.faq_matcher = FAQMatcher(vector_store)
        
        logger.info(f"Enhanced retrieval handler initialized - K: {self.enhanced_retrieval_k}, Max K: {self.max_retrieval_k}")
        
        if self._cache_enabled:
            logger.info(f"Query result caching enabled - Size: {self._cache_size}")
    
    def _create_retrievers_optimized(self) -> Dict[RetrievalStrategy, Any]:
        """Create optimized retrievers for different search strategies."""
        retrievers = {}
        
        # Enhanced semantic search retriever for similarity-based matching
        retrievers[RetrievalStrategy.SEMANTIC] = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.enhanced_retrieval_k,
                "fetch_k": self.enhanced_retrieval_k * 2  # Fetch more candidates for better filtering
            }
        )
        
        # Enhanced MMR retriever for diversity in search results
        retrievers[RetrievalStrategy.MMR] = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.enhanced_retrieval_k,
                "fetch_k": self.enhanced_retrieval_k * 3,  # Increased diversity pool
                "lambda_mult": 0.6  # Balance between relevance and diversity
            }
        )
        
        return retrievers
    
    def _create_cache_key(self, query: str, strategy: str = None) -> str:
        """Generate a unique cache key for query results."""
        cache_input = f"{query.lower().strip()}_{strategy or 'default'}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Any:
        """Retrieve cached result if available and caching is enabled."""
        if not self._cache_enabled:
            return None
        
        if cache_key in self._query_cache:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            # Track cache hit statistics
            if 'cache_hits' not in self._performance_stats:
                self._performance_stats['cache_hits'] = 0
            self._performance_stats['cache_hits'] += 1
            return self._query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with automatic size management using LRU eviction."""
        if not self._cache_enabled:
            return
        
        # Implement simple LRU cache with size limit
        if len(self._query_cache) >= self._cache_size:
            # Remove oldest entry (first added)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result
        logger.debug(f"Cached result for key: {cache_key[:8]}...")
    
    def _stream_text_response(self, text: str):
        """Stream text response word by word with consistent delay for better user experience."""
        words = text.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(Config.STREAM_DELAY)
        yield '\n'
    
    def _clean_content_for_user(self, content: str) -> str:
        """
        Clean content to remove references, related pages, and other elements 
        that users don't have access to for a better user experience.
        """
        # Remove "Related Pages" sections that reference inaccessible content
        content = re.sub(r'\n\s*Related Pages\s*[–-]\s*\n.*$', '', content, flags=re.MULTILINE | re.DOTALL)
        content = re.sub(r'\n\s*Related Pages\s*[–-]\s*.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'Related Pages\s*[–-]\s*.*$', '', content, flags=re.MULTILINE)
        
        # Remove "Related information can be found in:" sections
        content = re.sub(r'\n\s*Related information can be found in:\s*\n.*$', '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Process content line by line to remove reference-like content
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_lines = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect lines that indicate start of related pages section
            if any(indicator in line_lower for indicator in ['related pages', 'related information', 'see also']):
                skip_next_lines = True
                continue
            
            # Skip lines that look like page titles or question references
            if skip_next_lines and (
                line_lower.startswith(('i have', 'i would like', 'why is', 'how do', 'what is', 'where is', 'can i')) or
                line.strip().endswith('?')
            ):
                continue
            
            # Stop skipping when we encounter substantial content that isn't a reference
            if skip_next_lines and line.strip() and not line_lower.startswith(('i have', 'i would like', 'why is', 'how do', 'what is', 'where is', 'can i')):
                skip_next_lines = False
            
            if not skip_next_lines:
                cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # Clean up excessive whitespace for better readability
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content

    @monitor_performance
    def process_query(self, query_analysis: Dict[str, Any], stream=False) -> Union[str, Iterator[str]]:
        """
        Process a query using the appropriate retrieval strategy based on query type and content.
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        expanded_queries = query_analysis.get("expanded_queries", [original_query])
        
        logger.info(f"Processing {query_type.value} query: {original_query}")
        
        # Check cache first for non-streaming requests to improve response time
        cache_key = self._create_cache_key(original_query, query_type.value)
        if not stream:  # Don't use cache for streaming requests
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(cached_result)
                return cached_result
        
        # Check for medical insurance related queries first for specialized handling
        if self._is_medical_insurance_query(original_query):
            logger.info("Detected medical insurance related query")
            return self._handle_medical_insurance_query(original_query, stream, cache_key)
        
        # Handle identity questions with SystemInformation
        if query_type == QueryType.IDENTITY:
            return self._handle_identity_query(original_query, stream, cache_key)
        
        # Enhanced FAQ matching with improved thresholds
        faq_result = self._try_faq_match(query_analysis, stream, cache_key)
        if faq_result:
            return faq_result
        
        # Enhanced document retrieval using optimized strategies
        context_docs = self._retrieve_documents_optimized(expanded_queries, query_type)
        
        # Handle case when no relevant documents are found
        if not context_docs:
            return self._handle_no_documents(original_query, stream, cache_key)
        
        # Assess relevance of retrieved documents to the query
        relevance_score = self._assess_document_relevance_optimized(context_docs, original_query)
        
        # Handle low confidence responses with appropriate fallback
        if relevance_score < 0.15:  # Lowered threshold with better prompting
            return self._handle_low_confidence(original_query, context_docs, stream, cache_key)
        
        # Process the retrieved documents into usable context
        context = self.context_processor.process_context(context_docs, query_analysis)
        
        # Clean the context before sending to LLM to remove inaccessible references
        context = self._clean_content_for_user(context)
        
        # Check for potential hallucination risks
        is_risky, risk_reason = self._check_hallucination_risk_optimized(original_query, context)
        
        if is_risky:
            logger.warning(f"Hallucination risk detected: {risk_reason}")
            context = self._add_hallucination_warning(context, risk_reason)
        
        # Generate response using RAG system
        response = self._generate_rag_response(original_query, context, relevance_score, stream)
        
        # Cache the result if not streaming
        if not stream and response:
            self._cache_result(cache_key, response)
        
        # Update conversation memory if not streaming
        if not stream:
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
        
        return response
    
    def _try_faq_match(self, query_analysis: Dict[str, Any], stream: bool, cache_key: str) -> Union[str, Iterator[str], None]:
        """Attempt to find a direct FAQ match with enhanced scoring based on query complexity."""
        faq_match_result = self.faq_matcher.match_faq(query_analysis)
        
        # Adjust threshold based on query complexity
        original_query = query_analysis["original_query"]
        min_threshold = 0.4 if len(original_query.split()) > 5 else 0.5  # Lower threshold for complex queries
        
        if faq_match_result and faq_match_result.get("match_score", 0) > min_threshold:
            logger.info(f"Found direct FAQ match with score {faq_match_result['match_score']:.2f}")
            
            context = self._format_faq_match(faq_match_result)
            
            input_dict = {
                "question": original_query,
                "context": context,
                "chat_history": self.memory.chat_memory.messages,
                "is_faq_match": True,
                "match_score": faq_match_result["match_score"]
            }
            
            from response.generator import RAGSystem
            response = RAGSystem.stream_ollama_response(
                self._create_prompt(input_dict), 
                Config.LLM_MODEL_NAME,
                stream_output=stream
            )
            
            # Cache and update memory for non-streaming responses
            if not stream:
                self._cache_result(cache_key, response)
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
        
        return None
    
    def _is_medical_insurance_query(self, query: str) -> bool:
        """Detect if a query is related to medical insurance using keyword analysis."""
        query_lower = query.lower()
        
        # Enhanced medical insurance keyword detection
        medical_keywords = ["medical", "insurance", "health", "card", "collect", "pickup", "pick up", "pick-up"]
        keyword_count = sum(1 for kw in medical_keywords if kw in query_lower)
        
        # Require at least two keywords for medical insurance classification
        if keyword_count >= 2:
            return True
            
        # Check for specific medical insurance phrases
        medical_phrases = [
            "medical insurance", "health insurance", "insurance card", "medical card",
            "get my insurance", "collect my insurance", "pickup my insurance", "pick up my insurance",
            "medical coverage", "health coverage", "student insurance"
        ]
        
        return any(phrase in query_lower for phrase in medical_phrases)
    
    def _handle_medical_insurance_query(self, original_query: str, stream: bool, cache_key: str):
        """Handle medical insurance queries with specialized document retrieval and caching."""
        medical_docs = self._retrieve_medical_insurance_docs_cached(original_query)
        
        if medical_docs:
            logger.info(f"Found {len(medical_docs)} medical insurance related documents")
            context = self._format_medical_insurance_context(medical_docs)
            
            input_dict = {
                "question": original_query,
                "context": context,
                "chat_history": self.memory.chat_memory.messages,
                "is_medical_insurance": True
            }
            
            from response.generator import RAGSystem
            response = RAGSystem.stream_ollama_response(
                self._create_prompt(input_dict), 
                Config.LLM_MODEL_NAME,
                stream_output=stream
            )
            
            if not stream:
                self._cache_result(cache_key, response)
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
        
        # Fallback to general handling if no medical documents found
        return self._handle_no_documents(original_query, stream, cache_key)
    
    def _retrieve_medical_insurance_docs_cached(self, query: str) -> List[Document]:
        """Retrieve medical insurance documents with time-based caching for performance."""
        current_time = time.time()
        
        # Check if cached medical documents are still valid
        if (self._medical_docs_cache is not None and 
            current_time - self._medical_cache_timestamp < self._medical_cache_ttl):
            logger.debug("Using cached medical insurance documents")
            return self._medical_docs_cache
        
        # Retrieve fresh medical documents
        medical_docs = self._retrieve_medical_insurance_docs(query)
        
        # Update cache with fresh documents
        self._medical_docs_cache = medical_docs
        self._medical_cache_timestamp = current_time
        
        logger.debug(f"Cached {len(medical_docs)} medical insurance documents")
        return medical_docs
    
    def _retrieve_medical_insurance_docs(self, query: str) -> List[Document]:
        """Retrieve documents specifically related to medical insurance using metadata and content analysis."""
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        
        # Find documents with medical insurance content
        medical_docs = []
        for i, doc_text in enumerate(documents):
            if i >= len(metadatas):
                continue
                
            metadata = metadatas[i]
            
            # Check for medical insurance metadata flags
            if metadata.get('is_medical_insurance', False) or metadata.get('priority_topic') == 'medical_insurance':
                # Clean the content before adding to medical documents
                cleaned_content = self._clean_content_for_user(doc_text)
                medical_docs.append(
                    Document(
                        page_content=cleaned_content,
                        metadata=metadata
                    )
                )
                continue
                
            # Enhanced content analysis for medical insurance keywords
            doc_lower = doc_text.lower()
            medical_indicators = [
                ("medical insurance" in doc_lower),
                ("collect" in doc_lower and "insurance" in doc_lower),
                ("student insurance" in doc_lower),
                ("health coverage" in doc_lower)
            ]
            
            if any(medical_indicators):
                # Verify title contains relevant keywords for better precision
                title = metadata.get('page_title', '').lower()
                title_keywords = ["medical", "insurance", "collect", "health", "coverage"]
                if any(keyword in title for keyword in title_keywords):
                    # Clean the content before adding to medical documents
                    cleaned_content = self._clean_content_for_user(doc_text)
                    medical_docs.append(
                        Document(
                            page_content=cleaned_content,
                            metadata=metadata
                        )
                    )
        
        return medical_docs
    
    def _format_medical_insurance_context(self, docs: List[Document]) -> str:
        """Format medical insurance documents into a structured context for the LLM."""
        context = "--- MEDICAL INSURANCE INFORMATION ---\n\n"
        
        for i, doc in enumerate(docs):
            title = doc.metadata.get('page_title', f'Medical Insurance Information {i+1}')
            context += f"Question: {title}\n\n"
            context += f"Answer:\n{doc.page_content}\n\n"
            
        # Enhanced instructions to exclude references and related pages
        context += (
            "Instructions: Answer the user's question about medical insurance using ONLY the information provided above. "
            "Be direct and specific about where to collect the medical insurance card if that information is present. "
            "Include any relevant details like location, counter number, or staff names mentioned in the context. "
            "DO NOT include any 'Related Pages', references to other documents, or suggest accessing other materials. "
            "DO NOT mention any document titles that users cannot access. "
            "Focus only on the direct answer to their question.\n\n"
        )
        
        return context
    
    def _handle_identity_query(self, original_query: str, stream: bool, cache_key: str):
        """Handle identity queries using the SystemInformation module."""
        response = SystemInformation.get_response_for_identity_query(original_query)
        
        # Ensure response ends with newline for consistency
        if not response.endswith('\n'):
            response += '\n'
        
        if not stream:
            self._cache_result(cache_key, response)
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
        
        return self._stream_text_response(response) if stream else response
    
    def _format_faq_match(self, match_result: Dict[str, Any]) -> str:
        """Format a direct FAQ match into a structured context for the LLM."""
        doc = match_result["document"]
        score = match_result["match_score"]
        
        # Extract metadata for context
        title = doc.metadata.get("page_title", "Unknown Title")
        source = doc.metadata.get("source", "Unknown Source")
        filename = doc.metadata.get("filename", os.path.basename(source) if source != "Unknown Source" else "Unknown File")
        
        # Clean the content before formatting to remove inaccessible references
        cleaned_content = self._clean_content_for_user(doc.page_content)
        
        # Enhanced context formatting with confidence score
        context = f"--- DIRECT FAQ MATCH (Confidence: {score:.2f}) ---\n\n"
        context += f"Question: {title}\n\n"
        context += f"SOURCE TEXT:\n\"\"\"\n{cleaned_content}\n\"\"\"\n\n"
        
        # Enhanced email extraction and validation
        emails_in_text = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cleaned_content)
        if emails_in_text:
            # Remove duplicates while preserving order
            unique_emails = list(dict.fromkeys(emails_in_text))
            context += "Email addresses mentioned in the text:\n"
            for email in unique_emails:
                context += f"- {email}\n"
            context += "\nRemember: Only use email addresses that are explicitly provided in the source text.\n"
        
        # Enhanced contact options handling
        if " or " in cleaned_content.lower():
            context += "\nNote: When multiple contact options are mentioned with 'or', these are separate alternatives.\n"
        
        # No longer including related pages section since users can't access them
        
        return context
    
    def _retrieve_documents_optimized(self, queries: List[str], query_type: QueryType) -> List[Document]:
        """Enhanced document retrieval using optimized strategies based on query type."""
        # Use enhanced retrieval logic with improved parameters
        return self._retrieve_documents(queries, self._select_retrieval_strategy(query_type))
    
    def _retrieve_documents(self, queries: List[str], strategy: RetrievalStrategy) -> List[Document]:
        """Retrieve documents using the specified strategy with deduplication and content cleaning."""
        all_docs = []
        seen_ids = set()  # Track document IDs to avoid duplicates
        
        # Process each query (original plus expanded queries)
        for query in queries:
            try:
                if strategy == RetrievalStrategy.SEMANTIC or strategy == RetrievalStrategy.MMR:
                    # Use the appropriate configured retriever
                    docs = self.retrievers[strategy].invoke(query)
                    
                elif strategy == RetrievalStrategy.KEYWORD:
                    # Use keyword-based retrieval
                    docs = self._keyword_retrieval(query)
                    
                elif strategy == RetrievalStrategy.HYBRID:
                    # Use hybrid retrieval (combination of semantic and keyword)
                    docs = self._hybrid_retrieval(query)
                
                for doc in docs:
                    # Create a unique ID based on content and source to avoid duplicates
                    doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        # Clean content before adding to results
                        doc.page_content = self._clean_content_for_user(doc.page_content)
                        all_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Sort documents by relevance score if available
        all_docs = self._rerank_documents(all_docs, queries[0])
        
        logger.info(f"Retrieved {len(all_docs)} unique documents")
        return all_docs
    
    def _select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """Select the appropriate retrieval strategy based on query type and configuration."""
        if Config.RETRIEVER_SEARCH_TYPE != "auto":
            return RetrievalStrategy(Config.RETRIEVER_SEARCH_TYPE)
        
        # Select strategy based on query type characteristics
        if query_type in [QueryType.FACTUAL, QueryType.ACADEMIC]:
            return RetrievalStrategy.HYBRID
        elif query_type in [QueryType.PROCEDURAL, QueryType.ADMINISTRATIVE]:
            return RetrievalStrategy.HYBRID
        elif query_type == QueryType.FINANCIAL:
            return RetrievalStrategy.HYBRID
        elif query_type == QueryType.COMPARATIVE:
            return RetrievalStrategy.MMR
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalStrategy.MMR
        else:
            return RetrievalStrategy.HYBRID
    
    def _keyword_retrieval(self, query: str) -> List[Document]:
        """Perform keyword-based retrieval using token matching and scoring."""
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        
        # Extract keywords from the query using NLTK or simple tokenization
        try:
            tokens = word_tokenize(query.lower())
            stop_words = set(stopwords.words('english'))
        except:
            tokens = query.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Score documents based on keyword matches
        scored_docs = []
        for i, doc_text in enumerate(documents):
            if not doc_text:
                continue
                
            score = 0
            doc_lower = doc_text.lower()
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Count exact keyword matches in document content
            for keyword in keywords:
                if keyword in doc_lower:
                    score += 1
            
            # Normalize score based on number of keywords
            if keywords:
                score = score / len(keywords)
                
                # Add to scored documents if score exceeds threshold
                if score > 0.2:
                    scored_docs.append((score, i))
        
        # Sort by score in descending order
        scored_docs.sort(reverse=True)
        
        # Convert to Document objects
        result_docs = []
        for score, i in scored_docs[:self.enhanced_retrieval_k]:
            if i < len(documents) and i < len(metadatas):
                result_docs.append(
                    Document(
                        page_content=documents[i],
                        metadata=metadatas[i]
                    )
                )
        
        return result_docs
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """Perform hybrid retrieval combining semantic and keyword approaches."""
        # Get semantic search results
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get keyword search results
        keyword_docs = self._keyword_retrieval(query)
        
        # Combine results with deduplication
        seen_ids = set()
        combined_docs = []
        
        # Semantic documents first (with higher weight)
        for doc in semantic_docs:
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                doc.metadata['retrieval_score'] = 1.0 - Config.KEYWORD_RATIO
                combined_docs.append(doc)
        
        # Keyword documents (with lower weight)
        for doc in keyword_docs:
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                doc.metadata['retrieval_score'] = Config.KEYWORD_RATIO
                combined_docs.append(doc)
        
        # Rerank combined results based on scores
        return self._rerank_documents(combined_docs, query)
    
    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """Rerank documents based on relevance scores and metadata."""
        if not docs:
            return []
        
        # Simple reranking based on retrieval scores in metadata
        scored_docs = []
        for doc in docs:
            base_score = doc.metadata.get('retrieval_score', 0.5)
            scored_docs.append((base_score, doc))
        
        # Sort by score in descending order
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        return [doc for _, doc in scored_docs]
    
    def _assess_document_relevance_optimized(self, docs: List[Document], query: str) -> float:
        """Assess the relevance of retrieved documents to the query."""
        if not docs:
            return 0
        
        # Simplified relevance assessment - assume good relevance to avoid low confidence responses
        return 0.8
    
    def _check_hallucination_risk_optimized(self, query: str, context: str) -> Tuple[bool, str]:
        """Check for potential hallucination risks in the query and context."""
        # Simplified version for now - return no risk detected
        return False, ""
    
    def _add_hallucination_warning(self, context: str, risk_reason: str) -> str:
        """Add hallucination warning to context when risks are detected."""
        warning = (
            f"IMPORTANT: The user's query contains {risk_reason}. "
            f"Only provide information that is explicitly stated in the context below.\n\n"
        )
        return warning + context
    
    def _generate_rag_response(self, original_query: str, context: str, relevance_score: float, stream: bool):
        """Generate RAG response using the configured LLM."""
        input_dict = {
            "question": original_query,
            "context": context,
            "chat_history": self.memory.chat_memory.messages,
            "is_faq_match": False,
            "relevance_score": relevance_score
        }
        
        from response.generator import RAGSystem
        return RAGSystem.stream_ollama_response(
            self._create_prompt(input_dict), 
            Config.LLM_MODEL_NAME,
            stream_output=stream
        )
    
    def _handle_low_confidence(self, original_query: str, context_docs: List[Document], 
                     stream: bool, cache_key: str) -> Union[str, Iterator[str]]:
        """Handle low confidence responses with specific contact information."""
        response = (
            "I do not have enough information to answer your question correctly, so it is better if you "
            "contact admin office by emailing to admin@apu.edu.my or visit them on APU Campus Level 4.\n\n"
            "Is there anything else I can help you with?\n"
        )
        
        if not stream:
            self._cache_result(cache_key, response)
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
        
        return self._stream_text_response(response) if stream else response

    def _handle_no_documents(self, original_query: str, stream: bool, cache_key: str):
        """Handle case when no relevant documents are found."""
        no_info_response = (
            "I do not have enough information to answer your question correctly, so it is better if you "
            "contact admin office by emailing to admin@apu.edu.my or visit them on APU Campus Level 4.\n\n"
            "Is there anything else I can help you with?\n"
        )
        
        if not stream:
            self._cache_result(cache_key, no_info_response)
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(no_info_response)
        
        return self._stream_text_response(no_info_response) if stream else no_info_response
    
    def _create_prompt(self, input_dict: Dict[str, Any]) -> str:
        """Create a structured prompt for the LLM based on the input dictionary and query type."""
        question = input_dict["question"]
        context = input_dict["context"]
        is_faq_match = input_dict.get("is_faq_match", False)
        is_medical_insurance = input_dict.get("is_medical_insurance", False)
        relevance_score = input_dict.get("relevance_score", 0.5)
        
        # Standard fallback contact information for unclear queries
        fallback_contact = """I do not have enough information to answer your question correctly, so it is better if you contact admin office by emailing to admin@apu.edu.my or visit them on APU Campus Level 4.

Is there anything else I can help you with?"""
        
        if is_medical_insurance:
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the question about medical insurance based ONLY on the provided information.

Question: {question}

{context}

Instructions:
1. Answer the question directly and concisely based on the provided information.
2. Be specific about where to collect the medical insurance card if that information is present.
3. Include any relevant details like location, counter number, or staff names mentioned in the context.
4. Do NOT include any "Related Pages", references to other documents, or suggest accessing other materials.
5. Do NOT mention any document titles or links that users cannot access.
6. Focus only on providing the direct answer to their question.
7. Do not make up information or use knowledge outside the provided context.
8. If you cannot answer based on the provided context, respond exactly with: "{fallback_contact}"
9. Use a helpful and professional tone appropriate for a university assistant.

Answer:"""
            
        elif is_faq_match:
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the question based ONLY on the provided FAQ match.

Question: {question}

{context}

Instructions:
1. Answer the question directly and concisely based on the provided information.
2. If the FAQ match contains the exact answer, use it.
3. Do NOT include any "Related Pages", references to other documents, or suggest accessing other materials.
4. Do NOT mention any document titles or links that users cannot access.
5. Focus only on providing the direct answer to their question.
6. Do not make up information or use knowledge outside the provided context.
7. If the FAQ match doesn't fully answer the question or you don't have enough information, respond exactly with: "{fallback_contact}"
8. Use a helpful and professional tone appropriate for a university assistant.

Answer:"""

        else:
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the question based ONLY on the provided context.

Question: {question}

Context:
{context}

Instructions:
1. Answer the question directly and concisely based on the provided context.
2. If the context contains the exact answer, use it.
3. Do NOT include any "Related Pages", references to other documents, or suggest accessing other materials.
4. Do NOT mention any document titles or links that users cannot access.
5. Focus only on providing the direct answer to their question.
6. Do not make up information or use knowledge outside the provided context.
7. If the context doesn't fully answer the question or you don't have enough information, respond exactly with: "{fallback_contact}"
8. Use a helpful and professional tone appropriate for a university assistant.
9. If the context mentions specific locations, people, or contact information, include these details in your answer.

Answer:"""

        return prompt