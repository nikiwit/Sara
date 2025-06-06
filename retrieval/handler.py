"""
Main retrieval handler for processing queries and retrieving documents.
"""

import os
import re
import math
import time
import types
import logging
from typing import List, Dict, Any, Union, Iterator, Tuple
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
    
    def _create_retrievers(self) -> Dict[RetrievalStrategy, Any]:
        """Create retrievers for different strategies."""
        retrievers = {}
        
        # Semantic search retriever
        retrievers[RetrievalStrategy.SEMANTIC] = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVER_K}
        )
        
        # MMR retriever for diversity
        retrievers[RetrievalStrategy.MMR] = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.RETRIEVER_K,
                "fetch_k": Config.RETRIEVER_K * 3,
                "lambda_mult": 0.5
            }
        )
        
        # For keyword and hybrid search, we'll implement custom methods
        
        return retrievers
    
    def _stream_text_response(self, text: str):
        """Helper method to stream text word by word with consistent delay."""
        words = text.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(Config.STREAM_DELAY)
    
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
        
        logger.info(f"Processing {query_type.value} query: {original_query}")
        
        # Check for medical insurance related queries first
        if self._is_medical_insurance_query(original_query):
            logger.info("Detected medical insurance related query")
            medical_docs = self._retrieve_medical_insurance_docs(original_query)
            
            if medical_docs:
                logger.info(f"Found {len(medical_docs)} medical insurance related documents")
                # Format context with medical insurance docs
                context = self._format_medical_insurance_context(medical_docs)
                
                # Generate response
                input_dict = {
                    "question": original_query,
                    "context": context,
                    "chat_history": self.memory.chat_memory.messages,
                    "is_medical_insurance": True
                }
                
                # Generate response through LLM with streaming if requested
                from response.generator import RAGSystem
                response = RAGSystem.stream_ollama_response(
                    self._create_prompt(input_dict), 
                    Config.LLM_MODEL_NAME,
                    stream_output=stream
                )
                
                # If we're not streaming, update memory
                if not stream:
                    self.memory.chat_memory.add_user_message(original_query)
                    self.memory.chat_memory.add_ai_message(response)
                
                return response
        
        # Handle identity questions with SystemInformation
        if query_type == QueryType.IDENTITY:
            response = SystemInformation.get_response_for_identity_query(original_query)
                
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
            
            # If streaming, return character by character iterator instead of full response
            if stream:
                # Return an iterator that yields one character at a time with consistent delay
                return self._stream_text_response(response)
            else:
                return response
        
        # First try direct FAQ matching for a quick answer
        faq_match_result = self.faq_matcher.match_faq(query_analysis)
        
        # Lowered threshold from 0.7 to 0.5 to improve recall
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.5:
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
                Config.LLM_MODEL_NAME,
                stream_output=stream
            )
            
            # If we're not streaming, update memory
            if not stream:
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
        
        # If no good FAQ match, proceed with standard retrieval
        # Select retrieval strategy based on query type
        retrieval_strategy = self._select_retrieval_strategy(query_type)
        logger.info(f"Selected retrieval strategy: {retrieval_strategy.value}")
        
        # Retrieve relevant documents
        context_docs = self._retrieve_documents(expanded_queries, retrieval_strategy)
        
        # If no documents found, try a fallback strategy
        if not context_docs and retrieval_strategy != RetrievalStrategy.HYBRID:
            logger.info("No documents found, trying hybrid fallback strategy")
            context_docs = self._retrieve_documents(expanded_queries, RetrievalStrategy.HYBRID)
        
        # If still no relevant documents, return a "no information" response
        if not context_docs:
            no_info_response = (
                "I don't have specific information about that in the APU knowledge base. "
                "To get accurate information on this topic, I'd recommend contacting the appropriate "
                "department at APU directly."
            )
            
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(no_info_response)
            
            # If streaming, return as character stream for consistency
            if stream:
                return self._stream_text_response(no_info_response)
            else:
                return no_info_response
        
        # Assess document relevance - lowered threshold from 0.3 to 0.2
        relevance_score = self._assess_document_relevance(context_docs, original_query)
        
        # If relevance is too low, return low confidence response
        if relevance_score < 0.2:  # Lowered threshold for better recall
            low_confidence_response = (
                "I'm not confident I have the specific information you're looking for in the APU knowledge base. "
                "The information I found might not be directly relevant to your question. "
                "For accurate information, you may want to contact the appropriate APU department directly."
            )
            
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(low_confidence_response)
            
            # If streaming, return as character stream for consistency
            if stream:
                return self._stream_text_response(low_confidence_response)
            else:
                return low_confidence_response
        
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
            "relevance_score": relevance_score
        }
        
        # Generate response through LLM with streaming if requested
        # Import here to avoid circular imports
        from response.generator import RAGSystem
        response = RAGSystem.stream_ollama_response(
            self._create_prompt(input_dict), 
            Config.LLM_MODEL_NAME,
            stream_output=stream
        )
        
        # If we're not streaming, update memory
        if not stream:
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
        
        return response
    
    def _is_medical_insurance_query(self, query: str) -> bool:
        """
        Check if a query is related to medical insurance.
        
        Args:
            query: Query string
            
        Returns:
            True if the query is related to medical insurance, False otherwise
        """
        query_lower = query.lower()
        
        # Check for medical insurance keywords
        medical_keywords = ["medical", "insurance", "health", "card", "collect", "pickup", "pick up", "pick-up"]
        keyword_count = sum(1 for kw in medical_keywords if kw in query_lower)
        
        # If at least two keywords are present, it's likely a medical insurance query
        if keyword_count >= 2:
            return True
            
        # Check for specific phrases
        medical_phrases = [
            "medical insurance", 
            "health insurance", 
            "insurance card", 
            "medical card",
            "get my insurance",
            "collect my insurance",
            "pickup my insurance",
            "pick up my insurance"
        ]
        
        for phrase in medical_phrases:
            if phrase in query_lower:
                return True
                
        return False
    
    def _retrieve_medical_insurance_docs(self, query: str) -> List[Document]:
        """
        Retrieve documents specifically related to medical insurance.
        
        Args:
            query: Query string
            
        Returns:
            List of documents related to medical insurance
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        
        # Find documents with medical insurance metadata
        medical_docs = []
        for i, doc_text in enumerate(documents):
            if i >= len(metadatas):
                continue
                
            metadata = metadatas[i]
            
            # Check for medical insurance metadata
            if metadata.get('is_medical_insurance', False) or metadata.get('priority_topic') == 'medical_insurance':
                medical_docs.append(
                    Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                )
                continue
                
            # Also check content for medical insurance keywords
            doc_lower = doc_text.lower()
            if "medical insurance" in doc_lower or "collect" in doc_lower and "insurance" in doc_lower:
                # Check if title contains relevant keywords
                title = metadata.get('page_title', '').lower()
                if "medical" in title or "insurance" in title or "collect" in title:
                    medical_docs.append(
                        Document(
                            page_content=doc_text,
                            metadata=metadata
                        )
                    )
        
        return medical_docs
    
    def _format_medical_insurance_context(self, docs: List[Document]) -> str:
        """
        Format medical insurance documents into a context for the LLM.
        
        Args:
            docs: List of medical insurance documents
            
        Returns:
            Formatted context string
        """
        context = "--- MEDICAL INSURANCE INFORMATION ---\n\n"
        
        for doc in docs:
            title = doc.metadata.get('page_title', 'Medical Insurance Information')
            context += f"Question: {title}\n\n"
            context += f"Answer:\n{doc.page_content}\n\n"
            
        context += "Instructions: Answer the user's question about medical insurance using ONLY the information provided above. Be direct and specific about where to collect the medical insurance card if that information is present.\n\n"
        
        return context
    
    def _format_faq_match(self, match_result: Dict[str, Any]) -> str:
        """Format a direct FAQ match into a context for the LLM."""
        doc = match_result["document"]
        score = match_result["match_score"]
        
        # Get metadata
        title = doc.metadata.get("page_title", "Unknown Title")
        source = doc.metadata.get("source", "Unknown Source")
        filename = doc.metadata.get("filename", os.path.basename(source) if source != "Unknown Source" else "Unknown File")
        
        # Format the context with clear but not overly intimidating instructions
        context = f"--- DIRECT FAQ MATCH (Confidence: {score:.2f}) ---\n\n"
        context += f"Question: {title}\n\n"
        context += f"SOURCE TEXT:\n\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
        
        # Add valid email addresses that exist
        emails_in_text = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', doc.page_content)
        if emails_in_text:
            context += "Email addresses mentioned in the text:\n"
            for email in emails_in_text:
                context += f"- {email}\n"
            context += "\nRemember: Only use email addresses that are explicitly provided in the source text.\n"
        
        # Add explicit note about contact options if "or" appears
        if " or " in doc.page_content.lower():
            context += "\nNote: When multiple contact options are mentioned with 'or', these are separate alternatives.\n"
        
        # Add related pages if available
        related_pages = doc.metadata.get("related_pages", [])
        if related_pages:
            context += "\nRelated information can be found in:\n"
            for page in related_pages[:3]:
                context += f"- {page}\n"
        
        return context
    
    def _select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select the appropriate retrieval strategy based on query type.
        """
        if Config.RETRIEVER_SEARCH_TYPE != "auto":
            # Use the configured strategy if not set to auto
            return RetrievalStrategy(Config.RETRIEVER_SEARCH_TYPE)
        
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
                
                for doc in docs:
                    # Create a unique ID based on content and source
                    doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Sort docs by relevance score if available
        all_docs = self._rerank_documents(all_docs, queries[0])
        
        logger.info(f"Retrieved {len(all_docs)} unique documents")
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
            
            # Check for medical insurance priority
            if metadata.get('is_medical_insurance', False) or metadata.get('priority_topic') == 'medical_insurance':
                if any(kw in query.lower() for kw in ["medical", "insurance", "collect", "card"]):
                    score += 5.0  # Very high boost for medical insurance pages
            
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
        for score, i in scored_docs[:Config.RETRIEVER_K]:
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
        Perform hybrid retrieval (semantic + keyword).
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get semantic search results
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get keyword search results
        keyword_docs = self._keyword_retrieval(query)
        
        # Combine results with deduplication
        seen_ids = set()
        combined_docs = []
        
        # Semantic docs first (with higher weight)
        for doc in semantic_docs:
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Semantic score to metadata
                doc.metadata['retrieval_score'] = 1.0 - Config.KEYWORD_RATIO
                combined_docs.append(doc)
        
        # Keyword docs
        for doc in keyword_docs:
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Keyword score to metadata
                doc.metadata['retrieval_score'] = Config.KEYWORD_RATIO
                combined_docs.append(doc)
        
        # Rerank combined results
        return self._rerank_documents(combined_docs, query)
    
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
            
            # Check for medical insurance priority
            medical_boost = 0
            if doc.metadata.get('is_medical_insurance', False) or doc.metadata.get('priority_topic') == 'medical_insurance':
                if any(kw in query.lower() for kw in ["medical", "insurance", "collect", "card"]):
                    medical_boost = 2.0  # Very high boost for medical insurance pages
            
            # Calculate final score
            final_score = base_score + phrase_match + keyword_match + faq_boost + apu_kb_boost + medical_boost
            
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
        
        # Check for medical insurance matches
        medical_match_score = 0
        if any(kw in query.lower() for kw in ["medical", "insurance", "collect", "card"]):
            for doc in docs:
                if doc.metadata.get('is_medical_insurance', False) or doc.metadata.get('priority_topic') == 'medical_insurance':
                    medical_match_score = 1.0
                    break
                elif "medical insurance" in doc.page_content.lower() or ("collect" in doc.page_content.lower() and "insurance" in doc.page_content.lower()):
                    medical_match_score = 0.8
                    break
        
        # Calculate overall relevance
        relevance = max(
            exact_match_score,
            keyword_coverage * 0.7,
            faq_match_score,
            apu_kb_match_score,
            medical_match_score
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
        is_medical_insurance = input_dict.get("is_medical_insurance", False)
        relevance_score = input_dict.get("relevance_score", 0.5)
        
        if is_medical_insurance:
            # Medical insurance prompt (unchanged)
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the question about medical insurance based ONLY on the provided information.

    Question: {question}

    {context}

    Instructions:
    1. Answer the question directly and concisely based on the provided information.
    2. Be specific about where to collect the medical insurance card if that information is present.
    3. Include any relevant details like location, counter number, or staff names mentioned in the context.
    4. Do not make up information or use knowledge outside the provided context.
    5. Use a helpful and professional tone appropriate for a university assistant.

    Answer:"""
        
        elif is_faq_match:
            # Modified FAQ match prompt - more natural and direct
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the student's question directly and naturally.

    Question: {question}

    {context}

    Instructions:
    1. Provide a direct, helpful answer to the student's question using the information provided.
    2. Write as if you're speaking directly to the student - use "you" and be conversational.
    3. Do not mention "FAQ", "provided information", "based on", or reference sources.
    4. Give clear, actionable advice where applicable.
    5. If the information doesn't fully address their specific situation, suggest they contact APU directly for personalized guidance.
    6. Use a helpful and professional tone appropriate for a university assistant.

    Answer:"""

        elif relevance_score < 0.25:  # Low relevance - likely partial match
            # Enhanced prompt for partial matches
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
            # Regular prompt for good relevance (unchanged)
            prompt = f"""You are an AI assistant for APU (Asia Pacific University). Answer the question based ONLY on the provided context.

    Question: {question}

    Context:
    {context}

    Instructions:
    1. Answer the question directly and concisely based on the provided context.
    2. If the context contains the exact answer, use it.
    3. Do not make up information or use knowledge outside the provided context.
    4. If the context doesn't fully answer the question, acknowledge this and suggest contacting APU directly.
    5. Use a helpful and professional tone appropriate for a university assistant.
    6. If the context mentions specific locations, people, or contact information, include these details in your answer.

    Answer:"""

        return prompt