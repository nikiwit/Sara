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
        
        # Handle identity questions with SystemInformation
        if query_type == QueryType.IDENTITY:
            response = SystemInformation.get_response_for_identity_query(original_query)
                
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
            
            # If streaming, return character by character iterator instead of full response
            if stream:
                # Return an iterator that yields one character at a time with slight delay
                def character_stream(text):
                    for char in text:
                        yield char
                        time.sleep(0.01)  # Same delay as in ConversationHandler
                        
                return character_stream(response)
            else:
                return response
        
        # First try direct FAQ matching for a quick answer
        faq_match_result = self.faq_matcher.match_faq(query_analysis)
        
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.7:
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
            
            # If streaming, return as iterator
            if stream:
                return iter([no_info_response])
            else:
                return no_info_response
        
        # Assess document relevance
        relevance_score = self._assess_document_relevance(context_docs, original_query)
        
        # If relevance is too low, return low confidence response
        if relevance_score < 0.3:  # Low relevance threshold
            low_confidence_response = (
                "I'm not confident I have the specific information you're looking for in the APU knowledge base. "
                "The information I found might not be directly relevant to your question. "
                "For accurate information, you may want to contact the appropriate APU department directly."
            )
            
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(low_confidence_response)
            
            # If streaming, return as iterator
            if stream:
                return iter([low_confidence_response])
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
        
        # Add the valid email addresses that exist
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
                
                # Add unique documents to the result list
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
                # Count occurrences of the keyword
                count = doc_lower.count(keyword)
                score += count
                
                # Bonus for exact phrase match
                if query.lower() in doc_lower:
                    score += 5
                    
                # For APU KB pages, check title match
                if is_apu_kb and keyword in page_title.lower():
                    score += 3  # Boost score for title matches
            
            # Bonus for exact question match in FAQ pages
            if is_apu_kb and metadata.get('is_faq', False):
                question_similarity = self._calculate_question_similarity(query, page_title)
                score += question_similarity * 10  # High bonus for question similarity
            
            # Check for match in tags
            if is_apu_kb and 'tags' in metadata:
                for tag in metadata['tags']:
                    if any(kw in tag for kw in keywords):
                        score += 2  # Bonus for tag matches
            
            # Give a boost to direct question matches for FAQ pages
            if is_apu_kb and metadata.get('is_faq', False) and query.strip().endswith('?'):
                # Calculate similarity between query and page title
                title_similarity = self._calculate_question_similarity(query, page_title)
                if title_similarity > 0.7:  # High similarity
                    score += 15
                elif title_similarity > 0.5:  # Moderate similarity
                    score += 8
                
            if score > 0:
                doc_id = ids[i] if i < len(ids) else str(i)
                
                # Create Document object
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        **metadata,
                        'score': score,
                        'id': doc_id
                    }
                )
                scored_docs.append((score, doc))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract just the documents from the scored list
        result_docs = [doc for _, doc in scored_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _calculate_question_similarity(self, query: str, title: str) -> float:
        """
        Calculate similarity between a query and a question title.
        
        Args:
            query: The user query
            title: The FAQ title
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both strings
        query = query.lower().strip()
        title = title.lower().strip()
        
        # Remove question marks
        query = query.rstrip('?')
        title = title.rstrip('?')
        
        # Tokenize
        query_tokens = set(word_tokenize(query))
        title_tokens = set(word_tokenize(title))
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        query_tokens = {token for token in query_tokens if token not in stop_words}
        title_tokens = {token for token in title_tokens if token not in stop_words}
        
        # Return 0 if either set is empty after removing stop words
        if not query_tokens or not title_tokens:
            return 0
        
        # Calculate Jaccard similarity
        intersection = len(query_tokens.intersection(title_tokens))
        union = len(query_tokens.union(title_tokens))
        
        return intersection / union if union > 0 else 0
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Perform hybrid retrieval (semantic + keyword), optimized for APU KB.
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get semantic search results
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get keyword search results
        keyword_docs = self._keyword_retrieval(query)
        
        # Try direct FAQ matching
        faq_match_result = self.faq_matcher.match_faq({"original_query": query})
        faq_docs = []
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.5:
            faq_docs = [faq_match_result["document"]]
        
        # Combine results with weighting
        semantic_weight = 1 - Config.KEYWORD_RATIO - Config.FAQ_MATCH_WEIGHT
        keyword_weight = Config.KEYWORD_RATIO
        faq_weight = Config.FAQ_MATCH_WEIGHT
        
        # Track documents by ID to avoid duplicates
        combined_docs = {}
        
        # Add FAQ docs with highest priority
        for i, doc in enumerate(faq_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Use match score if available
            faq_score = faq_match_result.get("match_score", 0.8) * faq_weight
            
            # Store with score
            combined_docs[doc_id] = {
                "doc": doc,
                "score": faq_score
            }
        
        # Add semantic docs with their weights
        for i, doc in enumerate(semantic_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Inverse rank scoring (higher rank = lower score)
            semantic_score = semantic_weight * (1.0 - (i / max(1, len(semantic_docs))))
            
            # Check if this is an APU KB page and apply appropriate boosts
            if doc.metadata.get('content_type') == 'apu_kb_page':
                # Boost for FAQ pages
                if doc.metadata.get('is_faq', False):
                    semantic_score *= 1.2
                
                # Bonus for title match
                title = doc.metadata.get('page_title', '').lower()
                if any(word in title for word in query.lower().split()):
                    semantic_score *= 1.3
            
            # Store with score
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += semantic_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": semantic_score
                }
        
        for i, doc in enumerate(keyword_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Get original keyword score if available, otherwise use inverse rank
            keyword_score = (doc.metadata.get('score', 0) / max(1, max([d.metadata.get('score', 1) for d in keyword_docs])))
            
            # Adjust by weight
            keyword_score = keyword_weight * keyword_score
            
            # Update score if document already exists, otherwise add it
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += keyword_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": keyword_score
                }
        
        # Sort by combined score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x["score"], reverse=True)
        
        # Return top K documents
        result_docs = [item["doc"] for item in sorted_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to the query, optimized for APU KB.
        
        Args:
            documents: List of documents to rerank
            query: Original query string
            
        Returns:
            Reranked document list
        """
        if not documents:
            return []
        
        # Embed the query
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Score each document
            scored_docs = []
            for doc in documents:
                # Try to get precomputed embedding if available
                doc_embedding = None
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    doc_embedding = doc.embedding
                else:
                    # Compute embedding
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Get keyword score if available
                keyword_score = doc.metadata.get('score', 0)
                
                # Get base score
                base_score = (similarity * 0.6) + (keyword_score * 0.4)
                
                # Apply APU KB specific boosts
                if doc.metadata.get('content_type') == 'apu_kb_page':
                    # Boost FAQ pages for question-like queries
                    if doc.metadata.get('is_faq', False) and query.strip().endswith('?'):
                        base_score *= 1.3
                    
                    # Boost for title match
                    title = doc.metadata.get('page_title', '').lower()
                    if any(word in title for word in query.lower().split()):
                        base_score *= 1.2
                    
                    # Calculate question similarity for FAQ pages
                    if doc.metadata.get('is_faq', False):
                        question_similarity = self._calculate_question_similarity(query, title)
                        base_score += question_similarity * 0.5
                
                scored_docs.append((base_score, doc))
            
            # Sort by score (descending)
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            
            # Return reranked documents
            return [doc for _, doc in scored_docs]
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents  # Return original order if reranking fails
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def _create_prompt(self, input_dict):
        """Create a prompt for the LLM based on query type and context."""
        # Check if this is an FAQ match
        is_faq_match = input_dict.get("is_faq_match", False)
        match_score = input_dict.get("match_score", 0)
        question = input_dict.get("question", "").lower()
        relevance_score = input_dict.get("relevance_score", 1.0)
        
        debug_prefix = ""
        
        # Debug thinking for identity questions (which should be caught earlier)
        if any(term in question for term in ["model", "llm", "who are you", "what are you"]):
            debug_prefix = f"""
            <think>
            This appears to be a question about my identity or technical specifications. I should NOT search 
            the knowledge base for this answer. Instead, I should provide factual information about the
            system itself based on known configuration.
            
            System specifications:
            - LLM Model: {Config.LLM_MODEL_NAME}
            - Embedding Model: {Config.EMBEDDING_MODEL_NAME}
            - Search Type: {Config.RETRIEVER_SEARCH_TYPE}
            - Role: APU Knowledge Base Assistant
            </think>
            """
        
        # Debug thinking for low relevance scores    
        elif relevance_score < 0.5:
            debug_prefix = f"""
            <think>
            This query has a low relevance score ({relevance_score:.2f}), meaning the retrieved documents 
            may not be directly relevant to the question. I should be cautious and:
            
            1. Only provide information explicitly stated in the context
            2. Acknowledge if I don't have sufficient information to answer fully
            3. Not try to infer or guess information not present in the context
            4. Suggest contacting appropriate APU departments for more specific information
            </think>
            """
        
        # If this is an FAQ match with high confidence
        if is_faq_match and match_score > 0.85:
            template = """
            You are a helpful AI assistant answering questions about APU (Asia Pacific University).
            
            IMPORTANT GUIDELINES:
            1. Provide clear, step-by-step instructions when appropriate
            2. Use natural, conversational language
            3. NEVER invent email addresses - only use email addresses that appear in the source text
            4. If a person/role is mentioned without an email address, do NOT assign them an email address
            5. When the text says "contact A or B", these are separate contact options - do not combine them
            6. Base your answer on the SOURCE TEXT provided
            7. Offer helpful next steps when relevant
            8. DO NOT include meta-instructions in your response (e.g., "Important Note:", "Remember to:", "Please note that:")
            
            {debug_prefix}
            
            Context from APU knowledge base:
            {context}
            
            Question: {question}
            
            Provide a helpful, natural response with clear instructions:
            """
        # Check if financial question
        elif any(term in question for term in [
            "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
            "receipt", "outstanding", "due", "overdue", "installment",
            "scholarship", "loan", "charge", "refund", "deposit"
        ]):
            # Enhanced template for financial questions
            template = """
            You are a helpful AI assistant answering questions about financial matters at APU (Asia Pacific University) in Malaysia.
            
            CRITICAL INSTRUCTIONS:
            1. Focus specifically on providing detailed payment information including methods, deadlines, and procedures
            2. Include EXACT fee amounts, account numbers, and payment details if available in the context
            3. Specify payment locations, online systems, or bank accounts to use
            4. Mention what documentation students need when making payments
            5. Copy ALL email addresses and contact information EXACTLY as they appear in the context
            6. Do NOT invent or modify any payment methods or procedures not mentioned in the context
            7. Do NOT hallucinate any details not explicitly stated in the context
            8. DO NOT include meta-instructions in your response (e.g., "Important Note:", "Remember to:", "Please note that:")
            
            {debug_prefix}
            
            If specific payment information is not available in the context, clearly state:
            "The specific payment procedure for this situation is not detailed in my knowledge base. Please contact the Finance Office at finance@apu.edu.my or visit the Finance Counter during operating hours (Monday-Friday, 9am-5pm)."
            
            Context from APU knowledge base:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            """
        else:
            # Standard RAG prompt with enhanced anti-hallucination instructions
            template = """
            You are a helpful AI assistant answering questions about APU (Asia Pacific University) in Malaysia. 
            
            CRITICAL INSTRUCTIONS:
            1. ONLY answer using information directly stated in the context provided
            2. If you cannot find the specific answer in the context, say "I don't have specific information about that in the APU knowledge base"
            3. NEVER invent details, email addresses, phone numbers, deadlines, or procedures
            4. Copy ALL email addresses EXACTLY as they appear in the context (e.g., admin@apu.edu.my)
            5. Do NOT add any markdown formatting, bold text, or "Answer:" labels
            6. Provide only a single, straightforward response
            7. DO NOT include meta-instructions in your response (e.g., "Important Note:", "Remember to:", "Please note that:")
            
            {debug_prefix}
            
            Additional guidelines:
            - Focus on providing clear, action-oriented information when answering procedural questions
            - Use the specific terminology from APU (e.g., "EC" for "Extenuating Circumstances")
            - If information about specific fees is available, include the exact amount
            - If the answer isn't in the context but the question is about APU, suggest contacting the relevant department
            
            Context from APU knowledge base:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            """
        
        # Format chat history
        chat_history = ""
        for message in input_dict["chat_history"]:
            if isinstance(message, HumanMessage):
                chat_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                chat_history += f"AI: {message.content}\n"
        
        # Fill in the template
        prompt = template.format(
            context=input_dict["context"],
            chat_history=chat_history,
            question=input_dict["question"],
            debug_prefix=debug_prefix
        )
        
        return prompt
    
    def _assess_document_relevance(self, documents: List[Document], query: str) -> float:
        """
        Assess how relevant the retrieved documents are to the query.
        
        Args:
            documents: Retrieved documents
            query: Original query string
            
        Returns:
            Relevance score between 0 and 1
        """
        if not documents:
            return 0.0
            
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        query_keywords = [token for token in query_tokens if token not in stop_words and len(token) > 2]
        
        if not query_keywords:
            return 0.5  # No keywords to match
        
        # Extract bigrams for phrase matching
        query_bigrams = list(ngrams(query_tokens, 2)) if len(query_tokens) >= 2 else []
        query_bigram_phrases = [' '.join(bg) for bg in query_bigrams]
        
        # Count keyword and phrase matches in documents
        total_keyword_matches = 0
        total_phrase_matches = 0
        max_keyword_matches = len(query_keywords) * len(documents)
        max_phrase_matches = len(query_bigram_phrases) * len(documents) if query_bigram_phrases else 1
        
        # Track semantic similarity if embeddings are available
        semantic_scores = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Keyword matching
            for keyword in query_keywords:
                if keyword in content_lower:
                    total_keyword_matches += 1
            
            # Phrase matching (more important)
            for phrase in query_bigram_phrases:
                if phrase in content_lower:
                    total_phrase_matches += 1
            
            # Try to get semantic similarity if possible
            try:
                if hasattr(self, 'embeddings') and self.embeddings is not None:
                    # Attempt to calculate semantic similarity
                    query_embedding = self.embeddings.embed_query(query)
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    semantic_scores.append(similarity)
            except Exception as e:
                # Skip semantic scoring if it fails
                logger.debug(f"Semantic scoring failed: {e}")
        
        # Calculate lexical match score (keywords and phrases)
        keyword_score = total_keyword_matches / max_keyword_matches if max_keyword_matches > 0 else 0
        phrase_score = total_phrase_matches / max_phrase_matches if max_phrase_matches > 0 else 0
        lexical_score = (keyword_score * 0.4) + (phrase_score * 0.6)  # Phrases weighted higher
        
        # Calculate semantic score if available
        semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.5
        
        # Combine scores - semantic is more important for relevance
        if semantic_scores:
            final_score = (lexical_score * 0.3) + (semantic_score * 0.7)
        else:
            final_score = lexical_score
        
        return min(1.0, final_score)

    def _check_hallucination_risk(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Check if the query and context have high risk of hallucination.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Tuple of (is_risky, reason)
        """
        # 1. Check if query contains terms not found in context
        query_tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        query_keywords = [token for token in query_tokens if token not in stop_words and len(token) > 2]
        
        # Only check if we have meaningful keywords
        if query_keywords:
            context_lower = context.lower()
            missing_keywords = [kw for kw in query_keywords if kw not in context_lower]
            
            # If more than half of keywords are missing, high risk
            if len(missing_keywords) > len(query_keywords) / 2:
                return True, f"Key query terms missing from context: {', '.join(missing_keywords[:3])}"
        
        # 2. Check for questions about named entities not in context
        entity_patterns = [
            r'who is ([A-Z][a-z]+ [A-Z][a-z]+)',  # Person names
            r'what is ([A-Z][a-z]+ [A-Z][a-z]+)',  # Organization names
            r'where is ([A-Z][a-z]+ [A-Z][a-z]+)',  # Location names
            r'when is ([A-Z][a-z]+ [A-Z][a-z]+)',  # Event names
            r'how to ([A-Z][a-z]+ [A-Z][a-z]+)',   # Process names
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            for entity in matches:
                if entity.lower() not in context_lower:
                    return True, f"Named entity not found in context: {entity}"
        
        # 3. Check for questions requiring specific numbers/dates
        if re.search(r'\b(?:when|what year|what date|how many|how much)\b', query.lower()):
            # Look for numbers or dates in context
            has_numbers = bool(re.search(r'\d+', context))
            has_dates = bool(re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', context.lower()))
            has_years = bool(re.search(r'\b(?:19|20)\d{2}\b', context))
            
            if not (has_numbers or has_dates or has_years):
                return True, "Query requests specific data (numbers/dates) not found in context"
        
        # 4. Check for comparative questions without comparison terms
        if re.search(r'\b(?:compare|versus|vs|difference|better|worse|advantages?|disadvantages?)\b', query.lower()):
            comparison_terms = [
                r'\b(?:versus|vs|compared to|in comparison|on the other hand|however|while|whereas)\b',
                r'\b(?:better|worse|more|less|higher|lower|faster|slower)\b',
                r'\b(?:advantage|disadvantage|pro|con|benefit|drawback)\b'
            ]
            has_comparison = any(bool(re.search(pattern, context.lower())) for pattern in comparison_terms)
            if not has_comparison:
                return True, "Comparative query without comparison terms in context"
        
        # 5. Check for procedural questions without steps/instructions
        if re.search(r'\b(?:how to|steps?|procedure|process|guide|instructions?)\b', query.lower()):
            step_patterns = [
                r'\b(?:step|first|second|third|next|then|finally|lastly)\b',
                r'\b(?:1\.|2\.|3\.|4\.|5\.)\b',
                r'\b(?:begin|start|complete|finish|end)\b'
            ]
            has_steps = any(bool(re.search(pattern, context.lower())) for pattern in step_patterns)
            if not has_steps:
                return True, "Procedural query without clear steps in context"
        
        # 6. Check for questions about specific requirements/conditions
        if re.search(r'\b(?:require|need|must|should|condition|prerequisite|eligibility)\b', query.lower()):
            requirement_patterns = [
                r'\b(?:require|need|must|should|condition|prerequisite|eligibility)\b',
                r'\b(?:minimum|maximum|at least|no more than)\b',
                r'\b(?:if|when|unless|provided that|as long as)\b'
            ]
            has_requirements = any(bool(re.search(pattern, context.lower())) for pattern in requirement_patterns)
            if not has_requirements:
                return True, "Query about requirements without specific conditions in context"
        
        # 7. Check for questions about specific locations/places
        if re.search(r'\b(?:where|location|place|building|room|office|department)\b', query.lower()):
            location_patterns = [
                r'\b(?:building|room|floor|level|block|wing|campus)\b',
                r'\b(?:located|situated|found|positioned)\b',
                r'\b(?:near|next to|beside|behind|in front of)\b'
            ]
            has_locations = any(bool(re.search(pattern, context.lower())) for pattern in location_patterns)
            if not has_locations:
                return True, "Location query without specific place information in context"
        
        # 8. Check for questions about specific people/roles
        if re.search(r'\b(?:who|person|staff|faculty|lecturer|professor|student|admin)\b', query.lower()):
            person_patterns = [
                r'\b(?:contact|email|phone|extension|office hours)\b',
                r'\b(?:head|director|coordinator|manager|officer)\b',
                r'\b(?:department|faculty|school|division|unit)\b'
            ]
            has_person_info = any(bool(re.search(pattern, context.lower())) for pattern in person_patterns)
            if not has_person_info:
                return True, "Person/role query without specific contact information in context"
        
        # No high risks identified
        return False, ""