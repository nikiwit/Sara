"""
System information management for identity queries.
"""

from typing import Dict
from config import config

class SystemInformation:
    """Manages information about the RAG system itself."""
    
    @classmethod
    def get_system_info(cls) -> Dict[str, str]:
        """
        Get system information dictionary for answering identity questions.
        
        Returns:
            Dictionary mapping question patterns to responses
        """
        return {
            # Basic identity information
            "who are you": "I'm the APU Knowledge Base Assistant, designed to help you find information about Asia Pacific University's academic procedures, administrative processes, and university services.",
            "what are you": "I'm an AI-powered retrieval system specifically built to help APU students and staff access information from the APU knowledge base quickly and accurately.",
            "your name": "You can call me the APU Knowledge Base Assistant. I'm here to help you with any questions about APU.",
            
            # Technical information about the system
            "model": f"I'm powered by {config.LLM_MODEL_NAME} for answering questions, and I use the {config.EMBEDDING_MODEL_NAME} embedding model to understand and retrieve relevant information from the APU knowledge base.",
            "llm": f"I'm using {config.LLM_MODEL_NAME} as my language model to generate responses based on information retrieved from the APU knowledge base.",
            "embedding": f"I use the {config.EMBEDDING_MODEL_NAME} embedding model to convert text into numerical vectors for semantic search capabilities.",
            "how do you work": "I use a technique called Retrieval Augmented Generation (RAG) to find relevant information in the APU knowledge base and create helpful responses to your questions. First, I analyze your question, then search for relevant documents, and finally generate a response based on the retrieved information.",
            "technology": f"I'm built using the LangChain framework with {config.LLM_MODEL_NAME} as my language model and {config.EMBEDDING_MODEL_NAME} for embeddings. I use ChromaDB as my vector database to store and retrieve information efficiently.",
            "version": f"I'm running Enhanced CustomRAG version 1.0, an advanced Retrieval Augmented Generation system optimized for the APU knowledge base.",
            
            # Development information
            "who made you": "I was developed by the APU technology team to provide quick and accurate answers about university procedures and policies.",
            "what can you do": "I can answer questions about APU's academic procedures, administrative processes, fees, exams, and other university services. Just ask me anything related to APU!",
            "your purpose": "My purpose is to help APU students and staff quickly find accurate information about university procedures, policies, and services.",
        }
    
    @classmethod
    def get_response_for_identity_query(cls, query: str) -> str:
        """
        Get appropriate response for an identity query.
        
        Args:
            query: The user's query string
            
        Returns:
            Response string appropriate for the identity query
        """
        query_lower = query.lower()
        system_info = cls.get_system_info()
        
        # Check for model-specific questions 
        if any(term in query_lower for term in ["model", "llm", "language model"]):
            return system_info.get("model", system_info.get("llm"))
        
        # Check for embedding/vector questions
        if any(term in query_lower for term in ["embedding", "vector", "semantic"]):
            return system_info.get("embedding")
        
        # Check for technology questions
        if any(term in query_lower for term in ["tech", "technology", "stack", "framework", "built with"]):
            return system_info.get("technology")
        
        # Check for other identity questions
        for key, value in system_info.items():
            if key in query_lower:
                return value
        
        # Default response
        return f"I'm the APU Knowledge Base Assistant. I'm powered by the {config.LLM_MODEL_NAME} language model with {config.EMBEDDING_MODEL_NAME} embeddings for retrieval. I'm here to help you find information about APU's academic procedures, administrative processes, and university services."