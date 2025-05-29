"""
Local development configuration for APURAG system.
"""

import os
from config import Config, logger

class LocalConfig(Config):
    """Configuration optimized for local development environments."""
    
    # Override with local-specific settings
    
    # Use smaller models suitable for laptops
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "qwen2.5:3b-instruct")

    # Smaller chunk sizes for faster processing
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "400"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "100"))
    
    # Response streaming speed - faster for local development
    STREAM_DELAY = 0.015  # Faster than production for development
    
    # Reduced retrieval parameters
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "4"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "2"))  # Reduced for local
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "3000"))  # Smaller for local
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Session (chats) management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Limit resource usage
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "2"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "2G")
    
    # More verbose logging for development
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "DEBUG")
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # APU KB specific settings - reduced for local
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "2"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.0"))
    
    # Development-specific methods
    @classmethod
    def setup(cls):
        """Set up local development configuration."""
        super().setup()
        
        # Additional local-specific setup
        logger.info("Running in LOCAL development mode")
        logger.info(f"Using reduced resource settings: {cls.MAX_THREADS} threads, {cls.MAX_MEMORY} memory")
        logger.info(f"Stream delay: {cls.STREAM_DELAY}s (faster for development)")
        logger.info(f"Reduced context size: {cls.MAX_CONTEXT_SIZE} tokens")