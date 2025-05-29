"""
Production configuration for APURAG system optimized for HGX H100 G593-SD2.
"""

import os
from config import Config, logger

class ProductionConfig(Config):
    """Configuration optimized for production HGX H100 environment."""
    
    # Override with production-specific settings
    
    # Use larger models leveraging H100 capabilities
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "qwen2.5:7b-instruct")
    
    # Optimized chunk sizes
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "200"))
    
    # Response streaming speed - moderate for production user experience
    STREAM_DELAY = 0.025  # Balanced speed for production users
    
    # Enhanced retrieval parameters
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "8"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "3"))  # Full expansion for production
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "5000"))  # Larger for production
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Session (chats) management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Maximize resource usage for H100
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "32"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "64G")
    
    # Production-level logging
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "WARNING")
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # APU KB specific settings - enhanced for production
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.0"))
    
    # Production-specific methods
    @classmethod
    def setup(cls):
        """Set up production configuration."""
        super().setup()
        
        # Additional production-specific setup
        logger.info("Running in PRODUCTION mode on HGX H100")
        logger.info(f"Using enhanced resource settings: {cls.MAX_THREADS} threads, {cls.MAX_MEMORY} memory")
        logger.info(f"Stream delay: {cls.STREAM_DELAY}s (optimized for user experience)")
        logger.info(f"Enhanced context size: {cls.MAX_CONTEXT_SIZE} tokens")
        
        # Ensure absolute paths for production
        if not os.path.isabs(cls.DATA_PATH):
            logger.warning(f"Converting relative data path to absolute: {cls.DATA_PATH}")
            cls.DATA_PATH = os.path.abspath(cls.DATA_PATH)
            
        if not os.path.isabs(cls.PERSIST_PATH):
            logger.warning(f"Converting relative vector store path to absolute: {cls.PERSIST_PATH}")
            cls.PERSIST_PATH = os.path.abspath(cls.PERSIST_PATH)