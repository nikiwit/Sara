"""
Local development configuration for Sara system with enhanced model management.
"""

import os
from config import Config, logger

class LocalConfig(Config):
    """Configuration optimized for local development environments with enhanced model management for Sara."""
    
    # Override with local-specific settings
    
    # Use smaller models suitable for laptops
    EMBEDDING_MODEL_NAME = os.environ.get("SARA_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("SARA_LLM_MODEL", "qwen2.5:3b-instruct")
    
    # Reranker model - balanced performance for local development
    RERANKER_MODEL_NAME = os.environ.get("SARA_RERANKER_MODEL", "BAAI/bge-reranker-base")

    # Smaller chunk sizes for faster processing
    CHUNK_SIZE = int(os.environ.get("SARA_CHUNK_SIZE", "400"))
    CHUNK_OVERLAP = int(os.environ.get("SARA_CHUNK_OVERLAP", "100"))
    
    # Response streaming speed - faster for local development
    STREAM_DELAY = 0.015  # Faster than production for development
    
    # Reduced retrieval parameters
    RETRIEVER_K = int(os.environ.get("SARA_RETRIEVER_K", "4"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("SARA_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("SARA_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("SARA_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("SARA_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("SARA_EXPANSION_FACTOR", "2"))  # Reduced for local
    
    # Semantic enhancement settings - Explicitly enable for local development
    USE_ENHANCED_SEMANTICS = os.environ.get("SARA_USE_ENHANCED_SEMANTICS", "true").lower() == "true"
    # Use medium model for better similarity calculations (includes word vectors)
    SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_md")  # Medium model (~43MB) - better accuracy
    # SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_sm")  # Small model - no word vectors
    # SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_lg")  # Large model (~588MB) - recommended for production
    
    SEMANTIC_EXPANSION_LIMIT = int(os.environ.get("SARA_SEMANTIC_EXPANSION_LIMIT", "5"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("SARA_MAX_CONTEXT_SIZE", "7000"))  # Generous context for comprehensive responses
    USE_CONTEXT_COMPRESSION = os.environ.get("SARA_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Session (chats) management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("SARA_OLLAMA_URL", "http://localhost:11434")
    
    # Limit resource usage
    MAX_THREADS = int(os.environ.get("SARA_MAX_THREADS", "2"))
    MAX_MEMORY = os.environ.get("SARA_MAX_MEMORY", "2G")
    
    # More verbose logging for development
    LOG_LEVEL = os.environ.get("SARA_LOG_LEVEL", "DEBUG")
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # APU KB specific settings - reduced for local
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("SARA_APU_KB_ANSWER_SIZE", "2"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("SARA_APU_KB_EXACT_MATCH_BOOST", "2.0"))
    
    # Local Development Model Management - More frequent checks for testing
    MODEL_CHECK_INTERVAL_DAYS = int(os.environ.get("SARA_MODEL_CHECK_INTERVAL_DAYS", "7"))  # Weekly in dev
    MODEL_WARNING_AGE_DAYS = int(os.environ.get("SARA_MODEL_WARNING_AGE_DAYS", "30"))      # 1 month warning
    MODEL_CRITICAL_AGE_DAYS = int(os.environ.get("SARA_MODEL_CRITICAL_AGE_DAYS", "60"))     # 2 months critical
    MODEL_AUTO_UPDATE_PROMPT = os.environ.get("SARA_MODEL_AUTO_UPDATE_PROMPT", "True").lower() in ("true", "1", "t")
    MODEL_UPDATE_CHECK_ENABLED = os.environ.get("SARA_MODEL_UPDATE_CHECK_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_REQUIRE_APPROVAL = os.environ.get("SARA_MODEL_REQUIRE_APPROVAL", "True").lower() in ("true", "1", "t")
    MODEL_CACHE_CLEANUP = os.environ.get("SARA_MODEL_CACHE_CLEANUP", "False").lower() in ("true", "1", "t")  # Keep cache in dev
    MODEL_BACKUP_ENABLED = os.environ.get("SARA_MODEL_BACKUP_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_MAX_BACKUPS = int(os.environ.get("SARA_MODEL_MAX_BACKUPS", "2"))  # Fewer backups in dev
    
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
        
        # Log development model management settings
        if cls.MODEL_UPDATE_CHECK_ENABLED:
            logger.info(f"Development model management: checks every {cls.MODEL_CHECK_INTERVAL_DAYS} days")