"""
Production configuration for Sara system optimized for HGX H100 G593-SD2 with conservative model management.
"""

import os
from config import Config, logger

class ProductionConfig(Config):
    """Configuration optimized for production HGX H100 environment with conservative model management for Sara."""
    
    # Override with production-specific settings
    
    # Use larger models leveraging H100 capabilities
    EMBEDDING_MODEL_NAME = os.environ.get("SARA_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("SARA_LLM_MODEL", "qwen2.5:7b-instruct")
    
    # Optimized chunk sizes
    CHUNK_SIZE = int(os.environ.get("SARA_CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.environ.get("SARA_CHUNK_OVERLAP", "200"))
    
    # Response streaming speed - moderate for production user experience
    STREAM_DELAY = 0.025  # Balanced speed for production users
    
    # Enhanced retrieval parameters
    RETRIEVER_K = int(os.environ.get("SARA_RETRIEVER_K", "8"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("SARA_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("SARA_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("SARA_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("SARA_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("SARA_EXPANSION_FACTOR", "3"))  # Full expansion for production
    
    # Semantic enhancement settings - Production spaCy configuration
    USE_ENHANCED_SEMANTICS = os.environ.get("SARA_USE_ENHANCED_SEMANTICS", "true").lower() == "true"
    SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_sm")
    # SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_md")  # Medium model (~43MB) - better accuracy
    # SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_lg")  # Large model (~588MB) - recommended for production
    
    SEMANTIC_EXPANSION_LIMIT = int(os.environ.get("SARA_SEMANTIC_EXPANSION_LIMIT", "8"))  # Higher for production
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("SARA_MAX_CONTEXT_SIZE", "5000"))  # Larger for production
    USE_CONTEXT_COMPRESSION = os.environ.get("SARA_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Session (chats) management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("SARA_OLLAMA_URL", "http://localhost:11434")
    
    # Maximize resource usage for H100
    MAX_THREADS = int(os.environ.get("SARA_MAX_THREADS", "32"))
    MAX_MEMORY = os.environ.get("SARA_MAX_MEMORY", "64G")
    
    # Production-level logging
    LOG_LEVEL = os.environ.get("SARA_LOG_LEVEL", "WARNING")
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # APU KB specific settings - enhanced for production
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("SARA_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("SARA_APU_KB_EXACT_MATCH_BOOST", "2.0"))
    
    # Production Model Management - Conservative approach
    MODEL_CHECK_INTERVAL_DAYS = int(os.environ.get("SARA_MODEL_CHECK_INTERVAL_DAYS", "30"))      # Monthly checks
    MODEL_WARNING_AGE_DAYS = int(os.environ.get("SARA_MODEL_WARNING_AGE_DAYS", "90"))            # 3 months warning
    MODEL_CRITICAL_AGE_DAYS = int(os.environ.get("SARA_MODEL_CRITICAL_AGE_DAYS", "180"))          # 6 months critical
    MODEL_AUTO_UPDATE_PROMPT = os.environ.get("SARA_MODEL_AUTO_UPDATE_PROMPT", "False").lower() in ("true", "1", "t")  # No auto-prompts
    MODEL_UPDATE_CHECK_ENABLED = os.environ.get("SARA_MODEL_UPDATE_CHECK_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_REQUIRE_APPROVAL = os.environ.get("SARA_MODEL_REQUIRE_APPROVAL", "True").lower() in ("true", "1", "t")
    MODEL_CACHE_CLEANUP = os.environ.get("SARA_MODEL_CACHE_CLEANUP", "True").lower() in ("true", "1", "t")     # Auto-cleanup in prod
    MODEL_BACKUP_ENABLED = os.environ.get("SARA_MODEL_BACKUP_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_MAX_BACKUPS = int(os.environ.get("SARA_MODEL_MAX_BACKUPS", "3"))                        # Keep 3 backups
    MODEL_NOTIFICATION_EMAIL = os.environ.get("SARA_MODEL_NOTIFICATION_EMAIL", "")
    
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
        
        # Log production model management settings
        if cls.MODEL_UPDATE_CHECK_ENABLED:
            logger.info(f"Production model management: conservative approach")
            logger.info(f"Checks every {cls.MODEL_CHECK_INTERVAL_DAYS} days, warnings at {cls.MODEL_WARNING_AGE_DAYS} days")
            if not cls.MODEL_AUTO_UPDATE_PROMPT:
                logger.info(f"Manual approval required for all model updates")
            if cls.MODEL_NOTIFICATION_EMAIL:
                logger.info(f"Notifications enabled for: {cls.MODEL_NOTIFICATION_EMAIL}")