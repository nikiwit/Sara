"""
Production configuration for APURAG system optimized for HGX H100 G593-SD2.
"""

import os
from config import Config, logger

class ProductionConfig(Config):
    """Configuration optimized for production HGX H100 environment."""
    
    # Override with production-specific settings
    
    # Use larger models leveraging H100 capabilities
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-mpnet-base-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:7b")
    
    # Optimized chunk sizes
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "200"))
    
    # Enhanced retrieval parameters
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "8"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")
    
    # Maximize resource usage for H100
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "32"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "64G")
    
    # Production-level logging
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "WARNING")
    
    # Production-specific methods
    @classmethod
    def setup(cls):
        """Set up production configuration."""
        super().setup()
        
        # Additional production-specific setup
        logger.info("Running in PRODUCTION mode on HGX H100")
        logger.info(f"Using enhanced resource settings: {cls.MAX_THREADS} threads, {cls.MAX_MEMORY} memory")
        
        # Ensure absolute paths for production
        if not os.path.isabs(cls.DATA_PATH):
            logger.warning(f"Converting relative data path to absolute: {cls.DATA_PATH}")
            cls.DATA_PATH = os.path.abspath(cls.DATA_PATH)
            
        if not os.path.isabs(cls.PERSIST_PATH):
            logger.warning(f"Converting relative vector store path to absolute: {cls.PERSIST_PATH}")
            cls.PERSIST_PATH = os.path.abspath(cls.PERSIST_PATH)
