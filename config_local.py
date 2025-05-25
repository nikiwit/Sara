"""
Local development configuration for APURAG system.
"""

import os
from config import Config, logger

class LocalConfig(Config):
    """Configuration optimized for local development environments."""
    
    # Override with local-specific settings
    
    # Use smaller models suitable for laptops
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:1.5b")
    
    # Smaller chunk sizes for faster processing
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "400"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "100"))
    
    # Reduced retrieval parameters
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "4"))
    
    # Limit resource usage
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "2"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "2G")
    
    # More verbose logging for development
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "DEBUG")
    
    # Development-specific methods
    @classmethod
    def setup(cls):
        """Set up local development configuration."""
        super().setup()
        
        # Additional local-specific setup
        logger.info("Running in LOCAL development mode")
        logger.info(f"Using reduced resource settings: {cls.MAX_THREADS} threads, {cls.MAX_MEMORY} memory")
