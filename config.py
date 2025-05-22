"""
Configuration settings for the APURAG system.
"""

import os
import logging
from datetime import datetime

# Configure logging - production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customrag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

class Config:
    """Configuration settings for the RAG application."""
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:1.5b")
    
    # Chunking settings
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "150"))
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "6"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")  # Changed to hybrid as default
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))  # 40% weight to keywords by default for FAQ
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.5"))  # Weight for direct FAQ matches
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "3"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
    
    # APU filtering setting - set default to True to only process APU files
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']
    
    # APU KB specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.0"))

    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        # Set logging level based on configuration
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")
        logger.info(f"Search type: {cls.RETRIEVER_SEARCH_TYPE}")
        
        if cls.RETRIEVER_SEARCH_TYPE == "hybrid":
            logger.info(f"Keyword ratio: {cls.KEYWORD_RATIO}")
            logger.info(f"FAQ match weight: {cls.FAQ_MATCH_WEIGHT}")
        
        if cls.USE_QUERY_EXPANSION:
            logger.info(f"Query expansion enabled with factor: {cls.EXPANSION_FACTOR}")
            
        if cls.USE_CONTEXT_COMPRESSION:
            logger.info(f"Context compression enabled")
            
        # Log APU filtering status
        if cls.FILTER_APU_ONLY:
            logger.info("APU document filtering is ENABLED - only files starting with 'apu_' will be processed")
        else:
            logger.info("APU document filtering is DISABLED - all compatible files will be processed")