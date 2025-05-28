"""
Configuration settings for the APURAG system with environment-specific support.
"""

import os
import logging
import platform
from datetime import datetime
from dotenv import load_dotenv

# Determine environment
ENV = os.environ.get("APURAG_ENV", "local").lower()

# Load environment-specific .env file
if ENV == "production":
    env_file = ".env.production"
elif ENV == "local":
    env_file = ".env.local"
else:
    env_file = ".env"  # Fallback to default

# Load environment variables from appropriate .env file
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv()  # Fallback to default .env

# Configure logging
log_level_name = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customrag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

class Config:
    """Base configuration settings for the RAG application."""
    
    # Environment
    ENV = ENV
    
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
    
    # Response streaming speed
    STREAM_DELAY = 0.015  # Consistent streaming delay across all responses
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "6"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "3"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Resource settings
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "4"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "4G")
    
    # Hardware detection
    @classmethod
    def has_gpu(cls):
        """Detect if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = log_level_name
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']
    
    # APU KB specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.0"))

    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        # Log environment and configuration
        logger.info(f"Running in {cls.ENV} environment")
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")
        logger.info(f"Search type: {cls.RETRIEVER_SEARCH_TYPE}")
        logger.info(f"GPU available: {cls.has_gpu()}")
        
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

# Load environment-specific configuration
if ENV == "production":
    try:
        from config_production import ProductionConfig
        ConfigClass = ProductionConfig
        logger.info("Loaded production configuration")
    except ImportError:
        ConfigClass = Config
        logger.warning("Production configuration not found, using base configuration")
elif ENV == "local":
    try:
        from config_local import LocalConfig
        ConfigClass = LocalConfig
        logger.info("Loaded local development configuration")
    except ImportError:
        ConfigClass = Config
        logger.warning("Local configuration not found, using base configuration")
else:
    ConfigClass = Config
    logger.info("Using base configuration")

# Export the appropriate configuration class
config = ConfigClass
