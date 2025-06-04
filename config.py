"""
Configuration settings for the APURAG system with environment-specific support and model management.

This module provides comprehensive configuration management including:
- Environment-specific configuration loading
- Hardware detection and optimization
- Model lifecycle management settings
- Logging and error handling setup
- Path and resource management
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

def setup_nltk_with_fallback():
    """
    Setup NLTK with proper error handling and fallback options.
    
    Attempts to download required NLTK data with graceful fallback
    to basic text processing methods if NLTK is unavailable.
    """
    try:
        import nltk
        # Try to download required NLTK data with error handling
        try:
            nltk.data.find('corpora/wordnet')
            logger.info("NLTK WordNet data already available")
        except LookupError:
            try:
                logger.info("Attempting to download NLTK WordNet data...")
                nltk.download('wordnet', quiet=True, raise_on_error=True)
                logger.info("Successfully downloaded NLTK WordNet data")
            except Exception as e:
                logger.warning(f"Failed to download NLTK WordNet data: {e}")
                logger.info("NLTK will use fallback methods for text processing")
                
        # Try to download other useful NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True, raise_on_error=True)
                logger.debug("Downloaded NLTK punkt tokenizer")
            except Exception as e:
                logger.debug(f"Could not download NLTK punkt tokenizer: {e}")
                
    except ImportError:
        logger.info("NLTK not available - using basic text processing methods")
    except Exception as e:
        logger.warning(f"NLTK setup encountered an error: {e}")
        logger.info("Continuing with fallback text processing methods")

class Config:
    """
    Base configuration settings for the RAG application with model management.
    
    This class provides centralized configuration management with support for
    environment-specific overrides, hardware detection, and production-grade
    model lifecycle management.
    """
    
    # Class variable to track setup status
    _setup_completed = False
    
    # Environment
    ENV = ENV
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "qwen2.5:3b-instruct")
    
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
    
    # Session management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Resource settings
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", "4"))
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "4G")
    
    # Production Model Management Settings
    MODEL_CHECK_INTERVAL_DAYS = int(os.environ.get("CUSTOMRAG_MODEL_CHECK_INTERVAL_DAYS", "30"))
    MODEL_WARNING_AGE_DAYS = int(os.environ.get("CUSTOMRAG_MODEL_WARNING_AGE_DAYS", "60"))
    MODEL_CRITICAL_AGE_DAYS = int(os.environ.get("CUSTOMRAG_MODEL_CRITICAL_AGE_DAYS", "90"))
    MODEL_AUTO_UPDATE_PROMPT = os.environ.get("CUSTOMRAG_MODEL_AUTO_UPDATE_PROMPT", "True").lower() in ("true", "1", "t")
    MODEL_UPDATE_CHECK_ENABLED = os.environ.get("CUSTOMRAG_MODEL_UPDATE_CHECK_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_REQUIRE_APPROVAL = os.environ.get("CUSTOMRAG_MODEL_REQUIRE_APPROVAL", "True").lower() in ("true", "1", "t")
    MODEL_CACHE_CLEANUP = os.environ.get("CUSTOMRAG_MODEL_CACHE_CLEANUP", "False").lower() in ("true", "1", "t")
    MODEL_BACKUP_ENABLED = os.environ.get("CUSTOMRAG_MODEL_BACKUP_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_MAX_BACKUPS = int(os.environ.get("CUSTOMRAG_MODEL_MAX_BACKUPS", "3"))
    MODEL_NOTIFICATION_EMAIL = os.environ.get("CUSTOMRAG_MODEL_NOTIFICATION_EMAIL", "")

    @classmethod
    def has_gpu(cls):
        """
        Detect if GPU is available (CUDA or Apple Silicon MPS).
        
        Returns:
            bool: True if GPU acceleration is available
        """
        try:
            import torch
            # Check for CUDA first (for compatibility)
            if torch.cuda.is_available():
                return True
            # Check for Apple Silicon MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True
            else:
                return False
        except ImportError:
            return False
    
    @classmethod 
    def get_device_info(cls):
        """
        Get detailed device information for logging.
        
        Returns:
            tuple: (device_type, device_name) for hardware identification
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps", "Apple Silicon MPS"
            else:
                return "cpu", "CPU"
        except ImportError:
            return "cpu", "CPU (PyTorch not available)"
    
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
        """
        Set up the configuration and ensure directories exist.
        
        Performs one-time initialization including directory creation,
        NLTK setup, hardware detection, and configuration logging.
        """
        # Prevent duplicate setup logging
        if cls._setup_completed:
            logger.debug("Configuration setup already completed, skipping duplicate setup")
            return
            
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        # Setup NLTK with proper error handling
        setup_nltk_with_fallback()
        
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
        
        # Log model management settings
        if cls.MODEL_UPDATE_CHECK_ENABLED:
            logger.info("Production model management enabled")
            logger.info(f"Model update checks: every {cls.MODEL_CHECK_INTERVAL_DAYS} days")
            logger.info(f"Warning threshold: {cls.MODEL_WARNING_AGE_DAYS} days")
            logger.info(f"Critical threshold: {cls.MODEL_CRITICAL_AGE_DAYS} days")
            if cls.MODEL_AUTO_UPDATE_PROMPT:
                logger.info("Auto-prompts enabled for model updates")
            if cls.MODEL_REQUIRE_APPROVAL:
                logger.info("Manual approval required for updates")
            
        # Mark setup as completed
        cls._setup_completed = True
        logger.debug("Configuration setup completed successfully")

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