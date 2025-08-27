"""
Configuration settings for the Sara system with environment-specific support and model management.

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
from logging.handlers import RotatingFileHandler

# Disable ChromaDB telemetry globally before any ChromaDB imports
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('CHROMA_TELEMETRY_DISABLED', '1')

# Suppress ChromaDB telemetry error logs
def suppress_chromadb_telemetry_errors():
    """Suppress annoying ChromaDB telemetry error logs."""
    import logging
    chromadb_telemetry_logger = logging.getLogger("chromadb.telemetry.product.posthog")
    chromadb_telemetry_logger.setLevel(logging.CRITICAL)

# Apply telemetry log suppression immediately
suppress_chromadb_telemetry_errors()

# Determine environment
ENV = os.environ.get("SARA_ENV", "local").lower()

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

# Configure logging with rotation
log_level_name = os.environ.get("SARA_LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name.upper(), logging.INFO)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure log rotation settings
LOG_MAX_BYTES = int(os.environ.get("SARA_LOG_MAX_BYTES", "1073741824"))  # 1GB default
LOG_BACKUP_COUNT = int(os.environ.get("SARA_LOG_BACKUP_COUNT", "5"))  # Keep 5 backup files
LOG_USE_JSON = os.environ.get("SARA_LOG_USE_JSON", "False").lower() in ("true", "1", "t")

# Create rotating file handler with compression
rotating_handler = RotatingFileHandler(
    "logs/sara.log",
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding='utf-8'
)

# Configure JSON formatter for production if enabled
if LOG_USE_JSON:
    import json
    import datetime
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_entry)
    
    rotating_handler.setFormatter(JSONFormatter())
    console_formatter = JSONFormatter()
else:
    # Standard text format for development
    standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(standard_formatter)
    console_formatter = standard_formatter

# Configure console handler with appropriate formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

logging.basicConfig(
    level=log_level,
    handlers=[
        rotating_handler,
        console_handler
    ]
)
logger = logging.getLogger("Sara")

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
    Base configuration settings for the Sara application with model management.
    
    This class provides centralized configuration management with support for
    environment-specific overrides, hardware detection, and production-grade
    model lifecycle management for Sara.
    """
    
    # Class variable to track setup status
    _setup_completed = False
    
    # Environment
    ENV = ENV
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("SARA_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("SARA_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("SARA_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("SARA_LLM_MODEL", "qwen2.5:3b-instruct")
    
    # Reranker model settings
    RERANKER_MODEL_NAME = os.environ.get("SARA_RERANKER_MODEL", "BAAI/bge-reranker-base")
    
    # Chunking settings
    CHUNK_SIZE = int(os.environ.get("SARA_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("SARA_CHUNK_OVERLAP", "150"))
    
    # Response streaming speed
    STREAM_DELAY = 0.015  # Consistent streaming delay across all responses
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("SARA_RETRIEVER_K", "6"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("SARA_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("SARA_KEYWORD_RATIO", "0.4"))
    FAQ_MATCH_WEIGHT = float(os.environ.get("SARA_FAQ_MATCH_WEIGHT", "0.5"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("SARA_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("SARA_EXPANSION_FACTOR", "3"))
    
    # Semantic enhancement settings (Phase 4)
    USE_ENHANCED_SEMANTICS = os.environ.get("SARA_USE_ENHANCED_SEMANTICS", "true").lower() == "true"
    SEMANTIC_MODEL = os.environ.get("SARA_SEMANTIC_MODEL", "en_core_web_sm")
    SEMANTIC_CACHE_SIZE = int(os.environ.get("SARA_SEMANTIC_CACHE_SIZE", "1000"))
    SEMANTIC_EXPANSION_LIMIT = int(os.environ.get("SARA_SEMANTIC_EXPANSION_LIMIT", "5"))
    SEMANTIC_ERROR_THRESHOLD = int(os.environ.get("SARA_SEMANTIC_ERROR_THRESHOLD", "5"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("SARA_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("SARA_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Confidence and boundary detection settings
    CONFIDENCE_THRESHOLD = float(os.environ.get("SARA_CONFIDENCE_THRESHOLD", "0.4"))
    
    # Language Detection Settings
    LANGUAGE_DETECTION_ENABLED = os.environ.get("SARA_LANGUAGE_DETECTION", "True").lower() in ("true", "1", "t")
    SUPPORTED_LANGUAGES = {'en'}  # Only English supported currently
    LANGUAGE_DETECTION_CONFIDENCE = float(os.environ.get("SARA_LANG_CONFIDENCE", "0.8"))  # Minimum confidence threshold
    
    # Contact Information for Multilingual Support
    SUPPORT_PHONE = os.environ.get("SARA_SUPPORT_PHONE", "+603-8996-1000")
    SUPPORT_EMAIL = os.environ.get("SARA_SUPPORT_EMAIL", "info@apu.edu.my")
    SUPPORT_LOCATION = os.environ.get("SARA_SUPPORT_LOCATION", "APU Student Services (Level 1, New Campus)")
    
    # Ambiguity Detection Settings  
    AMBIGUITY_DETECTION_ENABLED = os.environ.get("SARA_AMBIGUITY_DETECTION", "True").lower() in ("true", "1", "t")
    AMBIGUITY_CONFIDENCE_THRESHOLD = float(os.environ.get("SARA_AMBIGUITY_THRESHOLD", "0.7"))
    
    # Session management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("SARA_OLLAMA_URL", "http://localhost:11434")
    
    # Resource settings
    MAX_THREADS = int(os.environ.get("SARA_MAX_THREADS", "4"))
    MAX_MEMORY = os.environ.get("SARA_MAX_MEMORY", "4G")
    
    # Production Model Management Settings
    MODEL_CHECK_INTERVAL_DAYS = int(os.environ.get("SARA_MODEL_CHECK_INTERVAL_DAYS", "30"))
    MODEL_WARNING_AGE_DAYS = int(os.environ.get("SARA_MODEL_WARNING_AGE_DAYS", "60"))
    MODEL_CRITICAL_AGE_DAYS = int(os.environ.get("SARA_MODEL_CRITICAL_AGE_DAYS", "90"))
    MODEL_AUTO_UPDATE_PROMPT = os.environ.get("SARA_MODEL_AUTO_UPDATE_PROMPT", "True").lower() in ("true", "1", "t")
    MODEL_UPDATE_CHECK_ENABLED = os.environ.get("SARA_MODEL_UPDATE_CHECK_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_REQUIRE_APPROVAL = os.environ.get("SARA_MODEL_REQUIRE_APPROVAL", "True").lower() in ("true", "1", "t")
    MODEL_CACHE_CLEANUP = os.environ.get("SARA_MODEL_CACHE_CLEANUP", "False").lower() in ("true", "1", "t")
    MODEL_BACKUP_ENABLED = os.environ.get("SARA_MODEL_BACKUP_ENABLED", "True").lower() in ("true", "1", "t")
    MODEL_MAX_BACKUPS = int(os.environ.get("SARA_MODEL_MAX_BACKUPS", "3"))
    MODEL_NOTIFICATION_EMAIL = os.environ.get("SARA_MODEL_NOTIFICATION_EMAIL", "")

    @classmethod
    def has_gpu(cls):
        """
        Detect if GPU is available (CUDA or Apple Silicon MPS).
        
        Returns:
            bool: True if GPU acceleration is available
        """
        try:
            import torch
            logger.debug(f"PyTorch version: {torch.__version__}")
            
            # Check for CUDA first (for compatibility)
            if torch.cuda.is_available():
                logger.debug(f"CUDA available: True, Device count: {torch.cuda.device_count()}")
                return True
            else:
                logger.debug("CUDA not available - checking reasons...")
                # Provide detailed CUDA diagnostics
                if hasattr(torch.cuda, 'is_available'):
                    logger.debug(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
                if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                    cuda_version = torch.version.cuda
                    logger.debug(f"PyTorch compiled with CUDA: {cuda_version if cuda_version else 'No'}")
                
                # Check if NVIDIA GPU is present but PyTorch doesn't have CUDA support
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_names = result.stdout.strip().split('\n')
                        logger.warning(f"NVIDIA GPU(s) detected: {gpu_names}")
                        logger.warning("PyTorch with CUDA support may not be installed. Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                    logger.debug(f"Could not check for NVIDIA GPUs: {e}")
            
            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.debug("Apple Silicon MPS available")
                return True
            else:
                if platform.system() == "Darwin":
                    logger.debug("Running on macOS but MPS not available")
                
            return False
            
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            logger.warning("Install PyTorch with: pip install torch torchvision torchaudio")
            return False
        except Exception as e:
            logger.error(f"Error during GPU detection: {e}")
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
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    return "cuda", f"{device_name} (+{device_count-1} more)"
                return "cuda", device_name
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps", "Apple Silicon MPS"
            else:
                # Try to get CPU info for better diagnostics
                cpu_info = "CPU"
                try:
                    if platform.system() == "Darwin":
                        import subprocess
                        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            cpu_info = f"CPU ({result.stdout.strip()})"
                    elif platform.system() == "Windows":
                        import subprocess
                        result = subprocess.run(['wmic', 'cpu', 'get', 'name', '/format:value'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if line.startswith('Name='):
                                    cpu_info = f"CPU ({line.split('=', 1)[1].strip()})"
                                    break
                except Exception:
                    pass  # Keep default CPU info
                return "cpu", cpu_info
        except ImportError:
            return "cpu", "CPU (PyTorch not available)"
        except Exception as e:
            logger.debug(f"Error getting device info: {e}")
            return "cpu", "CPU"
    
    # Logging settings
    LOG_LEVEL = log_level_name
    LOG_MAX_BYTES = LOG_MAX_BYTES
    LOG_BACKUP_COUNT = LOG_BACKUP_COUNT
    LOG_USE_JSON = LOG_USE_JSON
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("SARA_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    
    # APU filtering setting
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']
    
    # APU KB specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("SARA_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("SARA_APU_KB_EXACT_MATCH_BOOST", "2.0"))

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
        log_size_gb = cls.LOG_MAX_BYTES // 1073741824
        total_space_gb = log_size_gb * (cls.LOG_BACKUP_COUNT + 1)
        logger.info(f"Log rotation: {log_size_gb}GB max size, {cls.LOG_BACKUP_COUNT} backups (~{total_space_gb}GB total)")
        if cls.LOG_USE_JSON:
            logger.info("JSON logging enabled for structured log analysis")
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