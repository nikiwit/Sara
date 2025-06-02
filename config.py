"""
Configuration settings for the APURAG system with environment-specific support.
"""

import os
import logging
import platform
import multiprocessing
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

# Enhanced logging configuration
log_level_name = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name.upper(), logging.INFO)

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Enhanced logging with rotation and better formatting
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "apurag.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

# Add performance logging
perf_logger = logging.getLogger("APURAG.Performance")
perf_handler = logging.FileHandler(os.path.join(logs_dir, "performance.log"))
perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
perf_logger.addHandler(perf_handler)
perf_logger.setLevel(logging.INFO)

def setup_nltk_with_fallback():
    """Setup NLTK with proper error handling and fallback options."""
    try:
        import nltk
        
        # Set NLTK data path to a writable location
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data with better error handling
        required_datasets = ['wordnet', 'punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for dataset in required_datasets:
            try:
                nltk.data.find(f'corpora/{dataset}' if dataset in ['wordnet', 'stopwords'] else f'tokenizers/{dataset}')
                logger.debug(f"NLTK {dataset} data already available")
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {dataset} data...")
                    nltk.download(dataset, quiet=True, raise_on_error=True, download_dir=nltk_data_dir)
                    logger.info(f"Successfully downloaded NLTK {dataset} data")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK {dataset} data: {e}")
                    
    except ImportError:
        logger.info("NLTK not available - using basic text processing methods")
    except Exception as e:
        logger.warning(f"NLTK setup encountered an error: {e}")
        logger.info("Continuing with fallback text processing methods")

class Config:
    """Base configuration settings for the RAG application."""
    
    # Class variable to track setup status
    _setup_completed = False
    _performance_stats = {
        "startup_time": None,
        "config_load_time": None,
        "total_queries_processed": 0
    }
    
    # Environment
    ENV = ENV
    
    # Enhanced path management
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Additional directory paths
    LOGS_PATH = os.path.join(SCRIPT_DIR, "logs")
    CACHE_PATH = os.path.join(SCRIPT_DIR, "cache")
    TEMP_PATH = os.path.join(SCRIPT_DIR, "temp")
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "qwen2.5:3b-instruct")
    
    # Enhanced chunking settings
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "600"))  # Increased from 500
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "120"))  # Optimized overlap
    
    # Performance settings
    STREAM_DELAY = float(os.environ.get("CUSTOMRAG_STREAM_DELAY", "0.012"))  # Optimized delay
    
    # Enhanced retrieval settings
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "6"))  # Base setting
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))  # Balanced ratio
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.6"))  # Improved FAQ matching
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "2"))  # Conservative for stability
    
    # Context processing settings - major improvement
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "3000"))  # Base setting - will be overridden
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Enhanced context processing parameters
    CONTEXT_COMPRESSION_RATIO = float(os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION_RATIO", "0.75"))
    CONTEXT_PRIORITY_BOOST = float(os.environ.get("CUSTOMRAG_CONTEXT_PRIORITY_BOOST", "1.3"))
    CONTEXT_TARGET_UTILIZATION = float(os.environ.get("CUSTOMRAG_TARGET_UTILIZATION", "0.80"))  # Target 80%
    
    # Session management
    MAX_SESSIONS = int(os.environ.get("CUSTOMRAG_MAX_SESSIONS", "5"))
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Enhanced resource settings with automatic detection
    _cpu_cores = multiprocessing.cpu_count()
    MAX_THREADS = int(os.environ.get("CUSTOMRAG_MAX_THREADS", str(max(2, min(4, _cpu_cores // 2)))))  # Conservative base
    MAX_MEMORY = os.environ.get("CUSTOMRAG_MAX_MEMORY", "4G")  # Base setting
    
    # Performance optimization settings
    ENABLE_RESULT_CACHING = os.environ.get("CUSTOMRAG_ENABLE_CACHING", "True").lower() in ("true", "1", "t")
    CACHE_SIZE = int(os.environ.get("CUSTOMRAG_CACHE_SIZE", "1000"))
    ENABLE_RESPONSE_CACHING = os.environ.get("CUSTOMRAG_ENABLE_RESPONSE_CACHING", "True").lower() in ("true", "1", "t")
    RESPONSE_CACHE_SIZE = int(os.environ.get("CUSTOMRAG_RESPONSE_CACHE_SIZE", "100"))
    RESPONSE_CACHE_TTL = int(os.environ.get("CUSTOMRAG_RESPONSE_CACHE_TTL", "3600"))  # 1 hour
    
    # Enhanced embedding settings
    EMBEDDING_BATCH_SIZE = int(os.environ.get("CUSTOMRAG_EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_CACHE_SIZE = int(os.environ.get("CUSTOMRAG_EMBEDDING_CACHE_SIZE", "2000"))
    
    # LLM optimization settings
    LLM_TEMPERATURE = float(os.environ.get("CUSTOMRAG_LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P = float(os.environ.get("CUSTOMRAG_LLM_TOP_P", "0.9"))
    LLM_CONTEXT_LENGTH = int(os.environ.get("CUSTOMRAG_LLM_CONTEXT_LENGTH", "4096"))
    
    # Vector store persistence settings
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    ENABLE_VECTOR_STORE_PERSISTENCE = os.environ.get("CUSTOMRAG_ENABLE_PERSISTENCE", "True").lower() in ("true", "1", "t")
    VECTOR_STORE_CHECK_INTERVAL = int(os.environ.get("CUSTOMRAG_VS_CHECK_INTERVAL", "300"))  # 5 minutes
    
    # ChromaDB specific settings
    CHROMADB_PERSIST_RETRY_COUNT = int(os.environ.get("CUSTOMRAG_CHROMADB_RETRY", "3"))
    CHROMADB_PERSIST_RETRY_DELAY = float(os.environ.get("CUSTOMRAG_CHROMADB_RETRY_DELAY", "2.0"))
    CHROMADB_COLLECTION_NAME = os.environ.get("CUSTOMRAG_COLLECTION_NAME", "apu_kb_collection")
    CHROMADB_BATCH_SIZE = int(os.environ.get("CUSTOMRAG_CHROMADB_BATCH_SIZE", "50"))
    CHROMADB_CONNECTION_TIMEOUT = int(os.environ.get("CUSTOMRAG_CHROMADB_TIMEOUT", "30"))

    # Collection persistence settings
    COLLECTION_VERIFY_ON_LOAD = os.environ.get("CUSTOMRAG_VERIFY_COLLECTION", "True").lower() in ("true", "1", "t")
    COLLECTION_AUTO_RECOVER = os.environ.get("CUSTOMRAG_AUTO_RECOVER", "True").lower() in ("true", "1", "t")
    COLLECTION_BACKUP_ON_SHUTDOWN = os.environ.get("CUSTOMRAG_BACKUP_ON_SHUTDOWN", "True").lower() in ("true", "1", "t")

    @classmethod
    def get_chromadb_settings(cls):
        """Get ChromaDB-specific settings as a dictionary."""
        return {
            "persist_directory": cls.PERSIST_PATH,
            "collection_name": cls.CHROMADB_COLLECTION_NAME,
            "batch_size": cls.CHROMADB_BATCH_SIZE,
            "retry_count": cls.CHROMADB_PERSIST_RETRY_COUNT,
            "retry_delay": cls.CHROMADB_PERSIST_RETRY_DELAY,
            "connection_timeout": cls.CHROMADB_CONNECTION_TIMEOUT,
            "verify_on_load": cls.COLLECTION_VERIFY_ON_LOAD,
            "auto_recover": cls.COLLECTION_AUTO_RECOVER,
            "backup_on_shutdown": cls.COLLECTION_BACKUP_ON_SHUTDOWN
        }

    @classmethod
    def validate_chromadb_settings(cls):
        """Validate ChromaDB settings are reasonable."""
        issues = []
        
        if cls.CHROMADB_BATCH_SIZE < 10:
            issues.append("ChromaDB batch size too small - may impact performance")
        elif cls.CHROMADB_BATCH_SIZE > 1000:
            issues.append("ChromaDB batch size too large - may cause memory issues")
        
        if cls.CHROMADB_PERSIST_RETRY_COUNT < 1:
            issues.append("ChromaDB retry count must be at least 1")
        
        if cls.CHROMADB_CONNECTION_TIMEOUT < 10:
            issues.append("ChromaDB timeout too short - may cause connection failures")
        
        return len(issues) == 0, issues
    
    # Hardware detection - enhanced
    @classmethod
    def has_gpu(cls):
        """Detect if GPU is available with enhanced detection."""
        try:
            import torch
            if torch.cuda.is_available():
                return True
            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True
        except ImportError:
            pass
        return False
    
    @classmethod
    def get_device_info(cls):
        """Get detailed device information."""
        info = {
            "cpu_cores": cls._cpu_cores,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "has_gpu": cls.has_gpu()
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["total_memory_gb"] = round(memory.total / (1024**3), 2)
            info["available_memory_gb"] = round(memory.available / (1024**3), 2)
        except ImportError:
            info["memory_info"] = "psutil not available"
        
        return info
    
    # APU filtering - enabled by default
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "True").lower() in ("true", "1", "t")
    
    # Supported file types - enhanced
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub', '.html', '.htm']
    
    # APU KB specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.5"))  # Increased boost
    
    # Performance monitoring settings
    ENABLE_PERFORMANCE_MONITORING = os.environ.get("CUSTOMRAG_ENABLE_MONITORING", "True").lower() in ("true", "1", "t")
    PERFORMANCE_LOG_INTERVAL = int(os.environ.get("CUSTOMRAG_PERF_LOG_INTERVAL", "10"))
    SLOW_QUERY_THRESHOLD = float(os.environ.get("CUSTOMRAG_SLOW_QUERY_THRESHOLD", "2.0"))  # Log queries > 2 seconds
    
    # Vector store management methods
    @classmethod
    def should_rebuild_vector_store(cls):
        """Check if vector store needs rebuilding."""
        if cls.FORCE_REINDEX:
            logger.info("Force reindex enabled - will rebuild vector store")
            return True
            
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE:
            logger.info("Vector store persistence disabled - will rebuild")
            return True
            
        if not os.path.exists(cls.PERSIST_PATH):
            logger.info("Vector store directory doesn't exist - will create new")
            return True
            
        # Check if vector store is empty
        vs_files = [f for f in os.listdir(cls.PERSIST_PATH) if f.endswith('.sqlite3')]
        if not vs_files:
            logger.info("Vector store directory empty - will rebuild")
            return True
            
        # Check if documents are newer than vector store
        try:
            vs_modified = max([os.path.getmtime(os.path.join(cls.PERSIST_PATH, f)) for f in vs_files])
            
            if os.path.exists(cls.DATA_PATH):
                data_files = [f for f in os.listdir(cls.DATA_PATH) if f.endswith(tuple(cls.SUPPORTED_EXTENSIONS))]
                if data_files:
                    latest_doc_modified = max([os.path.getmtime(os.path.join(cls.DATA_PATH, f)) for f in data_files])
                    if latest_doc_modified > vs_modified:
                        logger.info("Documents newer than vector store - will rebuild")
                        return True
        except Exception as e:
            logger.warning(f"Error checking file timestamps: {e} - will rebuild to be safe")
            return True
            
        logger.info("Vector store exists and is up to date - will load existing")
        return False
    
    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        start_time = datetime.now()
        
        # Prevent duplicate setup logging
        if cls._setup_completed:
            logger.debug("Configuration setup already completed, skipping duplicate setup")
            return
        
        try:
            # Create all necessary directories
            directories = [cls.DATA_PATH, cls.PERSIST_PATH, cls.LOGS_PATH, cls.CACHE_PATH, cls.TEMP_PATH]
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            # Setup NLTK with proper error handling
            setup_nltk_with_fallback()
            
            # Get device information
            device_info = cls.get_device_info()
            
            # Comprehensive logging
            logger.info("=" * 60)
            logger.info("APURAG SYSTEM INITIALIZATION")
            logger.info("=" * 60)
            logger.info(f"Environment: {cls.ENV.upper()}")
            logger.info(f"Platform: {device_info['platform']} {device_info['architecture']}")
            logger.info(f"CPU Cores: {device_info['cpu_cores']} (using {cls.MAX_THREADS} threads)")
            
            if 'total_memory_gb' in device_info:
                logger.info(f"Memory: {device_info['total_memory_gb']}GB total, {device_info['available_memory_gb']}GB available")
            
            logger.info(f"GPU Available: {device_info['has_gpu']}")
            logger.info(f"Python Version: {device_info['python_version']}")
            
            logger.info("-" * 40)
            logger.info("DIRECTORY PATHS:")
            logger.info(f"  Data: {cls.DATA_PATH}")
            logger.info(f"  Vector Store: {cls.PERSIST_PATH}")
            logger.info(f"  Logs: {cls.LOGS_PATH}")
            logger.info(f"  Cache: {cls.CACHE_PATH}")
            
            logger.info("-" * 40)
            logger.info("MODEL CONFIGURATION:")
            logger.info(f"  Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
            logger.info(f"  LLM Model: {cls.LLM_MODEL_NAME}")
            logger.info(f"  Search Type: {cls.RETRIEVER_SEARCH_TYPE}")
            
            logger.info("-" * 40)
            logger.info("PERFORMANCE SETTINGS:")
            logger.info(f"  Max Context Size: {cls.MAX_CONTEXT_SIZE} tokens")
            logger.info(f"  Retrieval K: {cls.RETRIEVER_K} documents")
            logger.info(f"  Max Threads: {cls.MAX_THREADS}")
            logger.info(f"  Max Memory: {cls.MAX_MEMORY}")
            logger.info(f"  Target Context Utilization: {cls.CONTEXT_TARGET_UTILIZATION:.0%}")
            
            if cls.RETRIEVER_SEARCH_TYPE == "hybrid":
                logger.info(f"  Keyword Ratio: {cls.KEYWORD_RATIO}")
                logger.info(f"  FAQ Match Weight: {cls.FAQ_MATCH_WEIGHT}")
            
            logger.info("-" * 40)
            logger.info("OPTIMIZATION FEATURES:")
            logger.info(f"  Result Caching: {'Enabled' if cls.ENABLE_RESULT_CACHING else 'Disabled'} ({cls.CACHE_SIZE} entries)")
            logger.info(f"  Response Caching: {'Enabled' if cls.ENABLE_RESPONSE_CACHING else 'Disabled'} ({cls.RESPONSE_CACHE_SIZE} entries)")
            logger.info(f"  Query Expansion: {'Enabled' if cls.USE_QUERY_EXPANSION else 'Disabled'}")
            logger.info(f"  Context Compression: {'Enabled' if cls.USE_CONTEXT_COMPRESSION else 'Disabled'}")
            logger.info(f"  Performance Monitoring: {'Enabled' if cls.ENABLE_PERFORMANCE_MONITORING else 'Disabled'}")
            logger.info(f"  Vector Store Persistence: {'Enabled' if cls.ENABLE_VECTOR_STORE_PERSISTENCE else 'Disabled'}")
            
            # Log APU filtering status
            if cls.FILTER_APU_ONLY:
                logger.info("APU document filtering: ENABLED - only APU-related files will be processed")
            else:
                logger.info("APU document filtering: DISABLED - all compatible files will be processed")
            
            # Resource utilization analysis
            cpu_utilization = (cls.MAX_THREADS / cls._cpu_cores) * 100
            if cpu_utilization < 40:
                logger.warning(f"Low CPU utilization ({cpu_utilization:.0f}%) - consider increasing MAX_THREADS")
            elif cpu_utilization > 80:
                logger.warning(f"High CPU utilization ({cpu_utilization:.0f}%) - consider reducing MAX_THREADS")
            else:
                logger.info(f"Optimal CPU utilization: {cpu_utilization:.0f}%")
            
            # Mark setup as completed
            setup_time = (datetime.now() - start_time).total_seconds()
            cls._performance_stats["config_load_time"] = setup_time
            cls._setup_completed = True
            
            logger.info("=" * 60)
            logger.info(f"CONFIGURATION SETUP COMPLETED in {setup_time:.2f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Configuration setup failed: {e}")
            raise
    
    @classmethod
    def get_optimization_summary(cls):
        """Get a summary of optimization improvements."""
        return {
            "context_size_increase": f"{((cls.MAX_CONTEXT_SIZE-3000)/3000)*100:+.0f}%",
            "retrieval_increase": f"{((cls.RETRIEVER_K-4)/4)*100:+.0f}%",
            "thread_utilization": f"{(cls.MAX_THREADS/cls._cpu_cores)*100:.0f}%",
            "caching_enabled": cls.ENABLE_RESULT_CACHING,
            "compression_enabled": cls.USE_CONTEXT_COMPRESSION,
            "target_utilization": f"{cls.CONTEXT_TARGET_UTILIZATION:.0%}",
            "faq_weight_boost": f"{((cls.FAQ_MATCH_WEIGHT-0.5)/0.5)*100:+.0f}%",
            "apu_filtering": cls.FILTER_APU_ONLY,
            "persistence_enabled": cls.ENABLE_VECTOR_STORE_PERSISTENCE
        }
    
    @classmethod
    def log_performance_stats(cls):
        """Log current performance statistics."""
        if cls.ENABLE_PERFORMANCE_MONITORING:
            stats = cls._performance_stats.copy()
            stats.update(cls.get_optimization_summary())
            perf_logger.info(f"Performance Stats: {stats}")
    
    @classmethod
    def validate_configuration(cls):
        """Validate configuration settings and warn about potential issues."""
        warnings = []
        
        if cls.MAX_CONTEXT_SIZE > 8000:
            warnings.append("Very large context size may impact performance")
        
        if cls.MAX_THREADS > cls._cpu_cores:
            warnings.append("Thread count exceeds CPU cores")
        
        if cls.RETRIEVER_K > 20:
            warnings.append("Very high retrieval K may impact performance")
        
        if not cls.ENABLE_RESULT_CACHING:
            warnings.append("Result caching disabled - may impact response times")
        
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE and not cls.FORCE_REINDEX:
            warnings.append("Vector store persistence disabled but force reindex also disabled")
        
        for warning in warnings:
            logger.warning(f"Configuration Warning: {warning}")
        
        return len(warnings) == 0

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