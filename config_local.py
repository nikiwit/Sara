"""
Local development configuration for APURAG system.
"""

import os
import multiprocessing
from config import Config, logger

class LocalConfig(Config):
    """Configuration optimized for local development environments."""
    
    # Model configuration optimized for local development resources
    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    LLM_MODEL_NAME = "qwen2.5:3b-instruct"

    # Document chunking parameters for local development
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 100
    
    # Response streaming configuration for development
    STREAM_DELAY = 0.008
    
    # Retrieval parameters balanced for local performance
    RETRIEVER_K = 6  # Moderate document count for development
    RETRIEVER_SEARCH_TYPE = "hybrid"
    KEYWORD_RATIO = 0.4
    FAQ_MATCH_WEIGHT = 0.5
    
    # Query processing settings for development
    USE_QUERY_EXPANSION = True
    EXPANSION_FACTOR = 2
    
    # Context processing parameters for local environments
    MAX_CONTEXT_SIZE = 4000  # Increased from baseline for better responses
    USE_CONTEXT_COMPRESSION = True
    
    # Context optimization settings
    CONTEXT_COMPRESSION_RATIO = 0.75
    CONTEXT_PRIORITY_BOOST = 1.3
    CONTEXT_TARGET_UTILIZATION = 0.80
    
    # Session management for development
    MAX_SESSIONS = 5
    
    # Ollama API configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # Resource allocation for local development
    _cpu_cores = multiprocessing.cpu_count()
    MAX_THREADS = 6  # Conservative thread count for 10-core Apple Silicon
    MAX_MEMORY = "2G"
    
    # Vector store persistence settings for development
    FORCE_REINDEX = False  # Disabled to avoid slow development startup
    ENABLE_VECTOR_STORE_PERSISTENCE = True  # Enabled for fast development iteration
    
    # Performance optimization settings for local development
    ENABLE_RESULT_CACHING = True
    CACHE_SIZE = 1000
    ENABLE_RESPONSE_CACHING = True
    RESPONSE_CACHE_SIZE = 100
    RESPONSE_CACHE_TTL = 3600
    
    # Embedding processing settings for local resources
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_CACHE_SIZE = 2000
    
    # LLM settings optimized for development
    LLM_TEMPERATURE = 0.7
    LLM_TOP_P = 0.9
    LLM_CONTEXT_LENGTH = 4096
    
    # Development logging configuration
    LOG_LEVEL = "INFO"
    
    # APU content filtering for development quality
    FILTER_APU_ONLY = True  # Enabled to maintain content relevance
    
    # APU knowledge base specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = 3
    APU_KB_EXACT_MATCH_BOOST = 2.5
    
    # Performance monitoring for development debugging
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 10
    SLOW_QUERY_THRESHOLD = 2.0
    
    # ChromaDB settings optimized for local development
    CHROMADB_PERSIST_RETRY_COUNT = 3
    CHROMADB_PERSIST_RETRY_DELAY = 1.0  # Fast retries for development responsiveness
    CHROMADB_COLLECTION_NAME = "apu_kb_collection"
    CHROMADB_BATCH_SIZE = 32  # Smaller batches for local stability
    CHROMADB_CONNECTION_TIMEOUT = 20  # Reasonable timeout for local connections

    # Collection persistence settings for development reliability
    COLLECTION_VERIFY_ON_LOAD = True  # Always verify collection integrity locally
    COLLECTION_AUTO_RECOVER = True  # Auto-recover from collection issues
    COLLECTION_BACKUP_ON_SHUTDOWN = True  # Always backup on shutdown
    
    @classmethod
    def setup(cls):
        """Initialize local development configuration with optimizations."""
        # Initialize parent configuration first
        super().setup()
        
        # Log local development configuration details
        logger.info("LOCAL DEVELOPMENT - OPTIMIZED CONFIGURATION APPLIED")
        logger.info(f"Context Size UPGRADED: {cls.MAX_CONTEXT_SIZE} tokens ({((cls.MAX_CONTEXT_SIZE-3000)/3000)*100:+.0f}% vs baseline)")
        logger.info(f"Retrieval ENHANCED: {cls.RETRIEVER_K} documents ({((cls.RETRIEVER_K-4)/4)*100:+.0f}% vs baseline)")
        logger.info(f"CPU Threads OPTIMIZED: {cls.MAX_THREADS} threads ({(cls.MAX_THREADS/cls._cpu_cores)*100:.0f}% utilization)")
        logger.info(f"Memory INCREASED: {cls.MAX_MEMORY}")
        logger.info(f"FAQ Matching IMPROVED: {cls.FAQ_MATCH_WEIGHT} weight")
        logger.info(f"Caching ENABLED: {cls.CACHE_SIZE} query cache + {cls.RESPONSE_CACHE_SIZE} response cache")
        
        # Analyze CPU utilization and provide recommendations
        cpu_util = (cls.MAX_THREADS / cls._cpu_cores) * 100
        if cpu_util < 50:
            logger.warning("Running with limited threads - consider increasing MAX_THREADS for better performance")
        else:
            logger.info(f"Good CPU utilization: {cpu_util:.0f}%")
        
        # Check for platform-specific optimizations
        cls._check_apple_silicon_optimization()
        
        # Validate critical local development settings
        cls._validate_local_settings()
    
    @classmethod
    def _check_apple_silicon_optimization(cls):
        """Detect and report Apple Silicon specific optimizations."""
        try:
            import platform
            system_info = f"{platform.system()} {platform.machine()}"
            
            if platform.machine() == 'arm64' and platform.system() == 'Darwin':
                logger.info("Apple Silicon detected - using optimized settings")
                
                # Check for Metal Performance Shaders availability
                try:
                    import torch
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        logger.info("Apple MPS (Metal Performance Shaders) available for acceleration")
                except ImportError:
                    logger.debug("PyTorch not available for MPS check")
            else:
                logger.info(f"Running on: {system_info}")
                
        except Exception as e:
            logger.debug(f"Could not detect system architecture: {e}")
    
    @classmethod
    def _validate_local_settings(cls):
        """Validate that critical local development settings are properly configured."""
        issues = []
        
        # Validate minimum thread count for reasonable performance
        if cls.MAX_THREADS < 4:
            issues.append(f"Thread count too low: {cls.MAX_THREADS} (expected >= 4)")
        
        # Ensure development startup speed optimizations
        if cls.FORCE_REINDEX:
            issues.append("Force reindex is enabled - will cause slow startups")
        
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE:
            issues.append("Vector store persistence disabled - will cause rebuilds")
        
        # Validate content quality settings
        if not cls.FILTER_APU_ONLY:
            issues.append("APU filtering disabled - non-APU content will dilute search")
        
        # Validate performance optimization settings
        if not cls.ENABLE_RESULT_CACHING:
            issues.append("Result caching disabled - will impact performance")
        
        # Validate context and retrieval parameters
        if cls.MAX_CONTEXT_SIZE < 3500:
            issues.append(f"Context size too small: {cls.MAX_CONTEXT_SIZE} (expected >= 3500)")
        
        if cls.RETRIEVER_K < 5:
            issues.append(f"Retrieval count too low: {cls.RETRIEVER_K} (expected >= 5)")
        
        # Report validation results
        if issues:
            logger.warning("Local Configuration Issues Detected:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("Local configuration validation passed")
        
        return len(issues) == 0
    
    @classmethod
    def get_local_optimization_summary(cls):
        """Generate comprehensive summary of local development optimizations."""
        cpu_cores = multiprocessing.cpu_count()
        return {
            "environment": "local_development",
            "cpu_cores_available": cpu_cores,
            "cpu_cores_used": cls.MAX_THREADS,
            "cpu_utilization": f"{(cls.MAX_THREADS/cpu_cores)*100:.0f}%",
            "context_size": cls.MAX_CONTEXT_SIZE,
            "retrieval_documents": cls.RETRIEVER_K,
            "memory_limit": cls.MAX_MEMORY,
            "apu_filtering_enabled": cls.FILTER_APU_ONLY,
            "vector_store_persistence": cls.ENABLE_VECTOR_STORE_PERSISTENCE,
            "force_reindex_disabled": not cls.FORCE_REINDEX,
            "caching_enabled": cls.ENABLE_RESULT_CACHING,
            "performance_monitoring": cls.ENABLE_PERFORMANCE_MONITORING,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def should_rebuild_vector_store(cls):
        """Determine if vector store rebuild is required with local development considerations."""
        # Check parent class logic for file-based rebuild requirements
        parent_result = super().should_rebuild_vector_store()
        
        if parent_result:
            logger.info("Parent class recommends rebuilding vector store")
            return True
        
        # Additional local development specific checks
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE:
            logger.info("Local config: Vector store persistence disabled - will rebuild")
            return True
            
        # For local development, prioritize fast startup over rebuilds
        logger.info("Local config: Vector store persistence enabled and exists - will reuse")
        return False