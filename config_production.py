"""
Production configuration for APURAG system optimized for HGX H100 G593-SD2.
"""

import os
import multiprocessing
from config import Config, logger

class ProductionConfig(Config):
    """Configuration optimized for production HGX H100 environment."""
    
    # Model configuration for production workloads
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    LLM_MODEL_NAME = "qwen2.5:7b-instruct"
    
    # Document chunking parameters optimized for production
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 160  # 20% overlap for context continuity
    
    # Response streaming configuration for optimal user experience
    STREAM_DELAY = 0.015
    
    # Retrieval parameters optimized for production depth and accuracy
    RETRIEVER_K = 12  # Higher document count for comprehensive results
    RETRIEVER_SEARCH_TYPE = "hybrid"
    KEYWORD_RATIO = 0.25  # Favor semantic search in production
    FAQ_MATCH_WEIGHT = 0.8  # Higher precision for FAQ matching
    
    # Query processing settings for enhanced production results
    USE_QUERY_EXPANSION = True
    EXPANSION_FACTOR = 3  # Full expansion for production
    
    # Context processing parameters for production workloads
    MAX_CONTEXT_SIZE = 6000  # 100% increase from baseline for complex queries
    USE_CONTEXT_COMPRESSION = True
    
    # Context optimization settings
    CONTEXT_COMPRESSION_RATIO = 0.70
    CONTEXT_PRIORITY_BOOST = 1.5
    CONTEXT_TARGET_UTILIZATION = 0.75  # 75% utilization for stability
    
    # Session management for concurrent production users
    MAX_SESSIONS = 20
    
    # Ollama API configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # Resource allocation optimized for H100 hardware
    _cpu_cores = multiprocessing.cpu_count()
    MAX_THREADS = min(32, max(16, _cpu_cores * 8 // 10))  # 80% CPU utilization with bounds
    MAX_MEMORY = "32G"  # Aggressive memory allocation for H100 system
    
    # Vector store persistence settings for production uptime
    FORCE_REINDEX = False  # Disabled to prevent production downtime
    ENABLE_VECTOR_STORE_PERSISTENCE = True  # Enabled for fast startup times
    
    # Performance caching configuration for production
    ENABLE_RESULT_CACHING = True
    CACHE_SIZE = 5000  # Large cache for high-traffic production
    ENABLE_RESPONSE_CACHING = True
    RESPONSE_CACHE_SIZE = 1000  # Large response cache for repeated queries
    RESPONSE_CACHE_TTL = 7200  # 2 hours cache lifetime
    
    # Embedding processing optimization
    EMBEDDING_BATCH_SIZE = 64  # Large batches for better throughput
    EMBEDDING_CACHE_SIZE = 10000  # Massive cache for embedding reuse
    
    # LLM optimization settings for production quality
    LLM_TEMPERATURE = 0.6  # Slightly conservative for consistency
    LLM_TOP_P = 0.85
    LLM_CONTEXT_LENGTH = 8192  # Larger context window for 7b model
    
    # Production logging configuration
    LOG_LEVEL = "INFO"
    
    # APU content filtering for production quality assurance
    FILTER_APU_ONLY = True  # Enabled for production content quality
    
    # APU knowledge base specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = 5  # More context for production accuracy
    APU_KB_EXACT_MATCH_BOOST = 3.0  # Higher precision for exact matches
    
    # Production monitoring and reliability configuration
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 100  # Log performance every 100 queries
    SLOW_QUERY_THRESHOLD = 3.0  # 3 second threshold for slow query detection
    
    # Production reliability settings
    ENABLE_CIRCUIT_BREAKER = True
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0
    
    # Production health monitoring configuration
    ENABLE_HEALTH_CHECKS = True
    HEALTH_CHECK_INTERVAL = 300  # Health check every 5 minutes
    
    # Auto-scaling parameters for production load management
    AUTO_SCALE_THREADS = True
    SCALE_UP_THRESHOLD = 0.8  # Scale up at 80% load
    SCALE_DOWN_THRESHOLD = 0.3  # Scale down at 30% load
    
    # ChromaDB settings optimized for production stability
    CHROMADB_PERSIST_RETRY_COUNT = 3
    CHROMADB_PERSIST_RETRY_DELAY = 1.0  # Fast retry for production responsiveness
    CHROMADB_COLLECTION_NAME = "apu_kb_collection"
    CHROMADB_BATCH_SIZE = 32  # Optimized batch size for stability
    CHROMADB_CONNECTION_TIMEOUT = 20  # Connection timeout for reliability
    
    # Collection persistence settings for production reliability
    COLLECTION_VERIFY_ON_LOAD = True  # Always verify collection integrity
    COLLECTION_AUTO_RECOVER = True  # Auto-recover from collection issues
    COLLECTION_BACKUP_ON_SHUTDOWN = True  # Always backup on shutdown
    
    @classmethod
    def setup(cls):
        """Initialize and validate production configuration."""
        # Initialize parent configuration first
        super().setup()
        
        # Log production configuration startup
        logger.info("=" * 70)
        logger.info("PRODUCTION MODE - HGX H100 OPTIMIZED")
        logger.info("=" * 70)
        
        # Report resource allocation for production monitoring
        cpu_utilization = (cls.MAX_THREADS / cls._cpu_cores) * 100
        logger.info(f"Production Resource Allocation:")
        logger.info(f"   CPU Cores: {cls._cpu_cores} total → {cls.MAX_THREADS} threads ({cpu_utilization:.0f}% utilization)")
        logger.info(f"   Memory Limit: {cls.MAX_MEMORY}")
        logger.info(f"   Context Capacity: {cls.MAX_CONTEXT_SIZE} tokens ({((cls.MAX_CONTEXT_SIZE-3000)/3000)*100:+.0f}% vs baseline)")
        logger.info(f"   Retrieval Scope: {cls.RETRIEVER_K} documents ({((cls.RETRIEVER_K-6)/6)*100:+.0f}% vs baseline)")
        
        logger.info(f"Production Model Configuration:")
        logger.info(f"   Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"   LLM Model: {cls.LLM_MODEL_NAME}")
        logger.info(f"   LLM Context Length: {cls.LLM_CONTEXT_LENGTH} tokens")
        logger.info(f"   FAQ Match Weight: {cls.FAQ_MATCH_WEIGHT} ({((cls.FAQ_MATCH_WEIGHT-0.5)/0.5)*100:+.0f}% vs baseline)")
        
        logger.info(f"Production Performance Features:")
        logger.info(f"   Result Cache: {cls.CACHE_SIZE} entries")
        logger.info(f"   Response Cache: {cls.RESPONSE_CACHE_SIZE} entries (TTL: {cls.RESPONSE_CACHE_TTL//60}min)")
        logger.info(f"   Embedding Cache: {cls.EMBEDDING_CACHE_SIZE} entries")
        logger.info(f"   Batch Size: {cls.EMBEDDING_BATCH_SIZE}")
        
        logger.info(f"Production Reliability:")
        logger.info(f"   Circuit Breaker: {'Enabled' if cls.ENABLE_CIRCUIT_BREAKER else 'Disabled'}")
        logger.info(f"   Health Checks: {'Enabled' if cls.ENABLE_HEALTH_CHECKS else 'Disabled'}")
        logger.info(f"   Max Retries: {cls.MAX_RETRY_ATTEMPTS}")
        logger.info(f"   Monitoring: {'Enabled' if cls.ENABLE_PERFORMANCE_MONITORING else 'Disabled'}")
        logger.info(f"   Vector Store Persistence: {'Enabled' if cls.ENABLE_VECTOR_STORE_PERSISTENCE else 'Disabled'}")
        logger.info(f"   APU Content Filtering: {'Enabled' if cls.FILTER_APU_ONLY else 'Disabled'}")
        
        # Ensure absolute paths for production deployment
        if not os.path.isabs(cls.DATA_PATH):
            logger.warning(f"Converting relative data path to absolute: {cls.DATA_PATH}")
            cls.DATA_PATH = os.path.abspath(cls.DATA_PATH)
            
        if not os.path.isabs(cls.PERSIST_PATH):
            logger.warning(f"Converting relative vector store path to absolute: {cls.PERSIST_PATH}")
            cls.PERSIST_PATH = os.path.abspath(cls.PERSIST_PATH)
        
        # Validate production configuration for optimal performance
        cls.validate_production_config()
        
        logger.info("=" * 70)
        logger.info("PRODUCTION CONFIGURATION OPTIMIZED AND APPLIED")
        logger.info("=" * 70)
    
    @classmethod
    def get_production_stats(cls):
        """Retrieve comprehensive production statistics and metrics."""
        return {
            "environment": "production",  
            "hardware_target": "HGX H100",
            "context_size_boost": f"{((cls.MAX_CONTEXT_SIZE-3000)/3000)*100:+.0f}%",
            "retrieval_boost": f"{((cls.RETRIEVER_K-6)/6)*100:+.0f}%", 
            "thread_utilization": f"{(cls.MAX_THREADS/cls._cpu_cores)*100:.0f}%",
            "faq_boost": f"{((cls.FAQ_MATCH_WEIGHT-0.5)/0.5)*100:+.0f}%",
            "cache_capacity": {
                "result_cache": cls.CACHE_SIZE,
                "response_cache": cls.RESPONSE_CACHE_SIZE,
                "embedding_cache": cls.EMBEDDING_CACHE_SIZE
            },
            "reliability_features": {
                "circuit_breaker": cls.ENABLE_CIRCUIT_BREAKER,
                "health_checks": cls.ENABLE_HEALTH_CHECKS,
                "max_retries": cls.MAX_RETRY_ATTEMPTS,
                "vector_store_persistence": cls.ENABLE_VECTOR_STORE_PERSISTENCE,
                "apu_filtering": cls.FILTER_APU_ONLY
            },
            "performance_settings": {
                "max_context_size": cls.MAX_CONTEXT_SIZE,
                "max_threads": cls.MAX_THREADS,
                "max_memory": cls.MAX_MEMORY,
                "embedding_batch_size": cls.EMBEDDING_BATCH_SIZE,
                "slow_query_threshold": cls.SLOW_QUERY_THRESHOLD
            }
        }
    
    @classmethod
    def validate_production_config(cls):
        """Validate production configuration for optimal performance and reliability."""
        issues = []
        warnings = []
        
        # Validate critical production requirements
        if cls.MAX_THREADS < 8:
            issues.append(f"Thread count too low for production workload: {cls.MAX_THREADS}")
        
        if cls.MAX_CONTEXT_SIZE < 4000:
            issues.append(f"Context size too small for production complexity: {cls.MAX_CONTEXT_SIZE}")
        
        if not cls.ENABLE_RESULT_CACHING:
            issues.append("Result caching disabled - critical for production performance") 
        
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE:
            issues.append("Vector store persistence disabled - will cause slow production startups")
        
        if cls.FORCE_REINDEX:
            issues.append("Force reindex enabled - will cause slow production startups")
        
        if not cls.FILTER_APU_ONLY:
            issues.append("APU filtering disabled - will reduce response quality")
        
        # Check for performance optimization warnings
        if cls.RETRIEVER_K > 20:
            warnings.append(f"Very high retrieval count may impact latency: {cls.RETRIEVER_K}")
        
        if cls.MAX_CONTEXT_SIZE > 8000:
            warnings.append(f"Very large context size may impact response time: {cls.MAX_CONTEXT_SIZE}")
        
        # Validate CPU utilization is within optimal range
        cpu_utilization = (cls.MAX_THREADS / cls._cpu_cores) * 100
        if cpu_utilization > 90:
            warnings.append(f"High CPU utilization ({cpu_utilization:.0f}%) may cause contention")
        elif cpu_utilization < 50:
            warnings.append(f"Low CPU utilization ({cpu_utilization:.0f}%) - consider increasing threads")
        
        # Validate cache sizes are appropriate for production
        if cls.CACHE_SIZE < 1000:
            warnings.append(f"Small cache size for production: {cls.CACHE_SIZE}")
        
        if cls.RESPONSE_CACHE_SIZE < 500:
            warnings.append(f"Small response cache for production: {cls.RESPONSE_CACHE_SIZE}")
        
        # Log validation results
        if issues:
            logger.error("CRITICAL PRODUCTION CONFIG ISSUES:")
            for issue in issues:
                logger.error(f"   - {issue}")
        
        if warnings:
            logger.warning("PRODUCTION CONFIG WARNINGS:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
        
        if not issues and not warnings:
            logger.info("Production configuration validation passed - optimal settings detected")
        elif not issues:
            logger.info("Production configuration validation passed with minor warnings")
        
        return len(issues) == 0, issues, warnings
    
    @classmethod 
    def should_rebuild_vector_store(cls):
        """Determine if vector store rebuild is required with production-specific logic."""
        # Conservative approach for production to minimize downtime
        if cls.FORCE_REINDEX:
            logger.warning("PRODUCTION: Force reindex enabled - this will cause downtime!")
            return True
        
        if not cls.ENABLE_VECTOR_STORE_PERSISTENCE:
            logger.error("PRODUCTION: Vector store persistence disabled - this will cause frequent rebuilds!")
            return True
        
        # Use parent class logic for file-based rebuild checks
        parent_result = super().should_rebuild_vector_store()
        
        if parent_result:
            logger.warning("PRODUCTION: Vector store rebuild required - this may cause service interruption")
            
        return parent_result
    
    @classmethod
    def get_performance_recommendations(cls):
        """Generate performance optimization recommendations based on current configuration."""
        recommendations = []
        
        # Calculate current CPU utilization
        cpu_util = (cls.MAX_THREADS / cls._cpu_cores) * 100
        
        # Recommend thread count optimization
        if cpu_util < 60:
            recommendations.append(f"Consider increasing MAX_THREADS from {cls.MAX_THREADS} to {int(cls._cpu_cores * 0.7)} for better CPU utilization")
        
        # Recommend context size optimization
        if cls.MAX_CONTEXT_SIZE < 5000:
            recommendations.append(f"Consider increasing MAX_CONTEXT_SIZE from {cls.MAX_CONTEXT_SIZE} to 5000+ for better response quality")
        
        # Recommend cache size optimization
        if cls.CACHE_SIZE < 3000:
            recommendations.append(f"Consider increasing CACHE_SIZE from {cls.CACHE_SIZE} to 3000+ for better cache hit rates")
        
        # Recommend batch size optimization
        if cls.EMBEDDING_BATCH_SIZE < 64:
            recommendations.append(f"Consider increasing EMBEDDING_BATCH_SIZE from {cls.EMBEDDING_BATCH_SIZE} to 64+ for better throughput")
        
        return recommendations