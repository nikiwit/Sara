"""
Modern vector management package.

This package provides vector store management with:
- Clean separation of concerns
- Dependency injection
- Comprehensive error handling
- Type safety with modern Python features
- Async/await patterns for I/O operations
- Extensive testing coverage
"""

# Legacy imports for backward compatibility
from .chromadb_manager import ChromaDBManager
from .manager import VectorStoreManager

# Modern service imports (optional - gracefully handle missing dependencies)
try:
    from .models import (
        EmbeddingConfig,
        HealthCheckResult,
        ModelAgeInfo,
        ModelConfig,
        ModelMetadata,
        ModelStatus,
        UpdateInfo,
        VectorStoreStats,
        ModelCacheException,
        UpdateCheckException,
        HealthCheckException,
        VectorStoreException,
    )
    from .services import (
        EmbeddingModelService,
        ModelCacheService,
        ModelUpdateService,
        VectorStoreHealthService,
        VectorStoreManagerModern,
    )
    from .utils import (
        ensure_directory,
        get_optimal_device,
        sanitize_metadata_for_chroma,
        validate_model_path,
        PerformanceTimer,
    )
    MODERN_FEATURES_AVAILABLE = True
except ImportError as e:
    # Modern features not available due to missing dependencies
    import logging
    logger = logging.getLogger("Sara")
    logger.info(f"Modern vector management features not available: {e}")
    logger.info("Install additional dependencies for enhanced features: pip install aiohttp aiofiles pydantic")
    MODERN_FEATURES_AVAILABLE = False
    
    # Create placeholder classes to prevent import errors
    class EmbeddingConfig: pass
    class HealthCheckResult: pass
    class ModelAgeInfo: pass
    class ModelConfig: pass
    class ModelMetadata: pass
    class ModelStatus: pass
    class UpdateInfo: pass
    class VectorStoreStats: pass
    class ModelCacheException(Exception): pass
    class UpdateCheckException(Exception): pass
    class HealthCheckException(Exception): pass
    class VectorStoreException(Exception): pass
    class EmbeddingModelService: pass
    class ModelCacheService: pass
    class ModelUpdateService: pass
    class VectorStoreHealthService: pass
    class VectorStoreManagerModern: pass
    def ensure_directory(*args, **kwargs): pass
    def get_optimal_device(*args, **kwargs): pass
    def sanitize_metadata_for_chroma(*args, **kwargs): pass
    def validate_model_path(*args, **kwargs): pass
    class PerformanceTimer: pass

__all__ = [
    # Legacy compatibility
    'ChromaDBManager',
    'VectorStoreManager',
    
    # Models
    "EmbeddingConfig",
    "HealthCheckResult", 
    "ModelAgeInfo",
    "ModelConfig",
    "ModelMetadata",
    "ModelStatus",
    "UpdateInfo",
    "VectorStoreStats",
    
    # Exceptions
    "ModelCacheException",
    "UpdateCheckException", 
    "HealthCheckException",
    "VectorStoreException",
    
    # Services
    "EmbeddingModelService",
    "ModelCacheService",
    "ModelUpdateService", 
    "VectorStoreHealthService",
    "VectorStoreManagerModern",
    
    # Utilities
    "ensure_directory",
    "get_optimal_device",
    "sanitize_metadata_for_chroma", 
    "validate_model_path",
    "PerformanceTimer",
]

__version__ = "2.0.0"