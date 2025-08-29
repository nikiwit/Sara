"""
Service classes for vector store management.

This module implements the single responsibility principle by breaking down
the monolithic VectorStoreManager into focused service classes with proper
dependency injection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

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
)
from .utils import (
    PerformanceTimer,
    ensure_directory,
    fetch_model_info_async,
    find_model_cache_paths,
    get_optimal_device,
    safe_file_operation,
    safe_remove_directory,
    sanitize_metadata_for_chroma,
    validate_model_path,
)

logger = logging.getLogger("Sara")


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    
    @property
    def embedding_model_name(self) -> str: ...
    
    @property
    def persist_path(self) -> str: ...
    
    @property
    def chunk_size(self) -> int: ...
    
    @property
    def chunk_overlap(self) -> int: ...
    
    @property
    def env(self) -> str: ...


class ModelCacheService:
    """Service for managing model caching operations."""
    
    def __init__(self, cache_base_path: Path, config: ModelConfig):
        self.cache_base = ensure_directory(cache_base_path)
        self.config = config
        self.metadata_cache: dict[str, ModelMetadata] = {}
        
    def setup_cache_environment(self) -> None:
        """Setup HuggingFace cache environment variables."""
        import os
        
        # Set HuggingFace environment variables
        os.environ['HF_HOME'] = str(self.cache_base)
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_base / "sentence_transformers")
        
        # Performance optimizations
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        
        # Remove deprecated variables
        if 'TRANSFORMERS_CACHE' in os.environ:
            del os.environ['TRANSFORMERS_CACHE']
        
        logger.info(f"Model cache directory configured: {self.cache_base}")
    
    def cleanup_corrupted_files(self) -> int:
        """
        Clean up corrupted model cache files.
        
        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0
        
        # Clean incomplete downloads
        for incomplete_file in self.cache_base.rglob('*.incomplete'):
            try:
                incomplete_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed incomplete file: {incomplete_file}")
            except OSError as e:
                logger.warning(f"Could not remove incomplete file {incomplete_file}: {e}")
        
        # Clean lock files
        for lock_file in self.cache_base.rglob('*.lock'):
            try:
                lock_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed lock file: {lock_file}")
            except OSError as e:
                logger.warning(f"Could not remove lock file {lock_file}: {e}")
        
        # Clean temporary files
        for tmp_file in self.cache_base.rglob('*.tmp'):
            try:
                tmp_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed temporary file: {tmp_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {tmp_file}: {e}")
        
        if cleanup_count > 0:
            logger.info(f"Model cache cleanup completed: removed {cleanup_count} corrupted files")
        
        return cleanup_count
    
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if model is cached and valid.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is properly cached
        """
        cache_paths = find_model_cache_paths(model_name, self.cache_base)
        
        for path in cache_paths:
            if validate_model_path(path):
                logger.info(f"Model {model_name} found in cache at {path}")
                return True
        
        logger.debug(f"Model {model_name} not found in cache")
        return False
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get cached model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata or None if not found
        """
        if model_name in self.metadata_cache:
            return self.metadata_cache[model_name]
        
        metadata_file = self.cache_base / f"{model_name.replace('/', '_')}_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with safe_file_operation(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = ModelMetadata.from_dict(data)
                self.metadata_cache[model_name] = metadata
                return metadata
        except (json.JSONDecodeError, KeyError, ModelCacheException) as e:
            logger.warning(f"Failed to read metadata for {model_name}: {e}")
            return None
    
    def save_model_metadata(self, metadata: ModelMetadata) -> None:
        """
        Save model metadata to disk.
        
        Args:
            metadata: Metadata to save
        """
        metadata_file = self.cache_base / f"{metadata.model_name.replace('/', '_')}_metadata.json"
        
        try:
            with safe_file_operation(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update cache
            self.metadata_cache[metadata.model_name] = metadata
            logger.info(f"Saved metadata for model: {metadata.model_name} (usage: {metadata.usage_count})")
            
        except ModelCacheException as e:
            logger.error(f"Failed to save metadata for {metadata.model_name}: {e}")
    
    def register_model_usage(self, model_name: str, model_path: Optional[Path] = None) -> None:
        """
        Register model usage and update metadata.
        
        Args:
            model_name: Name of the model
            model_path: Optional path to the model
        """
        metadata = self.get_model_metadata(model_name)
        
        if metadata:
            metadata.increment_usage()
        else:
            # Create new metadata
            if model_path is None:
                cache_paths = find_model_cache_paths(model_name, self.cache_base)
                model_path = cache_paths[0] if cache_paths else None
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_path=model_path
            )
        
        self.save_model_metadata(metadata)
    
    def clear_model_cache(self, model_name: str) -> bool:
        """
        Clear cached model files.
        
        Args:
            model_name: Name of the model to clear
            
        Returns:
            True if successfully cleared
        """
        cache_paths = find_model_cache_paths(model_name, self.cache_base)
        removed_any = False
        
        for path in cache_paths:
            if safe_remove_directory(path):
                logger.info(f"Removed cached model at: {path}")
                removed_any = True
        
        if removed_any:
            # Remove metadata
            if model_name in self.metadata_cache:
                del self.metadata_cache[model_name]
            
            metadata_file = self.cache_base / f"{model_name.replace('/', '_')}_metadata.json"
            if metadata_file.exists():
                try:
                    metadata_file.unlink()
                except OSError as e:
                    logger.warning(f"Could not remove metadata file: {e}")
            
            logger.info(f"Successfully cleared cache for model: {model_name}")
        else:
            logger.warning(f"No cached files found for model: {model_name}")
        
        return removed_any


class ModelUpdateService:
    """Service for checking and managing model updates."""
    
    def __init__(self, cache_service: ModelCacheService, config: ModelConfig):
        self.cache_service = cache_service
        self.config = config
    
    def check_model_age(self, model_name: str) -> ModelAgeInfo:
        """
        Check model age and determine status.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Model age information
        """
        metadata = self.cache_service.get_model_metadata(model_name)
        
        if not metadata:
            return ModelAgeInfo(
                status=ModelStatus.UNKNOWN,
                age_days=0,
                should_check_updates=self.config.update_check_enabled,
                message="Model not registered, will check for updates"
            )
        
        age_days = (datetime.now(timezone.utc) - metadata.first_cached).days
        
        # Determine status based on age
        if age_days < self.config.check_interval_days:
            status = ModelStatus.FRESH
            should_check = False
            message = f"Model is fresh ({age_days} days old)"
        elif age_days < self.config.warning_age_days:
            status = ModelStatus.GOOD
            should_check = self.config.update_check_enabled
            message = f"Model is good ({age_days} days old)"
        elif age_days < self.config.critical_age_days:
            status = ModelStatus.AGING
            should_check = self.config.update_check_enabled
            message = f"Model is aging ({age_days} days old) - checking for updates recommended"
        else:
            status = ModelStatus.STALE
            should_check = self.config.update_check_enabled
            message = f"Model is stale ({age_days} days old) - update strongly recommended"
        
        return ModelAgeInfo(
            status=status,
            age_days=age_days,
            should_check_updates=should_check,
            message=message,
            first_cached=metadata.first_cached
        )
    
    async def check_for_updates_async(self, model_name: str) -> UpdateInfo:
        """
        Asynchronously check for model updates.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Update information
        """
        try:
            # Fetch model info from HuggingFace Hub
            with PerformanceTimer(f"Fetching model info for {model_name}"):
                model_info = await fetch_model_info_async(model_name)
            
            # Extract last modified date
            hub_last_modified = model_info.get('lastModified')
            if not hub_last_modified:
                return UpdateInfo(
                    has_updates=None,
                    message="Could not determine update status - no lastModified date from Hub"
                )
            
            # Parse hub date
            hub_date = datetime.fromisoformat(hub_last_modified.replace('Z', '+00:00'))
            
            # Get cached date
            metadata = self.cache_service.get_model_metadata(model_name)
            if not metadata:
                return UpdateInfo(
                    has_updates=None,
                    message="Could not determine update status - no cached date information",
                    hub_date=hub_date
                )
            
            cached_date = metadata.first_cached
            
            # Compare dates
            has_updates = hub_date > cached_date
            days_diff = (hub_date - cached_date).days if has_updates else 0
            
            message = (
                f"Model updated on Hub {days_diff} days after local cache" 
                if has_updates else "Local model is up to date"
            )
            
            return UpdateInfo(
                has_updates=has_updates,
                message=message,
                hub_date=hub_date,
                cached_date=cached_date,
                days_difference=days_diff
            )
            
        except UpdateCheckException as e:
            logger.error(f"Update check failed for {model_name}: {e}")
            return UpdateInfo(
                has_updates=None,
                message=f"Update check failed: {str(e)}"
            )
    
    def check_for_updates(self, model_name: str) -> UpdateInfo:
        """
        Synchronous wrapper for update checking.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Update information
        """
        try:
            return asyncio.run(self.check_for_updates_async(model_name))
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return UpdateInfo(
                has_updates=None,
                message=f"Update check failed: {str(e)}"
            )


class EmbeddingModelService:
    """Service for creating and managing embedding models."""
    
    def __init__(self, cache_service: ModelCacheService, config_provider: ConfigProvider):
        self.cache_service = cache_service
        self.config_provider = config_provider
        self._cached_embeddings: Optional[HuggingFaceEmbeddings] = None
        self._cached_model_name: Optional[str] = None
    
    def create_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """
        Create embedding model with configuration.
        
        Args:
            model_name: Name of the model (defaults to config value)
            
        Returns:
            Configured embedding model
        """
        if model_name is None:
            model_name = self.config_provider.embedding_model_name
        
        # Return cached embeddings if available
        if (self._cached_embeddings is not None and 
            self._cached_model_name == model_name):
            logger.info(f"Using cached embeddings for model: {model_name}")
            return self._cached_embeddings
        
        # Setup cache environment
        self.cache_service.setup_cache_environment()
        
        # Get optimal device
        device = get_optimal_device()
        
        # Create embedding configuration
        config = EmbeddingConfig.for_model(model_name, device)
        
        # Create embeddings
        with PerformanceTimer(f"Creating embeddings for {model_name}"):
            embeddings = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs={
                    'device': config.device,
                    'trust_remote_code': config.trust_remote_code,
                },
                encode_kwargs={
                    'normalize_embeddings': config.normalize_embeddings,
                    'batch_size': config.batch_size,
                    'use_fp16': config.use_fp16,
                }
            )
        
        # Register model usage
        self.cache_service.register_model_usage(model_name)
        
        # Cache for future use
        self._cached_embeddings = embeddings
        self._cached_model_name = model_name
        
        logger.info(f"Successfully created embeddings for model: {model_name}")
        return embeddings


class VectorStoreHealthService:
    """Service for vector store health checking and diagnostics."""
    
    @staticmethod
    def check_health(vector_store: Optional[Any]) -> HealthCheckResult:
        """
        Perform comprehensive health check on vector store.
        
        Args:
            vector_store: Vector store to check
            
        Returns:
            Health check result
        """
        result = HealthCheckResult(is_healthy=True)
        
        if not vector_store:
            result.add_issue("No vector store provided")
            return result
        
        # Check 1: Collection accessibility
        try:
            if hasattr(vector_store, '_collection') and vector_store._collection:
                collection = vector_store._collection
                count = collection.count()
                result.collection_accessible = True
                result.document_count = count
                logger.info(f"Collection reports {count} documents")
                
                if count == 0:
                    result.add_issue("Collection is accessible but empty")
            else:
                result.add_issue("Collection is not accessible")
        except Exception as e:
            result.add_issue(f"Collection access failed: {e}")
        
        # Check 2: Query functionality
        try:
            test_results = vector_store.similarity_search("test query APU university", k=1)
            if test_results:
                result.query_successful = True
                logger.info(f"Query test successful: retrieved {len(test_results)} documents")
            else:
                result.add_issue("Query returned no results")
        except Exception as e:
            result.add_issue(f"Query test failed: {e}")
        
        # Check 3: Document retrieval
        try:
            all_docs = vector_store.get()
            doc_count = len(all_docs.get('documents', []))
            result.documents_retrievable = True
            logger.info(f"get() method reports {doc_count} documents")
            
            if doc_count == 0:
                result.add_issue("Document retrieval returned no documents")
        except Exception as e:
            result.add_issue(f"Document retrieval failed: {e}")
        
        # Final health assessment
        result.is_healthy = (
            result.collection_accessible and 
            (result.query_successful or result.documents_retrievable) and
            result.document_count > 0
        )
        
        status = "PASSED" if result.is_healthy else "FAILED"
        logger.info(f"Vector store health check: {status}")
        
        return result
    
    @staticmethod
    def get_statistics(vector_store: Optional[Any]) -> Optional[VectorStoreStats]:
        """
        Get comprehensive statistics about vector store contents.
        
        Args:
            vector_store: Vector store to analyze
            
        Returns:
            Statistics or None if failed
        """
        if not vector_store:
            return None
        
        try:
            # Get all documents
            all_docs = vector_store.get()
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            if not documents:
                logger.warning("Vector store appears to be empty")
                return VectorStoreStats(
                    total_chunks=0,
                    unique_files=0,
                    document_counts={}
                )
            
            # Count documents by filename
            doc_counts: dict[str, int] = {}
            apu_kb_count = 0
            faq_count = 0
            most_recent_doc: Optional[tuple[str, datetime]] = None
            recent_timestamp = 0
            
            for metadata in metadatas:
                if not metadata:
                    continue
                
                # Count by filename
                if 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
                
                # Count APU KB pages
                if metadata.get('content_type') == 'apu_kb_page':
                    apu_kb_count += 1
                    if metadata.get('is_faq', False):
                        faq_count += 1
                
                # Track most recent document
                if 'timestamp' in metadata:
                    timestamp = metadata['timestamp']
                    if timestamp > recent_timestamp:
                        recent_timestamp = timestamp
                        filename = metadata.get('filename', 'Unknown')
                        most_recent_doc = (filename, datetime.fromtimestamp(timestamp))
            
            return VectorStoreStats(
                total_chunks=len(documents),
                unique_files=len(doc_counts),
                document_counts=doc_counts,
                apu_kb_count=apu_kb_count,
                faq_count=faq_count,
                most_recent_doc=most_recent_doc
            )
            
        except Exception as e:
            logger.error(f"Error getting vector store statistics: {e}")
            return None


class VectorStoreManagerModern:
    """
    Modern vector store manager using dependency injection and focused services.
    
    This class orchestrates the various services while maintaining a clean interface
    compatible with existing code.
    """
    
    def __init__(self, config_provider: ConfigProvider, cache_base_path: Optional[Path] = None):
        self.config_provider = config_provider
        
        # Setup cache path
        if cache_base_path is None:
            cache_base_path = Path(__file__).parent.parent / "model_cache" / "huggingface"
        
        # Initialize services
        model_config = ModelConfig()  # Could be injected from config_provider
        self.cache_service = ModelCacheService(cache_base_path, model_config)
        self.update_service = ModelUpdateService(self.cache_service, model_config)
        self.embedding_service = EmbeddingModelService(self.cache_service, config_provider)
        self.health_service = VectorStoreHealthService()
    
    # Compatibility methods for existing code
    def setup_model_cache(self) -> Path:
        """Setup model cache and return path."""
        self.cache_service.setup_cache_environment()
        return self.cache_service.cache_base
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is cached."""
        return self.cache_service.is_model_cached(model_name)
    
    def create_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """Create embedding model."""
        return self.embedding_service.create_embeddings(model_name)
    
    def check_vector_store_health(self, vector_store: Any) -> bool:
        """Check vector store health."""
        result = self.health_service.check_health(vector_store)
        return result.is_healthy
    
    def print_document_statistics(self, vector_store: Any) -> None:
        """Print document statistics."""
        stats = self.health_service.get_statistics(vector_store)
        if stats:
            print(stats.to_summary())
        else:
            print("Error retrieving document statistics.")
    
    def cleanup_corrupted_cache_files(self) -> int:
        """Clean up corrupted cache files."""
        return self.cache_service.cleanup_corrupted_files()