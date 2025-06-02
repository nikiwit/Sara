"""
ChromaDB client management and configuration for vector database operations.
Enhanced compatibility with newer ChromaDB versions and configuration integration.
"""

import logging
import time
import os
from datetime import datetime
import chromadb
from chromadb.config import Settings

# Configuration import for compatibility with updated system
from config import config as Config

logger = logging.getLogger("CustomRAG")

class ChromaDBManager:
    """Singleton manager for ChromaDB client lifecycle with enhanced compatibility."""
    _instance = None
    _client = None
    _client_version = None
    
    @classmethod
    def get_chromadb_version(cls):
        """Retrieve ChromaDB version for compatibility checks and logging."""
        try:
            import chromadb
            version = getattr(chromadb, '__version__', 'unknown')
            return version
        except Exception:
            return 'unknown'
    
    @classmethod
    def force_cleanup(cls):
        """Force comprehensive cleanup of all ChromaDB client resources with version compatibility."""
        if cls._client is not None:
            try:
                logger.info("Starting ChromaDB client cleanup...")
                
                # Try multiple cleanup methods based on ChromaDB version capabilities
                cleanup_methods = [
                    ('close', 'Closing client connection'),
                    ('reset', 'Resetting client state'),
                    ('clear_system_cache', 'Clearing system cache'),
                ]
                
                for method_name, description in cleanup_methods:
                    try:
                        if hasattr(cls._client, method_name):
                            logger.debug(f"{description}...")
                            method = getattr(cls._client, method_name)
                            method()
                            logger.debug(f"{description} completed")
                    except Exception as e:
                        logger.debug(f"{description} failed: {e}")
                
                # Clean up internal producer resources if available
                try:
                    if hasattr(cls._client, '_producer'):
                        if hasattr(cls._client._producer, 'stop'):
                            cls._client._producer.stop()
                        if hasattr(cls._client._producer, 'close'):
                            cls._client._producer.close()
                except Exception as e:
                    logger.debug(f"Producer cleanup failed: {e}")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                time.sleep(0.2)  # Brief pause for cleanup completion
                
            except Exception as e:
                logger.warning(f"Error during force cleanup: {e}")
            finally:
                cls._client = None
                cls._client_version = None
                logger.info("ChromaDB client cleanup completed")
    
    @classmethod
    def get_client(cls, persist_directory=None, reset=False, force_new=False):
        """Get or create a ChromaDB client with enhanced compatibility and configuration integration.
        
        Args:
            persist_directory: Path to the persistence directory
            reset: Force creation of a new client instance
            force_new: Force creation of completely new client for reindexing operations
        """
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # Ensure the persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create new client for reindexing operations or if requested
        if force_new or cls._client is None or reset:
            try:
                # Perform complete cleanup of existing client first
                if cls._client is not None:
                    cls.force_cleanup()
                
                # Wait for cleanup operations to complete
                time.sleep(0.5)
                
                # Get ChromaDB version for compatibility handling
                version = cls.get_chromadb_version()
                logger.info(f"Initializing ChromaDB client (version: {version})")
                
                # Create client with version-appropriate settings and config integration
                try:
                    # Attempt modern ChromaDB client initialization
                    settings = Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                    
                    # Apply performance settings from configuration if available
                    if hasattr(Config, 'MAX_THREADS') and Config.MAX_THREADS > 1:
                        logger.debug(f"Using {Config.MAX_THREADS} threads for ChromaDB operations")
                    
                    cls._client = chromadb.PersistentClient(
                        path=persist_directory,
                        settings=settings
                    )
                    cls._client_version = version
                    logger.info(f"Initialized modern ChromaDB client at {persist_directory}")
                    
                except Exception as modern_error:
                    logger.warning(f"Modern client initialization failed: {modern_error}")
                    
                    # Fallback to legacy initialization for older versions
                    try:
                        legacy_settings = Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=persist_directory,
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                        
                        cls._client = chromadb.Client(legacy_settings)
                        cls._client_version = version
                        logger.info(f"Initialized legacy ChromaDB client at {persist_directory}")
                        
                    except Exception as legacy_error:
                        logger.error(f"Both modern and legacy client initialization failed")
                        logger.error(f"Modern error: {modern_error}")
                        logger.error(f"Legacy error: {legacy_error}")
                        raise legacy_error
                        
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {e}")
                raise
        
        return cls._client
    
    @classmethod
    def get_or_create_collection(cls, client, name, metadata=None, embedding_function=None, force_new=False):
        """Get or create a collection with enhanced compatibility and configuration integration.
        
        Args:
            client: ChromaDB client instance
            name: Collection name
            metadata: Collection metadata dictionary
            embedding_function: Function for embeddings
            force_new: Force creation of new collection by deleting existing if present
        """
        if metadata is None:
            metadata = {}
            
        # Prepare comprehensive metadata that includes configuration information
        try:
            full_metadata = {
                "embedding_model": getattr(Config, 'EMBEDDING_MODEL_NAME', 'unknown'),
                "created_at": datetime.now().isoformat(),
                "app_version": "1.0",
                "environment": getattr(Config, 'ENV', 'unknown')
            }
            
            # Add configuration-specific metadata for tracking and debugging
            if hasattr(Config, 'FILTER_APU_ONLY'):
                full_metadata["apu_filtering_enabled"] = Config.FILTER_APU_ONLY
            
            if hasattr(Config, 'CHUNK_SIZE'):
                full_metadata["chunk_size"] = Config.CHUNK_SIZE
                
            if hasattr(Config, 'CHUNK_OVERLAP'):
                full_metadata["chunk_overlap"] = Config.CHUNK_OVERLAP
            
            # Add HNSW space configuration if supported by ChromaDB version
            try:
                full_metadata["hnsw:space"] = "cosine"
            except Exception:
                logger.debug("hnsw:space metadata not supported in this ChromaDB version")
            
            # Merge with provided metadata
            full_metadata.update(metadata)
            
        except Exception as e:
            logger.warning(f"Error preparing metadata: {e}")
            full_metadata = metadata or {}
        
        try:
            # Enhanced collection detection and loading
            collection = None
            collection_exists = False
            
            # Method 1: Direct collection access (most reliable)
            try:
                collection = client.get_collection(name=name)
                collection_exists = True
                logger.info(f"Found existing collection '{name}' via direct access")
            except Exception as e:
                logger.debug(f"Direct collection access failed: {e}")
                
                # Method 2: List all collections and search by name
                try:
                    all_collections = client.list_collections()
                    for c in all_collections:
                        if c.name == name:
                            collection_exists = True
                            collection = c
                            logger.info(f"Found existing collection '{name}' in collection list")
                            break
                except Exception as list_error:
                    logger.debug(f"Collection listing failed: {list_error}")
            
            # Handle collection deletion for force_new scenarios
            if force_new and collection_exists:
                logger.info(f"Force deleting existing collection: {name}")
                try:
                    client.delete_collection(name)
                    collection_exists = False
                    collection = None
                    logger.info(f"Successfully deleted collection: {name}")
                    time.sleep(0.5)  # Allow ChromaDB time to process deletion
                except Exception as e:
                    logger.warning(f"Could not delete existing collection: {e}")
                    # Continue anyway as recreation might still work
            
            # Create or use collection based on existence and requirements
            if collection_exists and not force_new and collection is not None:
                logger.info(f"Using existing collection: {name}")
                
                # Verify the existing collection is functional
                try:
                    count = collection.count()
                    logger.info(f"Existing collection has {count} documents")
                except Exception as e:
                    logger.warning(f"Existing collection verification failed: {e}")
                    collection_exists = False
                    collection = None
            
            # Create new collection if needed or requested
            if not collection_exists or force_new or collection is None:
                logger.info(f"Creating new collection: {name}")
                
                # Ensure clean state before creating new collection
                try:
                    # Attempt to delete if it exists (in case detection failed)
                    try:
                        client.delete_collection(name)
                        time.sleep(0.5)
                        logger.debug("Cleaned up any existing collection before creation")
                    except:
                        pass  # Ignore if collection doesn't exist
                    
                    # Create collection with retry logic for reliability
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            collection = client.create_collection(
                                name=name,
                                metadata=full_metadata,
                                embedding_function=None  # Use LangChain's embedding instead
                            )
                            logger.info(f"Successfully created collection: {name}")
                            break
                        except Exception as create_error:
                            if attempt < max_retries - 1:
                                logger.warning(f"Creation attempt {attempt + 1} failed: {create_error}")
                                time.sleep(1)
                            else:
                                raise create_error
                                
                except Exception as e:
                    logger.error(f"Error creating collection: {e}")
                    # Retry without metadata if full creation fails
                    try:
                        logger.info(f"Retrying collection creation without metadata")
                        collection = client.create_collection(name=name)
                        logger.info(f"Created collection without metadata: {name}")
                    except Exception as e2:
                        logger.error(f"Failed to create collection even without metadata: {e2}")
                        raise e2
            
            # Create LangChain wrapper with enhanced error handling
            try:
                from langchain_chroma import Chroma
                
                # Ensure we use the same client instance for consistency
                langchain_store = Chroma(
                    client=client,
                    collection_name=name,
                    embedding_function=embedding_function,
                    collection_metadata=full_metadata
                )
                
                # Verify the LangChain wrapper functionality
                try:
                    # Test the wrapper by attempting to access data
                    test_result = langchain_store.get()
                    logger.debug(f"LangChain wrapper verified, can access {len(test_result.get('ids', []))} documents")
                except Exception as e:
                    logger.warning(f"LangChain wrapper verification warning: {e}")
                
                logger.debug("Created LangChain wrapper")
                
            except Exception as e:
                logger.error(f"Error creating LangChain wrapper: {e}")
                raise
            
            return collection, langchain_store
            
        except Exception as e:
            logger.error(f"Error in get_or_create_collection for {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def close(cls):
        """Close the ChromaDB client connection with comprehensive cleanup procedures."""
        if cls._client is not None:
            try:
                logger.info("Closing ChromaDB client...")
                
                # Try multiple cleanup approaches for maximum compatibility
                cleanup_successful = False
                
                # Method 1: Standard close method
                if hasattr(cls._client, 'close'):
                    try:
                        cls._client.close()
                        cleanup_successful = True
                        logger.debug("Used standard close method")
                    except Exception as e:
                        logger.debug(f"Standard close failed: {e}")
                
                # Method 2: Persist method (for older ChromaDB versions)
                if not cleanup_successful and hasattr(cls._client, 'persist'):
                    try:
                        cls._client.persist()
                        cleanup_successful = True
                        logger.debug("Used persist method")
                    except Exception as e:
                        logger.debug(f"Persist method failed: {e}")
                
                # Method 3: Reset method (if available and needed)
                if hasattr(cls._client, 'reset'):
                    try:
                        # Only reset if we couldn't close properly
                        if not cleanup_successful:
                            cls._client.reset()
                            logger.debug("Used reset method")
                    except Exception as e:
                        logger.debug(f"Reset method failed: {e}")
                
                # Final cleanup and resource release
                cls._client = None
                cls._client_version = None
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                logger.info("ChromaDB client closed successfully")
                
            except Exception as e:
                logger.error(f"Error during ChromaDB client cleanup: {e}")
                # Force clear the reference even if cleanup fails
                cls._client = None
                cls._client_version = None
    
    @classmethod
    def health_check(cls):
        """Perform comprehensive health check on ChromaDB client with configuration awareness."""
        if cls._client is None:
            return False, "No client initialized"
        
        try:
            # Test basic ChromaDB operations
            collections = cls._client.list_collections()
            collection_count = len(collections)
            
            # Test collection creation and deletion functionality
            temp_name = f"health_check_{int(time.time())}"
            try:
                temp_collection = cls._client.create_collection(temp_name)
                cls._client.delete_collection(temp_name)
                create_delete_ok = True
            except Exception as e:
                create_delete_ok = False
                logger.debug(f"Create/delete test failed: {e}")
            
            # Compile comprehensive health status
            health_status = {
                "client_active": True,
                "collection_count": collection_count,
                "create_delete_ok": create_delete_ok,
                "version": cls._client_version or "unknown",
                "config_environment": getattr(Config, 'ENV', 'unknown'),
                "persistence_enabled": getattr(Config, 'ENABLE_VECTOR_STORE_PERSISTENCE', True),
                "apu_filtering": getattr(Config, 'FILTER_APU_ONLY', False)
            }
            
            return True, health_status
            
        except Exception as e:
            return False, f"Health check failed: {e}"
    
    @classmethod
    def get_performance_info(cls):
        """Retrieve performance-related information from configuration and client state."""
        try:
            perf_info = {
                "max_threads": getattr(Config, 'MAX_THREADS', 'unknown'),
                "max_memory": getattr(Config, 'MAX_MEMORY', 'unknown'),
                "embedding_batch_size": getattr(Config, 'EMBEDDING_BATCH_SIZE', 'unknown'),
                "chunk_size": getattr(Config, 'CHUNK_SIZE', 'unknown'),
                "retrieval_k": getattr(Config, 'RETRIEVER_K', 'unknown'),
                "client_version": cls._client_version or "unknown",
                "client_active": cls._client is not None
            }
            
            return perf_info
            
        except Exception as e:
            logger.debug(f"Error getting performance info: {e}")
            return {"error": str(e)}
    
    @classmethod
    def validate_config_compatibility(cls):
        """Validate that the current configuration is compatible with ChromaDB operations."""
        issues = []
        warnings = []
        
        try:
            # Check for required configuration attributes
            required_attrs = ['PERSIST_PATH', 'EMBEDDING_MODEL_NAME']
            for attr in required_attrs:
                if not hasattr(Config, attr):
                    issues.append(f"Missing required config attribute: {attr}")
            
            # Validate path accessibility and permissions
            if hasattr(Config, 'PERSIST_PATH'):
                try:
                    os.makedirs(Config.PERSIST_PATH, exist_ok=True)
                    if not os.access(Config.PERSIST_PATH, os.W_OK):
                        issues.append(f"Vector store path not writable: {Config.PERSIST_PATH}")
                except Exception as e:
                    issues.append(f"Cannot access vector store path: {e}")
            
            # Validate performance settings
            if hasattr(Config, 'MAX_THREADS'):
                if Config.MAX_THREADS < 1:
                    issues.append("MAX_THREADS must be at least 1")
                elif Config.MAX_THREADS > 32:
                    warnings.append("MAX_THREADS > 32 may cause resource contention")
            
            # Validate embedding batch size
            if hasattr(Config, 'EMBEDDING_BATCH_SIZE'):
                if Config.EMBEDDING_BATCH_SIZE < 1:
                    issues.append("EMBEDDING_BATCH_SIZE must be at least 1")
                elif Config.EMBEDDING_BATCH_SIZE > 100:
                    warnings.append("Large EMBEDDING_BATCH_SIZE may cause memory issues")
            
            # Log validation results
            if issues:
                logger.error("ChromaDB config compatibility issues:")
                for issue in issues:
                    logger.error(f"  - {issue}")
            
            if warnings:
                logger.warning("ChromaDB config compatibility warnings:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
            
            if not issues and not warnings:
                logger.info("ChromaDB config compatibility validation passed")
            
            return len(issues) == 0, issues, warnings
            
        except Exception as e:
            logger.error(f"Error during config compatibility validation: {e}")
            return False, [f"Validation error: {e}"], []