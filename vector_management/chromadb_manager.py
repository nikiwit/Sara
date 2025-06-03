"""
ChromaDB client management with optimized initialization and error handling.
"""

import os
import logging
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
import time
from datetime import datetime

logger = logging.getLogger("CustomRAG")

class ChromaDBManager:
    """Singleton manager for ChromaDB client lifecycle with optimized settings."""
    
    _instance = None
    _client = None
    
    @classmethod
    def get_client(cls, persist_directory=None, reset=False, force_new=False):
        """Get or create a ChromaDB client with proper configuration.
        
        Args:
            persist_directory: Path to the persistence directory
            reset: Force creation of a new client instance
            force_new: Force creation of completely new client (for reindexing)
        """
        if persist_directory is None:
            from config import Config
            persist_directory = Config.PERSIST_PATH
            
        # Force new client for reindexing operations
        if force_new or cls._client is None or reset:
            try:
                # Complete cleanup first
                cls.force_cleanup()
                
                # Wait for cleanup to complete
                time.sleep(0.5)
                
                # Ensure directory exists
                os.makedirs(persist_directory, exist_ok=True)
                
                # Create new client with optimized settings for faster startup
                cls._client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,  # Disable telemetry for faster startup
                        allow_reset=True,
                        is_persistent=True,
                        persist_directory=persist_directory  # Explicit persist directory
                    )
                )
                logger.info(f"Initialized fresh ChromaDB client for {persist_directory}")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {e}")
                raise
        
        return cls._client
    
    @classmethod
    def get_or_create_collection(cls, client, name, metadata=None, embedding_function=None, force_new=False):
        """Get or create a collection with metadata and optimized error handling.
        
        Args:
            client: ChromaDB client
            name: Collection name
            metadata: Collection metadata
            embedding_function: Function for embeddings
            force_new: Force creation of new collection (delete existing if present)
        """
        from config import Config
        
        if metadata is None:
            metadata = {}
            
        full_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
            "embedding_version": "1.0",
            "app_version": "1.0",
            "created_at": datetime.now().isoformat()
        }
        
        # Merge with provided metadata
        full_metadata.update(metadata)
        
        try:
            # Check if collection exists
            all_collections = client.list_collections()
            collection_exists = any(c.name == name for c in all_collections)
            
            # For reindexing, always create fresh collection
            if force_new and collection_exists:
                logger.info(f"Force deleting existing collection: {name}")
                try:
                    client.delete_collection(name)
                    collection_exists = False
                except Exception as e:
                    logger.warning(f"Could not delete existing collection: {e}")
            
            if collection_exists and not force_new:
                logger.info(f"Using existing collection: {name}")
                collection = client.get_collection(name)
                
                # Update metadata for existing collection (handle distance function warning)
                if hasattr(collection, 'modify') and full_metadata:
                    try:
                        # Only update non-conflicting metadata to avoid distance function warning
                        safe_metadata = {k: v for k, v in full_metadata.items() 
                                       if k not in ['hnsw:space']}  # Avoid distance function conflicts
                        if safe_metadata:
                            collection.modify(metadata=safe_metadata)
                            logger.debug(f"Updated collection metadata")
                    except Exception as e:
                        # Log as warning instead of error since this is expected for distance function
                        logger.warning(f"Could not update collection metadata: {e}")
            else:
                logger.info(f"Creating new collection: {name}")
                collection = client.create_collection(name=name, metadata=full_metadata)
            
            # Create LangChain wrapper
            langchain_store = Chroma(
                client=client,
                collection_name=name,
                embedding_function=embedding_function
            )
            
            return collection, langchain_store
            
        except Exception as e:
            logger.error(f"Error getting/creating collection {name}: {e}")
            raise
    
    @classmethod
    def force_cleanup(cls):
        """Force cleanup of all ChromaDB client resources."""
        if cls._client is not None:
            try:
                # Try multiple cleanup methods
                if hasattr(cls._client, 'close'):
                    cls._client.close()
                if hasattr(cls._client, 'clear_system_cache'):
                    cls._client.clear_system_cache()
                if hasattr(cls._client, '_producer') and hasattr(cls._client._producer, 'stop'):
                    cls._client._producer.stop()
                    
            except Exception as e:
                logger.warning(f"Error during force cleanup: {e}")
            finally:
                cls._client = None
                logger.info("Forced ChromaDB client cleanup completed")
                
                # Give time for cleanup
                import gc
                gc.collect()
                time.sleep(0.5)
    
    @classmethod
    def close(cls):
        """Close the ChromaDB client connection."""
        if cls._client is not None:
            try:
                # Different ChromaDB versions may have different cleanup methods
                if hasattr(cls._client, 'close'):
                    cls._client.close()
                elif hasattr(cls._client, 'persist'):
                    # Some versions use persist to ensure data is saved
                    cls._client.persist()
                # Simply clear the reference if no cleanup method is available
                cls._client = None
                logger.info("Released ChromaDB client resources")
            except Exception as e:
                logger.error(f"Error during ChromaDB client cleanup: {e}")
    
    @classmethod
    def reset_client(cls):
        """Reset the ChromaDB client (useful for testing or recovery)."""
        logger.info("Resetting ChromaDB client")
        cls.force_cleanup()
        
        # Clear any cached references
        cls._client = None