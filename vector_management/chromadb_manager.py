"""
ChromaDB client management and configuration.
"""

import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings
from config import Config

logger = logging.getLogger("CustomRAG")

class ChromaDBManager:
    """Singleton manager for ChromaDB client lifecycle."""
    _instance = None
    _client = None
    
    @classmethod
    def get_client(cls, persist_directory=None, reset=False):
        """Get or create a ChromaDB client with proper configuration.
        
        Args:
            persist_directory: Path to the persistence directory
            reset: Force creation of a new client instance
            
        Returns:
            Configured ChromaDB client
        """
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # Create a new client if requested or if none exists
        if cls._client is None or reset:
            try:
                # Close existing client if applicable
                if cls._client is not None and hasattr(cls._client, 'close'):
                    try:
                        cls._client.close()
                    except Exception as e:
                        logger.warning(f"Error closing existing ChromaDB client: {e}")
                
                # Create new client with proper settings
                from chromadb.config import Settings
                cls._client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                logger.info(f"Initialized ChromaDB client for {persist_directory}")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {e}")
                raise
        
        return cls._client
    
    @classmethod
    def get_or_create_collection(cls, client, name, metadata=None, embedding_function=None):
        """Get or create a collection with metadata.
        
        Args:
            client: ChromaDB client
            name: Collection name
            metadata: Collection metadata
            embedding_function: Function for embeddings
            
        Returns:
            Tuple of (collection, langchain_chroma_instance)
        """
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
            
            if collection_exists:
                logger.info(f"Using existing collection: {name}")
                collection = client.get_collection(name)
                
                # Update metadata for existing collection
                if hasattr(collection, 'modify') and full_metadata:
                    try:
                        collection.modify(metadata=full_metadata)
                    except Exception as e:
                        logger.warning(f"Could not update collection metadata: {e}")
            else:
                logger.info(f"Creating new collection: {name}")
                collection = client.create_collection(name=name, metadata=full_metadata)
            
            # Create LangChain wrapper
            from langchain_chroma import Chroma
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