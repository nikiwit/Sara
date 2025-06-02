"""
Vector store management operations and utilities for ChromaDB integration.
Enhanced ChromaDB compatibility and error handling with configuration support.
"""

import os
import sys
import shutil
import time
import logging
import torch
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration import for compatibility with updated system
from config import config as Config
from .chromadb_manager import ChromaDBManager

logger = logging.getLogger("CustomRAG")

class VectorStoreManager:
    """Manages vector database operations with comprehensive error handling and optimization."""
    
    _cached_embeddings = None
    _cached_embeddings_model = None
    
    @staticmethod
    def check_vector_store_health(vector_store) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive health check on vector store with detailed diagnostics."""
        if not vector_store:
            logger.warning("No vector store provided for health check")
            return False, {"error": "No vector store provided"}
            
        health_info = {
            "collection_info": False,
            "query_success": False,
            "get_success": False,
            "document_count": 0,
            "error_details": []
        }
        
        try:
            # Check collection access and document count
            try:
                if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                    collection = vector_store._collection
                    count = collection.count()
                    health_info["document_count"] = count
                    health_info["collection_info"] = True
                    logger.info(f"Collection reports {count} documents")
                else:
                    health_info["error_details"].append("No direct collection access")
                    logger.debug("No direct collection access available")
            except Exception as e:
                health_info["error_details"].append(f"Collection count failed: {e}")
                logger.debug(f"Collection count failed: {e}")
            
            # Test similarity search functionality
            try:
                test_results = vector_store.similarity_search("test query APU university", k=1)
                if test_results and len(test_results) > 0:
                    health_info["query_success"] = True
                    logger.info(f"Query test successful: retrieved {len(test_results)} documents")
                else:
                    health_info["error_details"].append("Query returned no results")
                    logger.debug("Query test returned no results")
            except Exception as e:
                health_info["error_details"].append(f"Query test failed: {e}")
                logger.debug(f"Query test failed: {e}")
            
            # Test document retrieval via get method
            try:
                all_docs = vector_store.get()
                if all_docs and all_docs.get('documents'):
                    doc_count = len(all_docs.get('documents', []))
                    health_info["get_success"] = True
                    health_info["document_count"] = max(health_info["document_count"], doc_count)
                    logger.info(f"get() method reports {doc_count} documents")
                else:
                    health_info["error_details"].append("get() method returned no documents")
                    logger.debug("get() method returned no documents")
            except Exception as e:
                health_info["error_details"].append(f"get() method failed: {e}")
                logger.debug(f"get() method failed: {e}")
            
            # Determine overall health status
            health_status = (
                health_info["collection_info"] or 
                (health_info["query_success"] and health_info["get_success"])
            ) and health_info["document_count"] > 0
            
            health_info["overall_health"] = health_status
            
            if health_status:
                logger.info("Vector store health check: PASSED")
            else:
                logger.warning("Vector store health check: FAILED")
                logger.warning(f"Error details: {health_info['error_details']}")
                
            return health_status, health_info
            
        except Exception as e:
            logger.error(f"Error during vector store health check: {e}")
            health_info["error_details"].append(f"Health check exception: {e}")
            return False, health_info
    
    @staticmethod
    def get_embedding_device():
        """Determine the optimal device for embedding computation based on available hardware."""
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA GPU for embeddings")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logger.info("Using Apple Silicon MPS for embeddings")
        else:
            device = 'cpu'
            logger.info("Using CPU for embeddings")
        
        return device
        
    @staticmethod
    def save_embeddings_backup(vector_store, filepath=None) -> bool:
        """Save comprehensive backup including embeddings for recovery purposes."""
        if not vector_store:
            logger.error("No vector store provided for backup")
            return False
            
        if filepath is None:
            backup_dir = os.path.dirname(Config.PERSIST_PATH)
            filepath = os.path.join(backup_dir, "embeddings_backup.pkl")
            
        try:
            logger.info("Starting embeddings backup with embeddings included...")
            
            # Retrieve data with embeddings explicitly included
            data = vector_store.get(include=['embeddings', 'metadatas', 'documents'])
            
            if not data or not data.get('ids') or len(data.get('ids', [])) == 0:
                logger.warning("No data to backup - vector store appears empty")
                return False
            
            # Validate that embeddings are actually included in backup
            embeddings_data = data.get('embeddings', [])
            if not embeddings_data or len(embeddings_data) == 0:
                logger.error("CRITICAL: No embeddings returned from vector store!")
                logger.error("This will create a broken backup!")
                
                # Attempt alternative method to retrieve embeddings
                try:
                    logger.info("Trying alternative method to get embeddings...")
                    if hasattr(vector_store, '_collection'):
                        collection = vector_store._collection
                        alt_data = collection.get(include=['embeddings', 'metadatas', 'documents'])
                        if alt_data.get('embeddings'):
                            data = alt_data
                            embeddings_data = data.get('embeddings', [])
                            logger.info("Retrieved embeddings using alternative method")
                        else:
                            logger.error("Alternative method also failed")
                            return False
                except Exception as e:
                    logger.error(f"Alternative method failed: {e}")
                    return False
            
            # Verify embedding dimensions match current model
            if embeddings_data and len(embeddings_data) > 0:
                actual_dim = len(embeddings_data[0])
                logger.info(f"Backup includes embeddings with dimension: {actual_dim}")
                
                # Validate dimensions against current model
                try:
                    current_embeddings = VectorStoreManager.create_embeddings()
                    test_embed = current_embeddings.embed_query("test")
                    expected_dim = len(test_embed)
                    
                    if actual_dim != expected_dim:
                        logger.error(f"DIMENSION MISMATCH: Backup={actual_dim}, Current={expected_dim}")
                        return False
                        
                    logger.info("Embedding dimensions match current model")
                        
                except Exception as e:
                    logger.warning(f"Could not validate dimensions: {e}")
            else:
                logger.error("No embeddings found after retrieval attempts")
                return False
            
            # Prepare comprehensive backup data
            backup_data = {
                'ids': data.get('ids', []),
                'documents': data.get('documents', []),
                'metadatas': data.get('metadatas', []),
                'embeddings': data.get('embeddings', []),  # Now actually included
                'backup_time': datetime.now().isoformat(),
                'doc_count': len(data.get('ids', [])),
                'embedding_model': Config.EMBEDDING_MODEL_NAME,
                'embedding_dimension': len(embeddings_data[0]) if embeddings_data else None,
                'version': '2.1',  # Version number for backup format
                'has_embeddings': len(embeddings_data) > 0  # Flag for verification
            }
            
            # Save backup with atomic write operation
            temp_filepath = filepath + '.tmp'
            try:
                with open(temp_filepath, 'wb') as f:
                    pickle.dump(backup_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Verify the saved backup integrity
                with open(temp_filepath, 'rb') as f:
                    verify_data = pickle.load(f)
                    
                if not verify_data.get('has_embeddings'):
                    raise ValueError("Backup verification failed - no embeddings saved")
                
                if verify_data.get('embedding_dimension') != backup_data['embedding_dimension']:
                    raise ValueError("Backup verification failed - dimension mismatch")
                
                # Atomic move to final location
                if os.path.exists(filepath):
                    os.replace(temp_filepath, filepath)
                else:
                    os.rename(temp_filepath, filepath)
                
                logger.info(f"Saved COMPLETE backup with {backup_data['doc_count']} documents")
                logger.info(f"   Model: {backup_data['embedding_model']}")
                logger.info(f"   Dimensions: {backup_data['embedding_dimension']}")
                logger.info(f"   Embeddings included: {backup_data['has_embeddings']}")
                return True
                
            except Exception as e:
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                raise e
                    
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def load_embeddings_backup(embeddings, filepath=None, collection_name="apu_kb_collection"):
        """Load embeddings from backup with enhanced compatibility and error handling."""
        if filepath is None:
            backup_dir = os.path.dirname(Config.PERSIST_PATH)
            filepath = os.path.join(backup_dir, "embeddings_backup.pkl")
            
        if not os.path.exists(filepath):
            logger.warning(f"No embeddings backup found at {filepath}")
            return None
            
        try:
            logger.info(f"Loading embeddings backup from {filepath}")
            
            # Load and validate backup data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            if not data or 'ids' not in data or len(data['ids']) == 0:
                logger.warning("Backup file exists but contains no valid data")
                return None
            
            backup_version = data.get('version', '1.0')
            doc_count = len(data['ids'])
            logger.info(f"Loading backup (version {backup_version}) with {doc_count} documents")
            
            # Create ChromaDB client and collection for backup restoration
            try:
                import chromadb
                from langchain_chroma import Chroma
                from chromadb.config import Settings
                
                # Create memory client for backup restoration
                client = chromadb.Client(Settings(anonymized_telemetry=False))
                
                # Create new collection with backup metadata
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"restored_from_backup": True, "backup_version": backup_version}
                )
                
                # Add data in batches for stability
                batch_size = 50  # Conservative batch size for reliability
                total_items = len(data['ids'])
                
                for i in range(0, total_items, batch_size):
                    end_idx = min(i + batch_size, total_items)
                    
                    # Prepare batch data with safe extraction
                    batch_ids = data['ids'][i:end_idx]
                    batch_documents = data['documents'][i:end_idx] if data.get('documents') else None
                    batch_metadatas = data['metadatas'][i:end_idx] if data.get('metadatas') else None
                    batch_embeddings = data['embeddings'][i:end_idx] if data.get('embeddings') else None
                   
                    # Add batch to collection with error handling
                    try:
                        add_kwargs = {
                            'ids': batch_ids,
                            'documents': batch_documents,
                            'metadatas': batch_metadatas
                        }
                        
                        # Include embeddings only if they exist
                        if batch_embeddings:
                            add_kwargs['embeddings'] = batch_embeddings
                        
                        collection.add(**add_kwargs)
                        
                        batch_num = i // batch_size + 1
                        total_batches = (total_items + batch_size - 1) // batch_size
                        logger.debug(f"Restored batch {batch_num}/{total_batches}: {len(batch_ids)} documents")
                        
                    except Exception as e:
                        logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                        # Continue with next batch rather than failing completely
                        continue
                
                # Create LangChain wrapper for restored vector store
                vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                
                # Verify successful restoration
                final_count = collection.count()
                logger.info(f"Successfully restored vector store from backup with {final_count} documents")
                
                return vector_store
                
            except Exception as e:
                logger.error(f"Error creating vector store from backup: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load embeddings backup: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
        
    @staticmethod
    def fix_chromadb_collection(vector_store):
        """Attempt to fix ChromaDB collections with modern API compatibility."""
        if not vector_store:
            return False
            
        try:
            logger.info("Attempting to fix ChromaDB collection...")
            
            # Check if collection exists but reports 0 documents
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                count = collection.count()
                
                if count == 0:
                    logger.warning("Collection exists but reports 0 documents - attempting fix")
                    
                    # Method 1: Test query to wake up the collection
                    try:
                        test_results = vector_store.similarity_search("test", k=1)
                        
                        # Check collection count after query test
                        count_after = collection.count()
                        logger.info(f"Collection count after query test: {count_after}")
                        
                        if count_after > 0:
                            logger.info("Collection fix successful via query test")
                            return True
                        
                    except Exception as e:
                        logger.debug(f"Query test fix failed: {e}")
                    
                    # Method 2: Direct document retrieval test
                    try:
                        all_docs = vector_store.get()
                        if all_docs and all_docs.get('documents'):
                            actual_doc_count = len(all_docs.get('documents', []))
                            logger.info(f"get() method found {actual_doc_count} documents despite count() showing 0")
                            
                            if actual_doc_count > 0:
                                logger.info("Collection has data - count() method may be unreliable")
                                return True
                                
                    except Exception as e:
                        logger.debug(f"get() method fix failed: {e}")
                    
                    # Method 3: Collection metadata and sample data check
                    try:
                        if hasattr(collection, 'get'):
                            sample_data = collection.get(limit=1)
                            if sample_data and sample_data.get('ids'):
                                logger.info("Collection has data accessible via direct get()")
                                return True
                                
                    except Exception as e:
                        logger.debug(f"Direct collection get() failed: {e}")
                
                else:
                    logger.info(f"Collection reports {count} documents - no fix needed")
                    return True
            
            logger.warning("Could not fix collection - no accessible data found")
            return False
            
        except Exception as e:
            logger.error(f"Error in fix_chromadb_collection: {e}")
            return False
        
    @staticmethod
    def sanitize_metadata(documents: List[Document]) -> List[Document]:
        """Sanitize document metadata for ChromaDB compatibility with comprehensive type handling."""
        sanitized_docs = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                # Create a copy of the metadata to avoid modifying original
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Process each metadata field for ChromaDB compatibility
                sanitized_metadata = {}
                for key, value in metadata.items():
                    try:
                        # Skip None values to avoid ChromaDB issues
                        if value is None:
                            continue
                        
                        # Handle different data types appropriately
                        if isinstance(value, (str, int, float, bool)):
                            # Primitive types are directly compatible
                            sanitized_metadata[key] = value
                        elif isinstance(value, list):
                            if value:  # Process non-empty lists
                                # Convert list to JSON string for storage
                                sanitized_metadata[key] = json.dumps(value)
                        elif isinstance(value, dict):
                            # Convert dictionary to JSON string
                            sanitized_metadata[key] = json.dumps(value)
                        else:
                            # Convert other types to string representation
                            sanitized_metadata[key] = str(value)
                            
                    except Exception as e:
                        logger.debug(f"Error processing metadata key '{key}' for document {doc_idx}: {e}")
                        # Skip problematic metadata fields rather than failing
                        continue
                
                # Create new document with sanitized metadata
                sanitized_doc = Document(
                    page_content=doc.page_content,
                    metadata=sanitized_metadata
                )
                sanitized_docs.append(sanitized_doc)
                
            except Exception as e:
                logger.error(f"Error sanitizing document {doc_idx}: {e}")
                # Create document with minimal metadata as fallback
                sanitized_doc = Document(
                    page_content=doc.page_content,
                    metadata={"sanitization_error": True}
                )
                sanitized_docs.append(sanitized_doc)
        
        logger.info(f"Sanitized metadata for {len(sanitized_docs)} documents")
        return sanitized_docs
    
    @staticmethod
    def create_embeddings(model_name=None):
        """Create embedding model with enhanced caching and device optimization."""
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL_NAME
        
        # Return cached embeddings if available for this model
        if (VectorStoreManager._cached_embeddings is not None and 
            VectorStoreManager._cached_embeddings_model == model_name):
            logger.debug(f"Using cached embeddings for model: {model_name}")
            return VectorStoreManager._cached_embeddings
            
        device = VectorStoreManager.get_embedding_device()
        
        try:
            logger.info(f"Creating new embeddings for model: {model_name}")
            
            # Configure embedding model with device-specific optimizations
            model_kwargs = {'device': device}
            encode_kwargs = {'normalize_embeddings': True}
            
            # Apply device-specific optimizations
            if device == 'cuda':
                model_kwargs['torch_dtype'] = torch.float16  # Use half precision for GPU efficiency
            elif device == 'mps':
                # Apple Silicon MPS specific settings
                batch_size = getattr(Config, 'EMBEDDING_BATCH_SIZE', 32)
                encode_kwargs['batch_size'] = min(batch_size, 32)  # Conservative batch size for MPS
            
            # Initialize HuggingFace embeddings with optimized settings
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Test embeddings functionality and validate dimensions
            try:
                test_embedding = embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    logger.info(f"Embeddings test successful - dimension: {len(test_embedding)}")
                else:
                    raise ValueError("Test embedding returned empty result")
            except Exception as e:
                logger.error(f"Embeddings test failed: {e}")
                raise
            
            # Cache the validated embeddings for reuse
            VectorStoreManager._cached_embeddings = embeddings
            VectorStoreManager._cached_embeddings_model = model_name
            logger.info(f"Cached embeddings for model: {model_name}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{model_name}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """Reset the ChromaDB environment with comprehensive cleanup and error handling."""
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Perform comprehensive cleanup process
        try:
            # Step 1: Force cleanup of any existing ChromaDB clients
            ChromaDBManager.force_cleanup()
            
            # Step 2: Release system resources via garbage collection
            import gc
            gc.collect()
            time.sleep(1.0)  # Allow time for resource cleanup
            
            # Step 3: Remove existing directory with retry logic
            if os.path.exists(persist_directory):
                success = VectorStoreManager._remove_directory_with_retry(persist_directory)
                if not success:
                    logger.error("Failed to remove existing vector store directory")
                    return False
            
            # Step 4: Create fresh directory structure with proper permissions
            try:
                os.makedirs(persist_directory, exist_ok=True)
                
                # Set appropriate permissions based on platform
                if sys.platform != 'win32':
                    os.chmod(persist_directory, 0o755)
                
                logger.info(f"Successfully reset vector store directory: {persist_directory}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create vector store directory: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error during vector store reset: {e}")
            return False
    
    @staticmethod
    def _remove_directory_with_retry(directory, max_attempts=3):
        """Remove directory with retry logic and permission fixing for stubborn files."""
        for attempt in range(max_attempts):
            try:
                # Fix permissions before attempting removal
                VectorStoreManager._fix_directory_permissions(directory)
                
                # Attempt directory removal
                shutil.rmtree(directory)
                
                # Verify successful removal
                if not os.path.exists(directory):
                    logger.info(f"Successfully removed directory on attempt {attempt + 1}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to remove directory failed: {e}")
                
                if attempt < max_attempts - 1:
                    time.sleep(1.0)  # Wait before retry
                else:
                    # Final attempt using system commands
                    try:
                        if sys.platform == 'win32':
                            os.system(f'rmdir /s /q "{directory}" 2>nul')
                        else:
                            os.system(f'chmod -R 777 "{directory}" 2>/dev/null || true')
                            os.system(f'rm -rf "{directory}"')
                        
                        # Verify system command success
                        time.sleep(0.5)
                        if not os.path.exists(directory):
                            logger.info("Successfully removed directory using system commands")
                            return True
                            
                    except Exception as sys_e:
                        logger.error(f"System command removal also failed: {sys_e}")
        
        logger.error(f"Failed to remove directory after {max_attempts} attempts")
        return False
    
    @staticmethod
    def _fix_directory_permissions(directory):
        """Fix permissions recursively with comprehensive error handling."""
        try:
            for root, dirs, files in os.walk(directory):
                # Fix directory permissions
                for dir_name in dirs:
                    try:
                        dir_path = os.path.join(root, dir_name)
                        if sys.platform == 'win32':
                            os.chmod(dir_path, 0o777)
                        else:
                            os.chmod(dir_path, 0o755)
                    except Exception as e:
                        logger.debug(f"Could not fix permissions for directory {dir_name}: {e}")
                
                # Fix file permissions
                for file_name in files:
                    try:
                        file_path = os.path.join(root, file_name)
                        if sys.platform == 'win32':
                            os.chmod(file_path, 0o777)
                        else:
                            os.chmod(file_path, 0o644)
                    except Exception as e:
                        logger.debug(f"Could not fix permissions for file {file_name}: {e}")
                        
        except Exception as e:
            logger.debug(f"Error fixing directory permissions: {e}")
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """Create or load vector store using intelligent configuration-based rebuild logic."""
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        collection_name = "apu_kb_collection"
        logger.info(f"Using collection name: {collection_name}")
        
        try:
            # Check and prepare vector store directory
            cls._check_vector_store_directory(persist_directory)
            
            # Use configuration's intelligent rebuild logic
            should_reset = False
            
            # If chunks are provided, we're creating a new store
            if chunks is not None:
                logger.info("Creating new vector store with provided chunks")
                should_reset = True
            else:
                # Use configuration's smart logic for rebuild decisions
                if hasattr(Config, 'should_rebuild_vector_store'):
                    should_reset = Config.should_rebuild_vector_store()
                    logger.info(f"Config rebuild decision: {should_reset}")
                else:
                    # Fallback logic if configuration method doesn't exist
                    should_reset = (
                        getattr(Config, 'FORCE_REINDEX', False) or 
                        not os.path.exists(persist_directory) or 
                        not os.listdir(persist_directory)
                    )
                    logger.info(f"Fallback rebuild decision: {should_reset}")
            
            if should_reset:
                logger.info("Resetting vector store directory")
                if not cls.reset_chroma_db(persist_directory):
                    raise Exception("Failed to reset vector store directory")
            
            # Get ChromaDB client with appropriate settings
            client = ChromaDBManager.get_client(persist_directory, force_new=should_reset)
            
            # Load existing or create new vector store based on circumstances
            if chunks is None and not should_reset:
                logger.info("Attempting to load existing vector store")
                vector_store = cls._load_existing_vector_store(client, collection_name, embeddings, persist_directory)
                
                if vector_store is None:
                    logger.warning("Could not load existing vector store - will try backup")
                    vector_store = VectorStoreManager.load_embeddings_backup(embeddings, collection_name=collection_name)
                
                return vector_store
                
            elif chunks:
                logger.info("Creating new vector store with provided chunks")
                return cls._create_new_vector_store(client, collection_name, chunks, embeddings)
            else:
                logger.error("Cannot create or load vector store - no chunks provided and no existing store")
                return None
                
        except Exception as e:
            logger.error(f"Error in get_or_create_vector_store: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @classmethod
    def _check_vector_store_directory(cls, directory):
        """Check vector store directory status and log relevant information."""
        if os.path.exists(directory):
            contents = os.listdir(directory)
            logger.info(f"Vector store directory exists with contents: {contents}")
            
            # Check for SQLite database file
            sqlite_path = os.path.join(directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                file_size = os.path.getsize(sqlite_path)
                logger.info(f"chroma.sqlite3 exists with size: {file_size} bytes")

    @classmethod
    def _load_existing_vector_store(cls, client, collection_name, embeddings, persist_directory):
        """Load existing vector store with enhanced error handling and persistence validation."""
        try:
            logger.info("Loading existing vector store")
            
            # Force client to reconnect to existing persistent data
            # Check for SQLite database with actual content
            sqlite_path = os.path.join(persist_directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                logger.info(f"Found existing SQLite database: {sqlite_path}")
                
                # Try multiple methods to load the collection
                vector_store = None
                
                # Method 1: Direct collection access
                try:
                    collection, vector_store = ChromaDBManager.get_or_create_collection(
                        client, collection_name, embedding_function=embeddings, force_new=False
                    )
                    
                    # Verify collection contains documents
                    if collection and hasattr(collection, 'count'):
                        count = collection.count()
                        logger.info(f"Collection reports {count} documents")
                        
                        if count > 0:
                            # Perform additional health verification
                            health_status, health_info = cls.check_vector_store_health(vector_store)
                            if health_status:
                                logger.info(f"Successfully loaded existing vector store with {count} documents")
                                return vector_store
                            else:
                                logger.warning("Vector store failed health check despite having documents")
                    
                except Exception as e:
                    logger.warning(f"Direct collection access failed: {e}")
                
                # Method 2: Restore from backup if direct load failed
                logger.info("Direct load failed, attempting to restore from backup")
                backup_path = os.path.join(os.path.dirname(persist_directory), "embeddings_backup.pkl")
                if os.path.exists(backup_path):
                    vector_store = VectorStoreManager.load_embeddings_backup(embeddings, collection_name=collection_name)
                    if vector_store:
                        # Save the restored vector store to persistent storage
                        logger.info("Successfully restored from backup, saving to persistent storage")
                        return vector_store
                
                logger.warning("All methods to load existing vector store failed")
                return None
                
            else:
                logger.warning(f"No SQLite database found at {sqlite_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @classmethod
    def _create_new_vector_store(cls, client, collection_name, chunks, embeddings):
        """Create new vector store with enhanced error handling and configuration integration."""
        try:
            logger.info(f"Creating new vector store with {len(chunks)} chunks")
            
            # Sanitize metadata before creating vector store
            sanitized_chunks = cls.sanitize_metadata(chunks)
            logger.info(f"Sanitized {len(sanitized_chunks)} chunks for vector store")
            
            # Prepare collection metadata using configuration values
            metadata = {
                "document_count": len(sanitized_chunks),
                "chunk_size": getattr(Config, 'CHUNK_SIZE', 600),
                "chunk_overlap": getattr(Config, 'CHUNK_OVERLAP', 120),
                "created_timestamp": datetime.now().isoformat(),
                "apu_filtering_enabled": getattr(Config, 'FILTER_APU_ONLY', False),
                "embedding_model": getattr(Config, 'EMBEDDING_MODEL_NAME', 'unknown')
            }
            
            # Get or create collection with force new to ensure clean state
            collection, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, metadata, embeddings, force_new=True
            )
            
            # Add documents in batches to manage memory usage effectively
            # Use configuration batch size with reasonable limits
            batch_size = getattr(Config, 'EMBEDDING_BATCH_SIZE', 100)
            batch_size = min(batch_size, 100)  # Cap at 100 for stability
            
            total_chunks = len(sanitized_chunks)
            
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                batch = sanitized_chunks[i:end_idx]
                
                try:
                    vector_store.add_documents(documents=batch)
                    batch_num = i // batch_size + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size
                    logger.info(f"Added batch {batch_num}/{total_batches}: {len(batch)} documents")
                    
                except Exception as e:
                    logger.error(f"Error adding batch {batch_num}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
            
            # Verify successful vector store creation
            final_count = collection.count()
            logger.info(f"Vector store created with {final_count} documents")
            
            # Create backup after successful creation if persistence is enabled
            if getattr(Config, 'ENABLE_VECTOR_STORE_PERSISTENCE', True):
                try:
                    VectorStoreManager.save_embeddings_backup(vector_store)
                    logger.info("Created backup of new vector store")
                except Exception as e:
                    logger.warning(f"Could not create backup after vector store creation: {e}")
            else:
                logger.info("Vector store persistence disabled - skipping backup creation")
            
            return vector_store
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """Print comprehensive document statistics with enhanced error handling and APU analysis."""
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            logger.info("Gathering document statistics...")
            
            # Perform health check first to ensure vector store is functional
            is_healthy, health_info = VectorStoreManager.check_vector_store_health(vector_store)
            
            if not is_healthy:
                logger.warning("Vector store health check failed")
                print("Vector store appears to have issues:")
                for error in health_info.get("error_details", []):
                    print(f"  - {error}")
                return
            
            # Initialize statistical counters
            doc_counts = {}
            apu_kb_count = 0
            faq_count = 0
            apu_documents = 0
            non_apu_documents = 0
            
            # Get document count from health check results
            document_count = health_info.get("document_count", 0)
            
            # Retrieve all documents for detailed statistical analysis
            try:
                all_docs = vector_store.get()
                documents = all_docs.get('documents', [])
                all_metadata = all_docs.get('metadatas', [])
                
                if len(documents) != document_count:
                    logger.info(f"Note: Health check reported {document_count} docs, get() returned {len(documents)} docs")
                
                # Analyze metadata with APU filtering awareness
                for metadata in all_metadata:
                    if metadata and isinstance(metadata, dict):
                        # Count documents by filename for detailed breakdown
                        filename = metadata.get('filename', 'unknown')
                        doc_counts[filename] = doc_counts.get(filename, 0) + 1
                        
                        # Categorize APU vs non-APU documents
                        is_apu = metadata.get('is_apu_file', False)
                        if is_apu:
                            apu_documents += 1
                        else:
                            non_apu_documents += 1
                        
                        # Count APU knowledge base specific content
                        if metadata.get('content_type') == 'apu_kb_page':
                            apu_kb_count += 1
                            if metadata.get('is_faq', False):
                                faq_count += 1
                
                total_chunks = len(documents)
                unique_files = len(doc_counts)
                
                # Display comprehensive statistics
                print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
                for filename, count in sorted(doc_counts.items()):
                    print(f"  - {filename}: {count} chunks")
                
                # Display APU-specific content analysis
                if apu_documents > 0 or non_apu_documents > 0:
                    print(f"\nAPU Content Analysis:")
                    print(f"  - APU documents: {apu_documents} chunks")
                    print(f"  - Non-APU documents: {non_apu_documents} chunks")
                    
                    # Show APU filtering status and effectiveness
                    filter_enabled = getattr(Config, 'FILTER_APU_ONLY', False)
                    if filter_enabled and non_apu_documents > 0:
                        print(f"  APU filtering enabled but {non_apu_documents} non-APU chunks found")
                    elif filter_enabled:
                        print(f"  APU filtering working correctly - pure APU content")
                    else:
                        print(f"  APU filtering disabled - mixed content")
                
                # Display APU knowledge base specific statistics
                if apu_kb_count > 0:
                    print(f"\nAPU Knowledge Base: {apu_kb_count} pages, including {faq_count} FAQs")
                
                # Display most recently added document if timestamp available
                recent_docs = []
                for metadata in all_metadata:
                    if metadata and 'timestamp' in metadata:
                        try:
                            timestamp = metadata['timestamp']
                            filename = metadata.get('filename', 'Unknown')
                            recent_docs.append((timestamp, filename))
                        except:
                            continue
                
                if recent_docs:
                    recent_docs.sort(reverse=True)
                    timestamp, filename = recent_docs[0]
                    try:
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"\nMost recently added document: {filename} (added: {date_str}).")
                    except:
                        print(f"\nMost recently added document: {filename}")
                        
            except Exception as e:
                logger.error(f"Error getting detailed statistics: {e}")
                print(f"\nVector store contains approximately {document_count} documents")
                print("  (Detailed statistics unavailable due to access issues)")
                    
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            print("Error retrieving document statistics.")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name) -> bool:
        """Verify if a specific document is properly indexed using multiple verification methods."""
        if not vector_store:
            logger.warning("No vector store provided for document verification")
            return False
            
        try:
            logger.info(f"Verifying if document '{doc_name}' is indexed")
            
            # Method 1: Search for the document using similarity search
            try:
                results = vector_store.similarity_search(f"information from {doc_name}", k=5)
                
                for doc in results:
                    filename = doc.metadata.get('filename', '')
                    if doc_name.lower() in filename.lower():
                        logger.info(f"Document '{doc_name}' found via similarity search")
                        return True
            except Exception as e:
                logger.debug(f"Similarity search failed: {e}")
            
            # Method 2: Check via direct get() method on all documents
            try:
                all_docs = vector_store.get()
                metadatas = all_docs.get('metadatas', [])
                
                for metadata in metadatas:
                    if metadata and isinstance(metadata, dict):
                        filename = metadata.get('filename', '')
                        if doc_name.lower() in filename.lower():
                            logger.info(f"Document '{doc_name}' found via get() method")
                            return True
            except Exception as e:
                logger.debug(f"get() method check failed: {e}")
            
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False