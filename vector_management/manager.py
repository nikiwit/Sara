"""
Vector store management operations and utilities.
"""

import os
import sys
import shutil
import time
import logging
import torch
import json
from datetime import datetime
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
from .chromadb_manager import ChromaDBManager

logger = logging.getLogger("CustomRAG")

class VectorStoreManager:
    """Manages the vector database operations."""
    
    @staticmethod
    def check_vector_store_health(vector_store):
        """Perform comprehensive health check on vector store."""
        if not vector_store:
            logger.warning("No vector store provided for health check")
            return False
            
        try:
            # Check 1: Can we get collection info?
            collection_info = False
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                try:
                    count = collection.count()
                    logger.info(f"Collection reports {count} documents")
                    if count > 0:
                        collection_info = True
                except Exception as e:
                    logger.warning(f"Failed to get collection count: {e}")
            
            # Check 2: Can we perform a test query?
            query_success = False
            try:
                test_results = vector_store.similarity_search("test query APU university", k=1)
                if test_results:
                    logger.info(f"Query test successful: retrieved {len(test_results)} documents")
                    query_success = True
            except Exception as e:
                logger.warning(f"Query test failed: {e}")
            
            # Check 3: Can we get all documents?
            get_success = False
            try:
                all_docs = vector_store.get()
                doc_count = len(all_docs.get('documents', []))
                logger.info(f"get() method reports {doc_count} documents")
                if doc_count > 0:
                    get_success = True
            except Exception as e:
                logger.warning(f"get() method failed: {e}")
            
            # Overall health assessment
            health_status = collection_info or (query_success and get_success)
            
            if health_status:
                logger.info("Vector store health check: PASSED")
            else:
                logger.warning("Vector store health check: FAILED")
                
            return health_status
            
        except Exception as e:
            logger.error(f"Error during vector store health check: {e}")
            return False
    
    @staticmethod
    def get_embedding_device():
        """Determine the best available device for embeddings."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for embeddings")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS for embeddings")
            return 'mps'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'
        
    @staticmethod
    def save_embeddings_backup(vector_store, filepath=None):
        """Save a backup of embeddings and metadata."""
        if not vector_store:
            return False
            
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
            
        try:
            data = vector_store.get()
            if not data or not data.get('ids') or len(data.get('ids', [])) == 0:
                logger.warning("No data to backup - vector store appears empty")
                return False
                
            # Add timestamp for versioning
            data['backup_time'] = datetime.now().isoformat()
            data['doc_count'] = len(data.get('ids', []))
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved embeddings backup with {data['doc_count']} documents to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def load_embeddings_backup(embeddings, filepath=None, collection_name="apu_kb_collection"):
        """Load embeddings from backup if main store is empty."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
            
        if not os.path.exists(filepath):
            logger.warning(f"No embeddings backup found at {filepath}")
            return None
            
        try:
            # Load backup
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Check if data appears valid
            if not data or 'ids' not in data or len(data['ids']) == 0:
                logger.warning("Backup file exists but contains no valid data")
                return None
                
            logger.info(f"Loaded embeddings backup with {len(data['ids'])} documents from {filepath}")
            
            # Create a new in-memory Chroma instance with this data
            from langchain_chroma import Chroma
            import chromadb
            from chromadb.config import Settings
            
            # Create a memory client
            client = chromadb.Client(Settings(anonymized_telemetry=False))
            
            # Create a new collection
            collection = client.create_collection(name=collection_name)
            
            # Add the data in batches to avoid memory issues
            batch_size = 100
            total_items = len(data['ids'])
            
            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)
                
                # Prepare batch
                batch_ids = data['ids'][i:end_idx]
                batch_embeddings = data['embeddings'][i:end_idx] if data.get('embeddings') else None
                batch_metadatas = data['metadatas'][i:end_idx] if data.get('metadatas') else None
                batch_documents = data['documents'][i:end_idx] if data.get('documents') else None
                
                # Add to collection
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                logger.info(f"Restored batch {i//batch_size + 1}/{(total_items+batch_size-1)//batch_size}: {len(batch_ids)} documents")
            
            # Create a LangChain wrapper around this collection
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            logger.info(f"Successfully restored vector store from backup with {total_items} documents")
            return vector_store
                
        except Exception as e:
            logger.error(f"Failed to load embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    @staticmethod
    def fix_chromadb_collection(vector_store):
        """
        Fix for ChromaDB collections that appear empty despite existing in the database.
        This is a workaround for a known issue with ChromaDB persistence.
        """
        if not vector_store:
            return False
            
        try:
            # Check if collection exists but reports 0 documents
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                count = vector_store._collection.count()
                if count == 0:
                    logger.warning("Collection exists but reports 0 documents - attempting fix")
                    
                    # Try to force a collection reload through direct access
                    if hasattr(vector_store._collection, '_client'):
                        client = vector_store._collection._client
                        collection_name = vector_store._collection.name
                        
                        # Try a direct query to wake up the collection
                        try:
                            logger.info("Attempting direct ChromaDB query to fix collection")
                            from chromadb.api.types import QueryResult
                            results = client.query(
                                collection_name=collection_name,
                                query_texts=["test query for collection fix"],
                                n_results=1,
                            )
                            logger.info(f"Direct query results: {results}")
                            
                            # Check collection again
                            count_after = vector_store._collection.count()
                            logger.info(f"Collection count after fix attempt: {count_after}")
                            
                            return count_after > 0
                        except Exception as e:
                            logger.error(f"Error during collection fix attempt: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error in fix_chromadb_collection: {e}")
            return False
        
    @staticmethod
    def sanitize_metadata(documents: List[Document]) -> List[Document]:
        """
        Sanitize document metadata to ensure compatibility with ChromaDB.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with sanitized metadata
        """
        sanitized_docs = []
        
        for doc in documents:
            # Create a copy of the metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # Process each metadata field
            for key, value in list(metadata.items()):
                # Convert lists to strings
                if isinstance(value, list):
                    if value:  # If list is not empty
                        metadata[key] = json.dumps(value)
                    else:
                        # Remove empty lists
                        metadata.pop(key)
                # Remove None values
                elif value is None:
                    metadata.pop(key)
                # Keep other primitive types as is
            
            # Create a new document with sanitized metadata
            sanitized_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            sanitized_docs.append(sanitized_doc)
        
        return sanitized_docs
    
    @staticmethod
    def create_embeddings(model_name=None):
        """Create the embedding model."""
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL_NAME
            
        device = VectorStoreManager.get_embedding_device()
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """Reset the ChromaDB environment - handles file system operations safely."""
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Remove existing directory if it exists
        if os.path.exists(persist_directory):
            try:
                # Make directory writable first (for Windows compatibility)
                if sys.platform == 'win32':
                    for root, dirs, files in os.walk(persist_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                
                # Try Python's built-in directory removal first
                shutil.rmtree(persist_directory)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\"")
                    else:
                        os.system(f"rm -rf \"{persist_directory}\"")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Create fresh directory structure
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set appropriate permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            
            # Create .chroma subdirectory for ChromaDB to recognize
            os.makedirs(os.path.join(persist_directory, ".chroma"), exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """Get existing vector store or create a new one with enhanced debugging."""
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # Use a consistent collection name
        collection_name = "apu_kb_collection"
        logger.info(f"Using collection name: {collection_name}")
        
        # Check vector store directory
        cls._check_vector_store_directory(persist_directory)
        
        # Reset directory if needed
        if Config.FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            cls.reset_chroma_db(persist_directory)
        
        # Get ChromaDB client
        client = ChromaDBManager.get_client(persist_directory)
        
        # Load existing vector store or create new one
        if chunks is None and os.path.exists(persist_directory) and os.listdir(persist_directory):
            return cls._load_existing_vector_store(client, collection_name, embeddings)
        elif chunks:
            return cls._create_new_vector_store(client, collection_name, chunks, embeddings)
        else:
            logger.error("Cannot create or load vector store - no chunks provided and no existing store")
            return None

    @classmethod
    def _check_vector_store_directory(cls, directory):
        """Check vector store directory and log information."""
        if os.path.exists(directory):
            logger.info(f"Vector store directory exists with contents: {os.listdir(directory)}")
            sqlite_path = os.path.join(directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                file_size = os.path.getsize(sqlite_path)
                logger.info(f"chroma.sqlite3 exists with size: {file_size} bytes")

    @classmethod
    def _load_existing_vector_store(cls, client, collection_name, embeddings):
        """Load existing vector store."""
        try:
            logger.info(f"Loading existing vector store")
            
            # Get collection and langchain wrapper
            collection, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, embedding_function=embeddings)
            
            # Verify collection has documents
            try:
                count = collection.count()
                logger.info(f"Vector store reports {count} documents after loading")
                
                if count > 0:
                    logger.info(f"Successfully loaded vector store with {count} documents")
                    return vector_store
                else:
                    logger.warning("Vector store exists but is empty - will try backup")
                    return None
            except Exception as e:
                logger.error(f"Error verifying collection: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def _create_new_vector_store(cls, client, collection_name, chunks, embeddings):
        """Create new vector store with chunks."""
        try:
            logger.info(f"Creating new vector store with {len(chunks)} chunks")
            
            # Sanitize metadata before creating vector store
            sanitized_chunks = cls.sanitize_metadata(chunks)
            
            # Get or create collection
            metadata = {
                "document_count": len(sanitized_chunks),
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP
            }
            
            _, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, metadata, embeddings)
            
            # Add documents to the vector store
            vector_store.add_documents(documents=sanitized_chunks)
            
            # Explicitly persist
            if hasattr(vector_store, 'persist'):
                vector_store.persist()
                logger.info(f"Vector store persisted successfully with {len(sanitized_chunks)} chunks")
            
            return vector_store
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """Print statistics about indexed documents."""
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            # Initialize counters
            doc_counts = {}
            apu_kb_count = 0
            faq_count = 0
            
            # Access collection directly first
            collection = None
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                logger.info("Accessing vector store statistics via direct collection")
                count = collection.count()
                logger.info(f"Collection reports {count} documents")
            
            # Get all documents
            all_docs = vector_store.get()
            documents = all_docs.get('documents', [])
            all_metadata = all_docs.get('metadatas', [])
            
            doc_count = len(documents)
            logger.info(f"Vector store get() method reports {doc_count} documents")
            
            if doc_count == 0:
                # Try a test query to see if documents can be retrieved
                try:
                    test_results = vector_store.similarity_search("test query", k=1)
                    if test_results:
                        logger.info(f"Found {len(test_results)} documents via search - data exists but get() not working")
                except Exception as e:
                    logger.error(f"Error during test search: {e}")
                    
            if doc_count == 0 and (collection is None or collection.count() == 0):
                logger.warning("Vector store appears to be empty")
                return
            
            # Count documents by filename
            for metadata in all_metadata:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
                
                # Count APU KB specific pages
                if metadata.get('content_type') == 'apu_kb_page':
                    apu_kb_count += 1
                    if metadata.get('is_faq', False):
                        faq_count += 1
            
            total_chunks = len(documents)
            unique_files = len(doc_counts)
            
            logger.info(f"Vector store contains {total_chunks} chunks from {unique_files} files")
            
            # Print file statistics
            print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
            for filename, count in sorted(doc_counts.items()):
                print(f"  - {filename}: {count} chunks")
            
            # Print APU KB specific statistics
            if apu_kb_count > 0:
                print(f"\nAPU Knowledge Base: {apu_kb_count} pages, including {faq_count} FAQs")
                
            # Print only the most recently added document if timestamp is available
            recent_docs = []
            for i, metadata in enumerate(all_metadata):
                if metadata and 'timestamp' in metadata:
                    recent_docs.append((metadata['timestamp'], metadata.get('filename', 'Unknown')))
            
            if recent_docs:
                recent_docs.sort(reverse=True)
                # Get only the most recent document
                timestamp, filename = recent_docs[0]
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nMost recently added document: {filename} (added: {date_str}).")
                    
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Add detailed error info
            print("Error retrieving document statistics.")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name):
        """Verify if a specific document is properly indexed."""
        if not vector_store:
            return False
            
        try:
            # Search for the document name in the vector store
            results = vector_store.similarity_search(f"information from {doc_name}", k=3)
            
            # Check if any results match this filename
            for doc in results:
                filename = doc.metadata.get('filename', '')
                if doc_name.lower() in filename.lower():
                    logger.info(f"Document '{doc_name}' is indexed in the vector store")
                    return True
                    
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False