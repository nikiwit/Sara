"""
Main CustomRAG application class.
"""

import os
import time
import types
import logging
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

from config import Config
from document_processing import DocumentProcessor
from input_processing import InputProcessor
from query_handling import QueryRouter, ConversationHandler, CommandHandler
from vector_management import VectorStoreManager, ChromaDBManager
from retrieval import ContextProcessor, RetrievalHandler
from response import RAGSystem

logger = logging.getLogger("CustomRAG")

class CustomRAG:
    """Main RAG application class for command line interface."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.input_processor = None
        self.context_processor = None
        self.retrieval_handler = None
        self.conversation_handler = None
        self.command_handler = None
        self.query_router = None
    
    def initialize(self):
        """Set up all components of the RAG system."""
        # FIXED: Remove duplicate Config.setup() call since main.py already handles this
        # Config.setup() - This was causing duplicate configuration logging
        
        # Check dependencies
        DocumentProcessor.check_dependencies()
        
        # Initialize components
        self.input_processor = InputProcessor()
        self.context_processor = ContextProcessor()
        
        # Create memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # System message to memory
        system_message = SystemMessage(content="I am an AI assistant that helps with answering questions about APU. I can provide information about academic procedures, administrative processes, and university services.")
        self.memory.chat_memory.messages.append(system_message)
        
        # Create embeddings with caching
        try:
            self.embeddings = VectorStoreManager.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
        
        # Initialize vector store with health checks
        vector_store_valid = self.initialize_vector_store()
        
        if not vector_store_valid:
            logger.error("Failed to initialize a valid vector store")
            return False
        
        # Print statistics
        VectorStoreManager.print_document_statistics(self.vector_store)
        
        # Initialize handlers
        self.conversation_handler = ConversationHandler(self.memory)
        self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
        self.command_handler = CommandHandler(self)
        
        # Initialize query router
        self.query_router = QueryRouter(
            self.conversation_handler,
            self.retrieval_handler,
            self.command_handler,
            memory=self.memory
        )
        
        return True

    def initialize_vector_store(self):
        """Initialize vector store with health checks and fallbacks."""
        # Path for backup
        backup_path = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
        backup_exists = os.path.exists(backup_path)
        
        # First try to load from ChromaDB (normal flow)
        vector_store_valid = False
        if not Config.FORCE_REINDEX and os.path.exists(Config.PERSIST_PATH):
            try:
                logger.info("Attempting to load existing vector store")
                self.vector_store = VectorStoreManager.get_or_create_vector_store(None, self.embeddings)
                
                # Check if it has documents and is healthy
                if self.vector_store:
                    vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
        
        # If ChromaDB load failed but we have a backup, try to restore from backup
        if not vector_store_valid and backup_exists:
            logger.info("Attempting to restore from embeddings backup")
            self.vector_store = VectorStoreManager.load_embeddings_backup(self.embeddings)
            if self.vector_store:
                vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
        
        # If both ChromaDB and backup failed, or we're forcing reindex, create from scratch
        if not vector_store_valid or Config.FORCE_REINDEX:
            logger.info("Creating new vector store from documents")
            
            # Load and process documents
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            # If successful, create a backup
            if self.vector_store:
                vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
                if vector_store_valid:
                    # Create backup
                    VectorStoreManager.save_embeddings_backup(self.vector_store)
        
        return vector_store_valid
    
    def reindex_documents(self):
        """Reindex all documents in the data directory with improved error handling."""
        logger.info("Reindexing documents")
        print("Reindexing documents. This may take a while...")
        
        # Complete cleanup of existing resources
        if self.vector_store is not None:
            try:
                # Force close ChromaDB client completely
                ChromaDBManager.force_cleanup()
                self.vector_store = None
                
                # Force garbage collection
                import gc
                gc.collect()
                time.sleep(1.5)  # Give more time for cleanup
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        
        # Reset the vector store directory with permissions
        if not VectorStoreManager.reset_chroma_db_with_permissions(Config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            print("Failed to reset vector store. Check permissions.")
            return False
        
        # Load and process documents
        try:
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                print("No documents found to index.")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                print("Failed to create document chunks.")
                return False
                
            # Get a completely fresh client for reindexing
            client = ChromaDBManager.get_client(force_new=True)
            
            # Reuse existing embeddings instead of recreating
            if self.embeddings is None:
                self.embeddings = VectorStoreManager.create_embeddings()
                
            # Create vector store with chunks using force_new flag
            collection, self.vector_store = ChromaDBManager.get_or_create_collection(
                client, "apu_kb_collection", 
                metadata={"reindexed_at": datetime.now().isoformat()},
                embedding_function=self.embeddings,
                force_new=True  # This ensures fresh collection
            )
            
            # Documents with better error handling
            try:
                # Sanitize metadata before adding
                sanitized_chunks = VectorStoreManager.sanitize_metadata(chunks)
                
                # Documents in batches to avoid memory issues
                batch_size = 50  # Smaller batches for stability
                total_chunks = len(sanitized_chunks)
                
                for i in range(0, total_chunks, batch_size):
                    batch = sanitized_chunks[i:i + batch_size]
                    try:
                        self.vector_store.add_documents(documents=batch)
                        logger.info(f"Added batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                    except Exception as e:
                        logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                        raise
                
                # Explicitly persist if supported
                if hasattr(self.vector_store, 'persist'):
                    self.vector_store.persist()
                    
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                print(f"Error adding documents to vector store: {e}")
                return False
            
            # Verify the vector store works
            if not VectorStoreManager.check_vector_store_health(self.vector_store):
                logger.error("Vector store health check failed after reindexing")
                print("Vector store health check failed after reindexing.")
                return False
                
            # Create a backup of the newly created vector store
            VectorStoreManager.save_embeddings_backup(self.vector_store)
                
            # Print statistics
            VectorStoreManager.print_document_statistics(self.vector_store)
            
            # Reinitialize handlers with new vector store
            self.retrieval_handler = RetrievalHandler(
                self.vector_store, self.embeddings, self.memory, self.context_processor)
            
            # Update query router with new retrieval handler
            self.query_router = QueryRouter(
                self.conversation_handler,
                self.retrieval_handler,
                self.command_handler,
                memory=self.memory
            )
                
            print(f"✅ Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"❌ Error during reindexing: {e}")
            return False
    
    def run_cli(self):
        """Run the interactive command line interface."""
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            return
            
        # Print banner and instructions
        print("\n" + "="*60)
        print("📚 APURAG - APU Knowledge Base Assistant 📚")
        print("="*60)
        print("Ask questions about APU using natural language.")
        print("Commands:")
        print("  - Type 'help' to see available commands")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document statistics")
        print("="*60 + "\n")
        
        # Main interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                start_time = time.time()
                
                print("\nThinking...\n")
                
                try:
                    # Process and route query
                    query_analysis = self.input_processor.analyze_query(query)
                    
                    # Route the query to appropriate handler with streaming enabled
                    response, should_continue = self.query_router.route_query(query_analysis, stream=True)
                    
                    if response:
                        # Handle streamed responses
                        if isinstance(response, types.GeneratorType) or hasattr(response, '__iter__'):
                            # print("Response: ", end="", flush=True)
                            full_response = ""
                            for token in response:
                                print(token, end="", flush=True)
                                full_response += token
                            print()  
                            
                            # Memory update is handled within the handlers now
                        else:
                            # Handle non-streamed responses
                            print(f"{response}")
                    
                    if not should_continue:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\n[Error occurred during response generation]")
                
                end_time = time.time()
                logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\nGoodbye! Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print("\n[An unexpected error occurred. Please try again.]")
                
    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info("Cleaning up resources")
        
        # Close ChromaDB client
        ChromaDBManager.close()
        
        # Clear vector store reference
        self.vector_store = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Cleanup completed")