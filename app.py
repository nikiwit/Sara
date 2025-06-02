"""
Main CustomRAG application class for APU knowledge base processing.
"""

import os
import time
import types
import logging
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from session_management import SessionManager

from config import config as Config
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
        """Initialize the RAG application with session management and core components."""
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.input_processor = None
        self.context_processor = None
        self.retrieval_handler = None
        self.conversation_handler = None
        self.command_handler = None
        self.query_router = None
        self.session_manager = SessionManager(max_sessions=Config.MAX_SESSIONS)
    
    def initialize(self):
        """Initialize all components of the RAG system with proper error handling."""
        
        # Verify required dependencies are available
        DocumentProcessor.check_dependencies()
        
        # Initialize core processing components
        self.input_processor = InputProcessor()
        self.context_processor = ContextProcessor()
        
        # Create or load default session
        if not self.session_manager.current_session:
            self.session_manager.create_session("Default Session")
        
        # Use session memory for conversation continuity
        self.memory = self.session_manager.current_session.memory
        
        # Initialize embeddings with caching support
        try:
            self.embeddings = VectorStoreManager.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
        
        # Initialize vector store with smart persistence logic
        vector_store_valid = self.initialize_vector_store_smart()
        
        if not vector_store_valid:
            logger.error("Failed to initialize a valid vector store")
            
            # Attempt recovery before failing completely
            if self.recover_vector_store():
                logger.info("Successfully recovered vector store")
                vector_store_valid = True
            else:
                return False
        
        # Display document statistics for verification
        VectorStoreManager.print_document_statistics(self.vector_store)
        
        # Initialize query processing handlers
        self.conversation_handler = ConversationHandler(self.memory)
        self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
        self.command_handler = CommandHandler(self)
        
        # Initialize query router for request distribution
        self.query_router = QueryRouter(
            self.conversation_handler,
            self.retrieval_handler,
            self.command_handler,
            memory=self.memory
        )
        
        return True

    def initialize_vector_store_smart(self):
        """Initialize vector store using intelligent persistence and recovery logic."""
        max_retries = getattr(Config, 'CHROMADB_PERSIST_RETRY_COUNT', 3)
        retry_delay = getattr(Config, 'CHROMADB_PERSIST_RETRY_DELAY', 2.0)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Determine if vector store rebuild is required based on configuration
                should_rebuild = Config.should_rebuild_vector_store()
                
                if not should_rebuild:
                    # Attempt to load existing vector store from persistence
                    logger.info(f"Attempting to load existing vector store (attempt {retry_count + 1}/{max_retries})")
                    self.vector_store = VectorStoreManager.get_or_create_vector_store(None, self.embeddings)
                    
                    if self.vector_store:
                        # Verify vector store health and integrity
                        is_healthy, health_info = VectorStoreManager.check_vector_store_health(self.vector_store)
                        
                        if is_healthy:
                            logger.info("Successfully loaded existing vector store")
                            
                            # Verify APU documents are indexed if filtering is enabled
                            if Config.FILTER_APU_ONLY:
                                apu_doc_found = VectorStoreManager.verify_document_indexed(self.vector_store, "apu_kb")
                                if not apu_doc_found:
                                    logger.warning("APU KB document not found in vector store - will rebuild")
                                    should_rebuild = True
                                else:
                                    return True
                            else:
                                return True
                        else:
                            logger.warning(f"Existing vector store failed health check (attempt {retry_count + 1})")
                            logger.warning(f"Health check details: {health_info}")
                            
                            # Attempt to repair the collection
                            if VectorStoreManager.fix_chromadb_collection(self.vector_store):
                                logger.info("Successfully fixed vector store collection")
                                return True
                            
                            # Try restoring from backup if repair fails
                            logger.info("Attempting to restore from backup...")
                            backup_store = VectorStoreManager.load_embeddings_backup(self.embeddings)
                            if backup_store:
                                self.vector_store = backup_store
                                logger.info("Successfully restored from backup")
                                return True
                            
                            should_rebuild = True
                    else:
                        logger.warning(f"Could not load existing vector store (attempt {retry_count + 1})")
                        
                        # Try backup restoration before rebuilding
                        logger.info("Attempting to restore from backup...")
                        backup_store = VectorStoreManager.load_embeddings_backup(self.embeddings)
                        if backup_store:
                            self.vector_store = backup_store
                            logger.info("Successfully restored from backup")
                            return True
                        
                        should_rebuild = True
                
                # Rebuild vector store if required or loading failed
                if should_rebuild:
                    logger.info("Creating new vector store from documents")
                    
                    # Load and process documents with comprehensive error handling
                    try:
                        documents = DocumentProcessor.load_documents(Config.DATA_PATH)
                        if not documents:
                            logger.error("No documents found to index")
                            
                            # Provide specific guidance for APU filtering issues
                            if Config.FILTER_APU_ONLY:
                                logger.error("APU filtering is enabled - ensure APU-related files exist in data directory")
                                logger.error(f"Looking for files starting with 'apu_' in: {Config.DATA_PATH}")
                                
                                # List available files for debugging
                                try:
                                    if os.path.exists(Config.DATA_PATH):
                                        files = os.listdir(Config.DATA_PATH)
                                        logger.info(f"Files in data directory: {files}")
                                        apu_files = [f for f in files if f.lower().startswith('apu_')]
                                        logger.info(f"APU files found: {apu_files}")
                                except Exception as e:
                                    logger.error(f"Error listing data directory: {e}")
                            
                            # Exit if no documents and retries exhausted
                            if retry_count >= max_retries - 1:
                                return False
                            
                            # Retry with delay
                            retry_count += 1
                            time.sleep(retry_delay)
                            continue
                            
                        logger.info(f"Successfully loaded {len(documents)} document sections")
                        
                        # Split documents into manageable chunks
                        chunks = DocumentProcessor.split_documents(documents)
                        if not chunks:
                            logger.error("Failed to create document chunks")
                            if retry_count >= max_retries - 1:
                                return False
                            retry_count += 1
                            time.sleep(retry_delay)
                            continue
                            
                        logger.info(f"Successfully created {len(chunks)} document chunks")
                        
                        # Create vector store from processed chunks
                        self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
                        
                        if not self.vector_store:
                            logger.error("Failed to create vector store from chunks")
                            if retry_count >= max_retries - 1:
                                return False
                            retry_count += 1
                            time.sleep(retry_delay)
                            continue
                        
                        # Verify newly created vector store integrity
                        is_healthy, health_info = VectorStoreManager.check_vector_store_health(self.vector_store)
                        
                        if is_healthy:
                            logger.info("Successfully created and verified new vector store")
                            
                            # Create backup after successful creation
                            try:
                                VectorStoreManager.save_embeddings_backup(self.vector_store)
                                logger.info("Created backup of new vector store")
                            except Exception as e:
                                logger.warning(f"Could not create backup: {e}")
                            
                            return True
                        else:
                            logger.error("Newly created vector store failed health check")
                            logger.error(f"Health check details: {health_info}")
                            if retry_count >= max_retries - 1:
                                return False
                            retry_count += 1
                            time.sleep(retry_delay)
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error during document loading/processing: {e}")
                        if retry_count >= max_retries - 1:
                            return False
                        retry_count += 1
                        time.sleep(retry_delay)
                        continue
                
                return False
                
            except Exception as e:
                logger.error(f"Error in vector store initialization (attempt {retry_count + 1}): {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                if retry_count >= max_retries - 1:
                    return False
                
                retry_count += 1
                time.sleep(retry_delay)
                continue
        
        # All retry attempts have been exhausted
        logger.error(f"Failed to initialize vector store after {max_retries} attempts")
        return False

    def initialize_vector_store(self):
        """Deprecated vector store initialization method maintained for compatibility."""
        logger.warning("Using deprecated initialize_vector_store method - should use initialize_vector_store_smart")
        return self.initialize_vector_store_smart()
    
    def recover_vector_store(self):
        """Attempt vector store recovery using multiple recovery strategies."""
        logger.info("Attempting vector store recovery...")
        
        # Check if auto-recovery is enabled in configuration
        if not getattr(Config, 'COLLECTION_AUTO_RECOVER', True):
            logger.warning("Auto-recovery is disabled in configuration")
            return False
        
        # Define recovery methods in order of preference
        recovery_methods = [
            ("backup", self._recover_from_backup),
            ("rebuild", self._recover_by_rebuild),
            ("memory", self._recover_from_memory)
        ]
        
        # Attempt each recovery method until one succeeds
        for method_name, method_func in recovery_methods:
            try:
                logger.info(f"Trying recovery method: {method_name}")
                if method_func():
                    logger.info(f"Successfully recovered using {method_name} method")
                    return True
            except Exception as e:
                logger.error(f"Recovery method {method_name} failed: {e}")
                continue
        
        logger.error("All recovery methods failed")
        return False

    def _recover_from_backup(self):
        """Recover vector store from backup file if available."""
        try:
            backup_store = VectorStoreManager.load_embeddings_backup(self.embeddings)
            if backup_store:
                self.vector_store = backup_store
                
                # Verify backup recovery was successful
                is_healthy, _ = VectorStoreManager.check_vector_store_health(self.vector_store)
                return is_healthy
        except Exception as e:
            logger.error(f"Backup recovery failed: {e}")
        return False

    def _recover_by_rebuild(self):
        """Recover by forcing a complete rebuild from source documents."""
        try:
            # Temporarily enable force rebuild
            original_force_reindex = Config.FORCE_REINDEX
            Config.FORCE_REINDEX = True
            success = self.initialize_vector_store_smart()
            Config.FORCE_REINDEX = original_force_reindex  # Reset to original value
            return success
        except Exception as e:
            logger.error(f"Rebuild recovery failed: {e}")
            Config.FORCE_REINDEX = False  # Ensure reset on error
        return False

    def _recover_from_memory(self):
        """Create minimal in-memory vector store as emergency fallback."""
        try:
            logger.warning("Creating minimal in-memory vector store as last resort")
            
            # Create temporary in-memory ChromaDB client
            import chromadb
            client = chromadb.Client()
            
            # Create minimal emergency collection
            collection = client.create_collection("emergency_collection")
            
            # Add emergency document to make collection functional
            collection.add(
                documents=["This is an emergency vector store. Please rebuild the index."],
                metadatas=[{"type": "emergency", "created": datetime.now().isoformat()}],
                ids=["emergency_doc_1"]
            )
            
            # Create LangChain wrapper for emergency collection
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                client=client,
                collection_name="emergency_collection",
                embedding_function=self.embeddings
            )
            
            logger.warning("Created emergency in-memory vector store")
            logger.warning("This is temporary - please rebuild the index properly")
            return True
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
        return False
    
    def reindex_documents(self):
        """Reindex all documents in the data directory with comprehensive error handling."""
        logger.info("Reindexing documents")
        print("Reindexing documents. This may take a while...")
        
        # Perform complete cleanup of existing resources
        if self.vector_store is not None:
            try:
                # Create backup before reindexing if enabled
                if getattr(Config, 'COLLECTION_BACKUP_ON_SHUTDOWN', True):
                    try:
                        VectorStoreManager.save_embeddings_backup(self.vector_store)
                        logger.info("Created backup before reindexing")
                    except Exception as e:
                        logger.warning(f"Could not create backup before reindexing: {e}")
                
                # Force complete ChromaDB client cleanup
                ChromaDBManager.force_cleanup()
                self.vector_store = None
                
                # Force garbage collection and wait for cleanup
                import gc
                gc.collect()
                time.sleep(1.5)  # Allow time for resource cleanup
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        
        # Reset the vector store directory for clean start
        if not VectorStoreManager.reset_chroma_db(Config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            print("Failed to reset vector store. Check permissions.")
            return False
        
        # Load and process documents for reindexing
        try:
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                print("No documents found to index.")
                
                # Provide specific guidance for APU filtering
                if Config.FILTER_APU_ONLY:
                    print("APU filtering is enabled - ensure files starting with 'apu_' exist in data directory")
                    print(f"Data directory: {Config.DATA_PATH}")
                
                return False
                
            # Split documents into processable chunks
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                print("Failed to create document chunks.")
                return False
                
            # Get fresh ChromaDB client for reindexing
            client = ChromaDBManager.get_client(Config.PERSIST_PATH, force_new=True)
            
            # Reuse existing embeddings to avoid reinitialization overhead
            if self.embeddings is None:
                self.embeddings = VectorStoreManager.create_embeddings()
                
            # Create new vector store with fresh collection
            collection_name = getattr(Config, 'CHROMADB_COLLECTION_NAME', 'apu_kb_collection')
            collection, self.vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, 
                metadata={"reindexed_at": datetime.now().isoformat()},
                embedding_function=self.embeddings,
                force_new=True  # Ensure completely fresh collection
            )
            
            # Add documents in batches to manage memory usage
            try:
                # Sanitize metadata to prevent ChromaDB issues
                sanitized_chunks = VectorStoreManager.sanitize_metadata(chunks)
                
                # Process documents in configurable batches
                batch_size = getattr(Config, 'CHROMADB_BATCH_SIZE', 50)
                total_chunks = len(sanitized_chunks)
                
                for i in range(0, total_chunks, batch_size):
                    batch = sanitized_chunks[i:i + batch_size]
                    try:
                        self.vector_store.add_documents(documents=batch)
                        batch_num = i//batch_size + 1
                        total_batches = (total_chunks + batch_size - 1)//batch_size
                        logger.info(f"Added batch {batch_num}/{total_batches}")
                        print(f"Progress: {batch_num}/{total_batches} batches completed")
                    except Exception as e:
                       logger.error(f"Error adding batch {batch_num}: {e}")
                       raise
                
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                print(f"Error adding documents to vector store: {e}")
                return False
            
            # Verify vector store integrity after reindexing
            is_healthy, health_info = VectorStoreManager.check_vector_store_health(self.vector_store)
            if not is_healthy:
                logger.error("Vector store health check failed after reindexing")
                logger.error(f"Health check details: {health_info}")
                print("Vector store health check failed after reindexing.")
                return False
                
            # Create backup of successfully reindexed vector store
            try:
                VectorStoreManager.save_embeddings_backup(self.vector_store)
                logger.info("Created backup of reindexed vector store")
            except Exception as e:
                logger.warning(f"Could not create backup after reindexing: {e}")
                
            # Display statistics for verification
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
                
            print(f"Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"Error during reindexing: {e}")
            return False

    def switch_session(self, session_id: str = None):
        """Switch to a different session or create a new one."""
        if session_id:
            session = self.session_manager.load_session(session_id)
            if not session:
                print(f"Session {session_id} not found.")
                return False
        else:
            # Create new session with user-specified title
            title = input("Enter session title (or press Enter for default): ").strip()
            session = self.session_manager.create_session(title if title else None)
        
        # Update memory reference to new session
        self.memory = session.memory
        
        # Update all handlers with new session memory
        if self.conversation_handler:
            self.conversation_handler.memory = self.memory
        if self.retrieval_handler:
            self.retrieval_handler.memory = self.memory
        if self.query_router:
            self.query_router.memory = self.memory
        
        print(f"Switched to session: {session.metadata.title} ({session.metadata.session_id[:8]})")
        return True

    def list_sessions_command(self):
        """Display all available sessions with detailed information."""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            print("No sessions found.")
            return
        
        print("\nAvailable Sessions:")
        print("-" * 80)
        for i, session in enumerate(sessions, 1):
            current_marker = "➤ " if (self.session_manager.current_session and 
                                    session.session_id == self.session_manager.current_session.metadata.session_id) else "  "
            print(f"{current_marker}{i}. {session.title}")
            print(f"     ID: {session.session_id[:8]}... | Messages: {session.message_count} | Last: {session.last_accessed.strftime('%Y-%m-%d %H:%M')}")
        print("-" * 80)

    def session_stats_command(self):
        """Display comprehensive session statistics."""
        stats = self.session_manager.get_session_statistics()
        current_session = self.session_manager.current_session
        
        print(f"\nSession Statistics:")
        print(f"   Sessions: {stats['total_sessions']}/{stats['max_sessions']}")
        print(f"   Total Messages: {stats['total_messages']}")
        if current_session:
            print(f"   Current Session: {current_session.metadata.title}")
            print(f"   Current ID: {stats['current_session_id'][:8]}...")
            print(f"   Current Messages: {stats['current_session_messages']}")
    
    def run_cli(self):
        """Run the interactive command line interface with comprehensive error handling."""
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            
            # Offer automatic recovery to user
            print("\nFailed to initialize the RAG system.")
            try:
                response = input("\nWould you like to try automatic recovery? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    print("\nAttempting automatic recovery...")
                    if self.recover_vector_store():
                        print("Recovery successful! Continuing...")
                        # Re-initialize handlers after successful recovery
                        self.conversation_handler = ConversationHandler(self.memory)
                        self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
                        self.command_handler = CommandHandler(self)
                        self.query_router = QueryRouter(
                            self.conversation_handler,
                            self.retrieval_handler,
                            self.command_handler,
                            memory=self.memory
                        )
                    else:
                        print("Recovery failed. Please check the logs and try reindexing manually.")
                        return
                else:
                    return
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return
            
        # Display welcome banner and available commands
        print("\n" + "="*60)
        print("APURAG - APU Knowledge Base Assistant")
        print("="*60)
        print("Ask questions about APU using natural language.")
        print("Commands:")
        print("  - Type 'help' to see available commands")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document statistics")
        print("  - Type 'new session' to create a new chat session")
        print("  - Type 'list sessions' to see all sessions")
        print("  - Type 'switch session' to change sessions")
        print("  - Type 'session stats' to see session statistics")
        print("  - Type 'clear session' to clear current session memory")
        print("="*60 + "\n")
        
        # Display current session information
        current_session = self.session_manager.current_session
        if current_session:
            print(f"Current Session: {current_session.metadata.title} ({current_session.metadata.session_id[:8]}...)")
            print(f"   Messages: {current_session.metadata.message_count} | Created: {current_session.metadata.created_at.strftime('%Y-%m-%d %H:%M')}\n")
        
        # Log current configuration for debugging
        logger.info(f"Running with configuration: {Config.__class__.__name__}")
        logger.info(f"APU filtering enabled: {Config.FILTER_APU_ONLY}")
        logger.info(f"Vector store persistence enabled: {getattr(Config, 'ENABLE_VECTOR_STORE_PERSISTENCE', True)}")
        logger.info(f"Max threads: {Config.MAX_THREADS}")
        logger.info(f"Max context size: {Config.MAX_CONTEXT_SIZE}")
        
        # Main user interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query with timing for performance monitoring
                start_time = time.time()
                
                print("\nThinking...\n")
                
                try:
                    # Analyze and route query to appropriate handler
                    query_analysis = self.input_processor.analyze_query(query)
                    
                    # Route query with streaming support enabled
                    response, should_continue = self.query_router.route_query(query_analysis, stream=True)
                    
                    if response:
                        # Handle streaming responses from generators
                        if isinstance(response, types.GeneratorType) or hasattr(response, '__iter__'):
                            full_response = ""
                            for token in response:
                                print(token, end="", flush=True)
                                full_response += token
                            print()  
                            
                            # Save conversation to current session (exclude commands)
                            if full_response.strip() and not query_analysis["original_query"].lower().startswith(('exit', 'quit', 'help', 'clear', 'reindex', 'stats', 'new session', 'list sessions', 'switch session', 'session stats', 'clear session')):
                                self.session_manager.add_conversation(query, full_response.strip())
                        else:
                            # Handle direct string responses
                            print(f"{response}")
                            
                            # Save conversation to current session (exclude commands)
                            if response and response.strip() and not query_analysis["original_query"].lower().startswith(('exit', 'quit', 'help', 'clear', 'reindex', 'stats', 'new session', 'list sessions', 'switch session', 'session stats', 'clear session')):
                                self.session_manager.add_conversation(query, response.strip())
                    
                    if not should_continue:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\n[Error occurred during response generation]")
                    
                    # Provide debug information if debug logging is enabled
                    if logger.isEnabledFor(logging.DEBUG):
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # Calculate and log query processing time
                end_time = time.time()
                query_time = end_time - start_time
                logger.info(f"Query processed in {query_time:.2f} seconds")
                
                # Log performance warning for slow queries
                if hasattr(Config, 'SLOW_QUERY_THRESHOLD') and query_time > Config.SLOW_QUERY_THRESHOLD:
                    logger.warning(f"Slow query detected: {query_time:.2f}s (threshold: {Config.SLOW_QUERY_THRESHOLD}s)")
                
                # Update performance statistics if tracking is enabled
                if hasattr(Config, '_performance_stats'):
                    Config._performance_stats['total_queries_processed'] += 1
                
            except KeyboardInterrupt:
                print("\nGoodbye! Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print("\n[An unexpected error occurred. Please try again.]")
                
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    logger.debug(traceback.format_exc())
                
    def cleanup(self):
        """Clean up resources and save state before application shutdown."""
        logger.info("Cleaning up resources")
        
        try:
            # Save current session state before cleanup
            if self.session_manager and self.session_manager.current_session:
                self.session_manager.save_current_session()
                logger.info("Saved current session")
        except Exception as e:
            logger.warning(f"Error saving session during cleanup: {e}")
        
        try:
            # Create final backup before shutdown if enabled
            if getattr(Config, 'COLLECTION_BACKUP_ON_SHUTDOWN', True) and self.vector_store:
                try:
                    VectorStoreManager.save_embeddings_backup(self.vector_store)
                    logger.info("Created final backup before shutdown")
                except Exception as e:
                    logger.warning(f"Could not create final backup: {e}")
        except Exception as e:
            logger.warning(f"Error checking backup settings: {e}")
        
        try:
            # Close ChromaDB client connections
            ChromaDBManager.close()
            logger.info("Closed ChromaDB client")
        except Exception as e:
            logger.warning(f"Error closing ChromaDB client: {e}")
        
        try:
            # Clear vector store reference and force garbage collection
            self.vector_store = None
            
            import gc
            gc.collect()
            
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Error during final cleanup: {e}")
        
        # Log performance statistics if available
        if hasattr(Config, 'log_performance_stats'):
            try:
                Config.log_performance_stats()
            except Exception as e:
                logger.debug(f"Error logging performance stats: {e}")