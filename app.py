"""
Main Sara application class with production model management.

This module provides the primary interface for the Sara system, including:
- Interactive command-line interface
- Session management and conversation history
- Production model lifecycle management
- Document indexing and retrieval operations
- System monitoring and statistics
"""

import os
import time
import types
import logging
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from session_management import SessionManager

from config import config
from document_processing import DocumentProcessor
from input_processing import InputProcessor
from query_handling import QueryRouter, ConversationHandler, CommandHandler
from vector_management import VectorStoreManager, ChromaDBManager
from retrieval import ContextProcessor, RetrievalHandler
from response import RAGSystem

logger = logging.getLogger("Sara")

class Sara:
    """
    Main Sara application class for command line interface with production features.
    
    This class orchestrates all components of the Sara system including document processing,
    query routing, conversation management, and model lifecycle operations.
    """
    
    def __init__(self):
        """Initialize the RAG application with session management."""
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.input_processor = None
        self.context_processor = None
        self.retrieval_handler = None
        self.conversation_handler = None
        self.command_handler = None
        self.query_router = None
        self.session_manager = SessionManager(max_sessions=config.MAX_SESSIONS)
    
    def initialize(self):
        """
        Set up all components of the RAG system.
        
        Initializes embeddings, vector store, session management, and all handlers
        required for the RAG pipeline to function properly.
        
        Returns:
            bool: True if initialization was successful
        """
        # Check dependencies
        DocumentProcessor.check_dependencies()
        
        # Initialize components
        self.input_processor = InputProcessor()
        self.context_processor = ContextProcessor()
        
        # Create or load a session
        if not self.session_manager.current_session:
            self.session_manager.create_session("Default Session")
        
        # Use session memory instead of creating new memory
        self.memory = self.session_manager.current_session.memory
        
        # Create embeddings with production caching and update management
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
        """
        Initialize vector store with health checks and fallbacks.
        
        Attempts to load existing vector store, falls back to backup restoration
        if needed, and creates new store from documents as last resort.
        
        Returns:
            bool: True if a valid vector store was initialized
        """
        # Path for backup
        backup_path = os.path.join(os.path.dirname(config.PERSIST_PATH), "embeddings_backup.pkl")
        backup_exists = os.path.exists(backup_path)
        
        # First try to load from ChromaDB (normal flow)
        vector_store_valid = False
        if not config.FORCE_REINDEX and os.path.exists(config.PERSIST_PATH):
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
        if not vector_store_valid or config.FORCE_REINDEX:
            logger.info("Creating new vector store from documents")
            
            # Load and process documents
            documents = DocumentProcessor.load_documents(config.DATA_PATH)
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

    def handle_production_commands(self, user_input: str) -> tuple[str, bool]:
        """
        Handle production model management commands.
        
        Processes commands related to model lifecycle management including
        status checking, cache management, and forced updates.
        
        Args:
            user_input: User command input
            
        Returns:
            tuple: (response_text, should_continue)
        """
        command = user_input.lower().strip()
        
        if command == "model report":
            report = VectorStoreManager.get_production_model_report()
            return f"\n{report}\n", True
        
        elif command == "model check":
            # Force a model update check
            model_name = config.EMBEDDING_MODEL_NAME
            age_info = VectorStoreManager._check_model_age_and_updates(model_name)
            update_info = VectorStoreManager._check_for_model_updates(model_name)
            
            response = [
                "\nModel Status Check",
                "=" * 30,
                f"Model: {model_name}",
                f"Age: {age_info['message']}",
                f"Updates: {update_info.get('message', 'Could not check')}",
                ""
            ]
            return "\n".join(response), True
        
        elif command == "model update":
            # Force clear cache and update
            model_name = config.EMBEDDING_MODEL_NAME
            response_lines = [f"\nClearing cache for {model_name}..."]
            
            if VectorStoreManager._clear_model_cache_for_update(model_name):
                response_lines.append("Cache cleared. Model will be re-downloaded on next startup.")
                response_lines.append("Tip: Restart the application to download the latest model.")
            else:
                response_lines.append("Failed to clear cache.")
            
            response_lines.append("")
            return "\n".join(response_lines), True
        
        elif command in ["model force update", "force model update"]:
            # Force update with immediate reload
            model_name = config.EMBEDDING_MODEL_NAME
            response_lines = [f"\nForce updating model {model_name}..."]
            
            if VectorStoreManager._clear_model_cache_for_update(model_name):
                response_lines.append("Cache cleared successfully.")
                response_lines.append("Downloading latest model...")
                
                try:
                    # Force reload embeddings
                    VectorStoreManager._cached_embeddings = None
                    VectorStoreManager._cached_embeddings_model = None
                    
                    # Create new embeddings (will download fresh)
                    self.embeddings = VectorStoreManager.create_embeddings()
                    
                    # Update retrieval handler with new embeddings
                    self.retrieval_handler.embeddings = self.embeddings
                    
                    response_lines.append("Model updated successfully!")
                    response_lines.append("System is now using the latest model.")
                    
                except Exception as e:
                    response_lines.append(f"Failed to reload model: {e}")
                    response_lines.append("Please restart the application manually.")
            else:
                response_lines.append("Failed to clear cache.")
            
            response_lines.append("")
            return "\n".join(response_lines), True
        
        elif command in ["help production", "production help"]:
            help_text = [
                "\nProduction Model Management Commands:",
                "=" * 50,
                "model report          - Show model age and usage statistics",
                "model check           - Check for model updates on HuggingFace Hub",
                "model update          - Clear cache (requires restart to update)",
                "model force update    - Clear cache and immediately download latest",
                "help production       - Show this help",
                "stats                 - Show document and system statistics",
                "",
                "Tips:",
                "  • Models are checked for updates automatically every 30 days",
                "  • You'll be prompted when models are aging (60+ days old)",
                "  • Use 'model force update' for immediate updates",
                ""
            ]
            return "\n".join(help_text), True
        
        return None, True  # Command not handled

    def handle_system_commands(self, user_input: str) -> tuple[str, bool]:
        """
        Handle system commands for application control and information.
        
        Processes commands for help, statistics, session management,
        and application lifecycle operations.
        
        Args:
            user_input: User command input
            
        Returns:
            tuple: (response_text, should_continue)
        """
        command = user_input.lower().strip()
        
        if command in ['exit', 'quit']:
            print("\nGoodbye! Have a great day!")
            return None, False
        
        elif command == 'help':
            help_text = [
                "\nSara Commands:",
                "=" * 40,
                "Query Commands:",
                "  Just type your question naturally!",
                "",
                "System Commands:",
                "  help                 - Show this help",
                "  help production      - Show production model commands",
                "  stats                - Show document statistics",
                "  reindex              - Rebuild document index",
                "  clear                - Reset conversation memory",
                "  exit/quit            - Exit application",
                "",
                "Session Commands:",
                "  new session          - Create a new chat session",
                "  list sessions        - Show all sessions",
                "  switch session       - Change to different session",
                "  session stats        - Show session statistics", 
                "  clear session        - Clear current session memory",
                "",
                "Production Commands:",
                "  model report         - Show model status and age",
                "  model check          - Check for model updates",
                "  model update         - Update model cache",
                ""
            ]
            return "\n".join(help_text), True
        
        elif command == 'clear':
            self.memory.clear()
            return "\nConversation memory cleared!", True
        
        elif command == 'reindex':
            print("\nStarting document reindexing...")
            success = self.reindex_documents()
            if success:
                return "\nDocument reindexing completed successfully!", True
            else:
                return "\nDocument reindexing failed. Check logs for details.", True
        
        elif command == 'stats':
            # Enhanced stats with model information
            print("\nSystem Statistics:")
            print("=" * 40)
            
            # Document statistics
            if self.vector_store:
                VectorStoreManager.print_document_statistics(self.vector_store)
            
            # Model statistics
            model_report = VectorStoreManager.get_production_model_report()
            print(f"\n{model_report}")
            
            # Session statistics
            stats = self.session_manager.get_session_statistics()
            current_session = self.session_manager.current_session
            
            print(f"\nSession Statistics:")
            print(f"   Sessions: {stats['total_sessions']}/{stats['max_sessions']}")
            print(f"   Total Messages: {stats['total_messages']}")
            if current_session:
                print(f"   Current Session: {current_session.metadata.title}")
                print(f"   Current Messages: {stats['current_session_messages']}")
            
            return "", True
        
        elif command == 'new session':
            return self.create_new_session_interactive(), True
        
        elif command == 'list sessions':
            self.list_sessions_command()
            return "", True
        
        elif command == 'switch session':
            return self.switch_session_interactive(), True
        
        elif command == 'session stats':
            self.session_stats_command()
            return "", True
        
        elif command == 'clear session':
            if self.session_manager.current_session:
                self.session_manager.clear_current_session()
                return "\nCurrent session memory cleared!", True
            else:
                return "\nNo active session to clear.", True
        
        return None, True  # Command not handled

    def create_new_session_interactive(self) -> str:
        """
        Interactive session creation with user input.
        
        Prompts the user for a session title and creates a new session,
        updating all handlers to use the new session's memory.
        
        Returns:
            str: Status message about session creation
        """
        try:
            title = input("Enter session title (or press Enter for default): ").strip()
            session = self.session_manager.create_session(title if title else None)
            
            # Update memory reference
            self.memory = session.memory
            
            # Update handlers with new memory
            if self.conversation_handler:
                self.conversation_handler.memory = self.memory
            if self.retrieval_handler:
                self.retrieval_handler.memory = self.memory
            if self.query_router:
                self.query_router.memory = self.memory
            
            return f"\nCreated and switched to new session: {session.metadata.title}"
        except (KeyboardInterrupt, EOFError):
            return "\nSession creation cancelled."

    def switch_session_interactive(self) -> str:
        """
        Interactive session switching with user selection.
        
        Displays available sessions and allows the user to switch
        by session number or ID prefix.
        
        Returns:
            str: Status message about session switching
        """
        try:
            # Show available sessions
            sessions = self.session_manager.list_sessions()
            if not sessions:
                return "\nNo sessions available to switch to."
            
            print("\nAvailable Sessions:")
            for i, session in enumerate(sessions, 1):
                current_marker = "➤ " if (self.session_manager.current_session and 
                                        session.session_id == self.session_manager.current_session.metadata.session_id) else "  "
                print(f"{current_marker}{i}. {session.title} (ID: {session.session_id[:8]}...)")
            
            choice = input("\nEnter session number or ID: ").strip()
            
            # Handle numeric choice
            if choice.isdigit():
                session_num = int(choice)
                if 1 <= session_num <= len(sessions):
                    target_session = sessions[session_num - 1]
                    return self.switch_to_session(target_session.session_id)
                else:
                    return f"\nInvalid session number. Please choose 1-{len(sessions)}."
            
            # Handle ID choice
            elif choice:
                for session in sessions:
                    if session.session_id.startswith(choice):
                        return self.switch_to_session(session.session_id)
                return f"\nNo session found with ID starting with '{choice}'."
            
            else:
                return "\nSwitch cancelled."
                
        except (KeyboardInterrupt, EOFError):
            return "\nSwitch cancelled."

    def switch_to_session(self, session_id: str) -> str:
        """
        Switch to a specific session by ID.
        
        Loads the specified session and updates all handlers to use
        the session's conversation memory.
        
        Args:
            session_id: Unique identifier of the session to switch to
            
        Returns:
            str: Status message about the switch operation
        """
        session = self.session_manager.load_session(session_id)
        if not session:
            return f"\nSession {session_id} not found."
        
        # Update memory reference
        self.memory = session.memory
        
        # Update handlers with new memory
        if self.conversation_handler:
            self.conversation_handler.memory = self.memory
        if self.retrieval_handler:
            self.retrieval_handler.memory = self.memory
        if self.query_router:
            self.query_router.memory = self.memory
        
        return f"\nSwitched to session: {session.metadata.title} ({session.metadata.session_id[:8]}...)"
    
    def reindex_documents(self):
        """
        Reindex all documents in the data directory with improved error handling.
        
        Performs a complete rebuild of the vector store by loading all documents
        from the configured data path, processing them into chunks, and creating
        a fresh vector database.
        
        Returns:
            bool: True if reindexing was successful
        """
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
        if not VectorStoreManager.reset_chroma_db_with_permissions(config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            print("Failed to reset vector store. Check permissions.")
            return False
        
        # Load and process documents
        try:
            documents = DocumentProcessor.load_documents(config.DATA_PATH)
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
            
            # Add documents with better error handling
            try:
                # Sanitize metadata before adding
                sanitized_chunks = VectorStoreManager.sanitize_metadata(chunks)
                
                # Process documents in batches to avoid memory issues
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
                
            print(f"Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"Error during reindexing: {e}")
            return False

    def list_sessions_command(self):
        """
        Display a formatted list of all available sessions.
        
        Shows session titles, IDs, message counts, and last access times
        with clear indication of the currently active session.
        """
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
        """
        Display comprehensive session statistics.
        
        Shows total sessions, message counts, and current session information
        to help users understand their usage patterns.
        """
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
        """
        Run the interactive command line interface with production features.
        
        Main entry point for the CLI application that handles initialization,
        displays the interface, and manages the interactive loop for user queries
        and commands.
        """
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            return
            
        # Print banner and instructions
        print("\n" + "="*60)
        print("Sara - Your APU Knowledge Assistant")
        print("="*60)
        print("Ask questions about APU using natural language. I'm Sara, your friendly assistant.")
        print("Commands:")
        print("  - Type 'help' to see available commands")
        print("  - Type 'help production' for model management commands")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document and model statistics")
        print("  - Type 'new session' to create a new chat session")
        print("  - Type 'list sessions' to see all sessions")
        print("  - Type 'switch session' to change sessions")
        print("  - Type 'session stats' to see session statistics")
        print("  - Type 'clear session' to clear current session memory")
        print("="*60 + "\n")
        
        # Show current session info
        current_session = self.session_manager.current_session
        if current_session:
            print(f"Current Session: {current_session.metadata.title} ({current_session.metadata.session_id[:8]}...)")
            print(f"   Messages: {current_session.metadata.message_count} | Created: {current_session.metadata.created_at.strftime('%Y-%m-%d %H:%M')}\n")
        
        # Main interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("I'd be happy to help! Please ask me a question about APU services, such as:")
                    print("• Fee payments and financial information")
                    print("• Reference letters and documentation")
                    print("• IT support (APKey password, timetable access)")
                    print("• Library services and resources")
                    print("• Parking information")
                    print("• Visa and immigration matters")
                    continue
                
                # Handle commands first
                # Try production commands
                response, should_continue = self.handle_production_commands(query)
                if response is not None:
                    print(response)
                    if not should_continue:
                        break
                    continue
                
                # Try system commands
                response, should_continue = self.handle_system_commands(query)
                if response is not None:
                    if response:  # Only print if there's content
                        print(response)
                    if not should_continue:
                        break
                    continue
                
                # Process as regular query
                start_time = time.time()
                
                print("\nThinking...\n")
                
                try:
                    # Extract conversation context from current session for better follow-up handling
                    conversation_context = self._get_conversation_context()
                    
                    # Process and route query with conversational context
                    query_analysis = self.input_processor.analyze_query(query, conversation_context)
                    
                    # Route the query to appropriate handler with streaming enabled
                    response, should_continue = self.query_router.route_query(query_analysis, stream=True)
                    
                    if response:
                        # Handle streamed responses
                        if isinstance(response, types.GeneratorType) or (hasattr(response, '__iter__') and not isinstance(response, str)):
                            full_response = ""
                            for token in response:
                                print(token, end="", flush=True)
                                full_response += token
                            print()  
                            
                            # Save conversation to current session
                            if full_response.strip():
                                self.session_manager.add_conversation(query, full_response.strip())
                        else:
                            # Handle non-streamed responses
                            print(f"{response}")
                            
                            # Save conversation to current session
                            if response and response.strip():
                                self.session_manager.add_conversation(query, response.strip())
                    
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
    
    def _get_conversation_context(self) -> str:
        """
        Extract recent conversation history for conversational context enhancement.
        
        Returns conversation history from current session for better follow-up handling.
        This follows 2024-2025 conversational RAG best practices for contextual query reformulation.
        
        Returns:
            String containing recent conversation history, or empty string if no context
        """
        try:
            current_session = self.session_manager.get_current_session()
            if not current_session or not current_session.memory:
                return ""
            
            # Get recent messages (last 6 messages = last 3 exchanges)
            messages = current_session.memory.chat_memory.messages
            if len(messages) <= 1:  # Only system message or empty
                return ""
            
            # Extract last 6 messages (3 user-assistant pairs) for context
            recent_messages = messages[-6:] if len(messages) > 6 else messages[1:]  # Skip system message
            
            context_parts = []
            for message in recent_messages:
                if hasattr(message, 'content'):
                    # Limit each message to prevent context explosion
                    content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    context_parts.append(content)
            
            context = " ".join(context_parts)
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to extract conversation context: {e}")
            return ""
                
    def cleanup(self):
        """
        Clean up resources before shutdown.
        
        Ensures proper cleanup of all resources including sessions,
        database connections, and memory to prevent resource leaks.
        """
        logger.info("Cleaning up resources")
        
        # Save current session before cleanup
        if self.session_manager and self.session_manager.current_session:
            self.session_manager.save_current_session()
        
        # Close ChromaDB client
        ChromaDBManager.close()
        
        # Clear vector store reference
        self.vector_store = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Cleanup completed")