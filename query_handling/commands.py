"""
System command handling for administrative operations.
"""

import io
import logging
from contextlib import redirect_stdout
from typing import Tuple

logger = logging.getLogger("Sara")

class CommandHandler:
    """Handles system commands."""
    
    def __init__(self, rag_system):
        """Initialize with a reference to the RAG system."""
        self.rag_system = rag_system
        self.api_mode = False  # Flag to indicate if running via API
    
    def set_api_mode(self, api_mode: bool):
        """Set whether the handler is running in API mode."""
        self.api_mode = api_mode
    
    def handle_command(self, command: str) -> Tuple[str, bool]:
        """
        Handles system commands.
        
        Args:
            command: The command string
            
        Returns:
            Tuple of (response, should_continue)
        """
        command_lower = command.lower().strip()
        
        # For API mode, convert administrative commands to knowledge base queries
        api_blocked_commands = [
            'reindex', 'stats', 'help', 'clear', 'exit', 'quit', 'bye', 'goodbye',
            'new session', 'create session', 'start new chat', 'list sessions', 
            'show sessions', 'sessions', 'switch session', 'change session', 
            'load session', 'session stats', 'session statistics', 'clear session', 
            'reset session', 'semantic stats', 'semantic statistics', 'menu', 'commands'
        ]
        
        if self.api_mode and command_lower in api_blocked_commands:
            if command_lower in ['help', 'menu', 'commands']:
                return "I'm SARA, your Academic Retrieval Assistant! I can answer questions about APU policies, procedures, and services. What would you like to know?", True
            elif command_lower == 'stats':
                return "I have access to comprehensive information about APU including academic procedures, library services, IT support, and more. What specific topic can I help you with?", True
            elif command_lower == 'reindex':
                return "I keep my knowledge base up to date automatically. Is there something specific about APU that I can help you find?", True
            elif command_lower in ['exit', 'quit', 'bye', 'goodbye']:
                return "Thanks for chatting! Feel free to ask me anything about APU anytime. Have a great day! 😊", True
            elif command_lower == 'clear':
                return "I'm ready for a fresh conversation! What would you like to know about APU?", True
            elif command_lower in ['new session', 'create session', 'start new chat']:
                return "Every conversation with me is fresh and focused! What APU topic can I help you with?", True
            elif command_lower in ['list sessions', 'show sessions', 'sessions']:
                return "I'm here to help you with any questions about APU. What would you like to know?", True
            elif command_lower in ['switch session', 'change session', 'load session']:
                return "You can always start a new topic with me! What APU information are you looking for?", True
            elif command_lower in ['session stats', 'session statistics']:
                return "I'm ready to provide you with information about APU. What specific area interests you?", True
            elif command_lower in ['clear session', 'reset session']:
                return "Starting fresh! How can I help you with APU information today?", True
            elif command_lower in ['semantic stats', 'semantic statistics']:
                return "I use advanced language processing to understand your questions about APU. What can I help you find?", True
            else:
                return "I'm here to help with questions about APU. What would you like to know?", True
        
        # Help command
        if command_lower in ["help", "menu", "commands"]:
            help_text = """
            Available Commands:
            - help: Display this help menu
            - exit, quit: Stop the application
            - clear: Reset the conversation memory
            - reindex: Reindex all documents
            - stats: See document statistics
            - semantic stats: Show semantic processor statistics
            - new session: Create a new chat session
            - list sessions: Show all available sessions
            - switch session: Change to a different session
            - session stats: Show session statistics
            - clear session: Clear current session memory
            """
            return help_text, True
        
        # Exit commands
        elif command_lower in ["exit", "quit", "bye", "goodbye"]:
            return "Goodbye! Have a great day!", False
            
        # Clear memory command
        elif command_lower == "clear":
            # Reset memory but keep system message
            system_message = self.rag_system.memory.chat_memory.messages[0] if self.rag_system.memory.chat_memory.messages else None
            self.rag_system.memory.clear()
            if system_message:
                self.rag_system.memory.chat_memory.messages.append(system_message)
            return "Conversation memory has been reset.", True
            
        # Stats command
        elif command_lower == "stats":
            # Import here to avoid circular import
            from vector_management.manager import VectorStoreManager
            
            # Capture printed output
            f = io.StringIO()
            with redirect_stdout(f):
                VectorStoreManager.print_document_statistics(self.rag_system.vector_store)
            
            output = f.getvalue()
            if not output.strip():
                output = "No document statistics available."
                
            return output, True
            
        # Reindex command
        elif command_lower == "reindex":
            result = self.rag_system.reindex_documents()
            if result:
                return "Documents have been successfully reindexed.", True
            else:
                return "Failed to reindex documents. Check the log for details.", True

        # Session management commands
        elif command_lower in ["new session", "create session", "start new chat"]:
            success = self.rag_system.switch_session()
            if success:
                return "✅ Created new session successfully!", True
            else:
                return "❌ Failed to create new session.", True

        elif command_lower in ["list sessions", "show sessions", "sessions"]:
            self.rag_system.list_sessions_command()
            return "", True

        elif command_lower in ["switch session", "change session", "load session"]:
            success = self._handle_switch_session_interactive()
            if success:
                return "✅ Session switched successfully!", True
            else:
                return "❌ Failed to switch session.", True

        elif command_lower in ["session stats", "session statistics"]:
            self.rag_system.session_stats_command()
            return "", True

        elif command_lower in ["clear session", "reset session"]:
            self.rag_system.session_manager.clear_current_session_memory()
            return "✅ Session memory cleared!", True
        
        elif command_lower in ["semantic stats", "semantic statistics"]:
            return self._handle_semantic_stats(), True
        
        # Unknown command
        else:
            return f"Unknown command: {command}. Type 'help' to see available commands.", True

    def _handle_switch_session_interactive(self):
        """Handle interactive session switching."""
        try:
            self.rag_system.list_sessions_command()
            session_input = input("\nEnter session ID (first 8 chars) or press Enter to create new: ").strip()
            
            if not session_input:
                # Create new session
                return self.rag_system.switch_session()
            
            # Find full session ID from partial
            sessions = self.rag_system.session_manager.list_sessions()
            for session in sessions:
                if session.session_id.startswith(session_input):
                    return self.rag_system.switch_session(session.session_id)
            
            print(f"Session starting with '{session_input}' not found.")
            return False
            
        except Exception as e:
            logger.error(f"Error in interactive session switch: {e}")
            return False
    
    def _handle_semantic_stats(self) -> str:
        """Handle semantic processor statistics command."""
        try:
            # Get stats from input processor
            input_processor = getattr(self.rag_system, 'input_processor', None)
            if not input_processor:
                return "❌ Input processor not available"
            
            spacy_processor = getattr(input_processor, 'spacy_processor', None)
            if not spacy_processor:
                return "ℹ️ Semantic processor not enabled or not available"
            
            stats = spacy_processor.get_statistics()
            
            report = "\n📊 Semantic Processor Statistics:\n"
            report += "=" * 50 + "\n"
            report += f"🔧 Status: {'✅ Healthy' if spacy_processor.is_healthy() else '❌ Unhealthy'}\n"
            report += f"🔄 Initialized: {'✅ Yes' if stats['initialized'] else '❌ No'}\n"
            report += f"🧠 Model loaded: {'✅ Yes' if stats['model_loaded'] else '❌ No'}\n"
            report += f"📂 Domain clusters: {stats['domain_clusters']}\n"
            report += f"🔨 Grammar patterns: {stats['grammar_patterns']}\n"
            report += f"⚠️ Error count: {stats['error_count']}/{stats['max_errors']}\n"
            
            # Processing stats
            processing_stats = stats['stats']
            report += f"\n📈 Processing Statistics:\n"
            report += f"   • Queries processed: {processing_stats['queries_processed']}\n"
            report += f"   • Errors encountered: {processing_stats['errors_encountered']}\n"
            report += f"   • Avg processing time: {processing_stats['average_processing_time']:.3f}s\n"
            
            if processing_stats['last_error']:
                report += f"   • Last error: {processing_stats['last_error']}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting semantic stats: {e}")
            return f"❌ Error retrieving semantic statistics: {e}"