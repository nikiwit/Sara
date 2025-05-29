"""
System command handling for administrative operations.
"""

import io
import logging
from contextlib import redirect_stdout
from typing import Tuple

logger = logging.getLogger("CustomRAG")

class CommandHandler:
    """Handles system commands."""
    
    def __init__(self, rag_system):
        """Initialize with a reference to the RAG system."""
        self.rag_system = rag_system
    
    def handle_command(self, command: str) -> Tuple[str, bool]:
        """
        Handles system commands.
        
        Args:
            command: The command string
            
        Returns:
            Tuple of (response, should_continue)
        """
        command_lower = command.lower().strip()
        
        # Help command
        if command_lower in ["help", "menu", "commands"]:
            help_text = """
            Available Commands:
            - help: Display this help menu
            - exit, quit: Stop the application
            - clear: Reset the conversation memory
            - reindex: Reindex all documents
            - stats: See document statistics
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