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
        
        # Unknown command
        else:
            return f"Unknown command: {command}. Type 'help' to see available commands.", True