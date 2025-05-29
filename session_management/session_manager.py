"""
Main session manager that handles session lifecycle and limits.
"""

import uuid
import logging
from typing import Optional, List
from datetime import datetime
from .session_storage import SessionStorage
from .session_types import ChatSession, SessionMetadata

logger = logging.getLogger("CustomRAG")

class SessionManager:
    """Manages chat sessions with automatic cleanup and limits."""
    
    def __init__(self, max_sessions: int = 5, storage_dir: str = "./sessions"):
        self.max_sessions = max_sessions
        self.storage = SessionStorage(storage_dir)
        self.current_session: Optional[ChatSession] = None
    
    def create_session(self, title: str = None) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, title)
        
        # Check if we need to remove old sessions
        self._enforce_session_limit()
        
        # Save the new session
        self.storage.save_session(session)
        self.current_session = session
        
        logger.info(f"Created new session: {session_id} ({session.metadata.title})")
        return session
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load an existing session."""
        session = self.storage.load_session(session_id)
        if session:
            self.current_session = session
            logger.info(f"Loaded session: {session_id} ({session.metadata.title})")
        else:
            logger.warning(f"Session {session_id} not found")
        return session
    
    def delete_session(self, session_id: str):
        """Delete a specific session."""
        if self.current_session and self.current_session.metadata.session_id == session_id:
            self.current_session = None
        
        self.storage.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
    
    def list_sessions(self) -> List[SessionMetadata]:
        """List all available sessions."""
        return self.storage.list_sessions()
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the currently active session."""
        return self.current_session
    
    def save_current_session(self):
        """Save the current session to disk."""
        if self.current_session:
            self.storage.save_session(self.current_session)
    
    def _enforce_session_limit(self):
        """Remove oldest sessions if we exceed the limit."""
        sessions = self.storage.list_sessions()
        
        while len(sessions) >= self.max_sessions:
            oldest_session_id = self.storage.get_oldest_session_id()
            if oldest_session_id:
                self.storage.delete_session(oldest_session_id)
                logger.info(f"Removed oldest session {oldest_session_id} to stay within limit")
                sessions = self.storage.list_sessions()
            else:
                break
    
    def add_conversation(self, human_message: str, ai_message: str):
        """Add a conversation exchange to the current session."""
        if self.current_session:
            self.current_session.add_message(human_message, ai_message)
            self.storage.save_session(self.current_session)
    
    def clear_current_session_memory(self):
        """Clear the memory of the current session."""
        if self.current_session:
            self.current_session.clear_memory()
            self.storage.save_session(self.current_session)
            logger.info(f"Cleared memory for session {self.current_session.metadata.session_id}")
    
    def get_session_statistics(self) -> dict:
        """Get statistics about sessions."""
        sessions = self.storage.list_sessions()
        total_messages = sum(s.message_count for s in sessions)
        
        return {
            "total_sessions": len(sessions),
            "max_sessions": self.max_sessions,
            "total_messages": total_messages,
            "current_session_id": self.current_session.metadata.session_id if self.current_session else None,
            "current_session_messages": self.current_session.metadata.message_count if self.current_session else 0
        }