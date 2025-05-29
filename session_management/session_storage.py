"""
Session storage and persistence management.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from .session_types import ChatSession, SessionMetadata

logger = logging.getLogger("CustomRAG")

class SessionStorage:
    """Handles session persistence to disk."""
    
    def __init__(self, storage_dir: str = "./sessions"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "sessions_metadata.json")
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist."""
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load sessions metadata from disk."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load sessions metadata: {e}")
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Dict]):
        """Save sessions metadata to disk."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving sessions metadata: {e}")
    
    def _get_session_file_path(self, session_id: str) -> str:
        """Get file path for a session."""
        return os.path.join(self.storage_dir, f"session_{session_id}.json")
    
    def save_session(self, session: ChatSession):
        """Save a session to disk."""
        try:
            # Save session data
            session_file = self._get_session_file_path(session.metadata.session_id)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata[session.metadata.session_id] = session.metadata.to_dict()
            self._save_metadata(metadata)
            
            logger.debug(f"Saved session {session.metadata.session_id}")
        except Exception as e:
            logger.error(f"Error saving session {session.metadata.session_id}: {e}")
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from disk."""
        try:
            session_file = self._get_session_file_path(session_id)
            if not os.path.exists(session_file):
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = ChatSession.from_dict(session_data)
            session.metadata.last_accessed = datetime.now()
            self.save_session(session)  # Update last accessed time
            
            logger.debug(f"Loaded session {session_id}")
            return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str):
        """Delete a session from disk."""
        try:
            # Remove session file
            session_file = self._get_session_file_path(session_id)
            if os.path.exists(session_file):
                os.remove(session_file)
            
            # Remove from metadata
            metadata = self._load_metadata()
            if session_id in metadata:
                del metadata[session_id]
                self._save_metadata(metadata)
            
            logger.info(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
    
    def list_sessions(self) -> List[SessionMetadata]:
        """List all sessions ordered by last accessed time."""
        metadata = self._load_metadata()
        sessions = []
        
        for session_data in metadata.values():
            try:
                sessions.append(SessionMetadata.from_dict(session_data))
            except Exception as e:
                logger.warning(f"Error parsing session metadata: {e}")
        
        # Sort by last accessed time (most recent first)
        sessions.sort(key=lambda s: s.last_accessed, reverse=True)
        return sessions
    
    def get_oldest_session_id(self) -> Optional[str]:
        """Get the ID of the oldest session."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[-1].session_id  # Last in the sorted list