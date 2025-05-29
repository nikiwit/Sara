"""
Data types and models for session management.
"""

import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

@dataclass
class SessionMetadata:
    """Metadata for a chat session."""
    session_id: str
    title: str
    created_at: datetime
    last_accessed: datetime
    message_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'message_count': self.message_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary."""
        return cls(
            session_id=data['session_id'],
            title=data['title'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            message_count=data['message_count']
        )

class ChatSession:
    """Represents a single chat session with isolated memory."""
    
    def __init__(self, session_id: str, title: str = None):
        self.metadata = SessionMetadata(
            session_id=session_id,
            title=title or f"Chat {session_id[:8]}",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            message_count=0
        )
        
        # Create isolated memory for this session
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Add system message
        system_message = SystemMessage(
            content="I am an AI assistant that helps with answering questions about APU. I can provide information about academic procedures, administrative processes, and university services."
        )
        self.memory.chat_memory.messages.append(system_message)
    
    def add_message(self, human_message: str, ai_message: str):
        """Add a conversation exchange to the session."""
        self.memory.chat_memory.add_user_message(human_message)
        self.memory.chat_memory.add_ai_message(ai_message)
        self.metadata.message_count += 1
        self.metadata.last_accessed = datetime.now()
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history for this session."""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear the conversation memory while keeping system message."""
        system_messages = [msg for msg in self.memory.chat_memory.messages 
                          if isinstance(msg, SystemMessage)]
        self.memory.chat_memory.clear()
        for msg in system_messages:
            self.memory.chat_memory.messages.append(msg)
        self.metadata.message_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        messages_data = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_data.append({"type": "ai", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages_data.append({"type": "system", "content": msg.content})
        
        return {
            "metadata": self.metadata.to_dict(),
            "messages": messages_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Deserialize session from dictionary."""
        metadata = SessionMetadata.from_dict(data["metadata"])
        session = cls(metadata.session_id, metadata.title)
        session.metadata = metadata
        
        # Restore memory
        session.memory.chat_memory.clear()
        for msg_data in data["messages"]:
            if msg_data["type"] == "human":
                session.memory.chat_memory.add_user_message(msg_data["content"])
            elif msg_data["type"] == "ai":
                session.memory.chat_memory.add_ai_message(msg_data["content"])
            elif msg_data["type"] == "system":
                session.memory.chat_memory.messages.append(
                    SystemMessage(content=msg_data["content"])
                )
        
        return session