"""
Conversational query handling for social interactions.
"""

import re
import random
import time
import logging

from config import Config

logger = logging.getLogger("CustomRAG")

class ConversationHandler:
    """Handles conversational queries that don't require document retrieval."""
    
    def __init__(self, memory, stream_delay=None):
        """Initialize with a memory for conversation history."""
        self.memory = memory
        self.stream_delay = stream_delay if stream_delay is not None else Config.STREAM_DELAY
        
        # Greeting patterns and responses
        self.greeting_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
        ]

        self.greeting_responses = [
            "Hello! I'm your APU knowledge base assistant. How can I help you with your questions about APU today?",
            "Hi there! I'm ready to help answer questions about APU. What would you like to know?",
            "Greetings! I'm here to assist with information about APU. What are you looking for?",
            "Hello! I'm your APU RAG assistant. I can help you find information about academics, administrative processes, and more.",
            "Hi! I'm ready to help you navigate APU-related questions. What would you like to learn about?"
        ]

        # Acknowledgement patterns and responses
        self.acknowledgement_patterns = [
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
            r'\b(?:awesome|great|cool|nice)\b',
            r'\bthat\s*(?:helps|helped)\b',
            r'\bgot\s*it\b',
        ]

        self.acknowledgement_responses = [
            "You're welcome! Is there anything else you'd like to know about APU?",
            "Happy to help! Let me know if you have any other questions about APU.",
            "My pleasure! Feel free to ask if you need anything else.",
            "Glad I could assist. Any other questions about APU?",
            "You're welcome! I'm here if you need more information about APU procedures, policies, or services."
        ]
    
    def handle_conversation(self, query: str, stream=False) -> str:
        """
        Handles conversational queries.
        
        Args:
            query: The user's query
            stream: Whether to stream the response
            
        Returns:
            Either a string response or an iterator for streaming
        """
        query_lower = query.lower().strip()
        response = None
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                response = random.choice(self.greeting_responses)
                break
        
        # Check for acknowledgements
        if not response:
            for pattern in self.acknowledgement_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    response = random.choice(self.acknowledgement_responses)
                    break
        
        # If no specific pattern matched, give a generic response
        if not response:
            response = "I'm here to help you with information about APU. What would you like to know?"
        
        # Update conversation memory - do this for non-streaming case
        if not stream:
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            return response
        
        # For streaming, use the Ollama streaming approach so UI shows smooth typing
        if stream:
            # Store response for updating memory after streaming completes
            self._last_response = response
            self._last_query = query
            
            # Return iterator that simulates streaming for simple responses
            return self._stream_response(response)

    def _stream_response(self, response: str):
        """
        Simulates streaming for conversational responses.
        
        Args:
            response: The full response to stream
            
        Returns:
            An iterator yielding tokens
        """
        # Update memory before starting to stream
        self.memory.chat_memory.add_user_message(self._last_query)
        self.memory.chat_memory.add_ai_message(response)
        
        # Stream word by word with consistent delay (like LLM tokens)
        words = response.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(self.stream_delay)