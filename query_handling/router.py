"""
Query routing logic for directing queries to appropriate handlers.
"""

import re
import time
import logging
from typing import Dict, Any, Tuple, Union, Iterator
from apurag_types import QueryType
from config import Config

logger = logging.getLogger("CustomRAG")

class QueryRouter:
    """Routes queries to appropriate handlers based on query type with conversational support."""
    
    def __init__(self, conversation_handler, retrieval_handler, command_handler, memory=None):
        """Initialize with handlers for different query types."""
        self.conversation_handler = conversation_handler
        self.retrieval_handler = retrieval_handler
        self.command_handler = command_handler
        self.memory = memory 
    
    def route_query(self, query_analysis: Dict[str, Any], stream=False) -> Tuple[Any, bool]:
        """
        Query routing with conversational priority handling.
        
        Args:
            query_analysis: Analysis output from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Tuple of (handler_result, should_continue)
        """
        original_query = query_analysis["original_query"]
        query_type = query_analysis.get("query_type", QueryType.UNKNOWN)
        
        if self._is_identity_query(original_query):
            logger.info(f"Processing system identity query: {original_query}")
            response = self._handle_identity_query(original_query, stream=stream)
            return response, True
        
        if self._is_help_request(original_query):
            logger.info(f"Processing help request: {original_query}")
            response = self._handle_help_request(original_query, stream=stream)
            return response, True
        
        # Check for conversational queries AFTER identity queries
        if self._is_conversational_query(original_query):
            logger.info(f"Processing conversational query: {original_query}")
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        if query_type == QueryType.COMMAND:
            logger.info(f"Processing system command: {original_query}")
            # Handle system commands (not streaming these)
            return self.command_handler.handle_command(original_query)
        
        elif query_type == QueryType.CONVERSATIONAL:
            logger.info(f"Processing legacy conversational query: {original_query}")
            # Handle conversational queries (now with streaming support)
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        else:
            logger.info(f"Processing {query_type.value} query through RAG pipeline: {original_query}")
            # Handle all other query types with retrieval system
            response = self.retrieval_handler.process_query(query_analysis, stream=stream)
            return response, True
    
    def _is_conversational_query(self, query: str) -> bool:
        """
        Conversational query detection that catches more patterns.
        Uses the ConversationHandler's detection logic for consistency.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's a conversational query that should be handled by ConversationHandler
        """
        # Delegate to the ConversationHandler's detection
        if hasattr(self.conversation_handler, 'is_conversational_query'):
            return self.conversation_handler.is_conversational_query(query)
        
        conversational_patterns = [
            # Greetings and social
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening))\b',
            r'\bhow\s+are\s+you\b',
            r'\bhow\s+are\s+you\s+doing\b',
            r'\bhow\s+is\s+your\s+day\b',
            r'\bnice\s+to\s+meet\s+you\b',
            r'\bgood\s+to\s+see\s+you\b',
            r'\bhave\s+a\s+good\s+day\b',
            
            r'\bbye\b',
            r'\bgoodbye\b',
            r'\bfarewell\b',
            r'\btake\s+care\b',
            r'\bsee\s+you\s+later\b',
            r'\bsee\s+you\s+soon\b',
            r'\bcatch\s+you\s+later\b',
            r'\bgotta\s+go\b',
            r'\bi\s+have\s+to\s+go\b',
            r'\bi\s+need\s+to\s+leave\b',
            r'\btalking\s+to\s+you\s+later\b',
            r'\buntil\s+next\s+time\b',
            
            # System knowledge/capabilities
            r'\bwhat\s+do\s+you\s+know\b',
            r'\bwhat\s+can\s+you\s+do\b',
            r'\bwhat\s+are\s+you\s+capable\s+of\b',
            r'\bwhat\s+information\s+do\s+you\s+have\b',
            r'\btell\s+me\s+about\s+yourself\b',
            r'\bwhat\s+are\s+your\s+capabilities\b',
            r'\bwhat\s+kind\s+of\s+questions\s+can\s+you\s+answer\b',
            r'\bwhat\s+topics\s+can\s+you\s+help\s+with\b',
            
            # Acknowledgments and responses
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
            r'\b(?:awesome|great|cool|nice|perfect|excellent|wonderful)\b',
            r'\bthat\s*(?:helps|helped)\b',
            r'\bgot\s*it\b',
            r'\bthat\'s\s+(?:great|good|perfect|helpful)\b',
            r'\bexactly\s+what\s+I\s+needed\b',
            
            # Clarification requests
            r'\bi\s+don\'t\s+understand\b',
            r'\bcan\s+you\s+explain\b',
            r'\bwhat\s+do\s+you\s+mean\b',
            r'\bi\'m\s+confused\b',
            r'\bcan\s+you\s+clarify\b',
            r'\bcan\s+you\s+help\s+me\s+understand\b',
            r'\bi\s+need\s+more\s+information\b',
            r'\bcan\s+you\s+be\s+more\s+specific\b',
        ]
        
        query_lower = query.lower().strip()
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.debug(f"Matched conversational pattern: {pattern} in query: {query}")
                return True
        
        return False
    
    def _is_identity_query(self, query: str) -> bool:
        """
        Check if query is asking about the system itself.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's an identity query
        """
        identity_patterns = [
            r'\bwho\s+are\s+you\b',
            r'\bwhat\s+are\s+you\b',
            r'\bwhat\s+is\s+your\s+name\b',
            r'\bwhat\'s\s+your\s+name\b',
            r'\bwhat\s+model\s+are\s+you\b',
            r'\bwhat\s+kind\s+of\s+(?:bot|assistant|ai)\s+are\s+you\b',
            r'\bintroduce\s+yourself\b',
            r'\btell\s+me\s+about\s+this\s+system\b',
            r'\bwhat\s+is\s+this\b',
            r'\bare\s+you\s+(?:human|real|ai|bot)\b',
            r'\bwhat\s+type\s+of\s+assistant\s+are\s+you\b',
        ]
        
        query_lower = query.lower()
        for pattern in identity_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.debug(f"Router matched identity pattern: {pattern} in query: {query}")
                return True
        
        logger.debug(f"Router found no identity patterns in: {query}")
        return False
    
    def _is_help_request(self, query: str) -> bool:
        """
        Check if query is a general help request.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's a help request
        """
        help_patterns = [
            r'\bhelp\s*me\b',
            r'\bi\s+need\s+help\b',
            r'\bcan\s+you\s+help\b',
            r'\bwhere\s+do\s+I\s+start\b',
            r'\bwhat\s+should\s+I\s+do\b',
            r'\bhow\s+do\s+I\s+begin\b',
            r'\bi\s+don\'t\s+know\s+where\s+to\s+start\b',
            r'\bi\'m\s+lost\b',
            r'\bi\s+need\s+guidance\b',
            r'\bshow\s+me\s+around\b',
            r'\bwhat\s+are\s+my\s+options\b',
        ]
        
        query_lower = query.lower()
        for pattern in help_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        return False
    
    def _handle_identity_query(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """
        Handle identity queries about the system with context awareness.
        
        Args:
            query: Original query string
            stream: Whether to stream the response
            
        Returns:
            Response string or iterator for streaming
        """
        # Check if user has interacted before
        is_return_user = self.memory and len(self.memory.chat_memory.messages) > 0
        
        if is_return_user:
            response = """Since we've chatted before, you know I'm an AI assistant built specifically for APU (Asia Pacific University)! 

I'm here to help with:
• Academic programs and course information
• Administrative procedures and services  
• Campus facilities and locations
• Student services and support
• Financial information and policies

I use APU's official knowledge base to provide accurate, up-to-date information. What specific APU topic can I help you explore further?"""
        else:
            response = """Hello! I'm an AI assistant built specifically for APU (Asia Pacific University). 

🎓 **My Purpose:** I'm designed to help students, staff, and visitors find information about APU services, procedures, and policies.

📚 **What I Can Help With:**
• Academic programs and courses
• Administrative procedures and services  
• Campus facilities and locations
• Student services and support
• Financial information and policies
• Medical insurance and benefits

🔍 **How I Work:** I use APU's official knowledge base to provide accurate information, so you can trust that what I tell you is current and official.

**Ready to help!** What would you like to know about APU?"""
        
        if not stream:
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
            return response
        else:
            # Update memory for streaming
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
            return self._stream_text_response(response)
    
    def _handle_help_request(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """
        Handle general help requests with personalized guidance.
        
        Args:
            query: Original query string
            stream: Whether to stream the response
            
        Returns:
            Response string or iterator for streaming
        """
        # Check conversation history for context
        is_return_user = self.memory and len(self.memory.chat_memory.messages) > 0
        recent_topics = self._extract_recent_topics() if is_return_user else ""
        
        if is_return_user and recent_topics:
            response = f"""I'd be happy to help! I see we were discussing {recent_topics}. 

**Would you like to:**
• Continue exploring {recent_topics}
• Ask about a different APU topic
• Get step-by-step guidance for a specific procedure

**Popular APU topics I can help with:**
📚 Academics - courses, programs, requirements
🏢 Administration - applications, registration, forms  
💰 Financial - fees, scholarships, payments
🏥 Services - medical insurance, student support
🏫 Campus - facilities, locations, contact info

What specific area would be most helpful for you right now?"""
        else:
            response = """I'm here to help with any APU-related questions! 

**Not sure where to start? Here are popular topics:**

🎓 **For Students:**
• Course information and academic requirements
• Registration and enrollment procedures
• Medical insurance and student services
• Fee payments and financial aid

🏢 **For Administrative Tasks:**
• Application processes and forms
• Campus facilities and locations
• Contact information for departments
• Policy and procedure questions

**Just ask me about any of these topics, or feel free to ask specific questions like:**
• "How do I collect my medical insurance card?"
• "What are the requirements for [specific program]?"
• "Where is [specific building/office]?"

What would you like to know about APU?"""
        
        if not stream:
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
            return response
        else:
            # Update memory for streaming
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
            return self._stream_text_response(response)
    
    def _extract_recent_topics(self) -> str:
        """
        Extract topics from recent conversation history to provide context.
        
        Returns:
            String describing recent topics discussed
        """
        if not self.memory or not self.memory.chat_memory.messages or len(self.memory.chat_memory.messages) < 2:
            return ""
        
        # Look at the last few messages for keywords
        recent_messages = self.memory.chat_memory.messages[-4:]  # Last 4 messages
        text_content = []
        
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                text_content.append(msg.content.lower())
        
        combined_text = " ".join(text_content)
        
        # Common APU topic keywords to identify
        topic_keywords = {
            'academic programs': ['course', 'program', 'degree', 'academic', 'study', 'class', 'subject', 'curriculum'],
            'administrative procedures': ['application', 'registration', 'enrollment', 'form', 'procedure', 'process', 'admin'],
            'financial matters': ['fee', 'payment', 'scholarship', 'financial', 'cost', 'tuition', 'money'],
            'student services': ['medical', 'insurance', 'card', 'collect', 'service', 'support'],
            'campus facilities': ['building', 'location', 'address', 'campus', 'library', 'lab', 'facility'],
        }
        
        # Find which topics were mentioned
        mentioned_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                mentioned_topics.append(topic)
        
        if mentioned_topics:
            if len(mentioned_topics) == 1:
                return mentioned_topics[0]
            elif len(mentioned_topics) == 2:
                return f"{mentioned_topics[0]} and {mentioned_topics[1]}"
            else:
                return f"{', '.join(mentioned_topics[:-1])}, and {mentioned_topics[-1]}"
        
        return "APU topics"
    
    def _stream_text_response(self, text: str) -> Iterator[str]:
        """
        Stream text word by word with consistent delay.
        
        Args:
            text: Full text to stream
            
        Returns:
            Iterator yielding text chunks
        """
        words = text.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(Config.STREAM_DELAY)