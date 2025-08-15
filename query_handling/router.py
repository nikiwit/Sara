"""
Query routing logic for directing queries to appropriate handlers.
"""

import re
import time
import logging
from typing import Dict, Any, Tuple, Union, Iterator
from sara_types import QueryType
from config import Config

logger = logging.getLogger("Sara")

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
        Query routing with conversational priority handling and improved error handling.
        
        Args:
            query_analysis: Analysis output from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Tuple of (handler_result, should_continue)
        """
        original_query = query_analysis["original_query"]
        query_type = query_analysis.get("query_type", QueryType.UNKNOWN)
        
        # Enhanced input validation and error handling
        try:
            # Handle empty or whitespace-only queries
            if not original_query or not original_query.strip():
                response = self._handle_empty_query(stream=stream)
                return response, True
            
            # Handle inappropriate content (cheating, academic dishonesty)
            if self._is_inappropriate_request(original_query):
                response = self._handle_inappropriate_request(original_query, stream=stream)
                return response, True
            
            # Handle very short queries that might be unclear (but exempt simple conversational ones)
            if len(original_query.strip()) < 3 and not self._is_simple_conversational_query(original_query):
                response = self._handle_unclear_query(original_query, stream=stream)
                return response, True
            
            # Handle very long queries that might be complex or contain multiple questions
            if len(original_query) > 500:
                response = self._handle_long_query(original_query, stream=stream)
                return response, True
                
        except Exception as e:
            logger.error(f"Error in query validation: {e}")
            response = self._handle_system_error(stream=stream)
            return response, True
        
        # Simplified routing logic with clear precedence (based on 2025 best practices)
        # 1. Priority routing for system queries
        if self._is_identity_query(original_query):
            logger.info(f"Processing system identity query: {original_query}")
            response = self._handle_identity_query(original_query, stream=stream)
            return response, True
        
        # 2. Simple conversational queries (greetings, thanks, etc.) get priority
        if self._is_simple_conversational_query(original_query):
            logger.info(f"Processing simple conversational query: {original_query}")
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        # 3. Help requests
        if self._is_help_request(original_query):
            logger.info(f"Processing help request: {original_query}")
            response = self._handle_help_request(original_query, stream=stream)
            return response, True
        
        # 4. Complex conversational queries (after simple ones)
        if self._is_conversational_query(original_query):
            logger.info(f"Processing conversational query: {original_query}")
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        # 5. Early boundary detection for out-of-scope queries
        if self._is_boundary_query(original_query):
            logger.info(f"Processing boundary query: {original_query}")
            response = self._handle_boundary_query(original_query, stream=stream)
            return response, True
        
        # Check for session commands explicitly before query_type check
        if self._is_session_command(original_query):
            logger.info(f"Processing system command: {original_query}")
            # Handle system commands (not streaming these)
            return self.command_handler.handle_command(original_query)
        
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
            # Handle query_type safely whether it's enum or string
            if hasattr(query_type, 'value'):
                query_type_str = query_type.value
            else:
                query_type_str = str(query_type)
            logger.info(f"Processing {query_type_str} query through RAG pipeline: {original_query}")
            # Handle all other query types with retrieval system
            response = self.retrieval_handler.process_query(query_analysis, stream=stream)
            return response, True
    
    def _is_simple_conversational_query(self, query: str) -> bool:
        """
        Check if query is a simple conversational query (greetings, thanks, farewells).
        These get priority routing to ensure consistent handling.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's a simple conversational query
        """
        simple_patterns = [
            # Simple greetings - highest priority
            r'^(?:hi+|h[ie]llo*|hey+|hiya?|sup|hai|helo+)$',
            r'^\s*(?:hi+|h[ie]llo*|hey+|hiya?|sup|hai|helo+)\s*$',
            # Simple thanks
            r'^(?:thanks?|thank\s*(?:you|u)|ty|thx|thanx)$',
            r'^\s*(?:thanks?|thank\s*(?:you|u)|ty|thx|thanx)\s*$',
            # Simple farewells
            r'^(?:bye|goodbye|farewell|see\s+(?:you|u))$',
            r'^\s*(?:bye|goodbye|farewell|see\s+(?:you|u))\s*$',
        ]
        
        query_clean = query.lower().strip()
        for pattern in simple_patterns:
            if re.match(pattern, query_clean):
                logger.debug(f"Matched simple conversational pattern: {pattern}")
                return True
        
        return False
    
    def _is_boundary_query(self, query: str) -> bool:
        """
        Check if query is an out-of-scope query that should get boundary response.
        This prevents these queries from going to retrieval and potentially failing.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's a boundary query that should be handled directly
        """
        query_lower = query.lower()
        
        # Boundary query keywords - queries that should NOT go to retrieval
        boundary_patterns = [
            # Weather and time
            ['weather', 'temperature', 'climate', 'rain', 'sunny', 'hot', 'cold'],
            ['time', 'current time', 'what time', 'clock'],
            ['date', 'today', 'tomorrow', 'yesterday', 'calendar', 'day of the week', 'what day'],
            
            # News and current events  
            ['news', 'current events', 'politics', 'sports', 'breaking news'],
            ['political', 'government', 'election', 'vote', 'politician'],
            ['trending', 'social media', 'twitter', 'facebook', 'instagram'],
            
            # Personal questions about the AI
            ['favorite', 'feelings', 'married', 'personal life', 'emotions'],
            ['fun', 'hobby', 'family', 'friends', 'relationship'],
            
            # Note: Removed library hours from boundary detection as this info exists in KB
            
            # Other common out-of-scope
            ['accommodation', 'hostel', 'housing', 'dormitory'],
            ['movies', 'entertainment', 'cinema', 'theater'],
            ['restaurant', 'food', 'dining', 'menu'],
        ]
        
        # Check for boundary patterns with context awareness
        for pattern_group in boundary_patterns:
            if any(pattern in query_lower for pattern in pattern_group):
                
                # Special handling for time-related queries
                if any(time_word in pattern_group for time_word in ['time', 'current time', 'what time', 'clock']):
                    # Check if this is a generic time query vs APU-specific time query
                    if self._is_generic_time_query(query_lower):
                        logger.debug(f"Matched generic time boundary pattern: {pattern_group}")
                        return True
                    else:
                        logger.debug(f"Time query appears APU-specific, allowing through: {query}")
                        continue
                
                # For other boundary patterns, apply normally
                logger.debug(f"Matched boundary pattern group: {pattern_group}")
                return True
        
        # Note: Removed library special case as this info exists in knowledge base
            
        return False
    
    def _is_generic_time_query(self, query_lower: str) -> bool:
        """
        Determine if a time-related query is generic (current time) vs APU-specific (schedules).
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            True if generic time query, False if APU-specific time query
        """
        # APU-specific time keywords
        apu_time_keywords = [
            'library', 'office', 'hours', 'open', 'close', 'closes', 'opening', 'closing',
            'schedule', 'timetable', 'class', 'lecture', 'exam', 'appointment',
            'deadline', 'submission', 'registration', 'application',
            'visit', 'counter', 'desk', 'service', 'operation', 'operating'
        ]
        
        # If the query contains APU-specific context, it's NOT a generic time query
        if any(keyword in query_lower for keyword in apu_time_keywords):
            return False
        
        # Generic time query patterns
        generic_time_patterns = [
            'what time is it',
            'current time',
            'what is the time',
            'tell me the time',
            'time now',
            'what time',
            'the time'
        ]
        
        # If it matches generic patterns without APU context, it's generic
        if any(pattern in query_lower for pattern in generic_time_patterns):
            return True
        
        # If query is just "time" without context, treat as generic
        if query_lower.strip() in ['time', 'what time', 'the time']:
            return True
        
        # Default: if it contains time but has other context, probably APU-specific
        return False
    
    def _handle_boundary_query(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """
        Handle boundary queries with appropriate out-of-scope responses.
        
        Args:
            query: Original query string
            stream: Whether to stream the response
            
        Returns:
            Response string or iterator for streaming
        """
        query_lower = query.lower()
        
        # Generate appropriate boundary response
        if any(word in query_lower for word in ['weather', 'temperature', 'climate', 'rain', 'sunny', 'hot', 'cold']):
            response = (
                "I'm designed to help with APU-specific information rather than weather updates. "
                "For weather information, please check your local weather app or website. "
                "Is there anything about APU services or procedures I can help you with instead?"
            )
        elif any(word in query_lower for word in ['time', 'current time', 'what time', 'clock']):
            response = (
                "I don't have access to current time information. Please check your device's clock. "
                "However, I can help you with APU schedules, timetables, and office hours. "
                "What APU-related timing information do you need?"
            )
        elif any(word in query_lower for word in ['date', 'today', 'tomorrow', 'yesterday', 'calendar', 'day of the week', 'what day']):
            response = (
                "I don't have access to current date information. Please check your device's calendar. "
                "However, I can help you with APU academic calendars, exam schedules, and important dates. "
                "What APU-related date information do you need?"
            )
        elif any(word in query_lower for word in ['news', 'current events', 'politics', 'sports', 'breaking news']):
            response = (
                "I focus on APU-related information rather than general news. "
                "For current events, please check news websites or apps. "
                "I'd be happy to help with APU announcements, procedures, or services instead!"
            )
        elif any(word in query_lower for word in ['political', 'government', 'election', 'vote', 'politician']):
            response = (
                "I don't provide political opinions or information. I'm designed to help with APU matters. "
                "For political information, please check reputable news sources. "
                "How can I help you with APU services or academic information instead?"
            )
        elif any(word in query_lower for word in ['trending', 'social media', 'twitter', 'facebook', 'instagram']):
            response = (
                "I don't have access to social media or trending information. "
                "For current trends, please check your social media apps directly. "
                "I can help you with APU's official communication channels and announcements though!"
            )
        elif any(word in query_lower for word in ['favorite', 'feelings', 'married', 'personal life', 'emotions']):
            response = (
                "I don't have personal experiences or feelings as I'm an AI assistant. "
                "I'm here to help with APU-related information and services. "
                "What can I help you with regarding your APU experience?"
            )
        elif any(word in query_lower for word in ['fun', 'hobby', 'family', 'friends', 'relationship']):
            response = (
                "I don't have personal experiences, but I can help you with APU student life information! "
                "I can provide details about student activities, clubs, facilities, and services. "
                "What aspect of APU student life would you like to know about?"
            )
        # Note: Removed library hours handling as this info exists in knowledge base
        elif any(word in query_lower for word in ['accommodation', 'hostel', 'housing', 'dormitory']):
            response = (
                "I don't have detailed information about accommodation options. "
                "For information about student housing and accommodation, please contact "
                "Student Services at student.services@apu.edu.my."
            )
        elif any(word in query_lower for word in ['movies', 'entertainment', 'cinema', 'theater']):
            response = (
                "I don't have information about entertainment venues or movie schedules. "
                "For entertainment options, please check local cinema websites or apps. "
                "I can help you with APU events, facilities, and student activities instead!"
            )
        elif any(word in query_lower for word in ['restaurant', 'food', 'dining', 'menu']):
            response = (
                "I don't have information about restaurant menus or dining options. "
                "For dining information, please check restaurant websites or food apps. "
                "I can help you with APU cafeteria information and campus dining facilities though!"
            )
        else:
            response = (
                "I don't have specific information about that topic in my knowledge base. "
                "For accurate and detailed information, I recommend contacting the appropriate "
                "APU department or checking the official APU website."
            )
        
        # Update memory and handle streaming
        if self.memory:
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
        
        if stream:
            return self._stream_text_response(response)
        return response
    
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
            # Enhanced greetings and social patterns
            r'\b(?:hi+|h[ie]llo*|hey+|hiya?|greetings?|howdy|good\s*(?:morning|afternoon|evening|day)|what\'?s\s*up|sup|hai|helo+)\b',
            r'\bhow\s+(?:are|r|is)\s+(?:you|u|ya?)\b',
            r'\bhow\s+(?:are|r)\s+(?:you|u)\s+(?:doing|going)\b',
            r'\bhow\s+is\s+(?:your|ur)\s+day\b',
            r'\bnice\s+to\s+meet\s+(?:you|u)\b',
            r'\bgood\s+to\s+see\s+(?:you|u)\b',
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
            
            # Addressing/naming queries
            r'\bhow\s+(?:can|do|should)\s+i\s+(?:call|address|refer\s+to)\s+you\b',
            r'\bwhat\s+(?:can|do|should)\s+i\s+call\s+you\b',
            r'\bwhat\s+should\s+i\s+call\s+you\b',
            r'\bhow\s+should\s+i\s+address\s+you\b',
            r'\bwhat\s+do\s+i\s+call\s+you\b',
            
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
    
    def _is_session_command(self, query: str) -> bool:
        """
        Check if query is a session management command.
        
        Args:
            query: Original query string
            
        Returns:
            True if it's a session command
        """
        session_command_patterns = [
            # Session management commands
            r'^new\s+session$',
            r'^create\s+session$', 
            r'^start\s+new\s+chat$',
            r'^list\s+sessions$',
            r'^show\s+sessions$',
            r'^sessions$',
            r'^switch\s+session$',
            r'^change\s+session$',
            r'^load\s+session$',
            r'^session\s+stats$',
            r'^session\s+statistics$',
            r'^clear\s+session$',
            r'^reset\s+session$',
            
            # Also handle commands with session IDs
            r'^switch\s+session\s+[a-f0-9-]+$',
            r'^load\s+session\s+[a-f0-9-]+$',
        ]
        
        query_lower = query.lower().strip()
        for pattern in session_command_patterns:
            if re.match(pattern, query_lower):
                logger.debug(f"Matched session command pattern: {pattern} in query: {query}")
                return True
        
        logger.debug(f"No session command patterns matched for: {query}")
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
            response = """Since we've chatted before, you know I'm Sara - your intelligent assistant for APU (Asia Pacific University)! 

I'm here to help with:
• Academic programs and course information
• Administrative procedures and services  
• Campus facilities and locations
• Student services and support
• Financial information and policies

I use APU's official knowledge base to provide accurate, up-to-date information. What specific APU topic can I help you explore?"""
        else:
            response = """Hello! I'm Sara, your intelligent assistant for APU (Asia Pacific University). 

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
    
    def _handle_empty_query(self, stream=False) -> Union[str, Iterator[str]]:
        """Handle empty or whitespace-only queries."""
        response = (
            "I'd be happy to help! Please ask me a question about APU services, "
            "such as fee payments, reference letters, IT support, library services, "
            "parking, or visa information."
        )
        
        if stream:
            return self._stream_text_response(response)
        return response
    
    def _handle_unclear_query(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """Handle very short or unclear queries."""
        response = (
            f"I received your message '{query}' but I'm not sure what you're asking about. "
            "Could you please provide more details? For example, you can ask about:\n\n"
            "• How to pay fees\n"
            "• Requesting reference letters\n"
            "• IT support (APKey password, timetable access)\n"
            "• Library services\n"
            "• Parking information\n"
            "• Visa and immigration matters"
        )
        
        if stream:
            return self._stream_text_response(response)
        return response
    
    def _handle_long_query(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """Handle very long queries that might contain multiple questions."""
        # Extract key topics from the long query
        key_topics = []
        topic_keywords = {
            'fees': ['fee', 'payment', 'pay', 'money', 'cost'],
            'reference letter': ['reference', 'letter', 'document', 'certificate'],
            'IT support': ['apkey', 'password', 'login', 'timetable', 'moodle'],
            'library': ['library', 'book', 'borrow', 'opac'],
            'parking': ['parking', 'park', 'car', 'vehicle'],
            'visa': ['visa', 'immigration', 'pass', 'permit']
        }
        
        query_lower = query.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                key_topics.append(topic)
        
        if key_topics:
            topics_text = ", ".join(key_topics)
            response = (
                f"I can see you're asking about several topics including {topics_text}. "
                "To give you the most accurate help, could you please ask about one topic at a time? "
                f"Which would you like to start with: {topics_text}?"
            )
        else:
            response = (
                "Your question is quite detailed. To provide the best help, could you please "
                "break it down into smaller, specific questions? This will help me give you "
                "more accurate and focused answers."
            )
        
        if stream:
            return self._stream_text_response(response)
        return response
    
    def _handle_system_error(self, stream=False) -> Union[str, Iterator[str]]:
        """Handle system errors gracefully."""
        response = (
            "I apologize, but I encountered a technical issue while processing your query. "
            "Please try asking your question again, or contact APU support directly if the "
            "problem persists."
        )
        
        if stream:
            return self._stream_text_response(response)
        return response
    
    def _is_inappropriate_request(self, query: str) -> bool:
        """Check if query contains inappropriate requests."""
        query_lower = query.lower()
        
        # Academic dishonesty keywords
        dishonesty_patterns = [
            'cheat', 'cheating', 'hack', 'hacking', 'bypass', 'skip',
            'fake', 'forge', 'plagiarize', 'copy answers', 'steal',
            'illegal', 'unauthorized access', 'break into'
        ]
        
        # Check for explicit academic dishonesty
        for pattern in dishonesty_patterns:
            if pattern in query_lower:
                return True
        
        # Check for suspicious phrases
        suspicious_phrases = [
            'help me cheat',
            'how to cheat',
            'bypass attendance',
            'fake documents',
            'get answers',
            'hack system',
            'access grades',
            'change my grade'
        ]
        
        for phrase in suspicious_phrases:
            if phrase in query_lower:
                return True
        
        return False
    
    def _handle_inappropriate_request(self, query: str, stream=False) -> Union[str, Iterator[str]]:
        """Handle inappropriate requests with clear boundaries."""
        response = (
            "I can't help with that request as it goes against academic integrity policies. "
            "Instead, I'd be happy to help you with legitimate APU services such as:\n\n"
            "• Academic support and study resources\n"
            "• Assignment submission procedures\n"
            "• Academic policies and guidelines\n"
            "• Student support services\n"
            "• IT help and technical support\n\n"
            "Is there something specific about APU services I can help you with?"
        )
        
        if stream:
            return self._stream_text_response(response)
        return response

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