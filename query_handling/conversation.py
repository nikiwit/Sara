"""
Conversational query handling for social interactions and system info.
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
            r'\bhow\s+are\s+you\s+doing\b',
            r'\bhow\s+do\s+you\s+do\b',
        ]

        self.greeting_responses = [
            "Hello! I'm your APU knowledge base assistant. How can I help you with your questions about APU today?",
            "Hi there! I'm ready to help answer questions about APU. What would you like to know?",
            "Greetings! I'm here to assist with information about APU. What are you looking for?",
            "Hello! I'm your APU RAG assistant. I can help you find information about academics, administrative processes, and more.",
            "Hi! I'm ready to help you navigate APU-related questions. What would you like to learn about?",
            "I'm doing well, thank you! I'm here to help you with any questions about APU. What can I assist you with?"
        ]

        # System knowledge/capability patterns
        self.knowledge_patterns = [
            r'\bwhat\s+do\s+you\s+know\b',
            r'\bwhat\s+can\s+you\s+do\b',
            r'\bwhat\s+are\s+you\s+capable\s+of\b',
            r'\bwhat\s+information\s+do\s+you\s+have\b',
            r'\btell\s+me\s+about\s+yourself\b',
            r'\bwhat\s+are\s+your\s+capabilities\b',
            r'\bwhat\s+kind\s+of\s+questions\s+can\s+you\s+answer\b',
            r'\bwhat\s+topics\s+can\s+you\s+help\s+with\b',
        ]

        self.knowledge_responses = [
            """I'm an AI assistant specialized in APU (Asia Pacific University) information. Here's what I can help you with:

📚 **Academic Information:**
- Course details, requirements, and schedules
- Program information and admission requirements
- Academic policies and procedures

🏢 **Administrative Services:**
- Student services and support
- Campus facilities and locations
- Registration and enrollment processes

💰 **Financial Information:**
- Fee structures and payment procedures
- Scholarship and financial aid information
- Medical insurance and benefits

📋 **Procedures & Policies:**
- Application processes
- Academic regulations
- Campus guidelines and rules

❓ **FAQ & Common Questions:**
- Frequently asked questions about APU
- Step-by-step guides for common procedures

Just ask me anything about APU and I'll do my best to help using the official APU knowledge base!""",

            """I'm your dedicated APU assistant with access to comprehensive information about Asia Pacific University. I can help you with:

• **Student Services** - Medical insurance, campus facilities, student support
• **Academic Programs** - Course information, requirements, schedules
• **Administrative Processes** - Applications, registrations, procedures
• **Campus Life** - Facilities, locations, services
• **Financial Information** - Fees, payments, scholarships

I use APU's official knowledge base to provide accurate, up-to-date information. Feel free to ask me specific questions about any APU-related topic!""",

            """Hello! I'm an AI assistant that specializes in APU information. My knowledge covers:

✅ **What I know about:**
- APU academic programs and courses
- Student services and administrative procedures  
- Campus facilities and locations
- Fee structures and financial information
- Medical insurance and student benefits
- Application and registration processes
- Academic policies and regulations

✅ **How I can help:**
- Answer specific questions about APU
- Guide you through procedures step-by-step
- Provide official information from APU's knowledge base
- Connect you with the right departments when needed

What would you like to know about APU?"""
        ]

        # Small talk patterns and responses
        self.small_talk_patterns = [
            r'\bhow\s+is\s+your\s+day\b',
            r'\bhow\s+was\s+your\s+day\b',
            r'\bnice\s+to\s+meet\s+you\b',
            r'\bnice\s+meeting\s+you\b',
            r'\bgood\s+to\s+see\s+you\b',
            r'\bpleasure\s+to\s+meet\s+you\b',
            r'\bhow\s+is\s+everything\b',
            r'\bhow\s+are\s+things\b',
            r'\bwhat\s+a\s+nice\s+day\b',
            r'\bhave\s+a\s+good\s+day\b',
            r'\bhave\s+a\s+great\s+day\b',
            r'\btake\s+care\b',
            r'\bsee\s+you\s+later\b',
            r'\bbye\b',
            r'\bgoodbye\b',
            r'\bfarewell\b',
        ]

        self.small_talk_responses = [
            "Thank you for asking! I'm having a great day helping APU students and staff. How can I assist you today?",
            "Nice to meet you too! I'm here to help with any questions about APU. What would you like to know?",
            "The pleasure is mine! I'm ready to assist with APU-related information. How can I help?",
            "Thank you! I hope you're having a wonderful day as well. What APU information can I help you find?",
            "Everything's going well on my end - ready to help with APU questions! What brings you here today?",
            "Thank you! Have a fantastic day, and feel free to come back if you need any APU information!",
            "Take care! Remember, I'm always here if you need help with APU-related questions.",
            "Goodbye! Don't hesitate to return if you have any questions about APU services or procedures.",
            "See you later! I'll be here whenever you need assistance with APU information."
        ]

        # Acknowledgement patterns and responses
        self.acknowledgement_patterns = [
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
            r'\b(?:awesome|great|cool|nice)\b',
            r'\bthat\s*(?:helps|helped)\b',
            r'\bgot\s*it\b',
            r'\bperfect\b',
            r'\bexcellent\b',
            r'\bwonderful\b',
            r'\bthat\'s\s+(?:great|good|perfect|helpful)\b',
            r'\bexactly\s+what\s+I\s+needed\b',
        ]

        self.acknowledgement_responses = [
            "You're welcome! Is there anything else you'd like to know about APU?",
            "Happy to help! Let me know if you have any other questions about APU.",
            "My pleasure! Feel free to ask if you need anything else.",
            "Glad I could assist. Any other questions about APU?",
            "You're welcome! I'm here if you need more information about APU procedures, policies, or services.",
            "Wonderful! I'm always here if you need more APU information.",
            "Great to hear! Feel free to ask about anything else related to APU.",
            "Perfect! Don't hesitate to reach out if you have more questions."
        ]

        # Clarification patterns for when users seem confused or need more help
        self.clarification_patterns = [
            r'\bi\s+don\'t\s+understand\b',
            r'\bcan\s+you\s+explain\b',
            r'\bwhat\s+do\s+you\s+mean\b',
            r'\bi\'m\s+confused\b',
            r'\bcan\s+you\s+clarify\b',
            r'\bcan\s+you\s+help\s+me\s+understand\b',
            r'\bi\s+need\s+more\s+information\b',
            r'\bcan\s+you\s+be\s+more\s+specific\b',
        ]

        self.clarification_responses = [
            "I'd be happy to clarify! Could you tell me specifically what aspect of APU you'd like to know more about?",
            "No problem! Let me help you find the right information. What specific APU topic or service are you interested in?",
            "I understand it can be confusing. Could you let me know what particular area you need help with - academics, administration, services, or something else?",
            "Of course! I'm here to make things clearer. What specific question about APU can I help answer for you?",
            "Absolutely! To give you the most helpful information, could you tell me what specific APU-related topic you're looking for?"
        ]
    
    def is_conversational_query(self, query: str) -> bool:
        """
        Check if a query is conversational and should be handled here.
        
        Args:
            query: The user's query
            
        Returns:
            True if it's a conversational query, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Check all pattern categories
        all_patterns = (
            self.greeting_patterns + 
            self.knowledge_patterns + 
            self.acknowledgement_patterns +
            self.small_talk_patterns +
            self.clarification_patterns
        )
        
        for pattern in all_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
                
        return False
    
    def handle_conversation(self, query: str, stream=False):
        """
        Handles conversational queries with context awareness.
        
        Args:
            query: The user's query
            stream: Whether to stream the response
            
        Returns:
            Either a string response or an iterator for streaming
        """
        query_lower = query.lower().strip()
        response = None
        
        # Context-aware greeting enhancement
        is_return_user = len(self.memory.chat_memory.messages) > 0
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if is_return_user:
                    # Returning user - more casual greeting
                    contextual_greetings = [
                        "Hello again! Good to see you back. How can I help you with APU today?",
                        "Hi there! Welcome back. What APU information can I assist you with now?",
                        "Hello! Nice to have you back. What would you like to know about APU this time?",
                        "Hi! Great to see you again. How can I help with your APU questions today?"
                    ]
                    response = random.choice(contextual_greetings)
                else:
                    # New user - standard greeting
                    response = random.choice(self.greeting_responses)
                break
        
        # Check for system knowledge/capability questions
        if not response:
            for pattern in self.knowledge_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Add context if user has asked questions before
                    if is_return_user:
                        base_response = random.choice(self.knowledge_responses)
                        response = base_response + "\n\nSince we've chatted before, feel free to ask follow-up questions or dive deeper into any APU topic!"
                    else:
                        response = random.choice(self.knowledge_responses)
                    break
        
        # Check for small talk
        if not response:
            for pattern in self.small_talk_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    response = random.choice(self.small_talk_responses)
                    break
        
        # Check for clarification requests
        if not response:
            for pattern in self.clarification_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Context-aware clarification
                    if is_return_user:
                        # Look at recent conversation for context
                        recent_topics = self._extract_recent_topics()
                        if recent_topics:
                            contextual_clarification = (
                                f"I'd be happy to clarify! I see we were discussing {recent_topics}. "
                                f"Would you like me to explain that in more detail, or is there something "
                                f"else about APU you'd like to understand better?"
                            )
                            response = contextual_clarification
                        else:
                            response = random.choice(self.clarification_responses)
                    else:
                        response = random.choice(self.clarification_responses)
                    break
        
        # Check for acknowledgements
        if not response:
            for pattern in self.acknowledgement_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Enhanced acknowledgement with context
                    if is_return_user and len(self.memory.chat_memory.messages) > 2:
                        contextual_acknowledgements = [
                            "You're very welcome! I enjoy helping with APU questions. Anything else you'd like to explore?",
                            "My pleasure! I'm glad I could help clarify things about APU. What else can I assist with?",
                            "Happy to help! Feel free to ask if you think of any other APU-related questions.",
                            "You're welcome! I'm here whenever you need more APU information or guidance."
                        ]
                        response = random.choice(contextual_acknowledgements)
                    else:
                        response = random.choice(self.acknowledgement_responses)
                    break
        
        # If no specific pattern matched, give a generic response with context
        if not response:
            if is_return_user:
                response = "I'm here to help with any APU questions you have. What would you like to know more about?"
            else:
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

    def _extract_recent_topics(self) -> str:
        """
        Extract topics from recent conversation history to provide context.
        
        Returns:
            String describing recent topics discussed
        """
        if not self.memory.chat_memory.messages or len(self.memory.chat_memory.messages) < 2:
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
            'academic': ['course', 'program', 'degree', 'academic', 'study', 'class', 'subject', 'curriculum'],
            'administrative': ['application', 'registration', 'enrollment', 'form', 'procedure', 'process', 'admin'],
            'financial': ['fee', 'payment', 'scholarship', 'financial', 'cost', 'tuition', 'money'],
            'student services': ['medical', 'insurance', 'card', 'collect', 'service', 'support', 'campus'],
            'facilities': ['building', 'location', 'address', 'campus', 'library', 'lab', 'facility'],
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