"""
Conversational query handling with fuzzy matching and personality.
"""

import re
import random
import time
import logging
from difflib import SequenceMatcher

from config import config
from .language_handler import LanguageHandler
from .ambiguity_handler import AmbiguityHandler

logger = logging.getLogger("Sara")

class ConversationHandler:
    """Handles conversational queries with natural personality and fuzzy matching."""
    
    def __init__(self, memory, stream_delay=None):
        """Initialize with a memory for conversation history."""
        self.memory = memory
        self.stream_delay = stream_delay if stream_delay is not None else config.STREAM_DELAY
        
        # Initialize handlers for best practices
        self.language_handler = LanguageHandler()
        self.ambiguity_handler = AmbiguityHandler()
        
        # Core conversational patterns with enhanced fuzzy matching support
        self.greeting_patterns = [
            # Enhanced greeting patterns with common typos and variations
            r'\b(?:hi+|h[ie]llo*|hey+|hiya?|greetings?|howdy|good\s*(?:morning|afternoon|evening|day)|what\'?s\s*up|sup|hai|helo+)\b',
            r'\bhow\s+(?:are|r|is)\s+(?:you|u|ya?)\b',
            r'\bhow\s+(?:are|r)\s+(?:you|u)\s+(?:doing|going)\b',
            r'\bhow\s+do\s+(?:you|u)\s+do\b',
        ]

        # Natural, personable greeting responses
        self.greeting_responses = [
            "Hey there! I'm doing great, thanks for asking! 😊 I'm Sara, ready to help you with anything APU-related. What's on your mind?",
            "Hi! I'm fantastic today - hope you are too! ☀️ I'm Sara, what can I help you find out about APU?",
            "Hello! I'm doing well, thank you! A bit busy helping students but I love it! 😄 I'm Sara, how can I assist you with APU today?",
            "Hey! I'm good, thanks for asking! The weather's been nice lately. 🌤️ I'm Sara, what APU questions do you have for me?",
            "Hi there! I'm doing awesome, thanks! Ready to dive into some APU information with you. What would you like to know?",
            "Hello! I'm great, thank you for asking! 😊 I'm Sara, always excited to help with APU questions. What brings you here today?",
            "Hey! I'm doing wonderful, thanks! Hope your day is going well too! I'm Sara, what can I help you with regarding APU?"
        ]

        # System knowledge patterns
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

        # More natural knowledge responses
        self.knowledge_responses = [
            "I'm Sara, your APU assistant! 😊 I can help with academics, admin procedures, fees, student services, and campus info. What would you like to know about APU?",
            
            "Hey! I'm Sara, here to help with APU questions! 😄 I know about courses, applications, campus facilities, and more. What can I help you with?",
            
            "I'm Sara! 🌟 I can assist with APU academics, student services, financial info, and campus procedures. What APU topic interests you?",
        ]

        # FIXED: Separate farewell patterns from general small talk
        self.farewell_patterns = [
            r'\bbye\b',
            r'\bgoodbye\b',
            r'\bfarewell\b',
            r'\bsee\s+you\s+later\b',
            r'\bsee\s+you\s+soon\b',
            r'\btake\s+care\b',
            r'\bcatch\s+you\s+later\b',
            r'\bgotta\s+go\b',
            r'\bi\s+have\s+to\s+go\b',
            r'\bi\s+need\s+to\s+leave\b',
            r'\btalking\s+to\s+you\s+later\b',
            r'\buntil\s+next\s+time\b',
        ]

        # FIXED: Dedicated farewell responses
        self.farewell_responses = [
            "Goodbye! Have a wonderful day ahead! 🌈 Feel free to come back anytime if you need APU info!",
            "Take care! 👋 Remember, I'm Sara, always here whenever you need help with APU stuff!",
            "Bye for now! 😊 Don't hesitate to return if you have any APU questions later!",
            "See you later! 👋 I'll be here whenever you need APU information or guidance!",
            "Farewell! 🌟 Hope I was able to help today. Come back anytime for APU assistance!",
            "Have a great day! 😊 Thanks for chatting - I'm here whenever you need APU support!",
            "Until next time! 👋 Feel free to return with any APU questions you might have!",
            "Goodbye and take care! 🌻 I enjoyed helping you today - see you again soon!"
        ]

        # Small talk patterns (excluding farewells)
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
        ]

        # Small talk responses (excluding farewells)
        self.small_talk_responses = [
            "My day's been great, thanks for asking! 😊 Helping students like you makes it even better. I'm Sara, what can I do for you?",
            "Nice to meet you too! 🤝 I'm Sara, excited to help you navigate APU. What would you like to know?",
            "Aw, thanks! 😄 I hope you're having a wonderful day too. I'm Sara, what APU info can I dig up for you?",
            "It's been a good day! Always enjoy chatting about APU stuff. 🌟 I'm Sara, what brings you here today?",
            "Everything's going smoothly on my end! ✨ I'm Sara, ready and eager to help with your APU questions. What's up?",
            "Thank you! I hope you have an amazing day too! 🌈 I'm Sara, what can I help you with regarding APU?",
        ]

        # Acknowledgement patterns with enthusiastic responses
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
            "You're so welcome! 😊 Happy I could help! Got any other questions for me?",
            "No problem at all! 🎉 That's what I'm here for! What else can I help you with?",
            "My pleasure! 😄 I'm Sara, and I love helping with APU stuff. Anything else on your mind?",
            "Glad I could help out! ✨ Feel free to ask if you need anything else!",
            "Awesome! 🙌 You're all set then! Come back anytime if you need more APU info!",
            "Perfect! 🎯 That's exactly what I like to hear! Any other questions?",
            "Woohoo! 🎉 Mission accomplished! Let me know if you need help with anything else!",
            "Yes! 🌟 So happy that hit the spot! I'm Sara, here if you need more APU assistance!"
        ]

        # Clarification patterns with encouraging responses
        self.clarification_patterns = [
            r'\bi\s+don\'t\s+understand\b',
            r'\bcan\s+you\s+explain\b',
            r'\bwhat\s+do\s+you\s+mean\b',
            r'\bi\'m\s+confused\b',
            r'\bcan\s+you\s+clarify\b',
            r'\bcan\s+you\s+help\s+me\s+understand\b',
            r'\bi\s+need\s+more\s+information\b',
            r'\bcan\s+you\s+be\s+more\s+specific\b',
            r'\bhuh\b',
            # FIXED: Removed overly broad \bwhat\b pattern that was matching everything
        ]

        # Addressing/naming patterns
        self.naming_patterns = [
            r'\bhow\s+(?:can|do|should)\s+i\s+(?:call|address|refer\s+to)\s+you\b',
            r'\bwhat\s+(?:can|do|should)\s+i\s+call\s+you\b',
            r'\bwhat\s+should\s+i\s+call\s+you\b',
            r'\bhow\s+should\s+i\s+address\s+you\b',
            r'\bwhat\s+do\s+i\s+call\s+you\b',
            r'\bwhat\s+is\s+your\s+name\b',
            r'\bwhat\'s\s+your\s+name\b',
        ]

        # Naming/addressing responses
        self.naming_responses = [
            "You can call me Sara! 😊 I'm your friendly APU assistant, here to help with any questions about Asia Pacific University. What can I help you with today?",
            "I'm Sara! 🌟 Feel free to call me Sara - I'm your go-to assistant for anything APU-related. What would you like to know?",
            "Just call me Sara! 😄 I'm here to help you navigate APU information and services. What brings you here today?",
            "You can call me Sara! 👋 I'm your APU assistant, ready to help with any questions about the university. How can I assist you?",
            "I'm Sara! 😊 That's what everyone calls me. I'm here to help with APU questions and information. What can I help you explore?",
        ]

        self.clarification_responses = [
            "No worries at all! 😊 Let me help clarify things. What specific topic would you like me to explain better?",
            "Of course! 🤗 I'd be happy to break it down for you. What particular topic needs more explanation?",
            "Totally understand! 💡 Sometimes information can be confusing. What specific area can I help make clearer?",
            "No problem! 😄 I'm here to make things crystal clear. What topic would you like me to dive deeper into?",
            "Absolutely! ✨ I love explaining things! What specific question can I help you understand better?"
        ]

        # Common typos and their corrections for fuzzy matching
        self.common_typos = {
            'hwo': 'how',
            'aer': 'are',
            'yuo': 'you',
            'yu': 'you',
            'ur': 'your',
            'ur': 'you are',
            'wat': 'what',
            'wher': 'where',
            'whre': 'where',
            'cna': 'can',
            'hlp': 'help',
            'teh': 'the',
            'adn': 'and',
            'helo': 'hello',
            'hi': 'hi',
            'hey': 'hey'
        }
    
    def _correct_spelling(self, query: str) -> str:
        """Correct common spelling mistakes in queries."""
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for comparison
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check direct typo mapping
            if clean_word in self.common_typos:
                # Keep original punctuation if any
                corrected = word.replace(clean_word, self.common_typos[clean_word])
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _fuzzy_match_patterns(self, query: str, patterns: list, threshold: float = 0.7) -> bool:
        """Use fuzzy matching to detect conversational patterns even with typos."""
        corrected_query = self._correct_spelling(query)
        
        # DEBUGGING: Log the corrected query
        logger.debug(f"Fuzzy matching: '{query}' -> '{corrected_query}'")
        
        # First try exact pattern matching on corrected query
        for pattern in patterns:
            if re.search(pattern, corrected_query, re.IGNORECASE):
                logger.debug(f"Matched exact pattern: {pattern}")
                return True
        
        # FIXED: Remove the problematic global fuzzy matching that was matching everything
        # The exact pattern matching above should be sufficient for conversational queries
        # If we need fuzzy matching, it should be pattern-category specific, not global
        
        logger.debug(f"No pattern matches found for: '{query}'")
        return False
    
    def is_conversational_query(self, query: str) -> bool:
        """
        Conversational query detection with pattern matching.
        FIXED: More specific detection that doesn't conflict with identity queries.
        """
        # DEBUGGING: Log the query being checked
        logger.debug(f"is_conversational_query checking: '{query}'")
        
        # Check each category individually to be more specific
        greeting_match = self._fuzzy_match_patterns(query, self.greeting_patterns, threshold=0.6)
        knowledge_match = self._fuzzy_match_patterns(query, self.knowledge_patterns, threshold=0.6)
        acknowledgement_match = self._fuzzy_match_patterns(query, self.acknowledgement_patterns, threshold=0.6)
        small_talk_match = self._fuzzy_match_patterns(query, self.small_talk_patterns, threshold=0.6)
        farewell_match = self._fuzzy_match_patterns(query, self.farewell_patterns, threshold=0.6)
        clarification_match = self._fuzzy_match_patterns(query, self.clarification_patterns, threshold=0.6)
        naming_match = self._fuzzy_match_patterns(query, self.naming_patterns, threshold=0.6)
        
        result = (greeting_match or knowledge_match or acknowledgement_match or 
                 small_talk_match or farewell_match or clarification_match or naming_match)
        
        logger.debug(f"is_conversational_query individual matches - greeting:{greeting_match}, knowledge:{knowledge_match}, ack:{acknowledgement_match}, small_talk:{small_talk_match}, farewell:{farewell_match}, clarification:{clarification_match}, naming:{naming_match}")
        logger.debug(f"is_conversational_query result: {result}")
        return result
    
    def handle_conversation(self, query: str, stream=False):
        """Handle conversational queries with language detection and ambiguity handling."""
        should_block, lang_response = self.language_handler.handle_query(query)
        if should_block:
            logger.info(f"Non-English query blocked: {query[:50]}...")
            return lang_response
        
        if self.ambiguity_handler.is_ambiguous(query):
            logger.info(f"Ambiguous query detected: {query[:50]}...")
            return self.ambiguity_handler.get_clarification(query)
        # Correct spelling first
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        response = None
        
        # Context-aware greeting enhancement
        is_return_user = len(self.memory.chat_memory.messages) > 0
        
        # DEBUGGING: Log the query and what patterns are being checked
        logger.debug(f"Processing conversational query: '{query}' -> '{corrected_query}'")
        
        # FIXED: Check each pattern category individually with proper logic
        greeting_match = self._fuzzy_match_patterns(corrected_query, self.greeting_patterns, threshold=0.6)
        knowledge_match = self._fuzzy_match_patterns(corrected_query, self.knowledge_patterns, threshold=0.6)
        acknowledgement_match = self._fuzzy_match_patterns(corrected_query, self.acknowledgement_patterns, threshold=0.6)
        small_talk_match = self._fuzzy_match_patterns(corrected_query, self.small_talk_patterns, threshold=0.6)
        clarification_match = self._fuzzy_match_patterns(corrected_query, self.clarification_patterns, threshold=0.6)
        farewell_match = self._fuzzy_match_patterns(corrected_query, self.farewell_patterns, threshold=0.6)
        naming_match = self._fuzzy_match_patterns(corrected_query, self.naming_patterns, threshold=0.6)
        
        logger.debug(f"Pattern matches - greeting:{greeting_match}, knowledge:{knowledge_match}, ack:{acknowledgement_match}, small_talk:{small_talk_match}, clarification:{clarification_match}, farewell:{farewell_match}, naming:{naming_match}")
        
        # Check for greetings FIRST (most common and should take priority)
        if greeting_match:
            logger.debug("Matched greeting patterns")
            if is_return_user:
                # Returning user - more casual and personalized
                contextual_greetings = [
                    "Hey again! 😊 Good to see you back! I'm doing great, thanks for asking. What can I help you with today?",
                    "Hi there! 👋 Welcome back! I'm having a wonderful day helping students. What APU info do you need?",
                    "Hello! 🌟 Nice to chat with you again! I'm doing fantastic. What would you like to explore about APU today?",
                    "Hey! 😄 Great to see you return! I'm doing awesome, hope you are too! How can I help with APU stuff?"
                ]
                response = random.choice(contextual_greetings)
            else:
                # New user - enthusiastic but welcoming
                response = random.choice(self.greeting_responses)
        
        # Check for system knowledge questions
        elif knowledge_match:
            logger.debug("Matched knowledge patterns")
            if is_return_user:
                base_response = random.choice(self.knowledge_responses)
                response = base_response + "\n\nSince we've chatted before, feel free to dive deeper into any topic! 🤿"
            else:
                response = random.choice(self.knowledge_responses)
        
        # Check for acknowledgements
        elif acknowledgement_match:
            logger.debug("Matched acknowledgement patterns")
            if is_return_user and len(self.memory.chat_memory.messages) > 2:
                contextual_acknowledgements = [
                    "You're very welcome! 😊 I absolutely love helping with APU questions! Anything else you'd like to explore?",
                    "My pleasure! 🎉 I'm so glad I could help clear things up about APU. What else can I assist with?",
                    "Happy to help! 😄 That's what makes my day! Feel free to ask about anything else APU-related!",
                    "You're welcome! ✨ I'm here whenever you need more APU information or guidance!"
                ]
                response = random.choice(contextual_acknowledgements)
            else:
                response = random.choice(self.acknowledgement_responses)
        
        # Check for small talk (excluding farewells)
        elif small_talk_match:
            logger.debug("Matched small talk patterns")
            response = random.choice(self.small_talk_responses)
        
        # Check for clarification requests
        elif clarification_match:
            logger.debug("Matched clarification patterns")
            if is_return_user:
                recent_topics = self._extract_recent_topics()
                if recent_topics:
                    response = f"No worries! 😊 I see we were talking about {recent_topics}. Would you like me to explain that better, or is there something else about APU you'd like to understand?"
                else:
                    response = random.choice(self.clarification_responses)
            else:
                response = random.choice(self.clarification_responses)
        
        # Check for naming/addressing queries
        elif naming_match:
            logger.debug("Matched naming patterns")
            response = random.choice(self.naming_responses)
        
        # Check for farewells LAST (to avoid catching other patterns accidentally)
        elif farewell_match:
            logger.debug("Matched farewell patterns")
            if is_return_user:
                # Returning user - more personal farewell
                contextual_farewells = [
                    "Goodbye! 👋 It was great chatting with you again. Feel free to return anytime for APU help!",
                    "Take care! 😊 Thanks for the conversation - I'm always here when you need APU assistance!",
                    "See you later! 🌟 I enjoyed helping you today. Come back whenever you have APU questions!",
                    "Farewell! 💫 Hope our chat was helpful. Don't hesitate to return for more APU info!"
                ]
                response = random.choice(contextual_farewells)
            else:
                response = random.choice(self.farewell_responses)
        
        # Generic fallback with personality
        if not response:
            logger.debug("No patterns matched, using fallback response")
            if is_return_user:
                response = "I'm here and ready to help! 😊 What APU questions are on your mind today?"
            else:
                response = "Hi there! 👋 I'm here to help you with anything APU-related. What would you like to know?"
        
        logger.debug(f"Selected response type based on pattern matching")
        
        # Update conversation memory
        if not stream:
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            return response
        else:
            self._last_response = response
            self._last_query = query
            return self._stream_response(response)

    def _extract_recent_topics(self) -> str:
        """Extract topics from recent conversation for context."""
        if not self.memory.chat_memory.messages or len(self.memory.chat_memory.messages) < 2:
            return ""
        
        recent_messages = self.memory.chat_memory.messages[-4:]
        text_content = []
        
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                text_content.append(msg.content.lower())
        
        combined_text = " ".join(text_content)
        
        topic_keywords = {
            'academic stuff': ['course', 'program', 'degree', 'academic', 'study', 'class', 'subject'],
            'administrative procedures': ['application', 'registration', 'enrollment', 'form', 'procedure'],
            'financial matters': ['fee', 'payment', 'scholarship', 'financial', 'cost', 'tuition'],
            'student services': ['medical', 'insurance', 'card', 'collect', 'service', 'support'],
            'campus facilities': ['building', 'location', 'address', 'campus', 'library', 'lab'],
        }
        
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
        """Stream the response naturally."""
        self.memory.chat_memory.add_user_message(self._last_query)
        self.memory.chat_memory.add_ai_message(response)
        
        words = response.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(self.stream_delay)