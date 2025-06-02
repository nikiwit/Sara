"""
Conversational query handling with fuzzy matching and personality.
Ensures no references or related pages are included in conversational responses.
"""

import re
import random
import time
import logging
from difflib import SequenceMatcher

from config import Config

logger = logging.getLogger("CustomRAG")

class ConversationHandler:
    """Handles conversational queries with natural personality and fuzzy matching."""
    
    def __init__(self, memory, stream_delay=None):
        """Initialize conversation handler with memory and optional stream delay configuration."""
        self.memory = memory
        self.stream_delay = stream_delay if stream_delay is not None else Config.STREAM_DELAY
        
        # Regular expression patterns for identifying greeting queries
        self.greeting_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
            r'\bhow\s+are\s+you\s+doing\b',
            r'\bhow\s+do\s+you\s+do\b',
        ]

        # Predefined greeting responses without document references
        self.greeting_responses = [
            "Hey there! I'm doing great, thanks for asking! Ready to help you with anything APU-related. What's on your mind?",
            "Hi! I'm fantastic today - hope you are too! What can I help you find out about APU?",
            "Hello! I'm doing well, thank you! A bit busy helping students but I love it! How can I assist you with APU today?",
            "Hey! I'm good, thanks for asking! The weather's been nice lately. What APU questions do you have for me?",
            "Hi there! I'm doing awesome, thanks! Ready to dive into some APU information with you. What would you like to know?",
            "Hello! I'm great, thank you for asking! Always excited to help with APU questions. What brings you here today?",
            "Hey! I'm doing wonderful, thanks! Hope your day is going well too! What can I help you with regarding APU?"
        ]

        # Patterns for detecting system knowledge and capability queries
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

        # Knowledge responses explaining system capabilities without document references
        self.knowledge_responses = [
            """Oh, I know quite a bit about APU! I'm like your friendly neighborhood APU encyclopedia. Here's what I can help you with:

**Academic Stuff:**
- Course details, requirements, schedules (the nitty-gritty details!)
- Program info and admission requirements
- Academic policies (the fine print, but explained simply!)

**Administrative Things:**
- Student services and campus support
- Where to find offices and facilities
- Registration and enrollment (I'll walk you through it!)

**Money Matters:**
- Fee structures and payment options
- Scholarships and financial aid info
- Medical insurance details

**Procedures & Policies:**
- How to apply for things (step by step!)
- Academic rules (translated from bureaucratic speak!)
- Campus guidelines

**Common Questions:**
- All those frequently asked questions
- Step-by-step guides for tricky procedures

Just ask me anything about APU - I'm here to make your life easier!""",

            """Hey! I'm your go-to APU assistant! Think of me as that helpful friend who somehow knows everything about the university. Here's my expertise:

• **Student Life** - Medical insurance, campus facilities, where to get help when you're stuck
• **Academics** - Course info, requirements, schedules (I love talking about courses!)
• **Admin Stuff** - Applications, registrations, procedures (I'll make it less confusing!)
• **Campus Navigation** - Facilities, locations, services (virtual tour guide!)
• **Financial Info** - Fees, payments, scholarships (money talk made simple!)

I use APU's official info, so you can trust what I tell you. Plus, I try to explain things in plain English instead of university jargon! 

What would you like to explore?""",

        ]

        # Patterns for detecting farewell messages
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

        # Farewell responses for ending conversations
        self.farewell_responses = [
            "Goodbye! Have a wonderful day ahead! Feel free to come back anytime if you need APU info!",
            "Take care! Remember, I'm always here whenever you need help with APU stuff!",
            "Bye for now! Don't hesitate to return if you have any APU questions later!",
            "See you later! I'll be here whenever you need APU information or guidance!",
            "Farewell! Hope I was able to help today. Come back anytime for APU assistance!",
            "Have a great day! Thanks for chatting - I'm here whenever you need APU support!",
            "Until next time! Feel free to return with any APU questions you might have!",
            "Goodbye and take care! I enjoyed helping you today - see you again soon!"
        ]

        # Patterns for casual conversation excluding farewells
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

        # Small talk responses for casual conversation
        self.small_talk_responses = [
            "My day's been great, thanks for asking! Helping students like you makes it even better. What can I do for you?",
            "Nice to meet you too! I'm excited to help you navigate APU. What would you like to know?",
            "Aw, thanks! I hope you're having a wonderful day too. What APU info can I dig up for you?",
            "It's been a good day! Always enjoy chatting about APU stuff. What brings you here today?",
            "Everything's going smoothly on my end! Ready and eager to help with your APU questions. What's up?",
            "Thank you! I hope you have an amazing day too! What can I help you with regarding APU?",
        ]

        # Patterns for acknowledgment and appreciation messages
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

        # Acknowledgment responses for user appreciation
        self.acknowledgement_responses = [
            "You're so welcome! Happy I could help! Got any other APU questions for me?",
            "No problem at all! That's what I'm here for! What else can I help you with?",
            "My pleasure! I love helping with APU stuff. Anything else on your mind?",
            "Glad I could help out! Feel free to ask if you need anything else about APU!",
            "Awesome! You're all set then! Come back anytime if you need more APU info!",
            "Perfect! That's exactly what I like to hear! Any other APU questions?",
            "Woohoo! Mission accomplished! Let me know if you need help with anything else!",
            "Yes! So happy that hit the spot! I'm here if you need more APU assistance!"
        ]

        # Patterns for clarification requests and confusion
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
        ]

        # Clarification responses for confused users
        self.clarification_responses = [
            "No worries at all! Let me help clarify things. What specific part about APU would you like me to explain better?",
            "Of course! I'd be happy to break it down for you. What particular APU topic needs more explanation?",
            "Totally understand! Sometimes APU stuff can be confusing. What specific area can I help make clearer?",
            "No problem! I'm here to make things crystal clear. What APU topic would you like me to dive deeper into?",
            "Absolutely! I love explaining things! What specific APU question can I help you understand better?"
        ]

        # Dictionary mapping common typos to their correct spellings
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
        """Apply spelling corrections to user queries based on common typos."""
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation from word for typo comparison
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Replace typo with correct spelling if found in dictionary
            if clean_word in self.common_typos:
                # Preserve original punctuation when replacing word
                corrected = word.replace(clean_word, self.common_typos[clean_word])
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _fuzzy_match_patterns(self, query: str, patterns: list, threshold: float = 0.7) -> bool:
        """Use fuzzy matching to detect conversational patterns even with spelling errors."""
        corrected_query = self._correct_spelling(query)
        
        # Log corrected query for debugging fuzzy matching
        logger.debug(f"Fuzzy matching: '{query}' -> '{corrected_query}'")
        
        # Test each regex pattern against the corrected query
        for pattern in patterns:
            if re.search(pattern, corrected_query, re.IGNORECASE):
                logger.debug(f"Matched exact pattern: {pattern}")
                return True
        
        logger.debug(f"No pattern matches found for: '{query}'")
        return False
    
    def is_conversational_query(self, query: str) -> bool:
        """
        Determine if a query is conversational rather than informational.
        Uses pattern matching to identify greetings, small talk, and system queries.
        """
        # Log query being analyzed for conversational patterns
        logger.debug(f"is_conversational_query checking: '{query}'")
        
        # Test query against all conversational pattern categories
        greeting_match = self._fuzzy_match_patterns(query, self.greeting_patterns, threshold=0.6)
        knowledge_match = self._fuzzy_match_patterns(query, self.knowledge_patterns, threshold=0.6)
        acknowledgement_match = self._fuzzy_match_patterns(query, self.acknowledgement_patterns, threshold=0.6)
        small_talk_match = self._fuzzy_match_patterns(query, self.small_talk_patterns, threshold=0.6)
        farewell_match = self._fuzzy_match_patterns(query, self.farewell_patterns, threshold=0.6)
        clarification_match = self._fuzzy_match_patterns(query, self.clarification_patterns, threshold=0.6)
        
        # Return true if any conversational pattern matches
        result = (greeting_match or knowledge_match or acknowledgement_match or 
                 small_talk_match or farewell_match or clarification_match)
        
        logger.debug(f"is_conversational_query individual matches - greeting:{greeting_match}, knowledge:{knowledge_match}, ack:{acknowledgement_match}, small_talk:{small_talk_match}, farewell:{farewell_match}, clarification:{clarification_match}")
        logger.debug(f"is_conversational_query result: {result}")
        return result
    
    def handle_conversation(self, query: str, stream=False):
        """
        Process conversational queries and generate appropriate responses.
        Ensures no document references are included in responses.
        """
        # Apply spelling corrections to improve pattern matching
        corrected_query = self._correct_spelling(query)
        query_lower = corrected_query.lower().strip()
        response = None
        
        # Check if user has previous conversation history
        is_return_user = len(self.memory.chat_memory.messages) > 0
        
        # Log query processing for debugging
        logger.debug(f"Processing conversational query: '{query}' -> '{corrected_query}'")
        
        # Test query against all pattern categories to determine response type
        greeting_match = self._fuzzy_match_patterns(corrected_query, self.greeting_patterns, threshold=0.6)
        knowledge_match = self._fuzzy_match_patterns(corrected_query, self.knowledge_patterns, threshold=0.6)
        acknowledgement_match = self._fuzzy_match_patterns(corrected_query, self.acknowledgement_patterns, threshold=0.6)
        small_talk_match = self._fuzzy_match_patterns(corrected_query, self.small_talk_patterns, threshold=0.6)
        clarification_match = self._fuzzy_match_patterns(corrected_query, self.clarification_patterns, threshold=0.6)
        farewell_match = self._fuzzy_match_patterns(corrected_query, self.farewell_patterns, threshold=0.6)
        
        logger.debug(f"Pattern matches - greeting:{greeting_match}, knowledge:{knowledge_match}, ack:{acknowledgement_match}, small_talk:{small_talk_match}, clarification:{clarification_match}, farewell:{farewell_match}")
        
        # Process greetings with priority (most common conversational pattern)
        if greeting_match:
            logger.debug("Matched greeting patterns")
            if is_return_user:
                # Personalized responses for returning users
                contextual_greetings = [
                    "Hey again! Good to see you back! I'm doing great, thanks for asking. What can I help you with today?",
                    "Hi there! Welcome back! I'm having a wonderful day helping students. What APU info do you need?",
                    "Hello! Nice to chat with you again! I'm doing fantastic. What would you like to explore about APU today?",
                    "Hey! Great to see you return! I'm doing awesome, hope you are too! How can I help with APU stuff?"
                ]
                response = random.choice(contextual_greetings)
            else:
                # Standard greeting responses for new users
                response = random.choice(self.greeting_responses)
        
        # Process system knowledge and capability questions
        elif knowledge_match:
            logger.debug("Matched knowledge patterns")
            if is_return_user:
                base_response = random.choice(self.knowledge_responses)
                response = base_response + "\n\nSince we've chatted before, feel free to dive deeper into any topic!"
            else:
                response = random.choice(self.knowledge_responses)
        
        # Process user acknowledgments and appreciation
        elif acknowledgement_match:
            logger.debug("Matched acknowledgement patterns")
            if is_return_user and len(self.memory.chat_memory.messages) > 2:
                # Enhanced acknowledgment responses for ongoing conversations
                contextual_acknowledgements = [
                    "You're very welcome! I absolutely love helping with APU questions! Anything else you'd like to explore?",
                    "My pleasure! I'm so glad I could help clear things up about APU. What else can I assist with?",
                    "Happy to help! That's what makes my day! Feel free to ask about anything else APU-related!",
                    "You're welcome! I'm here whenever you need more APU information or guidance!"
                ]
                response = random.choice(contextual_acknowledgements)
            else:
                response = random.choice(self.acknowledgement_responses)
        
        # Process casual conversation and small talk
        elif small_talk_match:
            logger.debug("Matched small talk patterns")
            response = random.choice(self.small_talk_responses)
        
        # Process clarification requests and confusion
        elif clarification_match:
            logger.debug("Matched clarification patterns")
            if is_return_user:
                # Provide context-aware clarification for returning users
                recent_topics = self._extract_recent_topics()
                if recent_topics:
                    response = f"No worries! I see we were talking about {recent_topics}. Would you like me to explain that better, or is there something else about APU you'd like to understand?"
                else:
                    response = random.choice(self.clarification_responses)
            else:
                response = random.choice(self.clarification_responses)
        
        # Process farewell messages (checked last to avoid pattern conflicts)
        elif farewell_match:
            logger.debug("Matched farewell patterns")
            if is_return_user:
                # Personalized farewell responses for returning users
                contextual_farewells = [
                    "Goodbye! It was great chatting with you again. Feel free to return anytime for APU help!",
                    "Take care! Thanks for the conversation - I'm always here when you need APU assistance!",
                    "See you later! I enjoyed helping you today. Come back whenever you have APU questions!",
                    "Farewell! Hope our chat was helpful. Don't hesitate to return for more APU info!"
                ]
                response = random.choice(contextual_farewells)
            else:
                response = random.choice(self.farewell_responses)
        
        # Fallback response when no specific patterns match
        if not response:
            logger.debug("No patterns matched, using fallback response")
            if is_return_user:
                response = "I'm here and ready to help! What APU questions are on your mind today?"
            else:
                response = "Hi there! I'm here to help you with anything APU-related. What would you like to know?"
        
        logger.debug(f"Selected response type based on pattern matching")
        
        # Handle response delivery based on streaming preference
        if not stream:
            # Store conversation in memory and return complete response
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            return response
        else:
            # Prepare for streaming response delivery
            self._last_response = response
            self._last_query = query
            return self._stream_response(response)

    def _extract_recent_topics(self) -> str:
        """Extract topic keywords from recent conversation history for contextual responses."""
        if not self.memory.chat_memory.messages or len(self.memory.chat_memory.messages) < 2:
            return ""
        
        # Analyze last 4 messages for topic context
        recent_messages = self.memory.chat_memory.messages[-4:]
        text_content = []
        
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                text_content.append(msg.content.lower())
        
        combined_text = " ".join(text_content)
        
        # Define topic categories and their associated keywords
        topic_keywords = {
            'academic stuff': ['course', 'program', 'degree', 'academic', 'study', 'class', 'subject'],
            'administrative procedures': ['application', 'registration', 'enrollment', 'form', 'procedure'],
            'financial matters': ['fee', 'payment', 'scholarship', 'financial', 'cost', 'tuition'],
            'student services': ['medical', 'insurance', 'card', 'collect', 'service', 'support'],
            'campus facilities': ['building', 'location', 'address', 'campus', 'library', 'lab'],
        }
        
        # Identify which topics were mentioned in recent conversation
        mentioned_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                mentioned_topics.append(topic)
        
        # Format topic list for natural language response
        if mentioned_topics:
            if len(mentioned_topics) == 1:
                return mentioned_topics[0]
            elif len(mentioned_topics) == 2:
                return f"{mentioned_topics[0]} and {mentioned_topics[1]}"
            else:
                return f"{', '.join(mentioned_topics[:-1])}, and {mentioned_topics[-1]}"
        
        return "APU topics"

    def _stream_response(self, response: str):
        """Deliver response as a stream of words with natural timing delays."""
        # Store conversation in memory before streaming
        self.memory.chat_memory.add_user_message(self._last_query)
        self.memory.chat_memory.add_ai_message(response)
        
        # Stream response word by word with configured delays
        words = response.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(self.stream_delay)