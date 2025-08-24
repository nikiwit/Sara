"""
Input processing for query analysis and enhancement.
"""

import re
import logging
from typing import List, Dict, Any
from collections import Counter, defaultdict

# NLP imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

from config import config
from sara_types import QueryType

logger = logging.getLogger("Sara")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class InputProcessor:
    """Processes user input to enhance retrieval quality."""
    
    def __init__(self):
        """Initialize the input processor with NLP components."""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add spaCy semantic processor (Phase 4: Full migration with monitoring)
        self.use_spacy_semantics = config.USE_ENHANCED_SEMANTICS
        
        if self.use_spacy_semantics:
            try:
                from spacy_semantic_processor import SpacySemanticProcessor
                self.spacy_processor = SpacySemanticProcessor()
                if self.spacy_processor.initialized:
                    logger.info("spaCy semantic processor loaded successfully")
                else:
                    self.use_spacy_semantics = False
                    logger.warning("spaCy processor failed to initialize")
            except Exception as e:
                logger.warning(f"spaCy semantic processor not available: {e}")
                self.spacy_processor = None
                self.use_spacy_semantics = False
        else:
            logger.info("Enhanced semantics disabled by configuration")
            self.spacy_processor = None
        
        
        
        pass
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query by converting to lowercase, fixing grammar issues,
        removing punctuation, and standardizing whitespace.
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Apply spaCy grammar correction first if available
        if self.use_spacy_semantics and self.spacy_processor:
            try:
                normalized = self.spacy_processor.correct_grammar(normalized)
            except Exception as e:
                logger.warning(f"spaCy grammar correction failed: {e}")
        
        # General login issue normalization (removed specific APSpace patterns)
        if re.search(r'\b(?:cannot|can not|unable to|trouble|problem|issue).*?(?:login|log in|sign in|access|signin)\b', normalized):
            normalized = re.sub(r'\b(?:cannot|can not|unable to|trouble|problem|issue).*?(?:login|log in|sign in|access|signin)\b', 'unable sign in', normalized)
        
        # Remove punctuation except apostrophes in contractions
        normalized = re.sub(r'[^\w\s\']', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand abbreviations in the query with their full forms.
        
        Args:
            query: Original query string
            
        Returns:
            Query with expanded abbreviations
        """
        return query
    
    def analyze_query(self, query: str, conversation_context: str = None) -> Dict[str, Any]:
        """
        Analyze the query to extract features for better retrieval.
        
        Args:
            query: Original user query
            conversation_context: Optional conversation history for context-aware processing
        
        Returns:
            A dictionary with query analysis including:
            - tokens: List of tokens
            - normalized_query: Normalized query text
            - lemmatized_tokens: Lemmatized tokens
            - keywords: Key terms (non-stopwords)
            - query_type: Type of query (enum)
            - expanded_queries: List of expanded query variants
            - extracted_terms: Education/APU-specific terms extracted
            - contextual_query: Context-enhanced query (if context provided)
        """
        # Apply conversational context enhancement if provided
        contextual_query = query
        if conversation_context:
            contextual_query = self._enhance_query_with_context(query, conversation_context)
        
        # Normalize the query (use contextual query if enhanced)
        normalized_query = self.normalize_query(contextual_query)
        
        # Expand abbreviations
        expanded_query = self.expand_abbreviations(normalized_query)
        
        # Tokenize
        tokens = word_tokenize(expanded_query)
        
        # Remove stopwords and get keywords
        keywords = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Generate n-grams for phrases (bigrams and trigrams)
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        
        # Extract bigram and trigram phrases
        bigram_phrases = [' '.join(bg) for bg in bigrams]
        trigram_phrases = [' '.join(tg) for tg in trigrams]
        
        # Extract education/APU-specific terms
        edu_terms = self._extract_education_terms(normalized_query, tokens)
        
        # Identify query type (enhanced with APU-specific types)
        query_type = self.classify_query_type(normalized_query, tokens, edu_terms)
        
        # Generate expanded queries if enabled
        expanded_queries = []
        if config.USE_QUERY_EXPANSION:
            expanded_queries = self.expand_query(expanded_query, keywords)
        
        return {
            "original_query": query,
            "contextual_query": contextual_query if conversation_context else query,
            "normalized_query": normalized_query,
            "expanded_query": expanded_query,
            "tokens": tokens,
            "keywords": keywords,
            "lemmatized_tokens": lemmatized_tokens,
            "bigram_phrases": bigram_phrases,
            "trigram_phrases": trigram_phrases,
            "query_type": query_type,
            "expanded_queries": expanded_queries,
            "edu_terms": edu_terms
        }
    
    def _extract_education_terms(self, query: str, tokens: List[str]) -> List[str]:
        """
        Extract education and APU-specific terms from the query.
        
        Args:
            query: Normalized query string
            tokens: List of tokens
            
        Returns:
            List of education/APU-specific terms
        """
        edu_terms = []
        
        # Common education terms
        edu_keywords = [
            "exam", "docket", "test", "assessment", "module", "course", "class",
            "lecturer", "referral", "resit", "retake", "fee", "payment", "transcript",
            "certificate", "registration", "visa", "EC", "extenuating", "circumstances",
            "attendance", "appeal", "degree", "diploma", "graduation", "intake", "semester",
            "programme", "program", "scholarship", "internship", "placement", "deferment",
            "withdrawal", "compensation", "supervisor", "submission", "deadline",
            "assignment", "project", "dissertation", "library", "campus", "accommodation",
            "timetable", "schedule", "result", "grade", "mark", "CGPA", "GPA"
        ]
        
        # Check for education terms in tokens
        for token in tokens:
            if token.lower() in edu_keywords:
                edu_terms.append(token.lower())
        
        # Check for course codes (e.g., APU1F2103CS)
        course_codes = re.findall(r'\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]{2,}\b', query)
        edu_terms.extend(course_codes)
        
        
        return list(set(edu_terms))  # Remove duplicates
    
    def classify_query_type(self, query: str, tokens: List[str], edu_terms: List[str]) -> QueryType:
        """
        Classify the query into different types, including APU-specific types.
        """
        # Check for command queries first
        command_patterns = [
            r'\b(?:exit|quit|bye|goodbye)\b',
            r'\bclear\b',
            r'\breindex\b',
            r'\bstats\b',
            r'\bhelp\b'
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMMAND
        
        # Check for identity queries (ENHANCED WITH MORE PATTERNS)
        identity_patterns = [
            # Basic identity questions
            r'\bwho\s+(?:are|r)\s+(?:you|u)\b',
            r'\bwhat\s+(?:are|r)\s+(?:you|u)\b',
            r'\byour\s+name\b',
            r'\bur\s+name\b',
            r'\bwhat\'s\s+your\s+name\b',
            r'\bintroduce\s+yourself\b',
            r'\babout\s+yourself\b',
            r'\bwho\s+(?:created|made|built|developed)\s+(?:you|u|this)\b',
            
            # Technical/system questions
            r'\bhow\s+(?:were|r)\s+(?:you|u)\s+(?:made|created|built|developed)\b',
            r'\bwhat\s+(?:can|do)\s+you\s+do\b',
            r'\byour\s+purpose\b',
            r'\bhow\s+do\s+you\s+work\b',
            r'\bwhat\s+(?:model|llm|language model|embedding model)\s+(?:are|is|do)\s+you\b',
            r'\bwhat\s+version\s+(?:are|is|do)\s+you\b',
            r'\bwhat\s+(?:are|is)\s+(?:your|ur)\s+(?:specs|capabilities|parameters|tokens|context)\b',
            r'\bwhat\s+(?:tech|technology|stack|framework)\s+(?:are|is|do)\s+you\s+use\b',
            r'\bwhich\s+(?:model|llm|language model)\s+(?:are|is|do)\s+you\b',
            r'\byour\s+(?:model|llm|technology|capabilities)\b',
            r'\bsystem\s+specs\b',
            r'\btell\s+me\s+about\s+(?:you|yourself|the system|this system)\b',
            
            # Key terms that should trigger identity detection
            r'\byou\s+(?:a|an)\s+(?:llm|language model|ai|model|bot)\b',
            r'\b(?:llm|language model|ai model|embedding|vector)\s+(?:used|using|powering)\b',
            r'\b(?:built|created|made)\s+(?:with|using)\b',
            r'\bare\s+you\s+(?:a|an|the)\s+(?:llm|language model|ai|chatbot|assistant)\b',
            r'\bhow\s+(?:large|big|small)\s+(?:are|is)\s+(?:you|your|the)\s+(?:model|context|window)\b'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.IDENTITY
        
        # Special case: queries containing "model" or "llm" without other context are likely identity questions
        model_terms = ["model", "llm", "language model", "embedding", "claude", "gpt", "openai", "anthropic", "huggingface"]
        if any(term in query.lower() for term in model_terms) and len(tokens) < 8:
            return QueryType.IDENTITY
        
        # Check for conversational queries - Enhanced with fuzzy matching support
        conversational_patterns = [
            # Improved greeting patterns with common typos and variations
            r'\b(?:hi+|h[ie]llo*|hey+|hiya?|greetings?|howdy|good\s*(?:morning|afternoon|evening|day)|what\'?s\s*up|sup|hai|helo+)\b',
            r'\bhow\s+(?:are|r|is)\s+(?:you|u|ya?)\b',
            r'\bhow\s+(?:are|r)\s+(?:you|u)\s+(?:doing|going)\b',
            r'\b(?:thanks?|thank\s*(?:you|u)|ty|thx|thanx)\b',
            r'\bappreciate\s*(?:it|that|this)\b',
            # Addressing/naming queries
            r'\bhow\s+(?:can|do|should)\s+i\s+(?:call|address|refer\s+to)\s+(?:you|u)\b',
            r'\bwhat\s+(?:can|do|should)\s+i\s+call\s+(?:you|u)\b',
            r'\bwhat\s+should\s+i\s+call\s+(?:you|u)\b',
            r'\bhow\s+should\s+i\s+address\s+(?:you|u)\b',
            r'\bwhat\s+do\s+i\s+call\s+(?:you|u)\b',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONVERSATIONAL
        
        # Check for APU-specific academic queries
        academic_patterns = [
            r'\b(?:exam|docket|test|assessment|module|course|class|lecture|assignment|submission)\b',
            r'\b(?:referral|resit|retake|grade|mark|result|CGPA|GPA)\b',
            r'\b(?:dissertation|project|FYP|lecturer|supervisor|class|timetable|schedule)\b',
            r'\b(?:registration|compensation|curriculum)\b'
        ]
        
        for pattern in academic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.ACADEMIC
        
        # Check for APU-specific administrative queries
        administrative_patterns = [
            r'\b(?:EC|extenuating circumstances|appeal|application|form|document|ID card)\b',
            r'\b(?:accommodation|housing|library|deadline|attendance|participation)\b',
            r'\b(?:visa|student pass|internship|placement|deferment|withdrawal|letter|certificate)\b',
            r'\b(?:transcript|admission|orientation|enrollment|transfer|VC\'s list)\b'
        ]
        
        for pattern in administrative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.ADMINISTRATIVE
        
        # Check for APU-specific financial queries
        financial_patterns = [
            r'\b(?:fee|payment|pay|cash|credit|debit|invoice|receipt|outstanding|due|overdue|installment)\b',
            r'\b(?:scholarship|funding|financial aid|bursary|loan|charge|refund|deposit)\b'
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FINANCIAL
        
        # Fall back to standard query classification if no APU-specific match
        
        # Check for factual queries
        factual_patterns = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+are\b',
            r'\bwho\s+is\b',
            r'\bwhen\s+\w+\b',
            r'\bwhere\s+\w+\b',
            r'\blist\b',
            r'\bdefine\b',
            r'\btell\s+me\s+about\b'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FACTUAL
        
        # Check for procedural queries
        procedural_patterns = [
            r'\bhow\s+to\b',
            r'\bhow\s+do\s+I\b',
            r'\bhow\s+can\s+I\b',
            r'\bsteps\s+to\b',
            r'\bguide\b',
            r'\btutorial\b',
            r'\bprocedure\b',
            r'\bprocess\b'
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.PROCEDURAL
        
        # Check for conceptual queries
        conceptual_patterns = [
            r'\bexplain\b',
            r'\bconcept\b',
            r'\btheory\b',
            r'\bwhy\s+\w+\b',
            r'\breasons?\s+for\b',
            r'\bmean\b',
            r'\bunderstand\b'
        ]
        
        for pattern in conceptual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONCEPTUAL
        
        # Check for comparative queries
        comparative_patterns = [
            r'\bcompare\b',
            r'\bcontrast\b',
            r'\bdifference\b',
            r'\bsimilar\b',
            r'\bversus\b',
            r'\bvs\b',
            r'\bbetter\b',
            r'\badvantages?\b',
            r'\bdisadvantages?\b'
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARATIVE
        
        # Check for exploratory queries
        exploratory_patterns = [
            r'\btell\s+me\s+more\b',
            r'\blearn\s+about\b',
            r'\bdiscover\b',
            r'\bexplore\b',
            r'\boverview\b',
            r'\bintroduction\b'
        ]
        
        for pattern in exploratory_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.EXPLORATORY
        
        # Default to unknown if no patterns match
        return QueryType.UNKNOWN
    
    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Expand the query with synonyms, semantic variations, and APU-specific terms.
        Enhanced for better matching of time-related and domain-specific queries.
        
        Returns:
            List of expanded query strings
        """
        # Try spaCy semantic processor first (following migration guide)
        if self.use_spacy_semantics and self.spacy_processor:
            try:
                spacy_expansions = self.spacy_processor.semantic_query_expansion(query)
                if spacy_expansions and len(spacy_expansions) > 1:
                    logger.info("Using spaCy semantic expansion")
                    return spacy_expansions
            except Exception as e:
                logger.warning(f"spaCy processor failed, falling back to traditional method: {e}")
        
        # Fallback to traditional method
        logger.info("Using traditional expansion method")
        return self._traditional_expand_query(query, keywords)
    
    def _traditional_expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """Traditional query expansion method as fallback."""
        expanded_queries = [query]  # Start with the original query
        
        # Skip expansion for very short queries or conversational queries
        if len(keywords) < 2:
            return expanded_queries
        
        # Add semantic variations for common query patterns
        semantic_variations = self._generate_semantic_variations(query, keywords)
        for variation in semantic_variations:
            if variation not in expanded_queries:
                expanded_queries.append(variation)
        
        
        
        # Generate contextual variations for specific domains
        domain_variations = self._generate_domain_variations(query, keywords)
        for variation in domain_variations:
            if variation not in expanded_queries:
                expanded_queries.append(variation)
        
        # If we haven't generated enough variations, try some combinations
        if len(expanded_queries) < config.EXPANSION_FACTOR and len(keywords) >= 3:
            # Generate variations by removing one non-essential keyword at a time
            for i in range(len(keywords)):
                variation = ' '.join(keywords[:i] + keywords[i+1:])
                if variation not in expanded_queries:
                    expanded_queries.append(variation)
                
                # Stop if we have enough variations
                if len(expanded_queries) >= config.EXPANSION_FACTOR:
                    break
        
        return expanded_queries[:config.EXPANSION_FACTOR]  # Limit to avoid too many variations
    
    def _generate_semantic_variations(self, query: str, keywords: List[str]) -> List[str]:
        """
        Generate semantic variations for common query patterns.
        
        Args:
            query: Original query
            keywords: Extracted keywords
            
        Returns:
            List of semantic variations
        """
        variations = []
        
        # Time-related query variations (for library hours, etc.)
        if any(word in keywords for word in ['when', 'time', 'hours', 'schedule']):
            if 'library' in keywords:
                variations.extend([
                    'library operating hours',
                    'library opening hours', 
                    'library schedule',
                    'when is library open',
                    'library operation time',
                    'what time library open'
                ])
            if 'close' in keywords or 'closes' in keywords:
                variations.append('operating hours')
                variations.append('opening time')
            if 'open' in keywords or 'opens' in keywords:
                variations.append('operating hours')
                variations.append('opening time')
        
        # General password/login related variations (removed APSpace-specific fixes)
        if any(word in keywords for word in ['password', 'login', 'forgot']):
            if 'forgot' in keywords and 'password' in keywords:
                variations.extend([
                    'password recovery',
                    'password reset',
                    'login credentials',
                    'authentication help'
                ])
        
        # Fee/payment related variations
        if any(word in keywords for word in ['fee', 'pay', 'payment']):
            variations.extend([
                'tuition payment',
                'fee payment process',
                'university fees',
                'payment procedure'
            ])
        
        return variations
    
    def _generate_domain_variations(self, query: str, keywords: List[str]) -> List[str]:
        """
        Generate domain-specific variations for APU queries.
        
        Args:
            query: Original query (currently unused but kept for future enhancements)
            keywords: Extracted keywords
            
        Returns:
            List of domain variations
        """
        variations = []
        
        # Library-specific variations
        if 'library' in keywords:
            variations.extend([
                'APU library',
                'university library',
                'learning resources center'
            ])
        
        # Academic-related variations
        if any(word in keywords for word in ['course', 'class', 'module']):
            variations.extend([
                'academic program',
                'study program',
                'university course'
            ])
        
        # Administrative variations
        if any(word in keywords for word in ['visa', 'immigration', 'pass']):
            variations.extend([
                'student pass',
                'immigration matters',
                'visa renewal'
            ])
        
        return variations
    
    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """
        Enhance the query using conversation context for better follow-up question handling.
        
        Args:
            query: Current user query
            context: Recent conversation history
            
        Returns:
            Enhanced query with conversational context
        """
        try:
            # Check if this is a follow-up question that needs context
            if not self._is_followup_question(query):
                return query
            
            # Extract entities and topics from recent context
            context_entities = self._extract_context_entities(context)
            context_topics = self._extract_context_topics(context)
            
            # Apply query reformulation based on context
            enhanced_query = self._reformulate_with_context(
                query, context_entities, context_topics
            )
            
            return enhanced_query
            
        except Exception as e:
            logger.debug(f"Context enhancement failed: {e}")
            return query
    
    
    def _is_followup_question(self, query: str) -> bool:
        """
        Determine if a query is likely a follow-up question needing context.
        
        Args:
            query: User query to analyze
            
        Returns:
            True if this appears to be a follow-up question
        """
        query_lower = query.lower().strip()
        
        # Common follow-up question patterns
        followup_patterns = [
            # Pronoun-based questions
            r'\b(?:how much|what about|what is)\s+(?:it|this|that|they?)\b',
            r'\b(?:it|this|that|they?)\s+(?:cost|costs?)\b',
            r'\b(?:how|what|when|where)\s+(?:about|for)\s+(?:it|this|that|they?)\b',
            
            # Generic cost/pricing questions
            r'^(?:how much|what(?:\'s| is) the)\s+(?:cost|price|fee)\??$',
            r'^(?:cost|price|fee)\??$',
            r'^how much\??$',
            
            # Generic time/schedule questions
            r'^(?:when|what time)\??$',
            r'^(?:schedule|timing|hours)\??$',
            
            # Generic location questions
            r'^(?:where|location)\??$',
            
            # Generic process questions
            r'^(?:how|process|procedure)\??$',
            
            # Other vague follow-ups
            r'^(?:what about|how about|and)\s+\w+\??$',
            r'^(?:more info|details|information)\??$'
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Additional heuristics: short questions with vague terms
        vague_terms = ['it', 'this', 'that', 'they', 'them', 'cost', 'price', 'when', 'where', 'how']
        words = query_lower.split()
        
        if (len(words) <= 4 and 
            any(term in words for term in vague_terms) and 
            query_lower.endswith('?') or len(words) <= 2):
            return True
        
        return False
    
    def _extract_context_entities(self, context: str) -> List[str]:
        """
        Extract key entities from conversation context.
        
        Args:
            context: Recent conversation history
            
        Returns:
            List of important entities mentioned in context
        """
        entities = []
        context_lower = context.lower()
        
        # APU-specific entities to look for
        apu_entities = {
            # Academic entities
            'visa renewal', 'student pass', 'visa extension', 'immigration',
            'fee payment', 'tuition fee', 'university fee', 'payment',
            'reference letter', 'recommendation letter', 'academic reference',
            'library hours', 'library services', 'library card',
            'parking permit', 'parking pass', 'vehicle registration',
            'apkey password', 'login credentials', 'student portal',
            'exam docket', 'examination slip', 'exam registration',
            'course registration', 'module enrollment', 'class schedule',
            'transcript request', 'academic transcript', 'official transcript',
            'graduation ceremony', 'convocation', 'degree certificate',
            
            # Services and facilities
            'student services', 'academic office', 'international office',
            'finance office', 'registry office', 'it services',
            'counseling services', 'career services', 'accommodation',
            
            # Academic programs
            'computer science', 'information technology', 'engineering',
            'business administration', 'accounting', 'finance',
        }
        
        # Look for multi-word entities first (more specific)
        for entity in sorted(apu_entities, key=len, reverse=True):
            if entity in context_lower:
                entities.append(entity)
        
        # Look for course codes and IDs
        course_codes = re.findall(r'\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]{2,}\b', context)
        entities.extend(course_codes)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities[:5]  # Limit to most recent/relevant entities
    
    def _extract_context_topics(self, context: str) -> List[str]:
        """
        Extract main topics from conversation context.
        
        Args:
            context: Recent conversation history
            
        Returns:
            List of main topics discussed
        """
        topics = []
        context_lower = context.lower()
        
        # Topic keywords mapping
        topic_mapping = {
            'visa_immigration': ['visa', 'immigration', 'student pass', 'permit', 'renewal', 'extension'],
            'fees_payment': ['fee', 'payment', 'tuition', 'cost', 'price', 'charge', 'invoice'],
            'library_services': ['library', 'book', 'borrow', 'study', 'resource', 'hours'],
            'parking': ['parking', 'vehicle', 'car', 'permit', 'registration'],
            'academic_records': ['transcript', 'certificate', 'record', 'grade', 'result'],
            'registration': ['registration', 'enrollment', 'course', 'module', 'class'],
            'it_support': ['password', 'apkey', 'login', 'portal', 'system', 'access'],
            'reference_letters': ['reference', 'letter', 'recommendation', 'document']
        }
        
        # Check which topics are present in context
        for topic, keywords in topic_mapping.items():
            if any(keyword in context_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _reformulate_with_context(self, query: str, entities: List[str], topics: List[str]) -> str:
        """
        Reformulate the query by incorporating contextual information.
        
        Args:
            query: Original user query
            entities: Relevant entities from context
            topics: Relevant topics from context
            
        Returns:
            Reformulated query with context
        """
        query_lower = query.lower().strip()
        
        # Handle cost/price questions
        if re.search(r'\b(?:how much|cost|price|fee)\b', query_lower):
            if any('visa' in entity or 'immigration' in entity for entity in entities):
                return f"visa renewal cost fee {query}"
            elif any('parking' in entity for entity in entities):
                return f"parking permit cost fee {query}"
            elif 'fees_payment' in topics:
                return f"university fee payment cost {query}"
        
        # Handle time/schedule questions
        if re.search(r'\b(?:when|time|hours|schedule)\b', query_lower):
            if any('library' in entity for entity in entities):
                return f"library opening hours schedule {query}"
            elif any('office' in entity for entity in entities):
                return f"office hours operating time {query}"
        
        # Handle location questions
        if re.search(r'\b(?:where|location)\b', query_lower):
            if entities:
                # Use the most specific entity
                primary_entity = entities[0]
                return f"{primary_entity} location address {query}"
        
        # Handle process/procedure questions
        if re.search(r'\b(?:how|process|procedure)\b', query_lower):
            if entities:
                primary_entity = entities[0]
                return f"{primary_entity} process procedure {query}"
        
        # General enhancement: add the most relevant entity if available
        if entities and len(query.split()) <= 3:
            return f"{entities[0]} {query}"
        
        return query