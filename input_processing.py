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

from config import Config
from apurag_types import QueryType

logger = logging.getLogger("CustomRAG")

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
        
        # Enhanced synonym dictionary with APU-specific terms
        self.synonyms = {
            "document": ["file", "text", "paper", "doc", "content"],
            "find": ["locate", "search", "discover", "retrieve", "get"],
            "explain": ["describe", "clarify", "elaborate", "detail", "elucidate"],
            "show": ["display", "present", "exhibit", "demonstrate"],
            "information": ["info", "data", "details", "facts", "knowledge"],
            "create": ["make", "generate", "produce", "build", "develop"],
            "modify": ["change", "alter", "adjust", "edit", "update"],
            "remove": ["delete", "eliminate", "erase", "take out"],
            "important": ["significant", "essential", "critical", "key", "vital"],
            "problem": ["issue", "challenge", "difficulty", "trouble", "obstacle"],
            "solution": ["answer", "resolution", "fix", "remedy", "workaround"],
            # APU specific synonyms
            "exam": ["examination", "test", "assessment", "final", "docket"],
            "course": ["module", "class", "subject", "unit"],
            "assignment": ["coursework", "homework", "project", "paper", "submission"],
            "fee": ["payment", "charge", "cost", "expense", "tuition"],
            "lecturer": ["professor", "teacher", "instructor", "tutor", "faculty"],
            "result": ["grade", "mark", "score", "outcome", "performance"],
            "deferment": ["postponement", "delay", "extension", "adjournment"],
            "referral": ["resit", "retake", "redo", "resubmission"],
            "attendance": ["presence", "participation", "turn up", "show up"],
            "certificate": ["diploma", "degree", "qualification", "credential"],
            "transcript": ["record", "academic record", "results"],
            "docket": ["exam slip", "hall ticket", "exam ticket"],
            "EC": ["extenuating circumstances", "special circumstances", "exception", "exemption"],
            "registration": ["enrollment", "signup", "joining", "application"],
            "scholarship": ["funding", "financial aid", "grant", "bursary"],
            "internship": ["industrial training", "placement", "practical training", "industry experience"],
            "visa": ["student pass", "immigration document", "permit"],
            "APU": ["Asia Pacific University", "university", "school", "campus"],
            "timetable": ["schedule", "class schedule", "calendar"],
            "intake": ["batch", "cohort", "entry", "enrollment period"],
        }
        
        # APU-specific abbreviations
        self.apu_abbreviations = {
            "EC": ["extenuating circumstances"],
            "APU": ["Asia Pacific University"],
            "APIIT": ["Asia Pacific Institute of Information Technology"],
            "VC": ["Vice Chancellor"],
            "MPU": ["Mata Pelajaran Umum"],
            "PL": ["Programme Leader"],
            "SIS": ["Student Information System"],
            "EB": ["Examination Board"],
            "UAC": ["University Appeals Committee"],
            "SAC": ["School Appeals Committee"],
            "FYP": ["Final Year Project"],
            "LRT": ["Light Rail Transit"],
            "MCO": ["Movement Control Order"],
            "APKey": ["APU Key", "student ID"],
            "HOS": ["Head of School"],
            "CGPA": ["Cumulative Grade Point Average"],
            "GPA": ["Grade Point Average"],
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query by converting to lowercase, removing punctuation,
        and standardizing whitespace.
        """
        normalized = query.lower()
        
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
        words = query.split()
        expanded_query = query
        
        # Check for abbreviations and expand them
        for abbr, expansions in self.apu_abbreviations.items():
            if abbr in words:
                for expansion in expansions:
                    if expansion not in expanded_query:
                        expanded_query += f" {expansion}"
        
        return expanded_query
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract features for better retrieval.
        
        Returns:
            A dictionary with query analysis including:
            - tokens: List of tokens
            - normalized_query: Normalized query text
            - lemmatized_tokens: Lemmatized tokens
            - keywords: Key terms (non-stopwords)
            - query_type: Type of query (enum)
            - expanded_queries: List of expanded query variants
            - extracted_terms: Education/APU-specific terms extracted
        """
        # Normalize the query
        normalized_query = self.normalize_query(query)
        
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
        
        # Identify query type
        query_type = self.classify_query_type(normalized_query, tokens, edu_terms)
        
        # Generate expanded queries if enabled
        expanded_queries = []
        if Config.USE_QUERY_EXPANSION:
            expanded_queries = self.expand_query(expanded_query, keywords)
        
        return {
            "original_query": query,
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
        
        # Check for common abbreviations
        for abbr in self.apu_abbreviations.keys():
            if abbr in query:
                edu_terms.append(abbr.lower())
        
        # Remove duplicates
        return list(set(edu_terms))
    
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
        
        # Check for identity queries with comprehensive patterns
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
        
        # Check for conversational queries
        conversational_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
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
        Expand the query with synonyms and variations, including APU-specific terms.
        
        Returns:
            List of expanded query strings
        """
        # Start with the original query
        expanded_queries = [query]
        
        # Skip expansion for very short queries or conversational queries
        if len(keywords) < 2:
            return expanded_queries
        
        # Expand using synonyms
        for i, keyword in enumerate(keywords):
            if keyword in self.synonyms:
                for synonym in self.synonyms[keyword][:Config.EXPANSION_FACTOR]:
                    # Replace the keyword with its synonym
                    new_keywords = keywords.copy()
                    new_keywords[i] = synonym
                    expanded_query = ' '.join(new_keywords)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # Expand abbreviations
        for i, keyword in enumerate(keywords):
            if keyword.upper() in self.apu_abbreviations:
                for expansion in self.apu_abbreviations[keyword.upper()]:
                    # Replace the abbreviation with its expanded form
                    new_keywords = keywords.copy()
                    new_keywords[i] = expansion
                    expanded_query = ' '.join(new_keywords)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # If we haven't generated enough variations, try some combinations
        if len(expanded_queries) < Config.EXPANSION_FACTOR and len(keywords) >= 3:
            # Generate variations by removing one non-essential keyword at a time
            for i in range(len(keywords)):
                variation = ' '.join(keywords[:i] + keywords[i+1:])
                if variation not in expanded_queries:
                    expanded_queries.append(variation)
                
                # Stop if we have enough variations
                if len(expanded_queries) >= Config.EXPANSION_FACTOR:
                    break
        
        # Limit to avoid too many variations
        return expanded_queries[:Config.EXPANSION_FACTOR]