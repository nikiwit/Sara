"""
Ambiguous query detection and clarification handler for SARA chatbot.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger("Sara")


class AmbiguityHandler:
    """Handle detection and clarification of ambiguous queries using ensemble approach."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
        # Initialize pronoun disambiguation system
        self.pronoun_disambiguator = self._initialize_pronoun_system()
        
        self.validation_methods = [
            self._pattern_based_validation,
            self._context_completeness_validation,
            self._semantic_clarity_validation
        ]
        
        # Patterns for detecting ambiguous queries (after pronoun disambiguation)
        self.ambiguous_patterns = {
            'pronoun_reference': [
                r'\bhow do i renew it\b',
                r'\bwhat is it\b',
                r'\bhow does it work\b',
                r'\bhow do i fix it\b',
                r'\bwhere do i find it\b',
                r'\bwhen do i submit it\b',
                # Remove patterns that are now handled by disambiguation
                r'\bcan i (have|use) (it|this|that)\b',
            ],
            'incomplete_context': [
                r'^what are the requirements\??$',
                r'^what.*requirements\??$(?!.*(for|to|of|visa|renewal|application|course|library|graduation))',
                r'^how do i apply\??$',
                r'^where do i submit\??$',
                r'^what documents.*needed\??$',
                r'^how much does it cost\??$',
                r'^what.*fees\??$',
                r'^where.*office\??$',
                r'^when.*deadline\??$',
                r'^how long.*takes\??$'
            ],
            'generic_questions': [
                r'^what.*process\??$',
                r'^how.*procedure\??$',
                r'^what.*steps\??$',
                r'^how do i get\??$',
                r'^where can i\??$',
                r'^what do i need\??$'
            ],
            'technical_vague': [
                r'^how do i install\??$',
                r'^how do i setup\??$',
                r'^how do i configure\??$',
                r'^what.*version\??$',
                r'^how do i access\??$'
            ]
        }
        
        # Context-specific clarification templates
        self.clarification_templates = {
            'renewal': {
                'message': "I'd be happy to help with renewal! What would you like to renew?",
                'options': [
                    "• Student visa renewal",
                    "• Library book renewal", 
                    "• APKey password reset",
                    "• Parking pass renewal"
                ]
            },
            'requirements': {
                'message': "I can help with requirements! What do you need requirements for?",
                'options': [
                    "• Visa application requirements",
                    "• Course registration requirements",
                    "• Library access requirements",
                    "• Graduation requirements"
                ]
            },
            'application': {
                'message': "I can help with applications! What would you like to apply for?",
                'options': [
                    "• Course registration",
                    "• Scholarship application",
                    "• Leave of absence",
                    "• Transcript request"
                ]
            },
            'documents': {
                'message': "I can help with document requirements! What documents do you need for?",
                'options': [
                    "• Visa renewal documents",
                    "• Course registration documents",
                    "• Scholarship application documents",
                    "• Official transcript request"
                ]
            },
            'technical': {
                'message': "I can help with technical setup! What software or system do you need help with?",
                'options': [
                    "• Software installation (Solidworks, SQL Server, etc.)",
                    "• System access (Azure, APSpace, APKey)",
                    "• Network and connectivity issues",
                    "• Hardware setup and configuration"
                ]
            },
            'fees': {
                'message': "I can help with fees and costs! What fees are you asking about?",
                'options': [
                    "• Tuition and course fees",
                    "• Visa renewal fees",
                    "• Library fines and charges",
                    "• Parking fees and rates"
                ]
            },
            'location': {
                'message': "I can help you find the right office or location! What are you looking for?",
                'options': [
                    "• Student Services (Level 1, New Campus)",
                    "• Library services and locations",
                    "• IT Support and help desk",
                    "• Academic department offices"
                ]
            },
            'generic': {
                'message': "I'd be happy to help! Could you provide more specific details about what you're looking for?",
                'options': [
                    "• Academic procedures (registration, grades, etc.)",
                    "• Visa and immigration services",
                    "• Library and IT services", 
                    "• Financial services (fees, scholarships)"
                ]
            }
        }
    
    def is_ambiguous(self, query: str) -> bool:
        """
        Check if query is ambiguous using ensemble validation methods.
        Includes pronoun disambiguation following industry patterns.
        
        Args:
            query: User input text
            
        Returns:
            True if query is ambiguous, False otherwise
        """
        # Step 1: Apply pronoun disambiguation
        disambiguated_query = self._apply_pronoun_disambiguation(query)
        
        # Step 2: If query was disambiguated, it's not ambiguous
        if disambiguated_query != query:
            logger.debug(f"Pronoun disambiguation applied: '{query}' -> '{disambiguated_query}'")
            return False
        
        # Step 3: Apply traditional ambiguity detection
        ambiguity_score = self.calculate_ambiguity_score(query)
        return ambiguity_score >= self.confidence_threshold
    
    def calculate_ambiguity_score(self, query: str) -> float:
        """
        Calculate ambiguity score using ensemble validation methods.
        
        Args:
            query: User input text
            
        Returns:
            Ambiguity score (0.0 to 1.0)
        """
        scores = []
        
        for validation_method in self.validation_methods:
            try:
                score = validation_method(query)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Validation method failed: {e}")
                scores.append(0.0)
        
        if not scores:
            return 0.0
            
        ensemble_score = sum(scores) / len(scores)
        logger.debug(f"Ambiguity ensemble score: {ensemble_score:.2f} for query: {query[:50]}...")
        return ensemble_score
    
    def _pattern_based_validation(self, query: str) -> float:
        """Pattern-based ambiguity detection."""
        query_lower = query.lower().strip()
        
        for category, patterns in self.ambiguous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Pattern match - Category: {category}")
                    return 1.0
        
        return 0.0
    
    def _context_completeness_validation(self, query: str) -> float:
        """Check if query has sufficient context."""
        query_lower = query.lower()
        
        context_indicators = [
            'visa', 'passport', 'student', 'library', 'course', 'registration',
            'transcript', 'graduation', 'scholarship', 'application', 'renewal',
            'apspace', 'apkey', 'fee', 'payment', 'exam', 'grade',
            'parking', 'zone', 'solidworks', 'sql', 'azure', 'consultation',
            'lecturer', 'swift', 'bank', 'transfer', 'overstay', 'penalty',
            'office', 'deadline', 'installation', 'software', 'system'
        ]
        
        context_count = sum(1 for indicator in context_indicators if indicator in query_lower)
        
        question_words = ['what', 'how', 'where', 'when', 'why']
        has_question = any(query_lower.startswith(word) for word in question_words)
        
        if has_question and context_count == 0:
            return 0.8
        elif has_question and context_count == 1:
            return 0.3
        elif has_question and context_count > 1:
            return 0.1
        
        return 0.2
    
    def _semantic_clarity_validation(self, query: str) -> float:
        """Analyze semantic clarity of the query."""
        ambiguous_pronouns = ['it', 'this', 'that', 'they', 'them']
        pronoun_count = sum(1 for pronoun in ambiguous_pronouns if f' {pronoun} ' in f' {query.lower()} ')
        
        word_count = len(query.split())
        if word_count == 0:
            return 1.0
            
        pronoun_ratio = pronoun_count / word_count
        return min(pronoun_ratio * 2.0, 1.0)
    
    def get_clarification(self, query: str) -> str:
        """
        Generate appropriate clarification response for ambiguous query.
        
        Args:
            query: User input text
            
        Returns:
            Clarification message with context options
        """
        query_lower = query.lower().strip()
        
        # Determine the most appropriate clarification type
        clarification_type = self._determine_clarification_type(query_lower)
        template = self.clarification_templates.get(clarification_type, self.clarification_templates['generic'])
        
        # Build clarification response
        response_parts = [template['message'], ""]
        response_parts.extend(template['options'])
        response_parts.extend([
            "",
            "Please let me know which option you're interested in, or provide more specific details about your question."
        ])
        
        return "\n".join(response_parts)
    
    def _determine_clarification_type(self, query: str) -> str:
        """
        Determine the most appropriate clarification template type.
        
        Args:
            query: Normalized query text
            
        Returns:
            Clarification type key
        """
        # Check for specific keywords to determine context
        keyword_mapping = {
            'renew': 'renewal',
            'renewal': 'renewal',
            'requirements': 'requirements',
            'requirement': 'requirements', 
            'apply': 'application',
            'application': 'application',
            'documents': 'documents',
            'document': 'documents',
            'papers': 'documents',
            'install': 'technical',
            'setup': 'technical',
            'configure': 'technical',
            'access': 'technical',
            'version': 'technical',
            'fees': 'fees',
            'cost': 'fees',
            'price': 'fees',
            'office': 'location',
            'location': 'location',
            'where': 'location'
        }
        
        for keyword, clarification_type in keyword_mapping.items():
            if keyword in query:
                return clarification_type
                
        return 'generic'
    
    def extract_context_clues(self, query: str) -> Dict[str, any]:
        """
        Extract any available context clues from the query.
        
        Args:
            query: User input text
            
        Returns:
            Dictionary of extracted context information
        """
        context = {
            'has_pronoun': bool(re.search(r'\b(it|this|that|they|them)\b', query.lower())),
            'question_type': self._identify_question_type(query),
            'domain_hints': self._extract_domain_hints(query),
            'urgency_indicators': self._extract_urgency_indicators(query)
        }
        
        return context
    
    def _identify_question_type(self, query: str) -> str:
        """Identify the type of question being asked."""
        query_lower = query.lower()
        
        if query_lower.startswith(('how', 'how do', 'how can')):
            return 'procedural'
        elif query_lower.startswith(('what', 'what is', 'what are')):
            return 'informational'
        elif query_lower.startswith(('where', 'where can', 'where do')):
            return 'locational'
        elif query_lower.startswith(('when', 'what time')):
            return 'temporal'
        elif query_lower.startswith(('why', 'why do')):
            return 'explanatory'
        else:
            return 'general'
    
    def _extract_domain_hints(self, query: str) -> List[str]:
        """Extract domain-specific hints from the query."""
        domain_keywords = {
            'academic': ['course', 'grade', 'exam', 'class', 'lecture', 'assignment', 'transcript', 'graduation', 'registration'],
            'visa': ['visa', 'passport', 'immigration', 'renewal', 'student pass', 'overstay', 'penalty'],
            'library': ['book', 'library', 'borrow', 'return', 'research', 'fine', 'overdue'],
            'financial': ['fee', 'payment', 'scholarship', 'cost', 'bank', 'transfer', 'swift', 'tuition'],
            'it': ['password', 'login', 'account', 'system', 'apspace', 'apkey', 'azure', 'access'],
            'technical': ['install', 'setup', 'configure', 'software', 'solidworks', 'sql', 'version'],
            'facilities': ['parking', 'zone', 'office', 'location', 'campus', 'building']
        }
        
        found_domains = []
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_domains.append(domain)
                
        return found_domains
    
    def _extract_urgency_indicators(self, query: str) -> str:
        """Extract urgency level from the query."""
        urgency_keywords = {
            'urgent': ['urgent', 'asap', 'immediately', 'emergency', 'deadline'],
            'soon': ['soon', 'quickly', 'fast', 'today', 'tomorrow'],
            'normal': []
        }
        
        query_lower = query.lower()
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return level
                
        return 'normal'
    
    def _initialize_pronoun_system(self) -> dict:
        """
        Initialize pronoun disambiguation system.
        Follows best practices from OpenAI, Anthropic, and NotebookLM.
        
        Returns:
            Dictionary of disambiguation patterns and corrections
        """
        return {
            'typo_corrections': {
                # Common "it" -> "a" typos in academic contexts
                r'\bget it ([a-z]+(?:\s+[a-z]+)*)': r'get a \1',
                r'\bhow can i get it ([a-z]+(?:\s+[a-z]+)*)': r'how can i get a \1',
                r'\bwhere can i get it ([a-z]+(?:\s+[a-z]+)*)': r'where can i get a \1',
                r'\bi need it ([a-z]+(?:\s+[a-z]+)*)': r'i need a \1',
                r'\bwant it ([a-z]+(?:\s+[a-z]+)*)': r'want a \1',
            },
            'contextual_patterns': {
                # University-specific context patterns
                'recommendation': [r'\b(?:recommendation|reference)\s+letter\b'],
                'transcript': [r'\b(?:official|interim)\s+transcript\b'], 
                'certificate': [r'\b(?:medical|insurance)\s+(?:card|certificate)\b'],
                'renewal': [r'\b(?:visa|passport|student\s+pass)\s+renewal\b'],
                'application': [r'\b(?:scholarship|course)\s+application\b']
            },
            'confidence_boost_terms': {
                # Terms that boost confidence for "a" interpretation
                'academic_terms': ['letter', 'transcript', 'certificate', 'card', 'form', 'document'],
                'action_terms': ['get', 'obtain', 'request', 'apply', 'submit'],
                'university_terms': ['recommendation', 'reference', 'official', 'medical', 'insurance']
            }
        }
    
    def _apply_pronoun_disambiguation(self, query: str) -> str:
        """
        Apply pronoun disambiguation.
        Uses contextual analysis and semantic understanding.
        
        Args:
            query: Input query text
            
        Returns:
            Disambiguated query text
        """
        original_query = query
        disambiguated = query.lower()
        
        # Step 1: Apply typo corrections with context awareness
        for pattern, replacement in self.pronoun_disambiguator['typo_corrections'].items():
            if re.search(pattern, disambiguated):
                # Check if this correction makes sense in context
                potential_correction = re.sub(pattern, replacement, disambiguated, flags=re.IGNORECASE)
                if self._validate_correction_context(original_query, potential_correction):
                    logger.debug(f"Applied typo correction: {pattern} -> {replacement}")
                    return potential_correction
        
        # Step 2: Contextual pronoun resolution
        if self._has_clear_academic_context(query):
            # In academic contexts, "it" often refers to documents/processes
            academic_corrections = {
                r'\bhow can i get it\b': 'how can i get a',
                r'\bwhere can i get it\b': 'where can i get a', 
                r'\bcan i get it\b': 'can i get a',
                r'\bi need it\b': 'i need a'
            }
            
            for pattern, replacement in academic_corrections.items():
                if re.search(pattern, disambiguated):
                    corrected = re.sub(pattern, replacement, disambiguated, flags=re.IGNORECASE)
                    logger.debug(f"Applied academic context correction: {pattern} -> {replacement}")
                    return corrected
        
        return original_query
    
    def _validate_correction_context(self, original: str, corrected: str) -> bool:
        """
        Validate that a pronoun correction makes sense in context.
        Uses semantic analysis to prevent false corrections.
        
        Args:
            original: Original query
            corrected: Proposed correction
            
        Returns:
            True if correction is contextually valid
        """
        # Check for confidence-boosting terms
        boost_terms = self.pronoun_disambiguator['confidence_boost_terms']
        
        word_sets = {
            'academic': set(boost_terms['academic_terms']),
            'action': set(boost_terms['action_terms']), 
            'university': set(boost_terms['university_terms'])
        }
        
        corrected_words = set(corrected.lower().split())
        
        # Calculate semantic overlap
        total_matches = 0
        for category, terms in word_sets.items():
            matches = len(corrected_words.intersection(terms))
            total_matches += matches
        
        # If we have 2+ relevant terms, the correction is likely valid
        return total_matches >= 2
    
    def _has_clear_academic_context(self, query: str) -> bool:
        """
        Check if query has clear academic/university context.
        
        Args:
            query: Query text to analyze
            
        Returns:
            True if query has academic context
        """
        academic_indicators = [
            'recommendation', 'letter', 'transcript', 'certificate', 'card',
            'medical', 'insurance', 'official', 'interim', 'reference',
            'visa', 'renewal', 'application', 'student', 'university'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for indicator in academic_indicators if indicator in query_lower)
        
        return matches >= 1