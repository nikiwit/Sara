"""
Language detection and handling module for SARA chatbot.
"""

import re
from typing import Tuple, Dict
from langdetect import DetectorFactory, detect_langs
import logging
from config import config

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger("Sara")


class LanguageHandler:
    """Handle language detection with ensemble approach and confidence thresholding."""
    
    def __init__(self, supported_languages: set = {'en'}, confidence_threshold: float = 0.65):
        self.supported_languages = supported_languages
        self.confidence_threshold = confidence_threshold
        self.mixed_language_patterns = [
            r'[a-zA-Z\s]+[\u4e00-\u9fff]+',  # English + Chinese
            r'[a-zA-Z\s]+[\u0400-\u04ff]+',  # English + Cyrillic
            r'[a-zA-Z\s]+[\u00c0-\u017f]+',  # English + European accents
        ]
        
        # Load English whitelist dynamically
        self.english_word_whitelist = self._load_english_whitelist()
        
        # Track statistics for monitoring
        self.stats = {
            'total_queries': 0,
            'english_queries': 0,
            'blocked_queries': 0,
            'whitelist_rescues': 0,
            'short_text_handled': 0,
            'llm_fallbacks': 0
        }
    
    def detect_language_with_confidence(self, query: str) -> tuple[str, float]:
        """
        Detect language with confidence scoring and whitelist checking.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        self.stats['total_queries'] += 1
        
        # Handle very short queries (1-3 characters)
        if len(query.strip()) < 3:
            self.stats['short_text_handled'] += 1
            return 'en', 1.0  # High confidence for short queries
        
        # Check English whitelist for single words or very short phrases
        if len(query.strip().split()) <= 2:
            query_words = {word.lower().strip('.,!?;:"()[]') for word in query.split()}
            if query_words.intersection(self.english_word_whitelist):
                self.stats['whitelist_rescues'] += 1
                self.stats['english_queries'] += 1
                logger.debug(f"Whitelist rescue for query: {query[:30]}...")
                return 'en', 0.95  # High confidence for whitelisted terms
            
        # Handle mixed language queries
        if self._is_mixed_language(query):
            return 'mixed', 0.9
        
        # Use ensemble approach for language detection
        return self._ensemble_language_detection(query)
    
    def _load_english_whitelist(self) -> set:
        """Load English whitelist from externalized configuration."""
        try:
            from nlp_config_loader import config_loader
            whitelist = config_loader.get_english_whitelist()
            logger.info(f"Loaded {len(whitelist)} terms in English whitelist")
            return whitelist
            
        except Exception as e:
            logger.error(f"Failed to load whitelist configuration: {e}")
            return {'help', 'support', 'please', 'thanks', 'hello', 'hi'}
    
    def _ensemble_language_detection(self, query: str) -> tuple[str, float]:
        """
        Enhanced language detection using ensemble approach with multiple validation strategies.
        Implements confidence scoring with multi-signal validation.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        try:
            # Primary detection using langdetect
            language_scores = detect_langs(query)
            if not language_scores:
                return 'unknown', 0.0
            
            primary_lang = language_scores[0]
            base_confidence = primary_lang.prob
            
            # Confidence adjustment based on query characteristics
            adjusted_confidence = self._calculate_adjusted_confidence(
                query, primary_lang.lang, base_confidence
            )
            
            # Multi-signal validation for accuracy
            final_lang, final_confidence = self._validate_with_multiple_signals(
                query, primary_lang.lang, adjusted_confidence
            )
            
            # Update statistics
            if final_lang == 'en':
                self.stats['english_queries'] += 1
                
            return final_lang, final_confidence
            
        except Exception as e:
            logger.warning(f"Ensemble language detection failed: {e}")
            
            # Fallback: if it looks English, assume English
            if self._looks_like_english(query):
                self.stats['whitelist_rescues'] += 1
                return 'en', 0.7
            
            return 'unknown', 0.0
    
    def _calculate_adjusted_confidence(self, query: str, detected_lang: str, base_confidence: float) -> float:
        """
        Calculate adjusted confidence using multiple signals.
        Follows best practices from OpenAI, Anthropic, and NotebookLM.
        
        Args:
            query: Input text
            detected_lang: Detected language code
            base_confidence: Base confidence from langdetect
            
        Returns:
            Adjusted confidence score
        """
        word_count = len(query.split())
        
        # Length-based confidence penalty (short text is unreliable)
        if word_count <= 3:
            base_confidence *= 0.6  # Heavy penalty for very short text
        elif word_count <= 6:
            base_confidence *= 0.8  # Moderate penalty for short text
        
        # Domain relevance boost for university terms
        query_words = set(query.lower().split())
        whitelist_overlap = len(query_words.intersection(self.english_word_whitelist))
        
        if whitelist_overlap >= 2:  # Multiple domain terms
            base_confidence = min(0.95, base_confidence * 1.3)
        elif whitelist_overlap >= 1:  # Single domain term
            base_confidence = min(0.9, base_confidence * 1.15)
        
        return base_confidence
    
    def _validate_with_multiple_signals(self, query: str, detected_lang: str, confidence: float) -> tuple[str, float]:
        """
        Multi-signal validation using contextual analysis.
        Implements patterns for production reliability.
        
        Args:
            query: Input text
            detected_lang: Detected language
            confidence: Current confidence
            
        Returns:
            Tuple of (final_language, final_confidence)
        """
        # Signal 0: Non-Latin script detection
        if self._contains_non_latin_scripts(query):
            # If we detect non-Latin scripts, trust langdetect unless it's clearly wrong
            if detected_lang != 'en':
                logger.debug(f"Non-Latin script detected, maintaining {detected_lang} classification: {query[:30]}...")
                return detected_lang, confidence
        
        # Signal 1: English pattern matching (only for Latin-based text)
        if self._looks_like_english(query) and detected_lang != 'en':
            # Additional safety check: ensure it's actually Latin script
            if self._is_primarily_latin_script(query):
                logger.debug(f"Multi-signal override: {detected_lang} -> en for: {query[:30]}...")
                self.stats['whitelist_rescues'] += 1
                return 'en', max(0.8, confidence)
        
        # Signal 2: Confidence threshold with contextual boost
        if detected_lang == 'en' and confidence < 0.7:
            # Check for English contextual indicators
            if self._has_english_context_indicators(query):
                boosted_confidence = min(0.85, confidence + 0.2)
                logger.debug(f"English context boost: {confidence:.2f} -> {boosted_confidence:.2f}")
                return 'en', boosted_confidence
        
        return detected_lang, confidence
    
    def _has_english_context_indicators(self, query: str) -> bool:
        """
        Check for contextual indicators that suggest English text.
        Uses configuration-driven semantic patterns.
        
        Args:
            query: Text to analyze
            
        Returns:
            True if text has English contextual indicators
        """
        # Use externalized semantic patterns
        try:
            from nlp_config_loader import config_loader
            patterns = config_loader.get_semantic_patterns()
            english_patterns = patterns.get('english_indicators', [])
            question_patterns = patterns.get('question_patterns', [])
            grammar_patterns = patterns.get('english_grammar_patterns', [])
            
            # Combine all available patterns
            all_patterns = english_patterns + question_patterns + grammar_patterns
            
        except Exception as e:
            logger.warning(f"Could not load semantic patterns: {e}")
            # Minimal fallback - use only whitelist overlap
            return self._calculate_whitelist_confidence(query) > 0.3
        
        # Apply patterns dynamically from configuration
        query_lower = query.lower()
        for pattern in all_patterns:
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.debug(f"Matched English pattern: {pattern}")
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern in config: {pattern}")
                continue
        
        # Fallback: statistical analysis of word overlap
        return self._calculate_whitelist_confidence(query) > 0.4
    
    def _looks_like_english(self, query: str) -> bool:
        """
        Heuristic check if text looks like English based on character distribution and patterns.
        
        Args:
            query: Text to check
            
        Returns:
            True if text appears to be English
        """
        # Remove punctuation and spaces for analysis
        clean_text = re.sub(r'[^a-zA-Z]', '', query).lower()
        
        if not clean_text:
            return False
        
        # Safety check: if text contains non-Latin scripts, it's not English
        if self._contains_non_latin_scripts(query):
            logger.debug(f"Non-Latin script detected in _looks_like_english: {query[:30]}...")
            return False
        
        # Check if text is primarily Latin alphabet
        if not self._is_primarily_latin_script(query):
            return False
        
        # Check for English patterns
        try:
            from nlp_config_loader import config_loader
            patterns = config_loader.get_semantic_patterns()
            english_indicators = patterns.get('english_indicators', [])
        except Exception:
            english_indicators = [r'\b\w+(ing|tion|ness|ment|able|ible)\b']
        
        for pattern in english_indicators:
            if re.search(pattern, query.lower()):
                return True
        
        # Statistical whitelist analysis
        return self._calculate_whitelist_confidence(query) > 0.5
    
    def _contains_non_latin_scripts(self, query: str) -> bool:
        """
        Detection of non-Latin scripts (Chinese, Arabic, Cyrillic, etc.).
        Follows Unicode standards for script detection.
        
        Args:
            query: Text to analyze
            
        Returns:
            True if text contains non-Latin scripts
        """
        # Unicode ranges for major non-Latin scripts
        non_latin_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Chinese, Japanese, Korean)
            (0x3400, 0x4DBF),   # CJK Extension A
            (0x20000, 0x2A6DF), # CJK Extension B
            (0x0400, 0x04FF),   # Cyrillic
            (0x0590, 0x05FF),   # Hebrew
            (0x0600, 0x06FF),   # Arabic
            (0x0900, 0x097F),   # Devanagari (Hindi)
            (0x3040, 0x309F),   # Hiragana
            (0x30A0, 0x30FF),   # Katakana
        ]
        
        for char in query:
            char_code = ord(char)
            for start, end in non_latin_ranges:
                if start <= char_code <= end:
                    logger.debug(f"Detected non-Latin character: {char} (U+{char_code:04X})")
                    return True
        
        return False
    
    def _is_primarily_latin_script(self, query: str) -> bool:
        """
        Check if text is primarily Latin script.
        
        Args:
            query: Text to analyze
            
        Returns:
            True if text is primarily Latin script
        """
        if not query.strip():
            return False
            
        # Remove punctuation and spaces (using standard punctuation instead of Unicode class)
        text_chars = re.sub(r'[\s\.,!?;:\"\'()\[\]{}\-_]', '', query)
        if not text_chars:
            return False
        
        latin_chars = 0
        for char in text_chars:
            char_code = ord(char)
            # Basic Latin (0000-007F) + Latin-1 Supplement (0080-00FF) + Latin Extended (0100-024F)
            if (0x0000 <= char_code <= 0x024F):
                latin_chars += 1
        
        latin_ratio = latin_chars / len(text_chars)
        return latin_ratio > 0.8
    
    def _calculate_whitelist_confidence(self, query: str) -> float:
        """
        Calculate confidence based on statistical analysis of whitelist overlap.
        Data-driven approach.
        
        Args:
            query: Text to analyze
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not query.strip():
            return 0.0
            
        # Clean and tokenize
        query_words = {word.lower().strip('.,!?;:"()[]') for word in query.split()}
        query_words = {word for word in query_words if word}  # Remove empty strings
        
        if not query_words:
            return 0.0
        
        # Calculate statistical overlap
        whitelist_matches = query_words.intersection(self.english_word_whitelist)
        overlap_ratio = len(whitelist_matches) / len(query_words)
        
        # Weight by match quality (longer matches = higher confidence)
        quality_weight = 1.0
        if whitelist_matches:
            avg_match_length = sum(len(word) for word in whitelist_matches) / len(whitelist_matches)
            if avg_match_length > 4:  # Longer words are more distinctive
                quality_weight = 1.2
        
        return min(1.0, overlap_ratio * quality_weight)
    
    def detect_language(self, query: str) -> str:
        """Legacy interface for backward compatibility."""
        lang, _ = self.detect_language_with_confidence(query)
        return lang
    
    def _is_mixed_language(self, query: str) -> bool:
        """Check if query contains mixed languages."""
        return any(re.search(pattern, query) for pattern in self.mixed_language_patterns)
    
    def handle_query(self, query: str) -> Tuple[bool, str, dict]:
        """
        Process query using hybrid approach.
        Returns blocking decision and optional LLM context for fallback.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (should_block, response_message, llm_context)
        """
        detected_lang, confidence = self.detect_language_with_confidence(query)
        
        # Handle low confidence cases intelligently
        if confidence < self.confidence_threshold:
            # Special case: Non-Latin scripts should be blocked regardless of confidence
            if self._contains_non_latin_scripts(query) and detected_lang != 'en':
                logger.info(f"Non-Latin script detected, blocking despite low confidence ({confidence:.2f}): {query[:50]}...")
                self.stats['blocked_queries'] += 1
                return True, self._get_language_message_by_code(detected_lang), {}
            
            # For ambiguous Latin-based text, defer to LLM
            logger.debug(f"Low confidence ({confidence:.2f}) language detection, deferring to LLM: {query[:50]}...")
            llm_context = {
                'language_uncertain': True,
                'detected_lang': detected_lang,
                'confidence': confidence,
                'instruction': 'If this query is not in English, politely redirect to English.'
            }
            self.stats['llm_fallbacks'] += 1
            return False, "", llm_context
        
        if detected_lang in self.supported_languages:
            self.stats['english_queries'] += 1
            return False, "", {}
            
        # High confidence non-English: Block immediately (performance optimization)
        self.stats['blocked_queries'] += 1
        logger.info(f"Blocked {detected_lang} query (confidence: {confidence:.2f}): {query[:50]}...")
            
        if detected_lang in ['zh', 'zh-cn', 'zh-tw']:
            return True, self._get_chinese_redirect_message(), {}
        elif detected_lang == 'es':
            return True, self._get_spanish_redirect_message(), {}
        elif detected_lang == 'mixed':
            return True, self._get_mixed_language_message(), {}
        else:
            return True, self._get_generic_language_message(), {}
    
    def _get_chinese_redirect_message(self) -> str:
        """Message for Chinese language queries."""
        return (
            "I can only assist in English. Please ask your question in English.\n\n"
            "For Chinese language support:\n"
            f"• Visit {config.SUPPORT_LOCATION}\n"
            f"• Call: {config.SUPPORT_PHONE}\n"
            f"• Email: {config.SUPPORT_EMAIL}"
        )
    
    def _get_spanish_redirect_message(self) -> str:
        """Message for Spanish language queries."""
        return (
            "I can only assist in English. Please ask your question in English.\n\n"
            "Para soporte en español:\n"
            f"• Visit {config.SUPPORT_LOCATION}\n"
            f"• Call: {config.SUPPORT_PHONE}\n"
            f"• Email: {config.SUPPORT_EMAIL}"
        )
    
    def _get_mixed_language_message(self) -> str:
        """Message for mixed language queries."""
        return (
            "I notice you're using multiple languages. Please ask your question in English only.\n\n"
            "For multilingual support:\n"
            f"• Visit {config.SUPPORT_LOCATION}\n"
            f"• Call: {config.SUPPORT_PHONE}\n"
            f"• Email: {config.SUPPORT_EMAIL}"
        )
    
    def _get_generic_language_message(self) -> str:
        """Generic message for unsupported languages."""
        return (
            "I can only assist in English. Please ask your question in English.\n\n"
            "For support in other languages:\n"
            f"• Visit {config.SUPPORT_LOCATION}\n"
            f"• Call: {config.SUPPORT_PHONE}\n"
            f"• Email: {config.SUPPORT_EMAIL}"
        )
    
    def _get_language_message_by_code(self, lang_code: str) -> str:
        """Get appropriate message based on detected language code."""
        if lang_code in ['zh', 'zh-cn', 'zh-tw']:
            return self._get_chinese_redirect_message()
        elif lang_code == 'es':
            return self._get_spanish_redirect_message()
        elif lang_code == 'mixed':
            return self._get_mixed_language_message()
        else:
            return self._get_generic_language_message()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get language detection statistics for monitoring."""
        total = max(self.stats['total_queries'], 1)  # Avoid division by zero
        
        return {
            'total_queries': self.stats['total_queries'],
            'english_queries': self.stats['english_queries'],
            'blocked_queries': self.stats['blocked_queries'],
            'whitelist_rescues': self.stats['whitelist_rescues'],
            'short_text_handled': self.stats['short_text_handled'],
            'english_rate': f"{(self.stats['english_queries'] / total * 100):.1f}%",
            'block_rate': f"{(self.stats['blocked_queries'] / total * 100):.1f}%",
            'whitelist_rescue_rate': f"{(self.stats['whitelist_rescues'] / total * 100):.1f}%"
        }
    
    def is_healthy(self) -> bool:
        """Check if language handler is functioning properly."""
        # Consider healthy if we're not blocking too many queries
        total = self.stats['total_queries']
        if total > 10:  # Only check after some queries
            block_rate = self.stats['blocked_queries'] / total
            return block_rate < 0.5  # Less than 50% blocked
        return True