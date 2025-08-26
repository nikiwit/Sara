"""
Language detection and handling module for SARA chatbot.
"""

import re
from typing import Tuple
from langdetect import DetectorFactory, detect_langs
import logging
from config import config

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


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
    
    def detect_language_with_confidence(self, query: str) -> tuple[str, float]:
        """
        Detect language with confidence scoring.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        # Handle very short queries
        if len(query.strip()) < 3:
            return 'en', 1.0  # High confidence for short queries
            
        # Handle mixed language queries
        if self._is_mixed_language(query):
            return 'mixed', 0.9
            
        try:
            language_scores = detect_langs(query)
            if language_scores:
                primary_lang = language_scores[0]
                return primary_lang.lang, primary_lang.prob
            else:
                return 'unknown', 0.0
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'unknown', 0.0
    
    def detect_language(self, query: str) -> str:
        """Legacy interface for backward compatibility."""
        lang, _ = self.detect_language_with_confidence(query)
        return lang
    
    def _is_mixed_language(self, query: str) -> bool:
        """Check if query contains mixed languages."""
        return any(re.search(pattern, query) for pattern in self.mixed_language_patterns)
    
    def handle_query(self, query: str) -> Tuple[bool, str]:
        """
        Process query using confidence thresholding.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (should_block, response_message)
        """
        detected_lang, confidence = self.detect_language_with_confidence(query)
        
        if confidence < self.confidence_threshold:
            logger.debug(f"Low confidence ({confidence:.2f}) language detection, allowing through: {query[:50]}...")
            return False, ""
        
        if detected_lang in self.supported_languages:
            return False, ""
            
        if detected_lang in ['zh', 'zh-cn', 'zh-tw']:
            return True, self._get_chinese_redirect_message()
        elif detected_lang == 'es':
            return True, self._get_spanish_redirect_message()
        elif detected_lang == 'mixed':
            return True, self._get_mixed_language_message()
        else:
            return True, self._get_generic_language_message()
    
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