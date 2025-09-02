"""
Configuration loader following 2025 NLP best practices.
Externalized, testable, predictable configuration management.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Set
from functools import lru_cache

logger = logging.getLogger("Sara")


class ConfigLoader:
    """Production-grade configuration loader for NLP systems."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "topics.yaml"
        self.config_path = Path(config_path)
        self._config_cache = None
    
    @lru_cache(maxsize=1)
    def load_config(self) -> Dict:
        """Load configuration with caching for performance."""
        try:
            if not self.config_path.exists():
                logger.error(f"Config file not found: {self.config_path}")
                return self._get_minimal_fallback()
                
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_minimal_fallback()
    
    def get_topic_definitions(self) -> Dict:
        """Get topic definitions for semantic processing."""
        config = self.load_config()
        return config.get('topics', {})
    
    def get_english_whitelist(self) -> Set[str]:
        """Get English language whitelist."""
        config = self.load_config()
        whitelist_config = config.get('english_whitelist', {})
        
        whitelist = set()
        for category in whitelist_config.values():
            if isinstance(category, list):
                whitelist.update(category)
        
        return whitelist
    
    def get_semantic_patterns(self) -> Dict[str, List[str]]:
        """Get semantic patterns for language detection."""
        config = self.load_config()
        return config.get('semantic_patterns', {})
    
    def _get_minimal_fallback(self) -> Dict:
        """Minimal fallback configuration."""
        return {
            'topics': {
                'general': {
                    'keywords': ['help', 'support', 'assistance'],
                    'entities': ['ORG'],
                    'confidence_threshold': 0.5
                }
            },
            'english_whitelist': {
                'common_terms': ['help', 'support', 'please', 'thanks']
            },
            'semantic_patterns': {
                'english_indicators': [r'\b\w+ing\b']
            }
        }
    
    def reload_config(self):
        """Force reload configuration (for testing/updates)."""
        self.load_config.cache_clear()
        logger.info("Configuration reloaded")


# Global config loader instance
config_loader = ConfigLoader()