"""
Response Cache Module for SARA
Implements intelligent caching for frequent queries to improve response times
"""

import hashlib
import json
import time
from typing import Dict, Optional, Any, List
import os
import pickle
from pathlib import Path
import logging

logger = logging.getLogger("Sara")

class ResponseCache:
    """
    Intelligent response cache with TTL support and persistence.
    """
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_file = self.cache_dir / "response_cache.pkl"
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'expired': 0
        }
        
        # Load existing cache from disk
        self._load_cache()
        
        logger.info(f"Initialized response cache with TTL {ttl}s")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a normalized cache key for a query."""
        # Normalize query: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(query.lower().strip().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - entry['timestamp'] > self.ttl
    
    def _load_cache(self):
        """Load cache from persistent storage."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.memory_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.memory_cache)} entries from cache")
            else:
                logger.info("No existing cache file found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.memory_cache = {}
    
    def _save_cache(self):
        """Save cache to persistent storage."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
            logger.debug("Cache saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for a query.
        
        Args:
            query: The user query
            
        Returns:
            Cached response if found and not expired, None otherwise
        """
        key = self._get_cache_key(query)
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            if self._is_expired(entry):
                # Remove expired entry
                del self.memory_cache[key]
                self.stats['expired'] += 1
                logger.debug(f"Cache entry expired for query: {query[:50]}...")
                return None
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry['response']
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Cache a response for a query.
        
        Args:
            query: The user query
            response: The bot response
            metadata: Optional metadata about the response
        """
        key = self._get_cache_key(query)
        
        entry = {
            'query': query,
            'response': response,
            'timestamp': time.time(),
            'metadata': metadata or {},
            'access_count': 1
        }
        
        # Update access count if entry exists
        if key in self.memory_cache:
            entry['access_count'] = self.memory_cache[key].get('access_count', 0) + 1
        
        self.memory_cache[key] = entry
        self.stats['writes'] += 1
        
        logger.debug(f"Cached response for query: {query[:50]}...")
        
        # Periodic cache cleanup and save (every 10 writes)
        if self.stats['writes'] % 10 == 0:
            self._cleanup_expired()
            self._save_cache()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.stats['expired'] += 1
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear(self):
        """Clear all cache entries."""
        self.memory_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.memory_cache),
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_writes': self.stats['writes'],
            'expired_entries': self.stats['expired'],
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Get approximate cache size in MB."""
        try:
            if self.cache_file.exists():
                return self.cache_file.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0.0
    
    def should_cache(self, query: str, response: str) -> bool:
        """
        Determine if a query-response pair should be cached.
        
        Args:
            query: The user query
            response: The bot response
            
        Returns:
            True if should be cached, False otherwise
        """
        # Don't cache very short queries or responses
        if len(query.strip()) < 10 or len(response.strip()) < 50:
            return False
        
        # Don't cache error responses or "I don't know" responses
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in [
            "i don't have", "i don't know", "error", "failed", "sorry"
        ]):
            return False
        
        # Don't cache very long responses (likely complex, context-specific)
        if len(response) > 2000:
            return False
        
        # Cache FAQs and common administrative queries
        query_lower = query.lower()
        if any(phrase in query_lower for phrase in [
            "how to", "how do i", "what is", "where is", "when is",
            "reference letter", "password", "fees", "payment", "login"
        ]):
            return True
        
        return True
    
    def get_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently accessed queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query information sorted by access count
        """
        query_info = []
        for key, entry in self.memory_cache.items():
            if not self._is_expired(entry):
                query_info.append({
                    'query': entry['query'],
                    'access_count': entry['access_count'],
                    'last_accessed': entry['timestamp']
                })
        
        # Sort by access count (descending)
        query_info.sort(key=lambda x: x['access_count'], reverse=True)
        return query_info[:limit]
    
    def preload_common_queries(self, common_queries: List[Dict[str, str]]):
        """
        Preload cache with common query-response pairs.
        
        Args:
            common_queries: List of {'query': str, 'response': str} dicts
        """
        preloaded = 0
        for item in common_queries:
            query = item.get('query')
            response = item.get('response')
            
            if query and response and self.should_cache(query, response):
                key = self._get_cache_key(query)
                if key not in self.memory_cache:  # Don't overwrite existing
                    self.set(query, response, {'preloaded': True})
                    preloaded += 1
        
        logger.info(f"Preloaded {preloaded} common queries into cache")
        self._save_cache()


class FrequentQueryCache:
    """
    Specialized cache for tracking and caching frequent queries.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.frequency_file = self.cache_dir / "query_frequency.json"
        self.frequency_data: Dict[str, int] = {}
        self._load_frequency_data()
    
    def _load_frequency_data(self):
        """Load query frequency data."""
        try:
            if self.frequency_file.exists():
                with open(self.frequency_file, 'r') as f:
                    self.frequency_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load frequency data: {e}")
            self.frequency_data = {}
    
    def _save_frequency_data(self):
        """Save query frequency data."""
        try:
            with open(self.frequency_file, 'w') as f:
                json.dump(self.frequency_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save frequency data: {e}")
    
    def track_query(self, query: str):
        """Track a query for frequency analysis."""
        key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        self.frequency_data[key] = self.frequency_data.get(key, 0) + 1
        
        # Save every 20 queries
        if sum(self.frequency_data.values()) % 20 == 0:
            self._save_frequency_data()
    
    def get_frequent_queries(self, threshold: int = 5) -> List[str]:
        """Get queries that have been asked frequently."""
        frequent = []
        for key, count in self.frequency_data.items():
            if count >= threshold:
                frequent.append(key)
        return frequent