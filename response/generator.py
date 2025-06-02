"""
Response generation using LLM integration with optimized connection pooling and caching.
Provides efficient interface to Ollama API with enhanced error handling and performance features.
"""

import os
import json
import logging
import requests
import time
import threading
from typing import List, Iterator, Union, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import Config

logger = logging.getLogger("CustomRAG")

class RAGSystem:
    """Manages the RAG processing pipeline with optimized LLM integration and performance enhancements."""
    
    # Class-level connection pooling for efficient HTTP management
    _session = None
    _session_lock = threading.Lock()
    _response_cache = {}
    _cache_lock = threading.Lock()
    
    @classmethod
    def _get_session(cls):
        """Get or create HTTP session with connection pooling and intelligent retry strategy."""
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:  # Double-check locking pattern
                    cls._session = requests.Session()
                    
                    # Configure comprehensive retry strategy for reliability
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=0.5,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["POST", "GET"]
                    )
                    
                    # Configure connection pooling for performance
                    adapter = HTTPAdapter(
                        max_retries=retry_strategy,
                        pool_connections=10,
                        pool_maxsize=20
                    )
                    
                    cls._session.mount("http://", adapter)
                    cls._session.mount("https://", adapter)
                    
                    logger.info("HTTP session initialized with connection pooling and retry strategy")
        
        return cls._session
    
    @staticmethod
    def format_docs(docs):
        """Format retrieved documents for inclusion in prompts with optimized structure and reduced overhead."""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        # Efficient document formatting with type-aware structure
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
            unique_filenames.add(filename)
            
            # Format based on document type for optimal LLM comprehension
            if doc.metadata.get('content_type') == 'apu_kb_page':
                title = doc.metadata.get('page_title', 'Untitled')
                if doc.metadata.get('is_faq', False):
                    formatted_text = f"Q{i+1}: {title}\nA: {doc.page_content}\n"
                else:
                    formatted_text = f"Topic {i+1}: {title}\n{doc.page_content}\n"
            else:
                # Simplified format for general documents
                formatted_text = f"Doc {i+1} ({filename}):\n{doc.page_content}\n"
                
            formatted_docs.append(formatted_text)

        # Concise summary header for context efficiency
        summary = f"Found {len(docs)} sources from {len(unique_filenames)} files.\n"
        
        return summary + "\n".join(formatted_docs)
    
    @staticmethod
    @lru_cache(maxsize=100)
    def _get_cached_connection_test(base_url: str) -> bool:
        """Cached connection test to avoid repeated network checks and improve performance."""
        test_url = f"{base_url}/api/tags"
        try:
            response = requests.get(test_url, timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    @staticmethod
    def _cache_response(prompt_hash: str, response: str):
        """Cache response with thread-safe implementation and automatic size management."""
        with RAGSystem._cache_lock:
            # Simple LRU cache implementation with size limits
            max_cache_size = getattr(Config, 'RESPONSE_CACHE_SIZE', 50)
            if len(RAGSystem._response_cache) >= max_cache_size:
                # Remove oldest entry for memory management
                oldest_key = next(iter(RAGSystem._response_cache))
                del RAGSystem._response_cache[oldest_key]
            
            RAGSystem._response_cache[prompt_hash] = {
                'response': response,
                'timestamp': time.time()
            }
    
    @staticmethod
    def _get_cached_response(prompt_hash: str) -> Optional[str]:
        """Retrieve cached response if available and not expired, with automatic cleanup."""
        with RAGSystem._cache_lock:
            if prompt_hash in RAGSystem._response_cache:
                cached = RAGSystem._response_cache[prompt_hash]
                # Check cache expiration based on configured TTL
                cache_ttl = getattr(Config, 'RESPONSE_CACHE_TTL', 3600)
                if time.time() - cached['timestamp'] < cache_ttl:
                    return cached['response']
                else:
                    # Remove expired entry for cache hygiene
                    del RAGSystem._response_cache[prompt_hash]
        
        return None
    
    @staticmethod
    def stream_ollama_response(prompt, model_name=None, base_url=None, stream_output=False, stream_delay=None):
        """
        Generate response from Ollama API with enhanced performance, caching, and comprehensive error handling.
        
        Args:
            prompt: The prompt to send to Ollama LLM
            model_name: The name of the model to use for generation
            base_url: The base URL for the Ollama API service
            stream_output: Whether to stream output in real-time or return complete response
            stream_delay: Delay between tokens when streaming for controlled output
            
        Returns:
            If stream_output is True, yields tokens as they are generated
            If stream_output is False, returns the complete response as a string
        """
        # Apply intelligent defaults from configuration
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
        if stream_delay is None:
            stream_delay = Config.STREAM_DELAY
        
        # Check response cache for non-streaming requests to improve performance
        prompt_hash = None
        if not stream_output and getattr(Config, 'ENABLE_RESPONSE_CACHING', False):
            import hashlib
            prompt_hash = hashlib.md5(f"{prompt}_{model_name}".encode()).hexdigest()
            cached_response = RAGSystem._get_cached_response(prompt_hash)
            if cached_response:
                logger.debug("Returning cached response")
                return cached_response
        
        # Prepare API request with optimized configuration
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": getattr(Config, 'LLM_TEMPERATURE', 0.7),
                "top_p": getattr(Config, 'LLM_TOP_P', 0.9),
                "num_ctx": getattr(Config, 'LLM_CONTEXT_LENGTH', 4096)
            }
        }

        full_response = ""
        start_time = time.time()

        try:
            # Verify API connectivity with cached connection test
            if not RAGSystem._get_cached_connection_test(base_url):
                error_msg = "Error: Could not connect to Ollama API. Make sure Ollama is running."
                logger.error("Ollama API unavailable")
                return error_msg if not stream_output else iter([error_msg])

            # Use optimized session with connection pooling
            session = RAGSystem._get_session()
            
            # Process streaming response with comprehensive error handling
            with session.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    error_msg = f"Error: Failed to generate response (HTTP {response.status_code})"
                    logger.error(f"Ollama API error: {response.status_code}")
                    return error_msg if not stream_output else iter([error_msg])

                token_count = 0
                last_log_time = time.time()
                
                # Process each token from the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                full_response += token
                                token_count += 1
                                
                                # Stream tokens with configurable delay for controlled output
                                if stream_output:
                                    yield token
                                    time.sleep(stream_delay)
                                
                                # Progress logging for long responses to aid debugging
                                current_time = time.time()
                                if current_time - last_log_time > 5:  # Log every 5 seconds
                                    logger.debug(f"Generated {token_count} tokens, {len(full_response)} characters")
                                    last_log_time = current_time

                            if json_line.get('done', False):
                                # Log completion statistics for performance monitoring
                                elapsed = time.time() - start_time
                                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                                logger.info(f"Response generated: {token_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/s)")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from Ollama API: {line} - {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing Ollama response: {e}")
                            break
        
        except requests.exceptions.Timeout:
            error_msg = "Error: Request to Ollama API timed out. Try a shorter prompt or check Ollama status."
            logger.error("Ollama request timeout")
            return error_msg if not stream_output else iter([error_msg])
        
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."
            logger.error("Ollama connection error")
            return error_msg if not stream_output else iter([error_msg])
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Unexpected error during Ollama request: {e}")
            return error_msg if not stream_output else iter([error_msg])
        
        # Cache successful responses for future performance improvement
        if not stream_output and full_response and prompt_hash:
            RAGSystem._cache_response(prompt_hash, full_response)
        
        # Return complete response for non-streaming requests
        if not stream_output:
            return full_response
    
    @staticmethod
    def generate_batch_responses(prompts: List[str], model_name=None, base_url=None, max_workers=3):
        """
        Generate responses for multiple prompts in parallel using thread pool for improved throughput.
        
        Args:
            prompts: List of prompts to process concurrently
            model_name: The name of the model to use for generation
            base_url: The base URL for the Ollama API service
            max_workers: Maximum number of concurrent requests to prevent overload
            
        Returns:
            List of responses corresponding to the input prompts in same order
        """
        if not prompts:
            return []
        
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
        
        logger.info(f"Processing {len(prompts)} prompts in parallel with {max_workers} workers")
        
        def process_single_prompt(prompt):
            return RAGSystem.stream_ollama_response(
                prompt, model_name, base_url, stream_output=False
            )
        
        # Use ThreadPoolExecutor for efficient parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_prompt, prompts))
        
        logger.info(f"Completed batch processing of {len(prompts)} prompts")
        return results
    
    @staticmethod
    def test_ollama_connection(base_url=None, model_name=None):
        """
        Test connection to Ollama API and return comprehensive status information for debugging.
        
        Args:
            base_url: The base URL for the Ollama API service
            model_name: The name of the model to test availability
            
        Returns:
            Dictionary with detailed connection status and diagnostic information
        """
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
        
        status = {
            'connected': False,
            'model_available': False,
            'response_time': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Test API connection and measure response time
            session = RAGSystem._get_session()
            response = session.get(f"{base_url}/api/tags", timeout=5)
            
            status['response_time'] = time.time() - start_time
            
            if response.status_code == 200:
                status['connected'] = True
                
                # Check if specific model is available
                try:
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    status['available_models'] = available_models
                    status['model_available'] = model_name in available_models
                    
                    if not status['model_available']:
                        status['error'] = f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
                except json.JSONDecodeError:
                    status['error'] = "Could not parse Ollama API response"
            else:
                status['error'] = f"API returned status code {response.status_code}"
                
        except requests.exceptions.Timeout:
            status['error'] = "Connection timeout"
        except requests.exceptions.ConnectionError:
            status['error'] = "Could not connect to Ollama API"
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Estimate token count for a text string using approximation algorithm.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens (approximate)
        """
        # Simple estimation: approximately 4 characters per token for English text
        # This is a rough estimate and actual tokenization may vary by model
        return max(1, len(text) // 4)
    
    @staticmethod
    def optimize_prompt_for_context(prompt: str, max_context_length: int = None) -> str:
        """
        Optimize prompt to fit within context length limits using intelligent truncation.
        
        Args:
            prompt: Original prompt text
            max_context_length: Maximum context length in tokens
            
        Returns:
            Optimized prompt that fits within specified limits
        """
        if max_context_length is None:
            max_context_length = getattr(Config, 'MAX_CONTEXT_SIZE', 4000)
        
        estimated_tokens = RAGSystem.estimate_token_count(prompt)
        
        if estimated_tokens <= max_context_length:
            return prompt
        
        logger.warning(f"Prompt too long ({estimated_tokens} tokens), optimizing for {max_context_length} tokens")
        
        # Intelligent truncation strategy preserving important content
        target_chars = max_context_length * 4  # Rough conversion back to characters
        
        if len(prompt) > target_chars:
            # Keep first 60% and last 20%, truncate middle section
            keep_start = int(target_chars * 0.6)
            keep_end = int(target_chars * 0.2)
            
            if keep_start + keep_end < len(prompt):
                optimized = (
                    prompt[:keep_start] + 
                    f"\n\n[... content truncated for length ...]\n\n" +
                    prompt[-keep_end:]
                )
                logger.info(f"Prompt optimized: {len(prompt)} -> {len(optimized)} characters")
                return optimized
        
        # Fallback to simple truncation if intelligent method fails
        return prompt[:target_chars]
    
    @staticmethod
    def get_performance_stats() -> Dict[str, Any]:
        """
        Retrieve comprehensive performance statistics for monitoring and optimization.
        
        Returns:
            Dictionary with detailed performance metrics and system status
        """
        with RAGSystem._cache_lock:
            cache_stats = {
                'cache_size': len(RAGSystem._response_cache),
                'cache_hits': sum(1 for entry in RAGSystem._response_cache.values() 
                                if time.time() - entry['timestamp'] < 3600)
            }
        
        return {
            'session_active': RAGSystem._session is not None,
            'cache_stats': cache_stats,
            'config': {
                'model_name': Config.LLM_MODEL_NAME,
                'base_url': Config.OLLAMA_BASE_URL,
                'stream_delay': Config.STREAM_DELAY,
                'max_context_size': Config.MAX_CONTEXT_SIZE
            }
        }
    
    @staticmethod
    def clear_cache():
        """Clear response cache to free memory and reset performance metrics."""
        with RAGSystem._cache_lock:
            RAGSystem._response_cache.clear()
        logger.info("Response cache cleared")
    
    @staticmethod
    def warm_up_connection(base_url=None):
        """
        Warm up the connection to Ollama API to reduce initial request latency.
        
        Args:
            base_url: The base URL for the Ollama API service
            
        Returns:
            Boolean indicating whether warmup was successful
        """
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
        
        logger.info("Warming up Ollama connection...")
        
        try:
            session = RAGSystem._get_session()
            response = session.get(f"{base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                logger.info("Ollama connection warmed up successfully")
                return True
            else:
                logger.warning(f"Ollama warmup returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to warm up Ollama connection: {e}")
            return False