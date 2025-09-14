"""
Response generation using LLM integration.
"""

import os
import json
import logging
import requests
import time  # Add this import
from typing import List, Iterator, Union

from config import config

logger = logging.getLogger("Sara")

class RAGSystem:
    """Manages the RAG processing pipeline."""
    
    @staticmethod
    def format_docs(docs):
        """Format retrieved documents for inclusion in the prompt."""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
            unique_filenames.add(filename)
            
            # Format differently for APU KB pages
            if doc.metadata.get('content_type') == 'apu_kb_page':
                title = doc.metadata.get('page_title', 'Untitled')
                main_url = doc.metadata.get('main_url', '')
                
                # Include source URL in the formatting if available
                url_info = f" (Source: {main_url})" if main_url else ""
                formatted_text = f"Document {i+1} (from {filename}, Topic: {title}{url_info}):\n{doc.page_content}\n\n"
            else:
                # Include metadata if available
                page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
                chunk_info = f"chunk {i+1}/{len(docs)}"
                
                metadata_line = f"Document {i+1} (from {filename} {page_info} {chunk_info}):\n"
                formatted_text = f"{metadata_line}{doc.page_content}\n\n"
                
            formatted_docs.append(formatted_text)

        # Summary of documents
        summary = f"Retrieved {len(docs)} chunks from {len(unique_filenames)} files: {', '.join(unique_filenames)}\n\n"
        
        return summary + "\n".join(formatted_docs)
    
    @staticmethod
    def stream_ollama_response(prompt, model_name=None, base_url=None, stream_output=False, stream_delay=None):
        """Stream response from Ollama API with token-by-token output.
        
        Args:
            prompt: The prompt to send to Ollama
            model_name: The name of the model to use
            base_url: The base URL for the Ollama API
            stream_output: Whether to stream output in real-time (yield tokens) or return full response
            stream_delay: Delay between tokens when streaming (default: config.STREAM_DELAY)
            
        Returns:
            If stream_output is True, yields tokens as they are generated
            If stream_output is False, returns the full response as a string
        """
        if stream_output:
            # Use the generator version for streaming
            return RAGSystem._stream_ollama_response_generator(prompt, model_name, base_url, stream_delay)
        else:
            # Use the non-generator version for non-streaming
            return RAGSystem._stream_ollama_response_full(prompt, model_name, base_url)
    
    @staticmethod
    def _stream_ollama_response_generator(prompt, model_name=None, base_url=None, stream_delay=None):
        """Generator version for streaming responses."""
        if model_name is None:
            model_name = config.LLM_MODEL_NAME
            
        if base_url is None:
            base_url = config.OLLAMA_BASE_URL
            
        if stream_delay is None:
            stream_delay = config.STREAM_DELAY
            
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": []
            }
        }

        try:
            # Test connection to Ollama API
            test_url = f"{base_url}/api/tags"
            try:
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code != 200:
                    error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running."
                    logger.error(f"Ollama API unavailable: HTTP {test_response.status_code}")
                    yield error_msg
                    return
            except requests.RequestException as e:
                error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."
                logger.error(f"Failed to connect to Ollama: {e}")
                yield error_msg
                return

            # Process the streaming response
            with requests.post(url, headers=headers, json=data, stream=True, timeout=3600) as response:
                if response.status_code != 200:
                    error_msg = f"Error: Failed to generate response (HTTP {response.status_code})"
                    logger.error(f"Ollama API error: {response.status_code}")
                    yield error_msg
                    return

                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                yield token
                                time.sleep(stream_delay)

                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from Ollama API: {line}")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error during Ollama request: {e}")
            yield error_msg
    
    @staticmethod
    def _stream_ollama_response_full(prompt, model_name=None, base_url=None):
        """Non-generator version for full responses."""
        if model_name is None:
            model_name = config.LLM_MODEL_NAME
            
        if base_url is None:
            base_url = config.OLLAMA_BASE_URL
            
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": []
            }
        }

        full_response = ""

        try:
            # Test connection to Ollama API
            test_url = f"{base_url}/api/tags"
            try:
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code != 200:
                    error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running."
                    logger.error(f"Ollama API unavailable: HTTP {test_response.status_code}")
                    return error_msg
            except requests.RequestException as e:
                error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."
                logger.error(f"Failed to connect to Ollama: {e}")
                return error_msg

            # Process the streaming response
            with requests.post(url, headers=headers, json=data, stream=True, timeout=3600) as response:
                if response.status_code != 200:
                    error_msg = f"Error: Failed to generate response (HTTP {response.status_code})"
                    logger.error(f"Ollama API error: {response.status_code}")
                    return error_msg

                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                full_response += token

                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from Ollama API: {line}")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error during Ollama request: {e}")
            return error_msg
        
        return full_response