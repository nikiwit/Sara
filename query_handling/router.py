"""
Query routing logic for directing queries to appropriate handlers.
"""

import logging
from typing import Dict, Any, Tuple
from apurag_types import QueryType

logger = logging.getLogger("CustomRAG")

class QueryRouter:
    """Routes queries to appropriate handlers based on query type."""
    
    def __init__(self, conversation_handler, retrieval_handler, command_handler):
        """Initialize with handlers for different query types."""
        self.conversation_handler = conversation_handler
        self.retrieval_handler = retrieval_handler
        self.command_handler = command_handler
    
    def route_query(self, query_analysis: Dict[str, Any], stream=False) -> Tuple[Any, bool]:
        """
        Route the query to the appropriate handler.
        
        Args:
            query_analysis: Analysis output from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Tuple of (handler_result, should_continue)
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        
        # Route based on query type
        if query_type == QueryType.COMMAND:
            # Handle system commands (not streaming these)
            return self.command_handler.handle_command(original_query)
        
        elif query_type == QueryType.CONVERSATIONAL:
            # Handle conversational queries (now with streaming support)
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        else:
            # Handle all other query types with retrieval system
            response = self.retrieval_handler.process_query(query_analysis, stream=stream)
            return response, True