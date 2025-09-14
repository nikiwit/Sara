from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import logging
import traceback
import threading
import os
import json
import time
import random
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Optional, Iterator

# Import Sara system components
from app import Sara
from query_handling import QueryRouter
from session_management import SessionManager

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sara")

# Constants
PROCESSING_MESSAGES = [
    "Reading through knowledge base...",
    "Thinking...",
    "Brewing the answer...",
    "Sifting through information...",
    "Assembling...",
    "Processing your request...",
    "Organizing thoughts...",
    "Searching for answers...",
    "Building response...",
    "Formulating answer...",
]

# Global state management
class APIState:
    def __init__(self):
        self._sara_instance = None
        self._sara_lock = threading.Lock()
        self._processing_status = {}
        self._status_lock = threading.Lock()
    
    def get_sara_instance(self):
        """Get or create Sara instance (thread-safe singleton)."""
        if self._sara_instance is None:
            with self._sara_lock:
                if self._sara_instance is None:
                    logger.info("Initializing Sara system...")
                    self._sara_instance = Sara()
                    if not self._sara_instance.initialize():
                        logger.error("Failed to initialize Sara system")
                        raise RuntimeError("Failed to initialize Sara system")
                    logger.info("Sara system initialized successfully")
        return self._sara_instance
    
    def set_processing_status(self, session_id: str, status: str):
        """Set processing status for a session."""
        with self._status_lock:
            self._processing_status[session_id] = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "message": random.choice(PROCESSING_MESSAGES) if status == "processing" else status
            }
    
    def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """Get processing status for a session."""
        with self._status_lock:
            return self._processing_status.get(session_id, {
                "status": "idle",
                "timestamp": datetime.now().isoformat(),
                "message": "Ready to help!"
            })
    
    def clear_processing_status(self, session_id: str):
        """Clear processing status for a session."""
        with self._status_lock:
            self._processing_status.pop(session_id, None)

api_state = APIState()

# Utility functions
def validate_request_data(data, required_fields):
    """Validate request data has required fields."""
    if not data:
        return False, "Missing request body"
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing '{field}' in request body"
    
    return True, None

def standardize_response(data=None, error=None, status_code=200):
    """Standardize API response format."""
    response_data = {
        "success": error is None,
        "timestamp": datetime.now().isoformat()
    }
    
    if data:
        response_data.update(data)
    
    if error:
        response_data["error"] = error
    
    return jsonify(response_data), status_code

def handle_response_format(response):
    """Ensure response is a string, not a generator."""
    if hasattr(response, '__iter__') and not isinstance(response, str):
        return ''.join(response) if response else ''
    return response

def setup_session_memory(sara, session):
    """Update Sara's memory references to a session."""
    sara.memory = session.memory
    for handler in [sara.conversation_handler, sara.retrieval_handler, sara.query_router]:
        if handler and hasattr(handler, 'memory'):
            handler.memory = sara.memory

def handle_session_management(sara, session_id, device_id):
    """Handle session creation and loading logic."""
    is_new_session = False
    
    if session_id:
        existing_session = sara.session_manager.load_session(session_id)
        if not existing_session:
            logger.info(f"Session {session_id} not found, creating new session")
            session_title = f"Chat Session"
            if device_id:
                session_title += f" ({device_id[:8]})"
            session = sara.session_manager.create_session(session_title)
            is_new_session = True
            session_id = session.metadata.session_id
        else:
            setup_session_memory(sara, existing_session)
    else:
        logger.info("No session_id provided, creating new session automatically")
        session_title = f"Chat Session"
        if device_id:
            session_title += f" ({device_id[:8]})"
        session = sara.session_manager.create_session(session_title)
        is_new_session = True
        session_id = session.metadata.session_id
        setup_session_memory(sara, session)
    
    return session_id, is_new_session

# Error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return standardize_response(error="Internal server error", status_code=500)

# Middleware for API mode security
def with_api_security(f):
    """Decorator to ensure API security mode."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            sara = api_state.get_sara_instance()
            if hasattr(sara, 'command_handler') and sara.command_handler:
                sara.command_handler.set_api_mode(True)
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return standardize_response(error=str(e), status_code=500)
    return decorated_function

# Chat endpoints
@app.route("/chat/query", methods=["POST"])
@with_api_security
def chat_query():
    """Handle user questions with automatic session management."""
    data = request.get_json()
    is_valid, error_msg = validate_request_data(data, ['query'])
    if not is_valid:
        return standardize_response(error=error_msg, status_code=400)
    
    query = data['query'].strip()
    if not query:
        return standardize_response(error="Query cannot be empty", status_code=400)
    
    session_id = data.get('session_id')
    device_id = data.get('device_id')
    
    sara = api_state.get_sara_instance()
    session_id, is_new_session = handle_session_management(sara, session_id, device_id)
    
    # Process query
    query_analysis = sara.input_processor.analyze_query(query)
    api_state.set_processing_status(session_id, "processing")
    
    # Check for streaming support
    accept_header = request.headers.get('Accept', '')
    wants_stream = 'text/event-stream' in accept_header
    
    if wants_stream:
        def generate_response_stream():
            yield f"data: {json.dumps({'type': 'status', 'message': api_state.get_processing_status(session_id)['message'], 'session_id': session_id})}\n\n"
            
            response_stream, _ = sara.query_router.route_query(query_analysis, stream=True)
            full_response = ""
            
            if hasattr(response_stream, '__iter__'):
                for chunk in response_stream:
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'session_id': session_id})}\n\n"
            else:
                full_response = str(response_stream)
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response, 'session_id': session_id})}\n\n"
            
            if full_response and full_response.strip():
                sara.session_manager.add_conversation(query, full_response.strip())
            
            api_state.clear_processing_status(session_id)
            yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id, 'timestamp': datetime.now().isoformat()})}\n\n"
        
        return Response(generate_response_stream(), mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
    
    # Non-streaming response
    response, _ = sara.query_router.route_query(query_analysis, stream=False)
    response = handle_response_format(response)
    api_state.clear_processing_status(session_id)
    
    # Save conversation
    if response and response.strip():
        sara.session_manager.add_conversation(query, response.strip())
    
    # Response metadata
    has_markdown = bool(response and any(marker in response for marker in ['**', '*', '#', '```', '[', '](']))
    has_links = bool(response and '](http' in response)
    
    return standardize_response({
        "response": response,
        "session_id": session_id,
        "is_new_session": is_new_session,
        "metadata": {
            "has_markdown": has_markdown,
            "has_links": has_links,
            "word_count": len(str(response).split()) if response else 0,
            "processing_status": "completed"
        }
    })

@app.route("/chat/conversation", methods=["POST"])
@with_api_security
def chat_conversation():
    """Handle conversational queries with automatic session management."""
    data = request.get_json()
    is_valid, error_msg = validate_request_data(data, ['query'])
    if not is_valid:
        return standardize_response(error=error_msg, status_code=400)
    
    query = data['query'].strip()
    session_id = data.get('session_id')
    device_id = data.get('device_id')
    
    sara = api_state.get_sara_instance()
    session_id, is_new_session = handle_session_management(sara, session_id, device_id)
    
    # Process query
    query_analysis = sara.input_processor.analyze_query(query)
    api_state.set_processing_status(session_id, "processing")
    
    # Check for streaming
    accept_header = request.headers.get('Accept', '')
    wants_stream = 'text/event-stream' in accept_header
    
    if wants_stream:
        def generate_conversation_stream():
            yield f"data: {json.dumps({'type': 'status', 'message': api_state.get_processing_status(session_id)['message'], 'session_id': session_id})}\n\n"
            
            response_stream = sara.conversation_handler.handle_conversation(query, stream=True)
            full_response = ""
            
            if hasattr(response_stream, '__iter__'):
                for chunk in response_stream:
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'session_id': session_id})}\n\n"
            else:
                full_response = str(response_stream)
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response, 'session_id': session_id})}\n\n"
            
            if full_response and full_response.strip():
                sara.session_manager.add_conversation(query, full_response.strip())
            
            api_state.clear_processing_status(session_id)
            yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id, 'timestamp': datetime.now().isoformat()})}\n\n"
        
        return Response(generate_conversation_stream(), mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
    
    # Non-streaming response
    response = sara.conversation_handler.handle_conversation(query, stream=False)
    response = handle_response_format(response)
    api_state.clear_processing_status(session_id)
    
    # Save conversation
    if response and response.strip():
        sara.session_manager.add_conversation(query, response.strip())
    
    # Response metadata
    has_markdown = bool(response and any(marker in response for marker in ['**', '*', '#', '```', '[', '](']))
    has_links = bool(response and '](http' in response)
    
    return standardize_response({
        "response": response,
        "session_id": session_id,
        "is_conversational": True,
        "is_new_session": is_new_session,
        "metadata": {
            "has_markdown": has_markdown,
            "has_links": has_links,
            "word_count": len(str(response).split()) if response else 0,
            "processing_status": "completed"
        }
    })

@app.route("/chat/stream", methods=["POST"])
@with_api_security
def chat_stream():
    """Stream chat responses with Server-Sent Events."""
    data = request.get_json()
    is_valid, error_msg = validate_request_data(data, ['query'])
    if not is_valid:
        return standardize_response(error=error_msg, status_code=400)
    
    query = data['query'].strip()
    if not query:
        return standardize_response(error="Query cannot be empty", status_code=400)
    
    session_id = data.get('session_id', f"stream_{int(time.time() * 1000)}")
    device_id = data.get('device_id')
    
    def generate_stream():
        try:
            sara = api_state.get_sara_instance()
            session_id_final, is_new_session = handle_session_management(sara, session_id, device_id)
            
            api_state.set_processing_status(session_id_final, "processing")
            yield f"data: {json.dumps({'type': 'status', 'message': api_state.get_processing_status(session_id_final)['message'], 'session_id': session_id_final})}\n\n"
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id_final, 'is_new_session': is_new_session})}\n\n"
            
            query_analysis = sara.input_processor.analyze_query(query)
            api_state.set_processing_status(session_id_final, "generating")
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...', 'session_id': session_id_final})}\n\n"
            
            response_stream, _ = sara.query_router.route_query(query_analysis, stream=True)
            full_response = ""
            
            if hasattr(response_stream, '__iter__'):
                for chunk in response_stream:
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'session_id': session_id_final})}\n\n"
            else:
                full_response = str(response_stream)
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response, 'session_id': session_id_final})}\n\n"
            
            if full_response and full_response.strip():
                sara.session_manager.add_conversation(query, full_response.strip())
            
            api_state.clear_processing_status(session_id_final)
            yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id_final, 'timestamp': datetime.now().isoformat()})}\n\n"
        
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            api_state.clear_processing_status(session_id)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}', 'session_id': session_id})}\n\n"
    
    return Response(generate_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

# Status endpoints
@app.route("/chat/status/<session_id>", methods=["GET"])
def chat_status(session_id):
    """Get processing status for a session."""
    status_info = api_state.get_processing_status(session_id)
    return standardize_response(status_info)

@app.route("/chat/history/<session_id>", methods=["GET"])
@with_api_security
def chat_history(session_id):
    """Get conversation history for a session."""
    sara = api_state.get_sara_instance()
    session = sara.session_manager.load_session(session_id)
    
    if not session:
        return standardize_response(error=f"Session {session_id} not found", status_code=404)
    
    messages = []
    if hasattr(session.memory, 'chat_memory') and hasattr(session.memory.chat_memory, 'messages'):
        for msg in session.memory.chat_memory.messages:
            messages.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
    
    return standardize_response({
        "messages": messages,
        "session_id": session_id,
        "message_count": len(messages)
    })

# Session management endpoints
@app.route("/chat/new-session", methods=["POST"])
@with_api_security
def create_new_chat_session():
    """Create a new chat session."""
    data = request.get_json() or {}
    device_id = data.get('device_id')
    
    sara = api_state.get_sara_instance()
    session_title = "Chat Session"
    if device_id:
        session_title += f" ({device_id[:8]})"
    
    session = sara.session_manager.create_session(session_title)
    setup_session_memory(sara, session)
    
    return standardize_response({
        "session_id": session.metadata.session_id,
        "title": session.metadata.title,
        "message": "New chat session created successfully"
    })

@app.route("/sessions", methods=["GET"])
@with_api_security
def list_sessions():
    """List all available chat sessions."""
    sara = api_state.get_sara_instance()
    sessions_data = sara.session_manager.list_sessions()
    
    sessions = []
    for session in sessions_data:
        sessions.append({
            "session_id": session.session_id,
            "title": session.title,
            "message_count": session.message_count,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat()
        })
    
    current_session = sara.session_manager.current_session
    current_session_id = current_session.metadata.session_id if current_session else None
    
    return standardize_response({
        "sessions": sessions,
        "current_session_id": current_session_id
    })

@app.route("/sessions", methods=["POST"])
@with_api_security
def create_session():
    """Create a new chat session."""
    data = request.get_json() or {}
    title = data.get('title')
    
    sara = api_state.get_sara_instance()
    session = sara.session_manager.create_session(title)
    setup_session_memory(sara, session)
    
    return standardize_response({
        "session_id": session.metadata.session_id,
        "title": session.metadata.title,
        "created_at": session.metadata.created_at.isoformat()
    })

@app.route("/sessions/<session_id>", methods=["PUT"])
@with_api_security
def switch_session(session_id):
    """Switch to a specific session."""
    sara = api_state.get_sara_instance()
    session = sara.session_manager.load_session(session_id)
    
    if not session:
        return standardize_response(error=f"Session {session_id} not found", status_code=404)
    
    setup_session_memory(sara, session)
    
    return standardize_response({
        "session_id": session.metadata.session_id,
        "title": session.metadata.title,
        "message_count": session.metadata.message_count
    })

@app.route("/sessions/<session_id>", methods=["DELETE"])
@with_api_security
def clear_session(session_id):
    """Clear a specific session's memory."""
    sara = api_state.get_sara_instance()
    
    # Check if this is the current session
    current_session = sara.session_manager.current_session
    if current_session and current_session.metadata.session_id == session_id:
        sara.session_manager.clear_current_session()
        return standardize_response({"message": "Current session memory cleared"})
    
    # Load and clear the specific session
    session = sara.session_manager.load_session(session_id)
    if not session:
        return standardize_response(error=f"Session {session_id} not found", status_code=404)
    
    session.memory.clear()
    sara.session_manager.save_session(session)
    
    return standardize_response({"message": f"Session {session_id} memory cleared"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8000)