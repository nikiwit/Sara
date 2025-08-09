from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import traceback
import threading
import os
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Optional
from timetable import timetable

# Import Sara system components
from app import Sara
from query_handling import QueryRouter
from session_management import SessionManager

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Admin operations are handled via CLI only for better security
# No admin API endpoints - admins use: python main.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SaraAPI")

# Global Sara instance (thread-safe singleton)
_sara_instance = None
_sara_lock = threading.Lock()

def get_sara_instance():
    """Get or create Sara instance (thread-safe singleton)."""
    global _sara_instance
    
    if _sara_instance is None:
        with _sara_lock:
            if _sara_instance is None:
                logger.info("Initializing Sara system...")
                _sara_instance = Sara()
                if not _sara_instance.initialize():
                    logger.error("Failed to initialize Sara system")
                    raise RuntimeError("Failed to initialize Sara system")
                logger.info("Sara system initialized successfully")
    
    return _sara_instance


# Existing timetable endpoint
@app.route("/get_timetable/<intake_code>/<group_number>")
def get_timetable(intake_code, group_number):
    ignored_modules = request.args.getlist("ignored")
    print(ignored_modules)
    class_list = timetable.get_timetable(intake_code, group_number, ignored_modules)
    return jsonify(class_list)


# Chatbot Endpoints

@app.route("/chat/query", methods=["POST"])
def chat_query():
    """
    Handle user questions to the chatbot with automatic session management.
    
    Expected JSON payload:
    {
        "query": "What are the admission requirements?",
        "session_id": "optional_session_id",  // If not provided, creates new session automatically
        "device_id": "optional_device_identifier"  // For device-specific session tracking
    }
    
    Returns:
    {
        "response": "Sara's response",
        "session_id": "current_session_id",
        "is_new_session": false,
        "timestamp": "2025-01-20T10:30:00",
        "success": true
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "success": False
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty",
                "success": False
            }), 400
        
        session_id = data.get('session_id')
        device_id = data.get('device_id')
        
        # Get Sara instance
        sara = get_sara_instance()
        
        is_new_session = False
        
        # Automatic session management
        if session_id:
            # Try to load existing session
            existing_session = sara.session_manager.load_session(session_id)
            if not existing_session:
                # Session not found, create new one
                logger.info(f"Session {session_id} not found, creating new session")
                session_title = f"Chat Session"
                if device_id:
                    session_title += f" ({device_id[:8]})"
                session = sara.session_manager.create_session(session_title)
                is_new_session = True
                session_id = session.metadata.session_id
            else:
                # Update Sara's memory reference to existing session
                sara.memory = existing_session.memory
                if sara.conversation_handler:
                    sara.conversation_handler.memory = sara.memory
                if sara.retrieval_handler:
                    sara.retrieval_handler.memory = sara.memory
                if sara.query_router:
                    sara.query_router.memory = sara.memory
        else:
            # No session provided, create new one automatically
            logger.info("No session_id provided, creating new session automatically")
            session_title = f"Chat Session"
            if device_id:
                session_title += f" ({device_id[:8]})"
            session = sara.session_manager.create_session(session_title)
            is_new_session = True
            session_id = session.metadata.session_id
            
            # Update Sara's memory reference to new session
            sara.memory = session.memory
            if sara.conversation_handler:
                sara.conversation_handler.memory = sara.memory
            if sara.retrieval_handler:
                sara.retrieval_handler.memory = sara.memory
            if sara.query_router:
                sara.query_router.memory = sara.memory
        
        # Process the query
        query_analysis = sara.input_processor.analyze_query(query)
        
        # Route the query (without streaming for API)
        response, _ = sara.query_router.route_query(query_analysis, stream=False)
        
        # Save conversation to session
        if response and response.strip():
            sara.session_manager.add_conversation(query, response.strip())
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "is_new_session": is_new_session,
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


@app.route("/chat/conversation", methods=["POST"])
def chat_conversation():
    """
    Handle conversational/small-talk queries with automatic session management.
    
    Expected JSON payload:
    {
        "query": "Hello! How are you?",
        "session_id": "optional_session_id",  // If not provided, creates new session automatically
        "device_id": "optional_device_identifier"
    }
    
    Returns:
    {
        "response": "Sara's conversational response",
        "session_id": "current_session_id",
        "is_conversational": true,
        "is_new_session": false,
        "timestamp": "2025-01-20T10:30:00",
        "success": true
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "success": False
            }), 400
        
        query = data['query'].strip()
        session_id = data.get('session_id')
        device_id = data.get('device_id')
        
        # Get Sara instance
        sara = get_sara_instance()
        
        is_new_session = False
        
        # Automatic session management (same logic as chat_query)
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
                sara.memory = existing_session.memory
                if sara.conversation_handler:
                    sara.conversation_handler.memory = sara.memory
        else:
            logger.info("No session_id provided, creating new session automatically")
            session_title = f"Chat Session"
            if device_id:
                session_title += f" ({device_id[:8]})"
            session = sara.session_manager.create_session(session_title)
            is_new_session = True
            session_id = session.metadata.session_id
            sara.memory = session.memory
            if sara.conversation_handler:
                sara.conversation_handler.memory = sara.memory
        
        # Handle conversational query
        response = sara.conversation_handler.handle_conversation(query, stream=False)
        
        # Save conversation to session
        if response and response.strip():
            sara.session_manager.add_conversation(query, response.strip())
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "is_conversational": True,
            "is_new_session": is_new_session,
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error processing conversational query: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


# New Session Creation Endpoint (for + button)

@app.route("/chat/new-session", methods=["POST"])
def create_new_chat_session():
    """
    Create a new chat session (for + button in app).
    
    Expected JSON payload:
    {
        "device_id": "optional_device_identifier"
    }
    
    Returns:
    {
        "session_id": "new_session_id",
        "title": "Chat Session",
        "message": "New chat session created",
        "success": true
    }
    """
    try:
        data = request.get_json() or {}
        device_id = data.get('device_id')
        
        sara = get_sara_instance()
        
        # Create session with device-specific title
        session_title = "Chat Session"
        if device_id:
            session_title += f" ({device_id[:8]})"
        
        session = sara.session_manager.create_session(session_title)
        
        # Update Sara's memory reference to new session
        sara.memory = session.memory
        if sara.conversation_handler:
            sara.conversation_handler.memory = sara.memory
        if sara.retrieval_handler:
            sara.retrieval_handler.memory = sara.memory
        if sara.query_router:
            sara.query_router.memory = sara.memory
        
        return jsonify({
            "session_id": session.metadata.session_id,
            "title": session.metadata.title,
            "message": "New chat session created successfully",
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error creating new chat session: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


# Session Management Endpoints

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """
    List all available chat sessions.
    
    Returns:
    {
        "sessions": [
            {
                "session_id": "abc123...",
                "title": "Session Title",
                "message_count": 5,
                "created_at": "2025-01-20T10:00:00",
                "last_accessed": "2025-01-20T10:30:00"
            }
        ],
        "current_session_id": "abc123...",
        "success": true
    }
    """
    try:
        sara = get_sara_instance()
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
        
        return jsonify({
            "sessions": sessions,
            "current_session_id": current_session_id,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


@app.route("/sessions", methods=["POST"])
def create_session():
    """
    Create a new chat session.
    
    Expected JSON payload:
    {
        "title": "Optional session title"
    }
    
    Returns:
    {
        "session_id": "new_session_id",
        "title": "Session Title",
        "created_at": "2025-01-20T10:30:00",
        "success": true
    }
    """
    try:
        data = request.get_json() or {}
        title = data.get('title')
        
        sara = get_sara_instance()
        session = sara.session_manager.create_session(title)
        
        # Update Sara's memory reference
        sara.memory = session.memory
        if sara.conversation_handler:
            sara.conversation_handler.memory = sara.memory
        if sara.retrieval_handler:
            sara.retrieval_handler.memory = sara.memory
        if sara.query_router:
            sara.query_router.memory = sara.memory
        
        return jsonify({
            "session_id": session.metadata.session_id,
            "title": session.metadata.title,
            "created_at": session.metadata.created_at.isoformat(),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


@app.route("/sessions/<session_id>", methods=["PUT"])
def switch_session(session_id):
    """
    Switch to a specific session.
    
    Returns:
    {
        "session_id": "session_id",
        "title": "Session Title",
        "message_count": 3,
        "success": true
    }
    """
    try:
        sara = get_sara_instance()
        session = sara.session_manager.load_session(session_id)
        
        if not session:
            return jsonify({
                "error": f"Session {session_id} not found",
                "success": False
            }), 404
        
        # Update Sara's memory reference
        sara.memory = session.memory
        if sara.conversation_handler:
            sara.conversation_handler.memory = sara.memory
        if sara.retrieval_handler:
            sara.retrieval_handler.memory = sara.memory
        if sara.query_router:
            sara.query_router.memory = sara.memory
        
        return jsonify({
            "session_id": session.metadata.session_id,
            "title": session.metadata.title,
            "message_count": session.metadata.message_count,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error switching to session {session_id}: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


@app.route("/sessions/<session_id>", methods=["DELETE"])
def clear_session(session_id):
    """
    Clear a specific session's memory.
    
    Returns:
    {
        "message": "Session memory cleared",
        "success": true
    }
    """
    try:
        sara = get_sara_instance()
        
        # Check if this is the current session
        current_session = sara.session_manager.current_session
        if current_session and current_session.metadata.session_id == session_id:
            sara.session_manager.clear_current_session()
            return jsonify({
                "message": "Current session memory cleared",
                "success": True
            })
        
        # Load and clear the specific session
        session = sara.session_manager.load_session(session_id)
        if not session:
            return jsonify({
                "error": f"Session {session_id} not found",
                "success": False
            }), 404
        
        session.memory.clear()
        sara.session_manager.save_session(session)
        
        return jsonify({
            "message": f"Session {session_id} memory cleared",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500


# Admin operations are handled via CLI only
# Admins use: python main.py
# Available CLI commands: stats, reindex, model report, model check, model update, clear, etc.

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5000)