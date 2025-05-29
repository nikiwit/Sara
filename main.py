"""
Main entry point for APURAG system with enhanced error handling.
"""

import os
import sys
import logging
import signal
from contextlib import contextmanager

# Import configuration first
from config import config

# Configure logging before other imports
logger = logging.getLogger("CustomRAG")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

@contextmanager
def error_context(operation_name):
    """Context manager for consistent error handling."""
    try:
        logger.debug(f"Starting {operation_name}")
        yield
        logger.debug(f"Completed {operation_name}")
    except Exception as e:
        logger.error(f"Error during {operation_name}: {e}", exc_info=True)
        raise

def setup_environment():
    """Setup the environment with proper error handling."""
    with error_context("environment setup"):
        # Setup configuration (this will only run once due to the fix)
        config.setup()
        
        # Setup resources if ResourceManager is available
        try:
            from resource_manager import ResourceManager
            with error_context("resource management setup"):
                ResourceManager.setup_resources()
                ResourceManager.optimize_for_environment()
        except ImportError:
            logger.warning("ResourceManager not available, using default resource settings")
        except Exception as e:
            logger.warning(f"Resource management setup failed: {e}")
            logger.info("Continuing with default resource settings")

def create_and_run_app():
    """Create and run the RAG application with error handling."""
    rag_app = None
    try:
        with error_context("application initialization"):
            from app import CustomRAG
            rag_app = CustomRAG()
        
        with error_context("application execution"):
            logger.info(f"Starting APURAG in {config.ENV} environment")
            rag_app.run_cli()
            
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        return 0
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed")
        return 1
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    finally:
        # Ensure cleanup always runs
        if rag_app:
            try:
                with error_context("application cleanup"):
                    rag_app.cleanup()
                    logger.info("Application cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)
        
        logger.info("APURAG shutdown complete")
    
    return 0

def main():
    """Main entry point for the application with comprehensive error handling."""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    try:
        # Setup environment
        setup_environment()
        
        # Create and run the application
        exit_code = create_and_run_app()
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user during startup")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception during startup: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    # Run main and exit with appropriate code
    exit_code = main()
    sys.exit(exit_code)