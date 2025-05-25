"""
Main entry point for APURAG system.
"""

import os
import sys
import logging
from app import CustomRAG
from config import config
from resource_manager import ResourceManager

# Configure logging
logger = logging.getLogger("CustomRAG")

def main():
    """Main entry point for the application."""
    # Setup configuration
    config.setup()
    
    # Setup resources
    ResourceManager.setup_resources()
    ResourceManager.optimize_for_environment()
    
    # Log environment
    logger.info(f"Starting APURAG in {config.ENV} environment")
    
    # Create and run the RAG application
    rag_app = CustomRAG()
    rag_app.run_cli()
    
    # Cleanup
    rag_app.cleanup()
    logger.info("APURAG shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
