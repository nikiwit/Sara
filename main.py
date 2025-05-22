"""
Entry point for the Enhanced CustomRAG application.

Enhanced CustomRAG - An Advanced Retrieval Augmented Generation System for APU Knowledge Base
----------------------------------------------------------------------------------
This application allows users to query the APU knowledge base using natural language.
It has been specially optimized for FAQ-style content organized in pages.

Features:
- Specialized APU knowledge base parsing and structure preservation
- Enhanced metadata extraction from structured content
- Education-specific query classification
- FAQ-optimized retrieval strategies
- Better context generation for Q&A content
- Improved direct question matching

Original Author: Nik
"""

import sys
import logging
from app import CustomRAG

logger = logging.getLogger("CustomRAG")

def main():
    """Entry point for the CustomRAG application."""
    app = None
    try:
        app = CustomRAG()
        app.run_cli()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up resources
        if app is not None:
            try:
                app.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())