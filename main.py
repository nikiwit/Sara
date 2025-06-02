"""
Main entry point for APURAG system with enhanced error handling.
"""

import os
import sys
import time
import logging
import signal
from contextlib import contextmanager

# Import configuration first and ensure proper setup
try:
    from config import config
    config.setup()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load configuration: {e}")
    sys.exit(1)

# Configure logging after configuration setup
logger = logging.getLogger("CustomRAG")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        # Try to cleanup gracefully
        try:
            from vector_management import ChromaDBManager
            ChromaDBManager.close()
        except Exception as e:
            logger.warning(f"Error during signal cleanup: {e}")
        
        # Verify cleanup on shutdown
        verify_cleanup_on_shutdown()
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
        logger.error(f"Error during {operation_name}: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        raise

def setup_environment():
    """Setup the environment with proper error handling."""
    with error_context("environment setup"):
        # Configuration is already setup in imports, but validate it
        try:
            # Validate critical settings
            validation_passed = config.validate_configuration()
            if not validation_passed:
                logger.warning("Configuration validation found issues - continuing with warnings")
            
            # Log optimization summary
            optimization_summary = config.get_optimization_summary()
            logger.info(f"Configuration optimizations applied: {optimization_summary}")
            
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            logger.info("Continuing with default settings")
        
        # Setup resources if ResourceManager is available
        try:
            from resource_manager import ResourceManager
            with error_context("resource management setup"):
                ResourceManager.setup_resources()
                ResourceManager.optimize_for_environment()
                logger.info("Resource management configured successfully")
        except ImportError:
            # This is fine - we'll use the config settings instead
            logger.debug("ResourceManager not available, using config-based resource settings")
        except Exception as e:
            logger.warning(f"Resource management setup failed: {e}")
            logger.info("Continuing with config-based resource settings")

def validate_environment():
    """Validate that the environment is properly set up."""
    issues = []
    
    # Check data directory
    if not os.path.exists(config.DATA_PATH):
        issues.append(f"Data directory not found: {config.DATA_PATH}")
    
    # Check for APU KB file if APU filtering is enabled
    if config.FILTER_APU_ONLY:
        apu_kb_path = os.path.join(config.DATA_PATH, "apu_kb.txt")
        if not os.path.exists(apu_kb_path):
            issues.append(f"APU KB file not found: {apu_kb_path} (APU filtering enabled)")
    
    # Check vector store directory permissions
    try:
        os.makedirs(config.PERSIST_PATH, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create vector store directory: {e}")
    
    # Log issues or success
    if issues:
        logger.warning("Environment validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("Environment validation passed")
        return True

def perform_preflight_checks():
    """Perform pre-flight checks before starting the application."""
    logger.info("Performing pre-flight checks...")
    
    checks_passed = True
    
    # Check vector store integrity
    try:
        if os.path.exists(config.PERSIST_PATH):
            sqlite_path = os.path.join(config.PERSIST_PATH, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                file_size = os.path.getsize(sqlite_path)
                # Less than 1KB is suspicious for a vector store
                if file_size < 1000:
                    logger.warning(f"Vector store SQLite file seems too small: {file_size} bytes")
                    
                    # Check if backup exists
                    backup_path = os.path.join(os.path.dirname(config.PERSIST_PATH), "embeddings_backup.pkl")
                    if os.path.exists(backup_path):
                        logger.info("Backup file exists - recovery possible")
                    else:
                        logger.warning("No backup file found - may need to rebuild")
                else:
                    logger.info(f"Vector store SQLite file exists: {file_size} bytes")
    except Exception as e:
        logger.error(f"Error checking vector store integrity: {e}")
        checks_passed = False
    
    # Check data directory for compatible files
    try:
        if os.path.exists(config.DATA_PATH):
            files = os.listdir(config.DATA_PATH)
            compatible_files = [f for f in files if any(f.endswith(ext) for ext in config.SUPPORTED_EXTENSIONS)]
            
            if not compatible_files:
                logger.warning("No compatible documents found in data directory")
                checks_passed = False
            else:
                logger.info(f"Found {len(compatible_files)} compatible documents")
                
                # Check for APU files if filtering is enabled
                if config.FILTER_APU_ONLY:
                    apu_files = [f for f in compatible_files if 'apu' in f.lower()]
                    if not apu_files:
                        logger.warning("APU filtering enabled but no APU files found")
                        checks_passed = False
                    else:
                        logger.info(f"Found {len(apu_files)} APU documents")
        else:
            logger.error(f"Data directory does not exist: {config.DATA_PATH}")
            checks_passed = False
    except Exception as e:
        logger.error(f"Error checking data directory: {e}")
        checks_passed = False
    
    # Check ChromaDB settings if validation method exists
    if hasattr(config, 'validate_chromadb_settings'):
        valid, issues = config.validate_chromadb_settings()
        if not valid:
            logger.warning("ChromaDB settings validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            checks_passed = False
        else:
            logger.info("ChromaDB settings validated")
    
    return checks_passed

def verify_cleanup_on_shutdown():
    """Verify cleanup operations completed successfully."""
    logger.info("Verifying cleanup operations...")
    
    cleanup_status = {
        "chromadb_closed": False,
        "backup_created": False,
        "sessions_saved": False
    }
    
    try:
        # Check if ChromaDB client is properly closed
        from vector_management import ChromaDBManager
        if ChromaDBManager._client is None:
            cleanup_status["chromadb_closed"] = True
            logger.info("ChromaDB client properly closed")
        else:
            logger.warning("ChromaDB client may not be properly closed")
        
        # Check if backup was created (if enabled)
        if getattr(config, 'COLLECTION_BACKUP_ON_SHUTDOWN', True):
            backup_path = os.path.join(os.path.dirname(config.PERSIST_PATH), "embeddings_backup.pkl")
            if os.path.exists(backup_path):
                # Check if backup is recent (within last 60 seconds)
                backup_time = os.path.getmtime(backup_path)
                if time.time() - backup_time < 60:
                    cleanup_status["backup_created"] = True
                    logger.info("Recent backup created")
                else:
                    logger.warning("Backup exists but may not be from this session")
        
        # Log final status
        all_good = all(cleanup_status.values())
        if all_good:
            logger.info("All cleanup operations completed successfully")
        else:
            logger.warning("Some cleanup operations may have failed")
            for operation, status in cleanup_status.items():
                if not status:
                    logger.warning(f"   - {operation}: Failed")
        
        return all_good
        
    except Exception as e:
        logger.error(f"Error during cleanup verification: {e}")
        return False

def create_and_run_app():
    """Create and run the RAG application with error handling."""
    rag_app = None
    try:
        # Perform pre-flight checks
        if not perform_preflight_checks():
            logger.warning("Pre-flight checks found issues - application may not work correctly")
            print("\nWARNING: Pre-flight checks found issues. Check the logs for details.")
            print("The application will attempt to start anyway...\n")
            # Give user time to read the warning
            time.sleep(2)
        
        # Validate environment before starting
        if not validate_environment():
            logger.warning("Environment validation failed - some features may not work correctly")
        
        with error_context("application initialization"):
            from app import CustomRAG
            rag_app = CustomRAG()
        
        with error_context("application execution"):
            logger.info(f"Starting APURAG in {config.ENV} environment")
            
            # Log the actual applied settings
            if hasattr(config, 'get_local_optimization_summary'):
                summary = config.get_local_optimization_summary()
                logger.info(f"Applied optimizations: {summary}")
            elif hasattr(config, 'get_production_stats'):
                summary = config.get_production_stats()
                logger.info(f"Production settings: {summary}")
            
            rag_app.run_cli()
            
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        return 0
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed")
        return 1
    except Exception as e:
        logger.error(f"Application error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.error(traceback.format_exc())
        return 1
    finally:
        # Ensure cleanup always runs
        if rag_app:
            try:
                with error_context("application cleanup"):
                    # Create backup if enabled
                    if getattr(config, 'COLLECTION_BACKUP_ON_SHUTDOWN', True):
                        try:
                            from vector_management import VectorStoreManager
                            if rag_app.vector_store:
                                VectorStoreManager.save_embeddings_backup(rag_app.vector_store)
                                logger.info("Created final backup before shutdown")
                        except Exception as e:
                            logger.warning(f"Could not create final backup: {e}")
                    
                    rag_app.cleanup()
                    logger.info("Application cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        # Verify cleanup
        verify_cleanup_on_shutdown()
        logger.info("APURAG shutdown complete")
    
    return 0

def print_startup_banner():
    """Print startup banner with configuration info."""
    try:
        device_info = config.get_device_info()
        
        print(f"""
┌{'─' * 68}┐
│  APURAG - APU Knowledge Base Assistant - OPTIMIZED VERSION     │  
├{'─' * 68}┤
│  Environment: {config.ENV.upper():<12} │ CPU Cores: {device_info.get('cpu_cores', 'Unknown'):<3} │ Threads: {config.MAX_THREADS:<3} │
│  Platform: {device_info.get('platform', 'Unknown'):<15} │ Memory: {config.MAX_MEMORY:<6} │ GPU: {'Yes' if device_info.get('has_gpu', False) else 'No':<3} │
│  APU Filter: {'ON' if config.FILTER_APU_ONLY else 'OFF':<8} │ Context: {config.MAX_CONTEXT_SIZE:<6} │ Cache: {'Yes' if config.ENABLE_RESULT_CACHING else 'No'}  │
└{'─' * 68}┘
        """)
    except Exception as e:
        logger.debug(f"Error printing banner: {e}")
        print("APURAG - APU Knowledge Base Assistant")

def main():
    """Main entry point for the application with comprehensive error handling."""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    try:
        # Print startup banner
        print_startup_banner()
        
        # Setup environment
        setup_environment()
        
        # Create and run the application
        exit_code = create_and_run_app()
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user during startup")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception during startup: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Run main and exit with appropriate code
    exit_code = main()
    sys.exit(exit_code)