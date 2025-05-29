"""
Resource management module for APURAG system.
Optimizes resource usage based on environment.
"""

import os
import logging
import threading
from config import config

logger = logging.getLogger("CustomRAG")

class ResourceManager:
    """Manages system resources based on environment configuration."""
    
    @classmethod
    def setup_resources(cls):
        """Configure system resources based on environment with error handling."""
        try:
            # Set thread limits
            threading.stack_size(1024 * 1024)  # 1MB stack size
            
            # Log resource availability
            try:
                import psutil
                logger.info(f"CPU cores available: {psutil.cpu_count()}")
                logger.info(f"Memory available: {cls.get_memory_gb():.2f} GB")
            except ImportError:
                logger.warning("psutil not available - resource monitoring disabled")
                return
            
            if config.has_gpu():
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                        
                        # Set PyTorch to use GPU
                        torch.set_default_device('cuda')
                    else:
                        logger.info("PyTorch available but no CUDA GPU detected")
                except ImportError:
                    logger.warning("PyTorch not available - GPU optimizations disabled")
            else:
                logger.info("No GPU detected, using CPU only")
                
        except Exception as e:
            logger.error(f"Error during resource setup: {e}")
            logger.info("Continuing with default resource settings")
    
    @staticmethod
    def get_memory_gb():
        """Get system memory in GB with error handling."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024 * 1024)
        except ImportError:
            logger.warning("psutil not available - cannot determine memory size")
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
            return 0.0
    
    @classmethod
    def optimize_for_environment(cls):
        """Apply environment-specific optimizations with error handling."""
        try:
            if config.ENV == "production":
                cls._optimize_for_production()
            else:
                cls._optimize_for_local()
        except Exception as e:
            logger.error(f"Error during environment optimization: {e}")
            logger.info("Continuing with default settings")
    
    @classmethod
    def _optimize_for_production(cls):
        """Apply production-specific optimizations."""
        try:
            if config.has_gpu():
                try:
                    import torch
                    
                    # Enable TF32 for better performance on Ampere GPUs
                    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        
                        # Enable cuDNN benchmarking
                        torch.backends.cudnn.benchmark = True
                        
                        logger.info("Applied production GPU optimizations")
                    else:
                        logger.info("TF32 optimizations not available on this PyTorch version")
                except ImportError:
                    logger.warning("PyTorch not available - skipping GPU optimizations")
        except Exception as e:
            logger.error(f"Error applying production optimizations: {e}")
    
    @classmethod
    def _optimize_for_local(cls):
        """Apply local-specific optimizations."""
        try:
            # Limit memory usage for local development
            try:
                import torch
                
                if config.has_gpu() and torch.cuda.is_available():
                    # Limit GPU memory usage
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.7)
                        logger.info("Limited GPU memory usage for local development")
                    except Exception as e:
                        logger.warning(f"Could not limit GPU memory: {e}")
            except ImportError:
                logger.info("PyTorch not available, skipping GPU optimizations")
        except Exception as e:
            logger.error(f"Error applying local optimizations: {e}")