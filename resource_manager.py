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
        """Configure system resources based on environment."""
        # Set thread limits
        threading.stack_size(1024 * 1024)  # 1MB stack size
        
        # Log resource availability
        import psutil
        logger.info(f"CPU cores available: {psutil.cpu_count()}")
        logger.info(f"Memory available: {cls.get_memory_gb():.2f} GB")
        
        if config.has_gpu():
            import torch
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Set PyTorch to use GPU
            torch.set_default_device('cuda')
        else:
            logger.info("No GPU detected, using CPU only")
    
    @staticmethod
    def get_memory_gb():
        """Get system memory in GB."""
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    @classmethod
    def optimize_for_environment(cls):
        """Apply environment-specific optimizations."""
        if config.ENV == "production":
            cls._optimize_for_production()
        else:
            cls._optimize_for_local()
    
    @classmethod
    def _optimize_for_production(cls):
        """Apply production-specific optimizations."""
        if config.has_gpu():
            import torch
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True
            
            logger.info("Applied production GPU optimizations")
    
    @classmethod
    def _optimize_for_local(cls):
        """Apply local-specific optimizations."""
        # Limit memory usage for local development
        try:
            import torch
            
            if config.has_gpu():
                # Limit GPU memory usage
                torch.cuda.set_per_process_memory_fraction(0.7)
                logger.info("Limited GPU memory usage for local development")
        except ImportError:
            logger.info("PyTorch not available, skipping GPU optimizations")
