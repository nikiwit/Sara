"""
Vector store management operations and utilities with optimized model caching and production management.

This module provides comprehensive vector database operations including:
- Production-grade model caching and lifecycle management
- Automatic model update detection and user prompting
- ChromaDB integration with health monitoring
- Document indexing and metadata management
- Backup and recovery functionality
"""

import os
import sys
import shutil
import time
import logging
import torch
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from .chromadb_manager import ChromaDBManager

logger = logging.getLogger("Sara")

class VectorStoreManager:
    """
    Manages vector database operations with optimized caching and production model management.
    
    This class provides a comprehensive suite of tools for managing embedding models,
    vector stores, and document indexing in both development and production environments.
    """
    
    _cached_embeddings = None
    _cached_embeddings_model = None
    
    @classmethod
    def get_model_config(cls):
        """
        Get model management configuration from config.
        
        Returns:
            dict: Configuration dictionary containing model management settings
        """
        return {
            'CHECK_INTERVAL_DAYS': config.MODEL_CHECK_INTERVAL_DAYS,
            'WARNING_AGE_DAYS': config.MODEL_WARNING_AGE_DAYS,
            'CRITICAL_AGE_DAYS': config.MODEL_CRITICAL_AGE_DAYS,
            'AUTO_UPDATE_PROMPT': config.MODEL_AUTO_UPDATE_PROMPT,
            'UPDATE_CHECK_ENABLED': config.MODEL_UPDATE_CHECK_ENABLED,
            'REQUIRE_APPROVAL': config.MODEL_REQUIRE_APPROVAL,
            'CACHE_CLEANUP': config.MODEL_CACHE_CLEANUP,
            'BACKUP_ENABLED': config.MODEL_BACKUP_ENABLED,
            'MAX_BACKUPS': config.MODEL_MAX_BACKUPS,
            'NOTIFICATION_EMAIL': config.MODEL_NOTIFICATION_EMAIL
        }
    
    @staticmethod
    def setup_model_cache():
        """
        Setup HuggingFace cache directory and ensure proper caching.
        
        Configures environment variables for optimal model caching and sets up
        the directory structure required for persistent model storage.
        Includes automatic cleanup of corrupted model files on startup.
        
        Returns:
            str: Path to the configured cache directory
        """
        # Set up HuggingFace cache directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "..", "model_cache", "huggingface")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Clean up corrupted files on startup
        VectorStoreManager._cleanup_corrupted_cache_files(cache_dir)
        
        # Set HuggingFace environment variables for caching
        os.environ['HF_HOME'] = cache_dir
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(cache_dir, "sentence_transformers")
        
        # Disable telemetry for faster loading
        os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow downloads but cache aggressively
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        
        # Remove deprecated TRANSFORMERS_CACHE if it exists to avoid warnings
        if 'TRANSFORMERS_CACHE' in os.environ:
            del os.environ['TRANSFORMERS_CACHE']
        
        logger.info(f"Model cache directory: {cache_dir}")
        return cache_dir
    
    @staticmethod
    def _cleanup_corrupted_cache_files(cache_dir):
        """
        Clean up corrupted model cache files on startup.
        
        Performs automatic cleanup of incomplete downloads, broken symlinks,
        and corrupted model files to prevent startup issues.
        
        Args:
            cache_dir: Path to the HuggingFace cache directory
        """
        cleanup_count = 0
        
        try:
            # Step 1: Remove incomplete download files
            incomplete_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.incomplete'):
                        file_path = os.path.join(root, file)
                        incomplete_files.append(file_path)
            
            if incomplete_files:
                logger.info(f"Cleaning up {len(incomplete_files)} incomplete download files")
                for file_path in incomplete_files:
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.debug(f"Removed incomplete file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove incomplete file {file_path}: {e}")
            
            # Step 2: Remove lock files that may prevent downloads
            lock_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.lock') or file.endswith('.tmp'):
                        file_path = os.path.join(root, file)
                        lock_files.append(file_path)
            
            if lock_files:
                logger.info(f"Cleaning up {len(lock_files)} lock/temporary files")
                for file_path in lock_files:
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.debug(f"Removed lock file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove lock file {file_path}: {e}")
            
            # Step 3: Check for broken symlinks in HF cache structure
            broken_symlinks = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.islink(file_path) and not os.path.exists(file_path):
                        broken_symlinks.append(file_path)
            
            if broken_symlinks:
                logger.info(f"Cleaning up {len(broken_symlinks)} broken symlinks")
                for file_path in broken_symlinks:
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.debug(f"Removed broken symlink: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove broken symlink {file_path}: {e}")
            
            # Step 4: Validate model directories and clean up corrupted ones
            problematic_models = VectorStoreManager._validate_model_cache_integrity(cache_dir)
            
            # Step 5: Remove corrupted model directories
            corrupted_cleanup_count = 0
            if problematic_models:
                logger.warning(f"Found {len(problematic_models)} models with potential issues")
                for model_name, issues in problematic_models.items():
                    logger.warning(f"Model {model_name}: {', '.join(issues)}")
                    
                    # Remove corrupted model directories
                    try:
                        # Use cache_dir directly to avoid recursion
                        possible_paths = [
                            os.path.join(cache_dir, "hub", f"models--{model_name.replace('/', '--')}"),
                            os.path.join(cache_dir, "sentence_transformers", f"sentence-transformers_{model_name.replace('/', '_')}"),
                            os.path.join(cache_dir, "sentence_transformers", f"models--{model_name.replace('/', '--')}")
                        ]
                        
                        for model_path in possible_paths:
                            if os.path.exists(model_path) and os.path.isdir(model_path):
                                logger.info(f"Removing corrupted model cache at {model_path}")
                                shutil.rmtree(model_path)
                                corrupted_cleanup_count += 1
                                logger.info(f"Successfully removed corrupted model: {model_name}")
                                break
                    except Exception as e:
                        logger.error(f"Failed to remove corrupted model {model_name}: {e}")
                
                if corrupted_cleanup_count > 0:
                    logger.info(f"Cleaned up {corrupted_cleanup_count} corrupted models")
                    cleanup_count += corrupted_cleanup_count
                else:
                    logger.info("Problematic models will be re-downloaded if needed")
            
            if cleanup_count > 0:
                logger.info(f"Model cache cleanup completed: removed {cleanup_count} corrupted files")
            elif cleanup_count == 0 and not incomplete_files and not lock_files and not broken_symlinks:
                logger.debug("Model cache is clean - no corrupted files found")
            
        except Exception as e:
            logger.error(f"Error during model cache cleanup: {e}")
    
    @staticmethod
    def _validate_model_cache_integrity(cache_dir):
        """
        Validate integrity of cached models.
        
        Checks for missing essential files in model cache directories
        and identifies models that may need re-downloading.
        
        Args:
            cache_dir: Path to the HuggingFace cache directory
            
        Returns:
            dict: Dictionary of model names and their issues
        """
        # Prevent infinite recursion during cleanup
        if getattr(VectorStoreManager, '_validating_cache', False):
            return {}
            
        VectorStoreManager._validating_cache = True
        
        try:
            problematic_models = {}
            
            # Check sentence-transformers cache
            st_cache_dir = os.path.join(cache_dir, "sentence_transformers")
            if os.path.exists(st_cache_dir):
                for item in os.listdir(st_cache_dir):
                    if item.startswith('models--'):
                        model_path = os.path.join(st_cache_dir, item)
                        if os.path.isdir(model_path):
                            issues = VectorStoreManager._check_model_directory_integrity(model_path)
                            if issues:
                                model_name = item.replace('models--', '').replace('--', '/')
                                problematic_models[model_name] = issues
            
            # Check hub cache
            hub_cache_dir = os.path.join(cache_dir, "hub")
            if os.path.exists(hub_cache_dir):
                for item in os.listdir(hub_cache_dir):
                    if item.startswith('models--'):
                        model_path = os.path.join(hub_cache_dir, item)
                        if os.path.isdir(model_path):
                            issues = VectorStoreManager._check_model_directory_integrity(model_path)
                            if issues:
                                model_name = item.replace('models--', '').replace('--', '/')
                                problematic_models[model_name] = issues
            
            return problematic_models
        
        except Exception as e:
            logger.debug(f"Error validating model cache integrity: {e}")
            return {}
        finally:
            VectorStoreManager._validating_cache = False
    
    @staticmethod
    def _check_model_directory_integrity(model_path):
        """
        Check integrity of a specific model directory.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            list: List of issues found (empty if model is fine)
        """
        issues = []
        
        try:
            # Check for essential files
            essential_files = ['config.json']
            model_files = ['pytorch_model.bin', 'model.safetensors', 'model.onnx']
            
            # Check snapshots directory (HF cache structure)
            snapshots_dir = os.path.join(model_path, 'snapshots')
            blobs_dir = os.path.join(model_path, 'blobs')
            
            has_snapshots = os.path.exists(snapshots_dir) and os.listdir(snapshots_dir)
            has_blobs = os.path.exists(blobs_dir)
            
            if has_snapshots and has_blobs:
                # Check for incomplete files in blobs
                if os.path.exists(blobs_dir):
                    incomplete_blobs = [f for f in os.listdir(blobs_dir) if f.endswith('.incomplete')]
                    if incomplete_blobs:
                        issues.append("incomplete blob files")
                
                # Check if snapshots have essential files
                has_config = False
                has_model = False
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if os.path.isdir(snapshot_path):
                        if any(os.path.exists(os.path.join(snapshot_path, f)) for f in essential_files):
                            has_config = True
                        if any(os.path.exists(os.path.join(snapshot_path, f)) for f in model_files):
                            has_model = True
                
                if not has_config:
                    issues.append("missing config files")
                if not has_model:
                    issues.append("missing model files")
            
            else:
                # Check root directory for direct files
                has_config = any(os.path.exists(os.path.join(model_path, f)) for f in essential_files)
                has_model = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
                
                if not has_config:
                    issues.append("missing config files")
                if not has_model:
                    issues.append("missing model files")
        
        except Exception as e:
            issues.append(f"error checking directory: {str(e)}")
        
        return issues
    
    @staticmethod
    def _get_model_registry_path():
        """
        Get the path to the model registry file.
        
        Returns:
            str: Full path to the production model registry JSON file
        """
        cache_dir = VectorStoreManager.setup_model_cache()
        return os.path.join(cache_dir, "production_model_registry.json")
    
    @staticmethod
    def _load_model_registry() -> Dict[str, Any]:
        """
        Load the production model registry.
        
        Returns:
            dict: Model registry data or empty dict if file doesn't exist
        """
        registry_path = VectorStoreManager._get_model_registry_path()
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model registry: {e}")
        return {}
    
    @staticmethod
    def _save_model_registry(registry: Dict[str, Any]):
        """
        Save the production model registry.
        
        Args:
            registry: Model registry data to save
        """
        try:
            registry_path = VectorStoreManager._get_model_registry_path()
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save model registry: {e}")
    
    @staticmethod
    def _register_model_usage(model_name, model_path):
        """
        Register model usage with proper error handling and date formatting.
        
        This method creates or updates metadata files for model tracking,
        preserving the original cache date while incrementing usage counts.
        
        Args:
            model_name: Name of the model to register
            model_path: Path to the cached model files
        """
        try:
            metadata_file = os.path.join(
                VectorStoreManager.setup_model_cache(), 
                f"{model_name.replace('/', '_')}_metadata.json"
            )
            
            # Get current timestamp in ISO format
            current_time = datetime.now().isoformat()
            
            metadata = {
                'model_name': model_name,
                'model_path': str(model_path) if model_path else None,
                'first_cached': current_time,
                'last_accessed': current_time,
                'usage_count': 1
            }
            
            # If metadata already exists, preserve first_cached and increment usage
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                        
                    # Preserve the original first_cached date
                    if 'first_cached' in existing_metadata and existing_metadata['first_cached']:
                        metadata['first_cached'] = existing_metadata['first_cached']
                        
                    # Increment usage count
                    if 'usage_count' in existing_metadata:
                        metadata['usage_count'] = existing_metadata.get('usage_count', 0) + 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not read existing metadata, creating new: {e}")
            
            # Write metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Registered model usage: {model_name} (usage count: {metadata['usage_count']})")
            
        except Exception as e:
            logger.error(f"Failed to register model usage for {model_name}: {e}")
            
    @staticmethod
    def _get_cached_model_metadata(model_name):
        """
        Get cached model metadata with improved error handling.
        
        Args:
            model_name: Name of the model to retrieve metadata for
            
        Returns:
            dict: Model metadata or None if not found/invalid
        """
        try:
            metadata_file = os.path.join(
                VectorStoreManager.setup_model_cache(), 
                f"{model_name.replace('/', '_')}_metadata.json"
            )
            
            if not os.path.exists(metadata_file):
                logger.debug(f"No metadata file found for {model_name}")
                return None
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Validate required fields
            if not metadata.get('first_cached'):
                logger.warning(f"Metadata file exists but missing first_cached date for {model_name}")
                return None
                
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to read cached model metadata for {model_name}: {e}")
            return None
        
    @staticmethod
    def _get_model_version_info(model_path: str) -> Dict[str, Any]:
        """
        Extract version information from cached model.
        
        Args:
            model_path: Path to the cached model directory
            
        Returns:
            dict: Model version information including size and configuration
        """
        if not model_path or not os.path.exists(model_path):
            return {}

        version_info = {}
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    version_info['model_type'] = config.get('model_type', 'unknown')
                    version_info['hidden_size'] = config.get('hidden_size', 'unknown')

            # Get cache stats
            total_size = 0
            file_count = 0
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                        file_count += 1

            version_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            version_info['file_count'] = file_count
            version_info['cached_date'] = datetime.now().isoformat()

            # Store first_cached if it's missing in the registry
            model_name = os.path.basename(model_path).replace("--", "/")
            
            # Use the metadata system instead of registry for consistency
            metadata = VectorStoreManager._get_cached_model_metadata(model_name)
            if not metadata or not metadata.get('first_cached'):
                # Register this model with proper metadata
                VectorStoreManager._register_model_usage(model_name, model_path)

        except Exception as e:
            logger.debug(f"Could not extract model version info: {e}")

        return version_info
    
    @staticmethod
    def _check_model_age_and_updates(model_name: str) -> Dict[str, Any]:
        """
        Check model age and determine if updates should be checked.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            dict: Model age information and update recommendations
        """
        model_config = VectorStoreManager.get_model_config()
        registry = VectorStoreManager._load_model_registry()
        model_entry = registry.get(model_name, {})
        
        # Calculate age
        first_cached = model_entry.get('first_cached')
        if not first_cached:
            return {
                'status': 'unknown',
                'age_days': 0,
                'should_check_updates': model_config['UPDATE_CHECK_ENABLED'],
                'message': 'Model not registered, will check for updates'
            }
        
        try:
            cached_date = datetime.fromisoformat(first_cached)
            age_days = (datetime.now() - cached_date).days
            
            # Determine status using config values
            if age_days < model_config['CHECK_INTERVAL_DAYS']:
                status = 'fresh'
                should_check = False
                message = f"Model is fresh ({age_days} days old)"
            elif age_days < model_config['WARNING_AGE_DAYS']:
                status = 'good'
                should_check = VectorStoreManager._should_check_for_updates(model_name)
                message = f"Model is good ({age_days} days old)"
            elif age_days < model_config['CRITICAL_AGE_DAYS']:
                status = 'aging'
                should_check = model_config['UPDATE_CHECK_ENABLED']
                message = f"Model is aging ({age_days} days old) - checking for updates recommended"
            else:
                status = 'stale'
                should_check = model_config['UPDATE_CHECK_ENABLED']
                message = f"Model is stale ({age_days} days old) - update strongly recommended"
            
            return {
                'status': status,
                'age_days': age_days,
                'should_check_updates': should_check,
                'message': message,
                'first_cached': first_cached
            }
            
        except Exception as e:
            logger.warning(f"Error checking model age: {e}")
            return {
                'status': 'error',
                'age_days': 0,
                'should_check_updates': model_config['UPDATE_CHECK_ENABLED'],
                'message': 'Could not determine model age, will check for updates'
            }
    
    @staticmethod
    def _should_check_for_updates(model_name: str) -> bool:
        """
        Determine if we should check for updates based on last check time.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if update check should be performed
        """
        model_config = VectorStoreManager.get_model_config()
        
        if not model_config['UPDATE_CHECK_ENABLED']:
            return False
            
        registry = VectorStoreManager._load_model_registry()
        model_entry = registry.get(model_name, {})
        
        last_check = model_entry.get('last_update_check')
        if not last_check:
            return True
        
        try:
            last_check_date = datetime.fromisoformat(last_check)
            check_interval = timedelta(days=model_config['CHECK_INTERVAL_DAYS'])
            return datetime.now() - last_check_date > check_interval
        except:
            return True

    @staticmethod
    def _check_for_model_updates(model_name):
        """
        Check for model updates with robust date handling.
        
        Compares the local cache date with the HuggingFace Hub's last modified date
        to determine if updates are available.
        
        Args:
            model_name: Name of the model to check for updates
            
        Returns:
            dict: Update status information including availability and dates
        """
        try:
            # Update the registry to record this check
            registry = VectorStoreManager._load_model_registry()
            if model_name not in registry:
                registry[model_name] = {}
            registry[model_name]['last_update_check'] = datetime.now().isoformat()
            VectorStoreManager._save_model_registry(registry)
            
            # Get HuggingFace Hub info
            from huggingface_hub import model_info
            
            logger.debug(f"Fetching model info from HuggingFace Hub for {model_name}")
            model_info_obj = model_info(model_name)
            
            # Get last modified date from Hub
            hub_last_modified = None
            if hasattr(model_info_obj, 'lastModified') and model_info_obj.lastModified:
                hub_last_modified = model_info_obj.lastModified
                logger.info(f"HuggingFace Hub lastModified: {hub_last_modified}")
            else:
                logger.warning(f"No lastModified date available from HuggingFace Hub for {model_name}")
                return {
                    'has_updates': None,
                    'message': "Could not determine update status - no lastModified date from Hub",
                    'hub_date': None,
                    'cached_date': None
                }
            
            # Get cached model metadata
            cached_metadata = VectorStoreManager._get_cached_model_metadata(model_name)
            
            if not cached_metadata or not cached_metadata.get('first_cached'):
                logger.warning(f"No cached metadata or first_cached date for {model_name}")
                # Try to find and register the model if it exists
                model_path = VectorStoreManager._find_model_cache_path(model_name)
                if model_path:
                    logger.info(f"Found existing model at {model_path}, registering metadata")
                    VectorStoreManager._register_model_usage(model_name, model_path)
                    cached_metadata = VectorStoreManager._get_cached_model_metadata(model_name)
                
                if not cached_metadata or not cached_metadata.get('first_cached'):
                    return {
                        'has_updates': None,
                        'message': "Could not determine update status - no cached date information",
                        'hub_date': hub_last_modified.isoformat() if hub_last_modified else None,
                        'cached_date': None
                    }
            
            cached_date_str = cached_metadata['first_cached']
            logger.info(f"Cached model first_cached: {cached_date_str}")
            
            # Parse dates with robust error handling
            try:
                # Parse Hub date (should be datetime object from huggingface_hub)
                if isinstance(hub_last_modified, str):
                    # If it's already a string, try to parse it
                    hub_date = datetime.fromisoformat(hub_last_modified.replace('Z', '+00:00'))
                else:
                    # If it's a datetime object, use it directly
                    hub_date = hub_last_modified
                    
                # Parse cached date (should be ISO format string)
                if cached_date_str.endswith('Z'):
                    cached_date_str = cached_date_str.replace('Z', '+00:00')
                    
                cached_date = datetime.fromisoformat(cached_date_str)
                
                # Make both timezone-aware for comparison
                if hub_date.tzinfo is None:
                    hub_date = hub_date.replace(tzinfo=timezone.utc)
                if cached_date.tzinfo is None:
                    cached_date = cached_date.replace(tzinfo=timezone.utc)
                    
                # Compare dates
                has_updates = hub_date > cached_date
                
                if has_updates:
                    days_diff = (hub_date - cached_date).days
                    message = f"Model updated on Hub {days_diff} days after local cache"
                else:
                    message = "Local model is up to date"
                    
                logger.info(f"Date comparison successful: Hub={hub_date.isoformat()}, Cached={cached_date.isoformat()}, Has updates={has_updates}")
                
                return {
                    'has_updates': has_updates,
                    'message': message,
                    'hub_date': hub_date.isoformat(),
                    'cached_date': cached_date.isoformat(),
                    'days_difference': (hub_date - cached_date).days if has_updates else 0
                }
                
            except (ValueError, TypeError) as e:
                logger.error(f"Date parsing failed: {e}")
                logger.error(f"Hub date: {hub_last_modified} (type: {type(hub_last_modified)})")
                logger.error(f"Cached date: {cached_date_str} (type: {type(cached_date_str)})")
                
                return {
                    'has_updates': None,
                    'message': f"Could not compare dates due to parsing error: {str(e)}",
                    'hub_date': str(hub_last_modified) if hub_last_modified else None,
                    'cached_date': cached_date_str
                }
                
        except Exception as e:
            logger.error(f"Failed to check for model updates: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'has_updates': None,
                'message': f"Update check failed: {str(e)}",
                'hub_date': None,
                'cached_date': None
            }

    @staticmethod
    def _prompt_user_for_model_update(model_name: str, age_info: Dict, update_info: Dict = None) -> bool:
        """
        Prompt user about model updates and return whether to proceed with update.
        
        Presents a user-friendly interface for deciding whether to update models
        based on age and availability of newer versions.
        
        Args:
            model_name: Name of the model to potentially update
            age_info: Information about model age and status
            update_info: Information about available updates
            
        Returns:
            bool: True if user wants to update the model
        """
        model_config = VectorStoreManager.get_model_config()
        
        # Skip prompting in production if auto-prompts are disabled
        if config.ENV == "production" and not model_config['AUTO_UPDATE_PROMPT']:
            logger.info("Model update available but auto-prompts disabled in production")
            logger.info("Use 'model check' and 'model update' commands to manage updates")
            return False
        
        print(f"\nModel Update Notification")
        print(f"=" * 50)
        print(f"Model: {model_name}")
        print(f"Status: {age_info['message']}")
        print(f"Environment: {config.ENV.upper()}")
        
        if update_info and update_info.get('has_updates'):
            print(f"Updates: {update_info['message']}")
        elif update_info and update_info.get('has_updates') is False:
            print(f"Updates: {update_info['message']}")
        else:
            print(f"Updates: Could not check for updates")
        
        print(f"\nRecommendations:")
        if age_info['status'] == 'stale':
            print(f"CRITICAL: Model is {age_info['age_days']} days old - update strongly recommended")
        elif age_info['status'] == 'aging':
            print(f"WARNING: Model is {age_info['age_days']} days old - consider updating")
        else:
            print(f"INFO: Model age check completed")
        
        # Show environment-specific advice
        if config.ENV == "production":
            print(f"\nProduction Environment:")
            print(f"   • Conservative update policy active")
            print(f"   • Manual approval required: {model_config['REQUIRE_APPROVAL']}")
            print(f"   • Cache cleanup enabled: {model_config['CACHE_CLEANUP']}")
        
        print(f"\nOptions:")
        print(f"1. Continue with current model (default)")
        print(f"2. Clear cache and download latest model")
        print(f"3. Skip age checks for this session")
        
        try:
            choice = input(f"\nEnter your choice (1-3, default=1): ").strip()
            
            if choice == '2':
                if model_config['REQUIRE_APPROVAL'] and config.ENV == "production":
                    confirm = input(f"Production update requires confirmation. Type 'CONFIRM' to proceed: ").strip()
                    if confirm != 'CONFIRM':
                        print(f"Update cancelled.")
                        return False
                return True  # User wants to update
            elif choice == '3':
                # Set a session flag to skip checks
                VectorStoreManager._skip_age_checks = True
                logger.info("Model age checks disabled for this session")
                return False
            else:
                return False  # Continue with current model
                
        except (KeyboardInterrupt, EOFError):
            print(f"\nUsing current model...")
            return False
    
    @staticmethod
    def _clear_model_cache_for_update(model_name: str) -> bool:
        """
        Clear cached model to force fresh download.
        
        Removes all cached files for a specific model to ensure a fresh download
        when the model is next requested.
        
        Args:
            model_name: Name of the model to clear from cache
            
        Returns:
            bool: True if cache was successfully cleared
        """
        try:
            cache_dir = VectorStoreManager.setup_model_cache()
            
            # Find and remove model cache directories
            possible_cache_dirs = [
                os.path.join(cache_dir, "hub"),
                os.path.join(cache_dir, "sentence_transformers"),
                cache_dir
            ]
            
            model_variants = [
                f"models--{model_name.replace('/', '--')}",
                model_name.replace('/', '_'),
                f"sentence-transformers_{model_name.replace('/', '_')}"
            ]
            
            removed_any = False
            for cache_base in possible_cache_dirs:
                if not os.path.exists(cache_base):
                    continue
                    
                for variant in model_variants:
                    model_path = os.path.join(cache_base, variant)
                    if os.path.exists(model_path):
                        logger.info(f"Removing cached model at: {model_path}")
                        shutil.rmtree(model_path)
                        removed_any = True
            
            if removed_any:
                # Update registry to reflect model was cleared
                registry = VectorStoreManager._load_model_registry()
                if model_name in registry:
                    del registry[model_name]
                    VectorStoreManager._save_model_registry(registry)
                    
                logger.info(f"Successfully cleared cache for model: {model_name}")
                return True
            else:
                logger.warning(f"No cached files found for model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear model cache: {e}")
            return False
    
    @staticmethod
    def _find_model_cache_path(model_name: str) -> Optional[str]:
        """
        Find the actual cache path for a model.
        
        Searches through multiple possible cache locations to find where
        a model is actually stored.
        
        Args:
            model_name: Name of the model to locate
            
        Returns:
            str: Path to the model cache directory, or None if not found
        """
        cache_base = os.environ.get('HF_HOME', VectorStoreManager.setup_model_cache())
        possible_paths = [
            os.path.join(cache_base, "hub", f"models--{model_name.replace('/', '--')}"),
            os.path.join(cache_base, "sentence_transformers", f"sentence-transformers_{model_name.replace('/', '_')}"),
            os.path.join(cache_base, "sentence_transformers", f"models--{model_name.replace('/', '--')}")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path
        return None
    
    @staticmethod
    def get_production_model_report() -> str:
        """
        Generate a comprehensive production model report.
        
        Returns:
            str: Formatted report showing model status, age, and usage statistics
        """
        registry = VectorStoreManager._load_model_registry()
        
        if not registry:
            return "No models registered in production cache"
        
        report_lines = [
            "Production Model Cache Report",
            "=" * 50
        ]
        
        for model_name, data in registry.items():
            age_info = VectorStoreManager._check_model_age_and_updates(model_name)
            usage_count = data.get('usage_count', 0)
            version_info = data.get('version_info', {})
            size_mb = version_info.get('total_size_mb', 'unknown')
            
            # Status indicator
            status_indicator = {
                'fresh': '[FRESH]',
                'good': '[GOOD]',
                'aging': '[AGING]',
                'stale': '[STALE]',
                'unknown': '[UNKNOWN]'
            }.get(age_info['status'], '[UNKNOWN]')
            
            report_lines.append(
                f"{status_indicator} {model_name}: {age_info['age_days']} days old, "
                f"used {usage_count} times, {size_mb}MB"
            )
        
        return "\n".join(report_lines)
    
    @staticmethod
    def is_model_cached(model_name):
        """
        Check if the model is already cached locally with improved detection.
        
        Searches through multiple cache locations and validates that essential
        model files are present before confirming the model is cached.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if model is found in cache with required files
        """
        # Get the HF_HOME directory
        hf_home = os.environ.get('HF_HOME')
        if not hf_home:
            logger.debug("HF_HOME not set, model not cached")
            return False
        
        # Check multiple possible cache locations
        possible_cache_dirs = [
            os.path.join(hf_home, "hub"),  # New HF cache structure
            os.path.join(hf_home, "sentence_transformers"),  # Sentence transformers cache
            hf_home  # Direct cache
        ]
        
        # Look for model in various formats
        model_variants = [
            f"models--{model_name.replace('/', '--')}",  # New HF format
            model_name.replace('/', '_'),  # Old format
            model_name,  # Direct name
            f"sentence-transformers_{model_name.replace('/', '_')}",  # ST format
            f"models--{model_name.replace('/', '--')}"  # ST directory format (redundant but explicit)
        ]
        
        found_path = None
        for cache_dir in possible_cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for variant in model_variants:
                model_path = os.path.join(cache_dir, variant)
                if os.path.exists(model_path):
                    # Check for essential files in both root and snapshot directories
                    essential_files = ['config.json']
                    model_files = ['pytorch_model.bin', 'model.safetensors', 'model.onnx']
                    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
                    
                    # First check root directory
                    has_config = any(os.path.exists(os.path.join(model_path, f)) for f in essential_files)
                    has_model = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
                    has_tokenizer = any(os.path.exists(os.path.join(model_path, f)) for f in tokenizer_files)
                    
                    # If not found in root, check snapshots directory (HF cache structure)
                    if not (has_config and (has_model or has_tokenizer)):
                        snapshots_dir = os.path.join(model_path, 'snapshots')
                        if os.path.exists(snapshots_dir):
                            for snapshot in os.listdir(snapshots_dir):
                                snapshot_path = os.path.join(snapshots_dir, snapshot)
                                if os.path.isdir(snapshot_path):
                                    # Check for files directly in snapshots (they may be symlinks)
                                    snapshot_has_config = any(os.path.exists(os.path.join(snapshot_path, f)) for f in essential_files)
                                    snapshot_has_model = any(os.path.exists(os.path.join(snapshot_path, f)) for f in model_files)
                                    snapshot_has_tokenizer = any(os.path.exists(os.path.join(snapshot_path, f)) for f in tokenizer_files)
                                    
                                    has_config = has_config or snapshot_has_config
                                    has_model = has_model or snapshot_has_model
                                    has_tokenizer = has_tokenizer or snapshot_has_tokenizer
                                    
                                    if has_config and (has_model or has_tokenizer):
                                        break
                    
                    # Additional check for HF blob integrity (symlinks should resolve)
                    if has_config and (has_model or has_tokenizer):
                        # Check if this is an HF cache with symlinks and verify they resolve
                        snapshots_dir = os.path.join(model_path, 'snapshots')
                        blobs_dir = os.path.join(model_path, 'blobs')
                        if os.path.exists(snapshots_dir) and os.path.exists(blobs_dir):
                            # Check for incomplete downloads in blobs directory
                            incomplete_files = [f for f in os.listdir(blobs_dir) if f.endswith('.incomplete')]
                            if incomplete_files:
                                logger.warning(f"Model {model_name} has incomplete downloads: {incomplete_files}")
                                logger.info(f"Cleaning up corrupted model cache for {model_name}")
                                # Remove the corrupted model directory
                                try:
                                    shutil.rmtree(model_path)
                                    logger.info(f"Removed corrupted model cache at {model_path}")
                                except Exception as e:
                                    logger.error(f"Failed to remove corrupted model cache: {e}")
                                has_model = False  # Mark as not properly cached
                    
                    if has_config and (has_model or has_tokenizer):
                        logger.info(f"Model {model_name} found in cache at {model_path}")
                        found_path = model_path
                        break
                    else:
                        logger.debug(f"Model {model_name} partially cached at {model_path}")
            
            if found_path:
                break
        
        # Also check if sentence-transformers has it loaded
        if not found_path:
            try:
                import sentence_transformers
                st_cache = os.path.join(hf_home, "sentence_transformers")
                if os.path.exists(st_cache):
                    for item in os.listdir(st_cache):
                        if model_name.replace('/', '_') in item or 'bge-base' in item.lower():
                            model_dir = os.path.join(st_cache, item)
                            if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 3:
                                logger.info(f"Model {model_name} found in sentence-transformers cache at {model_dir}")
                                found_path = model_dir
                                break
            except Exception as e:
                logger.debug(f"Error checking sentence-transformers cache: {e}")
        
        if found_path:
            # Always register the model when found to ensure metadata exists
            try:
                VectorStoreManager._register_model_usage(model_name, found_path)
            except Exception as e:
                logger.warning(f"Failed to register model usage for cached model: {e}")
            return True
        else:
            logger.debug(f"Model {model_name} not found in cache")
            return False
    
    @staticmethod
    def check_vector_store_health(vector_store):
        """
        Perform comprehensive health check on vector store.
        
        Tests multiple aspects of vector store functionality including
        collection access, query capability, and document retrieval.
        
        Args:
            vector_store: The vector store instance to check
            
        Returns:
            bool: True if all health checks pass
        """
        if not vector_store:
            logger.warning("No vector store provided for health check")
            return False
            
        try:
            # Check 1: Can we get collection info?
            collection_info = False
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                try:
                    count = collection.count()
                    logger.info(f"Collection reports {count} documents")
                    if count > 0:
                        collection_info = True
                except Exception as e:
                    logger.warning(f"Failed to get collection count: {e}")
            
            # Check 2: Can we perform a test query?
            query_success = False
            try:
                test_results = vector_store.similarity_search("test query APU university", k=1)
                if test_results:
                    logger.info(f"Query test successful: retrieved {len(test_results)} documents")
                    query_success = True
            except Exception as e:
                logger.warning(f"Query test failed: {e}")
            
            # Check 3: Can we get all documents?
            get_success = False
            try:
                all_docs = vector_store.get()
                doc_count = len(all_docs.get('documents', []))
                logger.info(f"get() method reports {doc_count} documents")
                if doc_count > 0:
                    get_success = True
            except Exception as e:
                logger.warning(f"get() method failed: {e}")
            
            # Overall health assessment
            health_status = collection_info or (query_success and get_success)
            
            if health_status:
                logger.info("Vector store health check: PASSED")
            else:
                logger.warning("Vector store health check: FAILED")
                
            return health_status
            
        except Exception as e:
            logger.error(f"Error during vector store health check: {e}")
            return False
    
    @staticmethod
    def get_embedding_device():
        """
        Determine the best available device for embeddings.
        
        Checks for CUDA GPU, Apple Silicon MPS, and falls back to CPU
        in order of preference for optimal performance.
        
        Returns:
            str: Device identifier ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for embeddings")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS for embeddings")
            return 'mps'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'
    
    @staticmethod
    def create_embeddings(model_name=None):
        """
        Create the embedding model with production-grade caching and update management.
        
        This method implements a comprehensive model lifecycle management system
        including age checking, update detection, and user prompting for updates.
        
        Args:
            model_name: Name of the embedding model to create (defaults to config value)
            
        Returns:
            HuggingFaceEmbeddings: Configured embedding model instance
        """
        if model_name is None:
            model_name = config.EMBEDDING_MODEL_NAME
        
        model_config = VectorStoreManager.get_model_config()
        
        # Check if we already have cached embeddings for this model
        if (VectorStoreManager._cached_embeddings is not None and 
            VectorStoreManager._cached_embeddings_model == model_name):
            logger.info(f"Using cached embeddings for model: {model_name}")
            return VectorStoreManager._cached_embeddings
        
        # Setup model cache directory
        cache_dir = VectorStoreManager.setup_model_cache()
        
        # Production model age and update checking (only if enabled)
        if (model_config['UPDATE_CHECK_ENABLED'] and 
            not getattr(VectorStoreManager, '_skip_age_checks', False)):
            
            age_info = VectorStoreManager._check_model_age_and_updates(model_name)
            
            # Log model status with appropriate severity
            if age_info['status'] == 'fresh':
                logger.info(f"Model status: {age_info['message']}")
            elif age_info['status'] == 'good':
                logger.info(f"Model status: {age_info['message']}")
            elif age_info['status'] == 'aging':
                logger.warning(f"Model status: {age_info['message']}")
            elif age_info['status'] == 'stale':
                logger.warning(f"Model status: {age_info['message']}")
            
            # Check for updates if needed
            update_info = None
            if age_info['should_check_updates']:
                logger.info("Checking for model updates...")
                update_info = VectorStoreManager._check_for_model_updates(model_name)
                
                if update_info.get('has_updates'):
                    logger.info(f"Update available: {update_info['message']}")
                elif update_info.get('has_updates') is False:
                    logger.info(f"Update status: {update_info['message']}")
                else:
                    logger.warning(f"Update check result: {update_info['message']}")
            
            # Prompt user for action if model is aging/stale or has updates
            should_update = False
            if (age_info['status'] in ['aging', 'stale'] or 
                (update_info and update_info.get('has_updates'))):
                
                should_update = VectorStoreManager._prompt_user_for_model_update(
                    model_name, age_info, update_info)
            
            # Clear cache if user requested update
            if should_update:
                logger.info("Clearing model cache for fresh download...")
                if VectorStoreManager._clear_model_cache_for_update(model_name):
                    logger.info("Model cache cleared successfully")
                else:
                    logger.warning("Could not clear model cache completely")
        
        # Check if model is already cached locally
        model_cached = VectorStoreManager.is_model_cached(model_name)
        
        if model_cached:
            logger.info(f"Model {model_name} found in local cache - skipping download")
        else:
            logger.info(f"Model {model_name} not cached - will download and cache for future use")
        
        device = VectorStoreManager.get_embedding_device()
        
        try:
            # Create embeddings with optimized parameters for faster loading
            logger.info(f"Creating embeddings for model: {model_name}")
            
            # Configure for faster loading - compatible with sentence-transformers
            model_kwargs = {
                'device': device,
                'trust_remote_code': False,  # Security best practice
            }
            
            # Configure encoding parameters for optimal performance
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 32,  # Optimize batch size for performance
            }
            
            # Add fp16 optimization for bge-large models  
            if 'bge-large' in model_name:
                encode_kwargs['use_fp16'] = True
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Register model usage for production tracking after successful creation
            model_path = VectorStoreManager._find_model_cache_path(model_name)
            VectorStoreManager._register_model_usage(model_name, model_path)
            
            # Cache the embeddings for future use
            VectorStoreManager._cached_embeddings = embeddings
            VectorStoreManager._cached_embeddings_model = model_name
            logger.info(f"Successfully cached embeddings for model: {model_name}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    @staticmethod
    def save_embeddings_backup(vector_store, filepath=None):
        """
        Save a backup of embeddings and metadata.
        
        Creates a pickle file containing all vector store data for recovery purposes.
        
        Args:
            vector_store: The vector store to backup
            filepath: Optional custom backup file path
            
        Returns:
            bool: True if backup was successful
        """
        if not vector_store:
            return False
            
        if filepath is None:
            filepath = os.path.join(os.path.dirname(config.PERSIST_PATH), "embeddings_backup.pkl")
            
        try:
            data = vector_store.get()
            if not data or not data.get('ids') or len(data.get('ids', [])) == 0:
                logger.warning("No data to backup - vector store appears empty")
                return False
                
            # Add timestamp for versioning
            data['backup_time'] = datetime.now().isoformat()
            data['doc_count'] = len(data.get('ids', []))
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved embeddings backup with {data['doc_count']} documents to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def load_embeddings_backup(embeddings, filepath=None, collection_name="apu_kb_collection"):
        """
        Load embeddings from backup if main store is empty.
        
        Restores vector store data from a previously created backup file.
        
        Args:
            embeddings: Embedding function to use for the restored store
            filepath: Optional custom backup file path
            collection_name: Name for the restored collection
            
        Returns:
            Chroma: Restored vector store instance or None if failed
        """
        if filepath is None:
            filepath = os.path.join(os.path.dirname(config.PERSIST_PATH), "embeddings_backup.pkl")
            
        if not os.path.exists(filepath):
            logger.warning(f"No embeddings backup found at {filepath}")
            return None
            
        try:
            # Load backup data
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Validate backup data
            if not data or 'ids' not in data or len(data['ids']) == 0:
                logger.warning("Backup file exists but contains no valid data")
                return None
                
            logger.info(f"Loaded embeddings backup with {len(data['ids'])} documents from {filepath}")
            
            # Create a new in-memory Chroma instance with this data
            from langchain_chroma import Chroma
            import chromadb
            from chromadb.config import Settings
            
            # Create a memory client
            client = chromadb.Client(Settings(anonymized_telemetry=False))
            
            # Create a new collection
            collection = client.create_collection(name=collection_name)
            
            # Add the data in batches to avoid memory issues
            batch_size = 100
            total_items = len(data['ids'])
            
            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)
                
                # Prepare batch data
                batch_ids = data['ids'][i:end_idx]
                batch_embeddings = data['embeddings'][i:end_idx] if data.get('embeddings') else None
                batch_metadatas = data['metadatas'][i:end_idx] if data.get('metadatas') else None
                batch_documents = data['documents'][i:end_idx] if data.get('documents') else None
                
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                logger.info(f"Restored batch {i//batch_size + 1}/{(total_items+batch_size-1)//batch_size}: {len(batch_ids)} documents")
            
            # Create a LangChain wrapper around this collection
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            logger.info(f"Successfully restored vector store from backup with {total_items} documents")
            return vector_store
                
        except Exception as e:
            logger.error(f"Failed to load embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    @staticmethod
    def fix_chromadb_collection(vector_store):
        """
        Fix for ChromaDB collections that appear empty despite existing in the database.
        
        This is a workaround for a known issue with ChromaDB persistence where
        collections may not be properly loaded on startup.
        
        Args:
            vector_store: The vector store to attempt to fix
            
        Returns:
            bool: True if the fix was successful
        """
        if not vector_store:
            return False
            
        try:
            # Check if collection exists but reports 0 documents
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                count = vector_store._collection.count()
                if count == 0:
                    logger.warning("Collection exists but reports 0 documents - attempting fix")
                    
                    # Try to force a collection reload through direct access
                    if hasattr(vector_store._collection, '_client'):
                        client = vector_store._collection._client
                        collection_name = vector_store._collection.name
                        
                        # Try a direct query to wake up the collection
                        try:
                            logger.info("Attempting direct ChromaDB query to fix collection")
                            from chromadb.api.types import QueryResult
                            results = client.query(
                                collection_name=collection_name,
                                query_texts=["test query for collection fix"],
                                n_results=1,
                            )
                            logger.info(f"Direct query results: {results}")
                            
                            # Check collection again
                            count_after = vector_store._collection.count()
                            logger.info(f"Collection count after fix attempt: {count_after}")
                            
                            return count_after > 0
                        except Exception as e:
                            logger.error(f"Error during collection fix attempt: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error in fix_chromadb_collection: {e}")
            return False
        
    @staticmethod
    def sanitize_metadata(documents: List[Document]) -> List[Document]:
        """
        Sanitize document metadata to ensure compatibility with ChromaDB.
        
        ChromaDB has specific requirements for metadata format. This method
        converts lists to JSON strings and removes None values.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List[Document]: Documents with sanitized metadata
        """
        sanitized_docs = []
        
        for doc in documents:
            # Create a copy of the metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # Process each metadata field
            for key, value in list(metadata.items()):
                # Convert lists to strings
                if isinstance(value, list):
                    if value:  # If list is not empty
                        metadata[key] = json.dumps(value)
                    else:
                        # Remove empty lists
                        metadata.pop(key)
                # Remove None values
                elif value is None:
                    metadata.pop(key)
                # Keep other primitive types as is
            
            # Create a new document with sanitized metadata
            sanitized_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            sanitized_docs.append(sanitized_doc)
        
        return sanitized_docs
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """
        Reset the ChromaDB environment - handles file system operations safely.
        
        Completely removes and recreates the vector store directory with
        proper error handling and permission management.
        
        Args:
            persist_directory: Path to the vector store directory
            
        Returns:
            bool: True if reset was successful
        """
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Remove existing directory if it exists
        if os.path.exists(persist_directory):
            try:
                # Make directory writable first (for Windows compatibility)
                if sys.platform == 'win32':
                    for root, dirs, files in os.walk(persist_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                
                # Try Python's built-in directory removal first
                shutil.rmtree(persist_directory)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\"")
                    else:
                        os.system(f"rm -rf \"{persist_directory}\"")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Create fresh directory structure
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set appropriate permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            
            # Create .chroma subdirectory for ChromaDB to recognize
            os.makedirs(os.path.join(persist_directory, ".chroma"), exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @classmethod
    def reset_chroma_db_with_permissions(cls, persist_directory):
        """
        Reset ChromaDB with permission handling and complete client cleanup.
        
        Enhanced version of reset_chroma_db that includes more aggressive
        cleanup procedures for stubborn file system issues.
        
        Args:
            persist_directory: Path to the vector store directory
            
        Returns:
            bool: True if reset was successful
        """
        logger.info(f"Resetting vector store at {persist_directory} with permission fixes")
        
        # Step 1: Force close all existing ChromaDB clients
        try:
            from .chromadb_manager import ChromaDBManager
            ChromaDBManager.force_cleanup()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
        
        # Step 2: Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(1.5)  # Give more time for file handles to close
        
        # Step 3: Remove existing directory completely
        if os.path.exists(persist_directory):
            try:
                # Fix permissions recursively before removal
                cls._fix_directory_permissions(persist_directory)
                
                # Force removal with multiple attempts
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(persist_directory)
                        logger.info("Successfully removed existing vector store directory")
                        break
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(1.0)
                        
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands with forced removal
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\" 2>nul")
                    else:
                        # Use chmod for stubborn files
                        os.system(f"chmod -R 777 \"{persist_directory}\" 2>/dev/null || true")
                        os.system(f"rm -rf \"{persist_directory}\"")
                        
                    # Verify removal
                    if os.path.exists(persist_directory):
                        logger.error("Failed to completely remove directory")
                        return False
                        
                    logger.info("Successfully removed directory using system commands")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Step 4: Create fresh directory structure with proper permissions
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set comprehensive permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            else:
                # Windows: ensure full control
                import stat
                os.chmod(persist_directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
            # Don't pre-create .chroma subdirectory - let ChromaDB do it
            logger.info("Successfully created fresh vector store directory with proper permissions")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @staticmethod
    def _fix_directory_permissions(directory):
        """
        Fix permissions recursively for directory removal.
        
        Ensures all files and directories have appropriate permissions
        for deletion operations across different operating systems.
        
        Args:
            directory: Directory to fix permissions for
        """
        try:
            if sys.platform == 'win32':
                # Windows permission fix
                for root, dirs, files in os.walk(directory):
                    for dir_name in dirs:
                        try:
                            dir_path = os.path.join(root, dir_name)
                            os.chmod(dir_path, 0o777)
                        except Exception as e:
                            logger.debug(f"Could not change permissions for directory {dir_name}: {e}")
                    for file_name in files:
                        try:
                            file_path = os.path.join(root, file_name)
                            os.chmod(file_path, 0o777)
                        except Exception as e:
                            logger.debug(f"Could not change permissions for file {file_name}: {e}")
            else:
                # Unix-like permission fix
                for root, dirs, files in os.walk(directory):
                    for dir_name in dirs:
                        try:
                            dir_path = os.path.join(root, dir_name)
                            os.chmod(dir_path, 0o755)
                        except Exception as e:
                            logger.debug(f"Could not change permissions for directory {dir_name}: {e}")
                    for file_name in files:
                        try:
                            file_path = os.path.join(root, file_name)
                            os.chmod(file_path, 0o644)
                        except Exception as e:
                            logger.debug(f"Could not change permissions for file {file_name}: {e}")
        except Exception as e:
            logger.warning(f"Error fixing directory permissions: {e}")
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """
        Get existing vector store or create a new one with debugging.
        
        Central method for vector store management that handles both
        loading existing stores and creating new ones from document chunks.
        
        Args:
            chunks: Document chunks to index (for new stores)
            embeddings: Embedding function to use
            persist_directory: Directory for persistent storage
            
        Returns:
            Chroma: Vector store instance or None if failed
        """
        if persist_directory is None:
            persist_directory = config.PERSIST_PATH
            
        collection_name = "apu_kb_collection"
        logger.info(f"Using collection name: {collection_name}")
        
        # Check vector store directory
        cls._check_vector_store_directory(persist_directory)
        
        # Reset directory if needed
        if config.FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            cls.reset_chroma_db(persist_directory)
        
        # Get ChromaDB client
        client = ChromaDBManager.get_client(persist_directory)
        
        # Load existing vector store or create new one
        if chunks is None and os.path.exists(persist_directory) and os.listdir(persist_directory):
            return cls._load_existing_vector_store(client, collection_name, embeddings)
        elif chunks:
            return cls._create_new_vector_store(client, collection_name, chunks, embeddings)
        else:
            logger.error("Cannot create or load vector store - no chunks provided and no existing store")
            return None

    @classmethod
    def _check_vector_store_directory(cls, directory):
        """
        Check vector store directory and log information.
        
        Performs diagnostic checks on the vector store directory to
        provide useful information for troubleshooting.
        
        Args:
            directory: Directory to check
        """
        if os.path.exists(directory):
            logger.info(f"Vector store directory exists with contents: {os.listdir(directory)}")
            sqlite_path = os.path.join(directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                file_size = os.path.getsize(sqlite_path)
                logger.info(f"chroma.sqlite3 exists with size: {file_size} bytes")

    @classmethod
    def _load_existing_vector_store(cls, client, collection_name, embeddings):
        """
        Load existing vector store.
        
        Attempts to load an existing ChromaDB collection and verify
        that it contains documents.
        
        Args:
            client: ChromaDB client instance
            collection_name: Name of the collection to load
            embeddings: Embedding function to use
            
        Returns:
            Chroma: Vector store instance or None if failed
        """
        try:
            logger.info("Loading existing vector store")
            
            # Get collection and langchain wrapper
            collection, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, embedding_function=embeddings)
            
            # Verify collection has documents
            try:
                count = collection.count()
                logger.info(f"Vector store reports {count} documents after loading")
                
                if count > 0:
                    logger.info(f"Successfully loaded vector store with {count} documents")
                    return vector_store
                else:
                    logger.warning("Vector store exists but is empty - will try backup")
                    return None
            except Exception as e:
                logger.error(f"Error verifying collection: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def _create_new_vector_store(cls, client, collection_name, chunks, embeddings):
        """
        Create new vector store with chunks.
        
        Creates a fresh ChromaDB collection and populates it with
        the provided document chunks.
        
        Args:
            client: ChromaDB client instance
            collection_name: Name for the new collection
            chunks: Document chunks to index
            embeddings: Embedding function to use
            
        Returns:
            Chroma: Vector store instance or None if failed
        """
        try:
            logger.info(f"Creating new vector store with {len(chunks)} chunks")
            
            # Sanitize metadata before creating vector store
            sanitized_chunks = cls.sanitize_metadata(chunks)
            
            # Get or create collection with metadata
            metadata = {
                "document_count": len(sanitized_chunks),
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP
            }
            
            _, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, metadata, embeddings)
            
            # Add documents to the vector store
            vector_store.add_documents(documents=sanitized_chunks)
            
            # Explicitly persist if supported
            if hasattr(vector_store, 'persist'):
                vector_store.persist()
                logger.info(f"Vector store persisted successfully with {len(sanitized_chunks)} chunks")
            
            return vector_store
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """
        Print statistics about indexed documents.
        
        Provides comprehensive information about the documents stored
        in the vector database including counts, types, and timestamps.
        
        Args:
            vector_store: The vector store to analyze
        """
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            # Initialize counters
            doc_counts = {}
            apu_kb_count = 0
            faq_count = 0
            
            # Access collection directly first
            collection = None
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                logger.info("Accessing vector store statistics via direct collection")
                count = collection.count()
                logger.info(f"Collection reports {count} documents")
            
            # Get all documents
            all_docs = vector_store.get()
            documents = all_docs.get('documents', [])
            all_metadata = all_docs.get('metadatas', [])
            
            doc_count = len(documents)
            logger.info(f"Vector store get() method reports {doc_count} documents")
            
            if doc_count == 0:
                # Try a test query to see if documents can be retrieved
                try:
                    test_results = vector_store.similarity_search("test query", k=1)
                    if test_results:
                        logger.info(f"Found {len(test_results)} documents via search - data exists but get() not working")
                except Exception as e:
                    logger.error(f"Error during test search: {e}")
                    
            if doc_count == 0 and (collection is None or collection.count() == 0):
                logger.warning("Vector store appears to be empty")
                return
            
            # Count documents by filename
            for metadata in all_metadata:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
                
                # Count APU KB specific pages
                if metadata.get('content_type') == 'apu_kb_page':
                    apu_kb_count += 1
                    if metadata.get('is_faq', False):
                        faq_count += 1
            
            total_chunks = len(documents)
            unique_files = len(doc_counts)
            
            logger.info(f"Vector store contains {total_chunks} chunks from {unique_files} files")
            
            # Print file statistics
            print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
            for filename, count in sorted(doc_counts.items()):
                print(f"  - {filename}: {count} chunks")
            
            # Print APU KB specific statistics
            if apu_kb_count > 0:
                print(f"\nAPU Knowledge Base: {apu_kb_count} pages, including {faq_count} FAQs")
                
            # Print only the most recently added document if timestamp is available
            recent_docs = []
            for i, metadata in enumerate(all_metadata):
                if metadata and 'timestamp' in metadata:
                    recent_docs.append((metadata['timestamp'], metadata.get('filename', 'Unknown')))
            
            if recent_docs:
                recent_docs.sort(reverse=True)
                # Get only the most recent document
                timestamp, filename = recent_docs[0]
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nMost recently added document: {filename} (added: {date_str}).")
                    
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            import traceback
            logger.error(traceback.format_exc()) 
            print("Error retrieving document statistics.")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name):
        """
        Verify if a specific document is properly indexed.
        
        Performs a search query to confirm that a document has been
        successfully indexed and is retrievable from the vector store.
        
        Args:
            vector_store: The vector store to search
            doc_name: Name of the document to verify
            
        Returns:
            bool: True if document is found in the index
        """
        if not vector_store:
            return False
            
        try:
            # Search for the document name in the vector store
            results = vector_store.similarity_search(f"information from {doc_name}", k=3)
            
            # Check if any results match this filename
            for doc in results:
                filename = doc.metadata.get('filename', '')
                if doc_name.lower() in filename.lower():
                    logger.info(f"Document '{doc_name}' is indexed in the vector store")
                    return True
                    
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False