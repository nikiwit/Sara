"""
Data models for vector store management.

This module defines structured data models using dataclasses and Pydantic
for type safety, validation, and clear data contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Optional import for enhanced validation
try:
    from pydantic import BaseModel, Field, validator
    HAS_PYDANTIC = True
except ImportError:
    # Fallback base class if pydantic not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        class Config:
            pass
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    HAS_PYDANTIC = False


class ModelStatus(Enum):
    """Enumeration of model status states."""
    FRESH = "fresh"
    GOOD = "good"
    AGING = "aging"
    STALE = "stale"
    UNKNOWN = "unknown"
    ERROR = "error"


class UpdateStatus(Enum):
    """Enumeration of model update status."""
    UP_TO_DATE = "up_to_date"
    UPDATES_AVAILABLE = "updates_available"
    CHECK_FAILED = "check_failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model management operations."""
    check_interval_days: int = 7
    warning_age_days: int = 30
    critical_age_days: int = 60
    auto_update_prompt: bool = True
    update_check_enabled: bool = True
    require_approval: bool = True
    cache_cleanup: bool = True
    backup_enabled: bool = True
    max_backups: int = 5
    notification_email: Optional[str] = None


@dataclass
class ModelMetadata:
    """Metadata for cached models with proper typing."""
    model_name: str
    model_path: Optional[Path] = None
    first_cached: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    version_info: dict[str, Any] = field(default_factory=dict)

    def increment_usage(self) -> None:
        """Increment usage count and update last accessed time."""
        self.usage_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path) if self.model_path else None,
            'first_cached': self.first_cached.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'usage_count': self.usage_count,
            'version_info': self.version_info
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create instance from dictionary."""
        return cls(
            model_name=data['model_name'],
            model_path=Path(data['model_path']) if data.get('model_path') else None,
            first_cached=datetime.fromisoformat(data.get('first_cached', datetime.now(timezone.utc).isoformat())),
            last_accessed=datetime.fromisoformat(data.get('last_accessed', datetime.now(timezone.utc).isoformat())),
            usage_count=data.get('usage_count', 0),
            version_info=data.get('version_info', {})
        )


class ModelAgeInfo(BaseModel):
    """Information about model age and status."""
    status: ModelStatus
    age_days: int = Field(ge=0, description="Age of model in days")
    should_check_updates: bool
    message: str
    first_cached: Optional[datetime] = None

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UpdateInfo(BaseModel):
    """Information about model updates."""
    has_updates: Optional[bool] = Field(None, description="True if updates available, None if unknown")
    message: str
    hub_date: Optional[datetime] = None
    cached_date: Optional[datetime] = None
    days_difference: int = Field(default=0, ge=0)

    @validator('hub_date', 'cached_date', pre=True)
    def parse_datetime(cls, v):
        """Parse datetime from various string formats."""
        if isinstance(v, str):
            # Handle timezone variations
            if v.endswith('Z'):
                v = v.replace('Z', '+00:00')
            return datetime.fromisoformat(v)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class VectorStoreStats:
    """Statistics about vector store contents."""
    total_chunks: int
    unique_files: int
    document_counts: dict[str, int]
    apu_kb_count: int = 0
    faq_count: int = 0
    most_recent_doc: Optional[tuple[str, datetime]] = None

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Knowledge base contains {self.unique_files} documents ({self.total_chunks} total chunks):"
        ]
        
        for filename, count in sorted(self.document_counts.items()):
            lines.append(f"  - {filename}: {count} chunks")
        
        if self.apu_kb_count > 0:
            lines.append(f"\nAPU Knowledge Base: {self.apu_kb_count} pages, including {self.faq_count} FAQs")
        
        if self.most_recent_doc:
            filename, timestamp = self.most_recent_doc
            date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            lines.append(f"\nMost recently added document: {filename} (added: {date_str}).")
        
        return "\n".join(lines)


class HealthCheckResult(BaseModel):
    """Result of vector store health check."""
    is_healthy: bool
    collection_accessible: bool = False
    query_successful: bool = False
    documents_retrievable: bool = False
    document_count: int = Field(default=0, ge=0)
    issues: list[str] = Field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """Add an issue to the list."""
        self.issues.append(issue)
        self.is_healthy = False


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding model creation."""
    model_name: str
    device: str = "cpu"
    trust_remote_code: bool = False
    normalize_embeddings: bool = True
    batch_size: int = 32
    use_fp16: bool = False
    
    @classmethod
    def for_model(cls, model_name: str, device: str = "cpu") -> EmbeddingConfig:
        """Create config for specific model."""
        use_fp16 = 'bge-large' in model_name and device != 'cpu'
        
        return cls(
            model_name=model_name,
            device=device,
            trust_remote_code=False,
            normalize_embeddings=True,
            batch_size=32,
            use_fp16=use_fp16
        )


class VectorStoreException(Exception):
    """Base exception for vector store operations."""
    pass


class ModelCacheException(VectorStoreException):
    """Exception for model caching operations."""
    pass


class HealthCheckException(VectorStoreException):
    """Exception for health check operations."""
    pass


class UpdateCheckException(VectorStoreException):
    """Exception for update check operations."""
    pass