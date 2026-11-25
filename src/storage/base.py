"""Abstract storage protocol so local disk can later be swapped for S3/MinIO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for saving and loading document artifacts and structured outputs."""

    def save_bytes(self, key: str, data: bytes) -> Path:
        """Persist raw bytes under key and return its resolved location."""
        ...

    def load_bytes(self, key: str) -> bytes:
        """Read raw bytes from key."""
        ...

    def save_text(self, key: str, content: str, encoding: str = "utf-8") -> Path:
        """Persist UTF-8 text under key."""
        ...

    def save_json(self, key: str, payload: Any) -> Path:
        """Serialize payload as UTF-8 JSON under key."""
        ...

    def load_json(self, key: str) -> Any:
        """Load JSON payload from key."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if key exists in storage."""
        ...

    def resolve_path(self, key: str) -> Path:
        """Resolve key to a safe path inside the storage root."""
        ...
