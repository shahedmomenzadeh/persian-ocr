"""Local filesystem implementation of StorageBackend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalStorageBackend:
    """Stores artifacts under a local base directory with path-traversal protection."""

    def __init__(self, base_dir: Path | str = Path("./data")) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, key: str) -> Path:
        """Resolve relative key inside base_dir, rejecting traversal outside it."""
        clean_key = key.lstrip("/\\")
        target = (self.base_dir / clean_key).resolve()
        if target != self.base_dir and self.base_dir not in target.parents:
            raise ValueError(f"Unsafe storage key escapes base_dir: {key!r}")
        return target

    def save_bytes(self, key: str, data: bytes) -> Path:
        target = self.resolve_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def load_bytes(self, key: str) -> bytes:
        return self.resolve_path(key).read_bytes()

    def save_text(self, key: str, content: str, encoding: str = "utf-8") -> Path:
        target = self.resolve_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return target

    def save_json(self, key: str, payload: Any) -> Path:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.save_text(key, serialized, encoding="utf-8")

    def load_json(self, key: str) -> Any:
        text = self.resolve_path(key).read_text(encoding="utf-8")
        return json.loads(text)

    def exists(self, key: str) -> bool:
        return self.resolve_path(key).exists()
