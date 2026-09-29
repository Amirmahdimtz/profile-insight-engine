from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from src.infrastructure.di.inject import inject


class ImageStorageError(RuntimeError):
    """Raised when temporary canonical-image storage cannot be safely managed."""


@inject
class LocalImageStorage:
    """Concrete local temporary storage for canonical images only; raw uploads are never persisted."""

    def __init__(self):
        self._root = Path(tempfile.gettempdir()) / "profile_insight_engine"

    async def create_scope_async(self, analysis_id: str) -> str:
        return await asyncio.to_thread(self._create_scope, analysis_id)

    async def write_canonical_async(
        self, scope_reference: str, content_hash: str, content: bytes
    ) -> str:
        return await asyncio.to_thread(
            self._write_canonical, scope_reference, content_hash, content
        )

    async def cleanup_scope_async(self, scope_reference: str) -> None:
        await asyncio.to_thread(self._cleanup_scope, scope_reference)

    def _create_scope(self, analysis_id: str) -> str:
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        analysis_token = hashlib.sha256(analysis_id.encode("utf-8")).hexdigest()[:16]
        path = Path(tempfile.mkdtemp(prefix=f"{analysis_token}-", dir=self._root))
        try:
            path.chmod(0o700)
        except OSError as exc:
            shutil.rmtree(path, ignore_errors=True)
            raise ImageStorageError("unable to secure temporary image storage scope") from exc
        return str(path)

    def _owned_scope(self, scope_reference: str) -> Path:
        scope = Path(scope_reference).resolve()
        root = self._root.resolve()
        if scope.parent != root or not scope.name:
            raise ImageStorageError("invalid temporary image storage scope")
        return scope

    def _write_canonical(
        self, scope_reference: str, content_hash: str, content: bytes
    ) -> str:
        if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
            raise ImageStorageError("invalid canonical content hash")
        scope = self._owned_scope(scope_reference)
        if not scope.is_dir():
            raise ImageStorageError("temporary image storage scope does not exist")
        path = scope / f"{content_hash}.png"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            return str(path)
        except OSError as exc:
            raise ImageStorageError("unable to create canonical image file") from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            path.unlink(missing_ok=True)
            raise ImageStorageError("unable to write canonical image file") from exc
        return str(path)

    def _cleanup_scope(self, scope_reference: str) -> None:
        scope = self._owned_scope(scope_reference)
        if scope.exists():
            shutil.rmtree(scope)
