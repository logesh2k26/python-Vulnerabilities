"""Input validation and sanitization helpers."""
import os
import re
import logging
from pathlib import Path
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_CODE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_FILES_PER_REQUEST = 20
MAX_FILENAME_LENGTH = 255

# Characters allowed in a sanitized filename (alphanumeric, underscore, hyphen, dot)
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def sanitize_filename(raw_name: str) -> str:
    """Strip path components and dangerous characters from a filename.

    Returns a safe basename suitable for writing to disk.
    Raises HTTPException if the result is empty or invalid.
    """
    # 1. Take only the base name (defeats ../ traversal)
    base = os.path.basename(raw_name)

    # 2. Remove any remaining unsafe characters
    safe = _SAFE_FILENAME_RE.sub("_", base)

    # 3. Strip leading dots (hidden files / traversal remnants)
    safe = safe.lstrip(".")

    # 4. Enforce length
    if not safe or len(safe) > MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return safe


def validate_resolved_path(dest: Path, allowed_root: Path) -> None:
    """Ensure *dest* resolves to a child of *allowed_root*.

    Raises HTTPException on traversal attempt.
    """
    try:
        resolved = dest.resolve()
        allowed = allowed_root.resolve()
        if not str(resolved).startswith(str(allowed)):
            logger.warning("Path traversal blocked: %s", dest)
            raise HTTPException(status_code=400, detail="Path traversal blocked")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid file path")


def validate_file_size(content: bytes, filename: str = "") -> None:
    """Reject files bigger than MAX_FILE_SIZE_BYTES."""
    if len(content) > MAX_FILE_SIZE_BYTES:
        logger.warning("File too large: %s (%d bytes)", filename, len(content))
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)",
        )


def validate_code_content(content: str) -> None:
    """Reject code submissions bigger than MAX_CODE_SIZE_BYTES."""
    if len(content.encode("utf-8", errors="replace")) > MAX_CODE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Code content too large (max {MAX_CODE_SIZE_BYTES // (1024*1024)} MB)",
        )


def validate_file_count(count: int) -> None:
    """Reject requests with too many files."""
    if count > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files (max {MAX_FILES_PER_REQUEST})",
        )


def validate_python_extension(filename: str) -> bool:
    """Return True if filename ends with .py."""
    return filename.lower().endswith(".py")
