"""API Key authentication for the vulnerability detector."""
import os
import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Header scheme
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_header():
    """Return the header scheme (for OpenAPI docs)."""
    return _api_key_header


async def verify_api_key(
    api_key: str = Security(_api_key_header),
) -> str:
    """Validate the API key from the X-API-Key header.

    The expected key is read from the ``API_SECRET_KEY`` environment variable
    (also loadable via ``pydantic-settings`` in ``config.py``).
    """
    expected = os.environ.get("API_SECRET_KEY", "")

    if not expected:
        # If no key is configured, allow all requests (dev mode).
        # In production API_SECRET_KEY MUST be set.
        return "dev-mode"

    if not api_key:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != expected:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
