"""Application configuration settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    API_TITLE: str = "Python Vulnerability Detector"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False  # Must be explicitly enabled via env

    # Security
    API_SECRET_KEY: str = ""  # Set in .env for production
    OPENROUTER_API_KEY: str = ""  # Set in .env for production

    # CORS Settings — allow all for production (or specify exact domains)
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST"]
    CORS_ALLOW_HEADERS: List[str] = ["Content-Type", "Authorization", "X-API-Key"]

    # Model Settings
    MODEL_PATH: Path = Path(__file__).parent.parent / "pretrained" / "vulnerability_gnn.pt"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    NUM_CLASSES: int = 12  # 11 vuln types + safe

    NUM_ATTENTION_HEADS: int = 4
    NUM_GNN_LAYERS: int = 3
    DROPOUT: float = 0.3

    # Device Settings
    USE_GPU: bool = True

    # Analysis Settings
    MAX_FILE_SIZE_MB: int = 10
    MAX_BATCH_SIZE: int = 50
    CONFIDENCE_THRESHOLD: float = 0.5

    # WebSocket
    MAX_WS_CONNECTIONS: int = 100

    # Rate Limiting (string format for slowapi)
    RATE_LIMIT_ANALYZE: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_TRAIN: str = "2/hour"

    # Vulnerability Types
    VULNERABILITY_TYPES: List[str] = [
        "safe",
        "eval_exec",
        "command_injection",
        "unsafe_deserialization",
        "hardcoded_secrets",
        "sql_injection",
        "path_traversal",
        "ssrf",
        "insecure_cryptography",
        "xxe",
        "redos",
        "xss"
    ]

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()

# Determine device
import torch
DEVICE = torch.device(
    "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
)
