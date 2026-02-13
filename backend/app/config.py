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
    DEBUG: bool = True
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    
    # Model Settings
    MODEL_PATH: Path = Path(__file__).parent.parent / "pretrained" / "vulnerability_gnn.pt"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    NUM_CLASSES: int = 7  # 6 vuln types + safe
    NUM_ATTENTION_HEADS: int = 4
    NUM_GNN_LAYERS: int = 3
    DROPOUT: float = 0.3
    
    # Device Settings
    USE_GPU: bool = True
    
    # Analysis Settings
    MAX_FILE_SIZE_MB: int = 10
    MAX_BATCH_SIZE: int = 50
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Vulnerability Types
    VULNERABILITY_TYPES: List[str] = [
        "safe",
        "eval_exec",
        "command_injection",
        "unsafe_deserialization",
        "hardcoded_secrets",
        "sql_injection",
        "path_traversal"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Determine device
import torch
DEVICE = torch.device(
    "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
)
