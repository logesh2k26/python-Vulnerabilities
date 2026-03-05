"""Pydantic schemas for API responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class VulnerabilityItem(BaseModel):
    type: str
    confidence: float
    severity: str
    description: str
    remediation: str
    affected_lines: List[int]
    code_snippet: str
    status: str = "unsafe"  # unsafe, mitigated, safe
    mitigations: List[Dict] = Field(default_factory=list)
    taint_path: Optional[Dict] = None
    metadata: Dict = Field(default_factory=dict)


class HighlightedLine(BaseModel):
    line: int
    score: float
    content: str


class AnalysisStats(BaseModel):
    total_nodes: int
    total_edges: int
    lines_of_code: int


class AnalysisResult(BaseModel):
    filename: str
    is_vulnerable: bool
    overall_confidence: float
    label: str
    vulnerabilities: List[VulnerabilityItem]
    ml_predictions: Dict[str, float]
    highlighted_lines: List[HighlightedLine]
    stats: AnalysisStats
    error: Optional[str] = None


class FileInput(BaseModel):
    filename: str
    content: str


class BatchAnalysisRequest(BaseModel):
    files: List[FileInput]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class CategoryInfo(BaseModel):
    count: int
    valid: int = 0
    invalid: int = 0


class DatasetStatus(BaseModel):
    status: str  # OK, WARNINGS, ERROR
    total_files: int
    valid_files: int = 0
    invalid_files: int = 0
    duplicate_files: int = 0
    categories: Dict[str, CategoryInfo] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class TrainingRequest(BaseModel):
    epochs: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, gt=0, lt=1)


class TrainingResult(BaseModel):
    status: str  # success, error
    epochs_completed: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    final_f1: float = 0.0
    model_path: str = ""
    message: str = ""
