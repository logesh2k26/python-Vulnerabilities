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
