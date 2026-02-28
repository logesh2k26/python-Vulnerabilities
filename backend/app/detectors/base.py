"""Base detector class for vulnerability detection."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from app.core.ast_parser import ASTNode


@dataclass
class DetectionResult:
    """Result of a vulnerability detection."""
    vulnerability_type: str
    confidence: float
    affected_lines: List[int]
    affected_nodes: List[int]
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    remediation: str
    code_snippet: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "vulnerability_type": self.vulnerability_type,
            "confidence": self.confidence,
            "affected_lines": self.affected_lines,
            "affected_nodes": self.affected_nodes,
            "description": self.description,
            "severity": self.severity,
            "remediation": self.remediation,
            "code_snippet": self.code_snippet,
            "metadata": self.metadata
        }


class BaseDetector(ABC):
    """Abstract base class for vulnerability detectors."""
    
    name: str = "base"
    description: str = "Base detector"
    
    def __init__(self):
        self.source_lines: List[str] = []
    
    def set_source(self, source_code: str):
        self.source_lines = source_code.split('\n')
    
    def get_line_content(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""
    
    def get_code_snippet(self, start: int, end: int) -> str:
        lines = []
        for i in range(max(1, start), min(len(self.source_lines) + 1, end + 1)):
            lines.append(f"{i}: {self.source_lines[i - 1]}")
        return '\n'.join(lines)
    
    @abstractmethod
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        """Run detection on AST nodes."""
        pass
    
    def _find_data_sources(self, nodes: List[ASTNode], target_node: ASTNode) -> List[ASTNode]:
        """Find potential untrusted data sources connected to target."""
        sources = []
        input_funcs = {'input', 'request', 'get', 'post', 'recv', 'read'}
        input_vars = {'request', 'data', 'params', 'query', 'user_input', 'args'}
        
        # Simple proximity check: only look for inputs BEFORE the target call
        # In a real engine, we'd use the DFG, but this heuristic reduces FP
        for node in nodes:
            if node.lineno > target_node.lineno:
                continue
                
            if node.node_type == "Call":
                func_name = node.attributes.get("func_name", "")
                if func_name in input_funcs:
                    sources.append(node)
            elif node.node_type == "Name":
                if node.name and node.name.lower() in input_vars:
                    sources.append(node)
        return sources
