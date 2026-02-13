"""Unsafe deserialization vulnerability detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class DeserializationDetector(BaseDetector):
    """Detect unsafe deserialization vulnerabilities."""
    
    name = "deserialization"
    description = "Detects unsafe deserialization using pickle, yaml, marshal"
    
    DANGEROUS_PATTERNS = {
        ("pickle", "loads"): ("critical", "Arbitrary code execution via pickle"),
        ("pickle", "load"): ("critical", "Arbitrary code execution via pickle"),
        ("cPickle", "loads"): ("critical", "Arbitrary code execution via cPickle"),
        ("cPickle", "load"): ("critical", "Arbitrary code execution via cPickle"),
        ("marshal", "loads"): ("high", "Arbitrary code execution via marshal"),
        ("marshal", "load"): ("high", "Arbitrary code execution via marshal"),
        ("yaml", "load"): ("critical", "YAML deserialization with unsafe loader"),
        ("yaml", "unsafe_load"): ("critical", "Explicit unsafe YAML loading"),
        ("shelve", "open"): ("high", "Shelve uses pickle internally"),
        ("dill", "loads"): ("critical", "Dill can deserialize arbitrary code"),
        ("dill", "load"): ("critical", "Dill can deserialize arbitrary code"),
    }
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            key = (module, func_name)
            if key not in self.DANGEROUS_PATTERNS:
                continue
            
            severity, desc = self.DANGEROUS_PATTERNS[key]
            
            # Check for safe YAML loader
            if module == "yaml" and func_name == "load":
                if self._has_safe_loader(nodes, node):
                    continue
            
            sources = self._find_data_sources(nodes, node)
            confidence = 0.95 if sources else 0.75
            
            results.append(DetectionResult(
                vulnerability_type="unsafe_deserialization",
                confidence=confidence,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description=desc,
                severity=severity,
                remediation=self._get_remediation(module, func_name),
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                metadata={"function": f"{module}.{func_name}"}
            ))
        
        return results
    
    def _has_safe_loader(self, nodes: List[ASTNode], call_node: ASTNode) -> bool:
        safe_loaders = {"SafeLoader", "FullLoader", "safe_load"}
        for node in nodes:
            if node.parent_id == call_node.node_id and node.name in safe_loaders:
                return True
        return False
    
    def _get_remediation(self, module: str, func: str) -> str:
        if module in ("pickle", "cPickle", "dill"):
            return "Never unpickle data from untrusted sources. Use JSON or other safe formats."
        if module == "yaml":
            return "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)"
        if module == "marshal":
            return "Marshal is not designed for security. Use JSON for data interchange."
        return "Use safe serialization formats like JSON"
