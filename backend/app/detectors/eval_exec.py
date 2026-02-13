"""Eval/Exec vulnerability detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class EvalExecDetector(BaseDetector):
    """Detect unsafe eval(), exec(), compile() usage."""
    
    name = "eval_exec"
    description = "Detects unsafe usage of eval, exec, and compile"
    
    DANGEROUS_FUNCS = {"eval", "exec", "compile", "execfile"}
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            if func_name not in self.DANGEROUS_FUNCS:
                continue
            
            # Check if input might be from untrusted source
            sources = self._find_data_sources(nodes, node)
            confidence = 0.95 if sources else 0.7
            
            severity = "critical" if func_name in ("eval", "exec") else "high"
            
            results.append(DetectionResult(
                vulnerability_type="eval_exec",
                confidence=confidence,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description=f"Usage of {func_name}() can execute arbitrary code",
                severity=severity,
                remediation=f"Avoid {func_name}(). Use ast.literal_eval() for data parsing or implement safe alternatives.",
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                metadata={"function": func_name, "has_untrusted_input": bool(sources)}
            ))
        
        return results
