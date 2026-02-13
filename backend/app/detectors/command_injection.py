"""Command injection vulnerability detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class CommandInjectionDetector(BaseDetector):
    """Detect command injection vulnerabilities."""
    
    name = "command_injection"
    description = "Detects OS command injection vulnerabilities"
    
    DANGEROUS_PATTERNS = {
        ("os", "system"): "critical",
        ("os", "popen"): "critical",
        ("os", "spawn"): "high",
        ("os", "spawnl"): "high",
        ("os", "spawn"): "high",
        ("os", "spawnl"): "high",
        # subprocess functions handled separately via SHELL_TRUE_FUNCS
        ("commands", "getoutput"): "critical",
        ("commands", "getstatusoutput"): "critical",
    }
    
    SHELL_TRUE_FUNCS = {"call", "run", "Popen"}
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            key = (module, func_name)
            if key not in self.DANGEROUS_PATTERNS:
                # Check for shell=True
                if func_name in self.SHELL_TRUE_FUNCS:
                    if self._has_shell_true(nodes, node):
                        results.append(self._create_shell_result(node))
                continue
            
            severity = self.DANGEROUS_PATTERNS[key]
            sources = self._find_data_sources(nodes, node)
            confidence = 0.9 if sources else 0.65
            
            results.append(DetectionResult(
                vulnerability_type="command_injection",
                confidence=confidence,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description=f"{module}.{func_name}() can execute shell commands",
                severity=severity,
                remediation="Use subprocess with shell=False and pass arguments as list. Validate and sanitize all inputs.",
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                metadata={"function": f"{module}.{func_name}", "has_untrusted_input": bool(sources)}
            ))
        
        return results
    
    def _has_shell_true(self, nodes: List[ASTNode], call_node: ASTNode) -> bool:
        for node in nodes:
            if node.parent_id == call_node.node_id:
                if node.node_type == "keyword" and node.name == "shell":
                    return True
        return False
    
    def _create_shell_result(self, node: ASTNode) -> DetectionResult:
        return DetectionResult(
            vulnerability_type="command_injection",
            confidence=0.85,
            affected_lines=[node.lineno],
            affected_nodes=[node.node_id],
            description="subprocess with shell=True can be exploited for command injection",
            severity="high",
            remediation="Use shell=False and pass command as a list of arguments",
            code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno)
        )
