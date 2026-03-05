"""Cross-Site Scripting (XSS) detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class XSSDetector(BaseDetector):
    """Detects potential XSS in web application responses."""

    name = "xss"
    description = "Detects Cross-Site Scripting vulnerabilities"

    UNSAFE_FUNCTIONS = {
        "Markup", "render_template_string", "Response", "HTMLResponse", 
        "send_response", "safe", "render_template", "execute_html"
    }
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            
            if func_name in self.UNSAFE_FUNCTIONS:
                # For these high-risk functions, we flag them if they occur
                # A more advanced version would use the taint analyzer here
                results.append(DetectionResult(
                    vulnerability_type="xss",
                    confidence=0.8,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description=f"Potential XSS: Dangerous function '{func_name}' used to process HTML",
                    severity="high",
                    remediation="Always escape user input before rendering in HTML; use templating engines with auto-escaping",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"function": func_name}
                ))
        return results
