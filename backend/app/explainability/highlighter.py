"""Code highlighter for vulnerability visualization."""
from typing import List, Dict
from app.core.ast_parser import ASTNode


class CodeHighlighter:
    """Generate HTML/JSON for highlighting vulnerable code."""
    
    SEVERITY_COLORS = {
        "critical": "#ff4757",
        "high": "#ff6b6b",
        "medium": "#ffa726",
        "low": "#66bb6a"
    }
    
    def __init__(self, source_code: str):
        self.lines = source_code.split('\n')
    
    def generate_highlights(
        self,
        vulnerabilities: List[Dict],
        line_importance: Dict[int, float]
    ) -> List[Dict]:
        """Generate highlight data for frontend."""
        highlights = []
        
        for vuln in vulnerabilities:
            for line_num in vuln.get("affected_lines", []):
                highlights.append({
                    "line": line_num,
                    "type": vuln["type"],
                    "severity": vuln["severity"],
                    "color": self.SEVERITY_COLORS.get(vuln["severity"], "#ffa726"),
                    "message": vuln["description"],
                    "importance": line_importance.get(line_num, 0.5),
                    "content": self.lines[line_num - 1] if line_num <= len(self.lines) else ""
                })
        
        return highlights
    
    def generate_annotated_code(
        self,
        vulnerabilities: List[Dict],
        line_importance: Dict[int, float]
    ) -> str:
        """Generate annotated source with markers."""
        vuln_lines = {}
        for vuln in vulnerabilities:
            for ln in vuln.get("affected_lines", []):
                if ln not in vuln_lines:
                    vuln_lines[ln] = []
                vuln_lines[ln].append(vuln)
        
        output = []
        for i, line in enumerate(self.lines, 1):
            importance = line_importance.get(i, 0)
            marker = "!!" if i in vuln_lines else (">" if importance > 0.5 else " ")
            output.append(f"{marker} {i:4d} | {line}")
            
            if i in vuln_lines:
                for vuln in vuln_lines[i]:
                    output.append(f"       ^ {vuln['type']}: {vuln['description']}")
        
        return '\n'.join(output)
