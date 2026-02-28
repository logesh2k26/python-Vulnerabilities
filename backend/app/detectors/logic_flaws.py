"""SQL injection and path traversal detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class LogicFlawDetector(BaseDetector):
    """Detect SQL injection, path traversal, SSRF vulnerabilities."""
    
    name = "logic_flaws"
    description = "Detects SQL injection, path traversal, and SSRF"
    
    SQL_FUNCTIONS = {"execute", "executemany", "executescript", "raw", "rawquery"}
    PATH_FUNCTIONS = {"open", "read", "write", "join", "abspath"}
    REQUEST_FUNCTIONS = {"get", "post", "put", "delete", "request", "urlopen"}
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        results.extend(self._detect_sql_injection(nodes))
        results.extend(self._detect_path_traversal(nodes))
        results.extend(self._detect_ssrf(nodes))
        return results
    
    def _detect_sql_injection(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            if func_name not in self.SQL_FUNCTIONS:
                continue
            
            # Check for string formatting in SQL
            has_format = self._has_string_formatting(node, nodes)
            sources = self._find_data_sources(nodes, node)
            
            if has_format or sources:
                confidence = 0.9 if has_format and sources else 0.7
                results.append(DetectionResult(
                    vulnerability_type="sql_injection",
                    confidence=confidence,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description="SQL query constructed with string formatting",
                    severity="critical",
                    remediation="Use parameterized queries with placeholders",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"function": func_name, "has_formatting": has_format}
                ))
        return results
    
    def _detect_path_traversal(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            if func_name not in self.PATH_FUNCTIONS:
                continue
            
            sources = self._find_data_sources(nodes, node)
            if not sources:
                continue
            
            results.append(DetectionResult(
                vulnerability_type="path_traversal",
                confidence=0.8,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description="File path constructed from user input",
                severity="high",
                remediation="Validate paths, use os.path.basename(), prevent ../ sequences",
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                metadata={"function": func_name}
            ))
        return results
    
    def _detect_ssrf(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            if func_name not in self.REQUEST_FUNCTIONS:
                continue
            if module not in ("requests", "urllib", "urllib2", "http", "httpx", "aiohttp"):
                continue
            
            sources = self._find_data_sources(nodes, node)
            if not sources:
                continue
            
            results.append(DetectionResult(
                vulnerability_type="ssrf",
                confidence=0.75,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description="HTTP request URL from user input (SSRF risk)",
                severity="high",
                remediation="Validate and allowlist URLs, block internal IPs",
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                metadata={"function": f"{module}.{func_name}"}
            ))
        return results
    def _has_string_formatting(self, call_node: ASTNode, nodes: List[ASTNode]) -> bool:
        start_line = call_node.lineno
        end_line = call_node.end_lineno or start_line
        
        for node in nodes:
            if not (start_line <= node.lineno <= end_line):
                continue
            if node.node_type in ("JoinedStr", "FormattedValue"):
                return True
            if node.node_type == "BinOp" and node.attributes.get("op") == "Mod":
                return True
            if node.node_type == "Call" and node.attributes.get("func_name") == "format":
                return True
        return False
