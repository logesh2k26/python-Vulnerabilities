"""Hardcoded secrets detector."""
import re
import math
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class HardcodedSecretsDetector(BaseDetector):
    """Detect hardcoded secrets, API keys, passwords."""
    
    name = "hardcoded_secrets"
    description = "Detects hardcoded passwords, API keys, and secrets"
    
    SECRET_PATTERNS = [
        (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "password"),
        (r'(?i)(api_?key|apikey)\s*=\s*["\'][^"\']+["\']', "api_key"),
        (r'(?i)(secret_?key|secretkey)\s*=\s*["\'][^"\']+["\']', "secret_key"),
        (r'(?i)(auth_?token|token)\s*=\s*["\'][^"\']+["\']', "token"),
        (r'(?i)(aws_?access|aws_?secret)\s*=\s*["\'][^"\']+["\']', "aws_credential"),
        (r'(?i)bearer\s+[a-zA-Z0-9\-_.]+', "bearer_token"),
        (r'(?i)basic\s+[a-zA-Z0-9+/=]+', "basic_auth"),
        (r'ghp_[a-zA-Z0-9]{36}', "github_token"),
        (r'xox[baprs]-[0-9]{10,13}-[a-zA-Z0-9-]+', "slack_token"),
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "private_key"),
    ]
    
    SECRET_VAR_NAMES = {
        "password", "passwd", "pwd", "secret", "api_key", "apikey",
        "auth_token", "token", "private_key", "secret_key", "credentials"
    }
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        
        for node in nodes:
            if node.node_type == "Assign":
                result = self._check_assignment(node, nodes)
                if result:
                    results.append(result)
            elif node.node_type == "Constant" and isinstance(node.name, str):
                result = self._check_string_constant(node)
                if result:
                    results.append(result)
        
        return results
    
    def _check_assignment(self, node: ASTNode, nodes: List[ASTNode]) -> DetectionResult:
        targets = node.attributes.get("targets", [])
        
        for target in targets:
            target_lower = target.lower() if isinstance(target, str) else ""
            
            if any(secret in target_lower for secret in self.SECRET_VAR_NAMES):
                # Find the value being assigned
                value_node = self._find_value_node(node, nodes)
                
                if value_node and isinstance(value_node.name, str):
                    if len(value_node.name) > 4 and value_node.name not in ("None", "True", "False"):
                        confidence = self._calculate_confidence(target_lower, value_node.name)
                        return DetectionResult(
                            vulnerability_type="hardcoded_secrets",
                            confidence=confidence,
                            affected_lines=[node.lineno],
                            affected_nodes=[node.node_id],
                            description=f"Hardcoded secret in variable '{target}'",
                            severity="high",
                            remediation="Use environment variables or a secrets manager",
                            code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                            metadata={"variable": target, "secret_type": self._identify_secret_type(target_lower)}
                        )
        return None
    
    def _check_string_constant(self, node: ASTNode) -> DetectionResult:
        value = node.name
        if not isinstance(value, str) or len(value) < 8:
            return None
        
        for pattern, secret_type in self.SECRET_PATTERNS:
            if re.search(pattern, value):
                return DetectionResult(
                    vulnerability_type="hardcoded_secrets",
                    confidence=0.85,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description=f"Potential {secret_type} found in string",
                    severity="high",
                    remediation="Remove secrets from source code",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"secret_type": secret_type}
                )
        
        # Check entropy for random-looking strings
        if len(value) >= 16 and self._calculate_entropy(value) > 4.0:
            return DetectionResult(
                vulnerability_type="hardcoded_secrets",
                confidence=0.6,
                affected_lines=[node.lineno],
                affected_nodes=[node.node_id],
                description="High-entropy string may be a secret",
                severity="medium",
                remediation="Review if this string contains sensitive data",
                code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno)
            )
        return None
    
    def _find_value_node(self, assign_node: ASTNode, nodes: List[ASTNode]) -> ASTNode:
        for node in nodes:
            if node.parent_id == assign_node.node_id and node.node_type == "Constant":
                return node
        return None
    
    def _calculate_confidence(self, var_name: str, value: str) -> float:
        confidence = 0.7
        if any(kw in var_name for kw in ["password", "secret", "key"]):
            confidence += 0.15
        if len(value) > 10:
            confidence += 0.05
        if self._calculate_entropy(value) > 3.5:
            confidence += 0.1
        return min(confidence, 0.98)
    
    def _calculate_entropy(self, s: str) -> float:
        if not s:
            return 0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0
        for count in freq.values():
            p = count / len(s)
            entropy -= p * math.log2(p)
        return entropy
    
    def _identify_secret_type(self, var_name: str) -> str:
        if "password" in var_name or "passwd" in var_name or "pwd" in var_name:
            return "password"
        if "api" in var_name and "key" in var_name:
            return "api_key"
        if "token" in var_name:
            return "token"
        if "secret" in var_name:
            return "secret"
        return "credential"
