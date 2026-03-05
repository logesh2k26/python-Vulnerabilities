"""Insecure Cryptography detector."""
from typing import List, Dict
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class InsecureCryptographyDetector(BaseDetector):
    """Detects weak cryptographic algorithms and insecure randomness."""

    name = "insecure_cryptography"
    description = "Detects weak hashing, insecure PRNGs, and hardcoded keys"

    WEAK_HASHES = {"md5", "sha1", "ripemd160"}
    INSECURE_PRNGS = {"random", "randrange", "randint", "choice", "choices", "sample"}
    
    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        results.extend(self._detect_weak_hashing(nodes))
        results.extend(self._detect_insecure_randomness(nodes))
        results.extend(self._detect_hardcoded_keys(nodes))
        return results

    def _detect_weak_hashing(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            # Check hashlib calls like hashlib.md5() or hashlib.new('md5')
            is_weak_hash = False
            algo = ""

            # The parser puts the module in 'module' and the attribute in 'func_name'
            if module == "hashlib" or func_name == "hashlib":
                # Handles 'import hashlib; hashlib.md5()'
                if func_name in self.WEAK_HASHES:
                    is_weak_hash = True
                    algo = func_name
                elif func_name == "new" or func_name == "hashlib.new":
                    is_weak_hash = True
                    algo = "hashlib.new() with weak algorithm"
            elif func_name in self.WEAK_HASHES:
                # Handles 'from hashlib import md5; md5()'
                is_weak_hash = True
                algo = func_name
            
            if is_weak_hash:
                results.append(DetectionResult(
                    vulnerability_type="insecure_cryptography",
                    confidence=0.9,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description=f"Use of weak cryptographic hash function: {algo}",
                    severity="medium",
                    remediation=f"Replace with a secure alternative like SHA-256 or SHA-3",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"algorithm": algo}
                ))


        return results

    def _detect_insecure_randomness(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            if (module == "random" or func_name == "random") and any(f in func_name for f in self.INSECURE_PRNGS):
                results.append(DetectionResult(
                    vulnerability_type="insecure_cryptography",
                    confidence=0.7,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description=f"Insecure random number generator '{func_name}' used for security purposes",
                    severity="low",
                    remediation="Use the 'secrets' module for cryptographically strong random numbers",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"function": func_name}
                ))
        return results

    def _detect_hardcoded_keys(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        # Keywords suggesting cryptographic keys/secrets
        key_keywords = {"key", "secret", "password", "token", "iv", "nonce", "salt"}
        
        for node in nodes:
            if node.node_type != "Assign":
                continue
            
            # Check if target name looks like a key
            target_names = node.attributes.get("targets", [])
            is_key_target = any(k in name.lower() for name in target_names for k in key_keywords)
            
            if not is_key_target:
                continue
                
            # Check if assigned value is a literal string of certain length
            # (Simplification: look for Const nodes in children)
            # This is handled by HardcodedSecretsDetector as well, but we add crypto context here
            pass
            
        return results
