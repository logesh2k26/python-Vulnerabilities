"""Sanitizer Detection for identifying mitigation patterns."""
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from app.core.ast_parser import ASTNode
import logging

logger = logging.getLogger(__name__)


class SanitizerType(Enum):
    """Types of sanitization patterns."""
    PATH_NORMALIZATION = "path_normalization"
    PATH_BASENAME = "path_basename"
    PATH_PREFIX_CHECK = "path_prefix_check"
    COMMAND_WHITELIST = "command_whitelist"
    SHELL_FALSE = "shell_false"
    PARAMETERIZED_QUERY = "parameterized_query"
    INPUT_VALIDATION = "input_validation"
    REGEX_VALIDATION = "regex_validation"
    TYPE_CHECK = "type_check"
    LENGTH_CHECK = "length_check"
    SAFE_YAML_LOADER = "safe_yaml_loader"
    LITERAL_EVAL = "literal_eval"
    ESCAPE_FUNCTION = "escape_function"


@dataclass
class SanitizerMatch:
    """Represents a detected sanitization pattern."""
    sanitizer_type: SanitizerType
    line: int
    node_id: int
    variable: Optional[str]
    description: str
    effectiveness: float  # 0.0 to 1.0, how effective is this mitigation
    mitigates: List[str]  # List of vulnerability types this mitigates
    
    def to_dict(self) -> Dict:
        return {
            "type": self.sanitizer_type.value,
            "line": self.line,
            "variable": self.variable,
            "description": self.description,
            "effectiveness": self.effectiveness,
            "mitigates": self.mitigates
        }


class SanitizerDetector:
    """Detects sanitization and validation patterns in code."""
    
    # Path sanitization patterns
    PATH_SANITIZERS = {
        "resolve": ("pathlib.Path.resolve", SanitizerType.PATH_NORMALIZATION, 0.7),
        "abspath": ("os.path.abspath", SanitizerType.PATH_NORMALIZATION, 0.6),
        "realpath": ("os.path.realpath", SanitizerType.PATH_NORMALIZATION, 0.7),
        "normpath": ("os.path.normpath", SanitizerType.PATH_NORMALIZATION, 0.5),
        "basename": ("os.path.basename", SanitizerType.PATH_BASENAME, 0.9),
    }
    
    # Command sanitization patterns
    COMMAND_SANITIZERS = {
        "shlex.quote": (SanitizerType.ESCAPE_FUNCTION, 0.8),
        "shlex.split": (SanitizerType.ESCAPE_FUNCTION, 0.6),
        "pipes.quote": (SanitizerType.ESCAPE_FUNCTION, 0.8),
    }
    
    # SQL sanitization patterns
    SQL_SANITIZERS = {
        "?": (SanitizerType.PARAMETERIZED_QUERY, 0.95),  # Placeholder
        "%s": (SanitizerType.PARAMETERIZED_QUERY, 0.95), # Placeholder
        ":param": (SanitizerType.PARAMETERIZED_QUERY, 0.95),
    }
    
    # Validation patterns
    VALIDATION_PATTERNS = [
        (r"isinstance\s*\(", SanitizerType.TYPE_CHECK, 0.7),
        (r"\.isdigit\s*\(\)", SanitizerType.TYPE_CHECK, 0.8),
        (r"\.isalnum\s*\(\)", SanitizerType.INPUT_VALIDATION, 0.8),
        (r"\.isalpha\s*\(\)", SanitizerType.INPUT_VALIDATION, 0.8),
        (r"re\.match\s*\(", SanitizerType.REGEX_VALIDATION, 0.7),
        (r"re\.fullmatch\s*\(", SanitizerType.REGEX_VALIDATION, 0.85),
        (r"len\s*\([^)]+\)\s*[<>=]", SanitizerType.LENGTH_CHECK, 0.5),
    ]
    
    def __init__(self):
        self.sanitizers: List[SanitizerMatch] = []
        self.sanitized_vars: Dict[str, List[SanitizerMatch]] = {}
    
    def detect(self, nodes: List[ASTNode], source_code: str) -> List[SanitizerMatch]:
        """
        Detect sanitization patterns in code.
        
        Args:
            nodes: List of ASTNode from parser
            source_code: Original source code
            
        Returns:
            List of detected sanitizer matches
        """
        self.sanitizers = []
        self.sanitized_vars = {}
        lines = source_code.split('\n')
        
        # Detect various sanitizer patterns
        self._detect_path_sanitizers(nodes)
        self._detect_command_sanitizers(nodes)
        self._detect_sql_sanitizers(nodes, lines)
        self._detect_validation_patterns(nodes, lines)
        self._detect_whitelist_patterns(nodes)
        self._detect_yaml_safe_loader(nodes)
        self._detect_literal_eval(nodes)
        self._detect_shell_false(nodes)
        self._detect_prefix_checks(nodes, lines)
        
        return self.sanitizers
    
    def _detect_path_sanitizers(self, nodes: List[ASTNode]):
        """Detect path normalization and sanitization."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            
            if func_name in self.PATH_SANITIZERS:
                desc, stype, eff = self.PATH_SANITIZERS[func_name]
                var = self._find_assigned_var_for_call(node, nodes)
                
                match = SanitizerMatch(
                    sanitizer_type=stype,
                    line=node.lineno,
                    node_id=node.node_id,
                    variable=var,
                    description=f"Path sanitization via {desc}",
                    effectiveness=eff,
                    mitigates=["path_traversal"]
                )
                self.sanitizers.append(match)
                if var:
                    self._track_sanitized_var(var, match)
    
    def _detect_command_sanitizers(self, nodes: List[ASTNode]):
        """Detect command argument sanitization."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            full_name = f"{module}.{func_name}" if module else func_name
            
            if full_name in self.COMMAND_SANITIZERS:
                stype, eff = self.COMMAND_SANITIZERS[full_name]
                var = self._find_assigned_var_for_call(node, nodes)
                
                match = SanitizerMatch(
                    sanitizer_type=stype,
                    line=node.lineno,
                    node_id=node.node_id,
                    variable=var,
                    description=f"Command argument escaped via {full_name}",
                    effectiveness=eff,
                    mitigates=["command_injection"]
                )
                self.sanitizers.append(match)
                if var:
                    self._track_sanitized_var(var, match)
    
    def _detect_sql_sanitizers(self, nodes: List[ASTNode], lines: List[str]):
        """Detect parameterized SQL queries."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            if func_name not in ("execute", "executemany"):
                continue
            
            # Check if line contains parameterized query patterns
            if node.lineno <= len(lines):
                line = lines[node.lineno - 1]
                
                # Check for placeholders
                if "?" in line or "%s" in line or ":param" in line.lower():
                    # Also need to verify there's a second argument
                    num_args = node.attributes.get("num_args", 0)
                    if num_args >= 2:
                        match = SanitizerMatch(
                            sanitizer_type=SanitizerType.PARAMETERIZED_QUERY,
                            line=node.lineno,
                            node_id=node.node_id,
                            variable=None,
                            description="SQL query uses parameterized placeholders",
                            effectiveness=0.95,
                            mitigates=["sql_injection"]
                        )
                        self.sanitizers.append(match)
    
    def _detect_validation_patterns(self, nodes: List[ASTNode], lines: List[str]):
        """Detect input validation patterns in source code."""
        for i, line in enumerate(lines):
            lineno = i + 1
            for pattern, stype, eff in self.VALIDATION_PATTERNS:
                if re.search(pattern, line):
                    # Find the node at this line
                    node_id = self._find_node_at_line(nodes, lineno)
                    
                    match = SanitizerMatch(
                        sanitizer_type=stype,
                        line=lineno,
                        node_id=node_id,
                        variable=None,
                        description=f"Input validation: {stype.value}",
                        effectiveness=eff,
                        mitigates=["eval_exec", "command_injection", "sql_injection"]
                    )
                    self.sanitizers.append(match)
    
    def _detect_whitelist_patterns(self, nodes: List[ASTNode]):
        """Detect whitelist-based validation patterns."""
        for node in nodes:
            # Look for "if x in whitelist" or "if x in allowed_values" patterns
            if node.node_type == "Compare":
                ops = node.attributes.get("ops", [])
                if "In" in ops or "NotIn" in ops:
                    # This is a containment check - likely whitelist
                    match = SanitizerMatch(
                        sanitizer_type=SanitizerType.COMMAND_WHITELIST,
                        line=node.lineno,
                        node_id=node.node_id,
                        variable=None,
                        description="Whitelist validation check",
                        effectiveness=0.9,
                        mitigates=["command_injection", "eval_exec"]
                    )
                    self.sanitizers.append(match)
    
    def _detect_yaml_safe_loader(self, nodes: List[ASTNode]):
        """Detect safe YAML loader usage."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            
            if func_name == "safe_load":
                match = SanitizerMatch(
                    sanitizer_type=SanitizerType.SAFE_YAML_LOADER,
                    line=node.lineno,
                    node_id=node.node_id,
                    variable=None,
                    description="Using yaml.safe_load instead of yaml.load",
                    effectiveness=0.95,
                    mitigates=["unsafe_deserialization"]
                )
                self.sanitizers.append(match)
            elif func_name == "load":
                # Check for SafeLoader argument
                # Look for child nodes that might be SafeLoader
                for child in nodes:
                    if child.parent_id == node.node_id and child.name in ("SafeLoader", "FullLoader"):
                        match = SanitizerMatch(
                            sanitizer_type=SanitizerType.SAFE_YAML_LOADER,
                            line=node.lineno,
                            node_id=node.node_id,
                            variable=None,
                            description=f"Using yaml.load with {child.name}",
                            effectiveness=0.9,
                            mitigates=["unsafe_deserialization"]
                        )
                        self.sanitizers.append(match)
                        break
    
    def _detect_literal_eval(self, nodes: List[ASTNode]):
        """Detect ast.literal_eval usage (safe alternative to eval)."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            if func_name == "literal_eval" or (module == "ast" and func_name == "literal_eval"):
                var = self._find_assigned_var_for_call(node, nodes)
                match = SanitizerMatch(
                    sanitizer_type=SanitizerType.LITERAL_EVAL,
                    line=node.lineno,
                    node_id=node.node_id,
                    variable=var,
                    description="Using ast.literal_eval (safe eval alternative)",
                    effectiveness=0.95,
                    mitigates=["eval_exec"]
                )
                self.sanitizers.append(match)
    
    def _detect_shell_false(self, nodes: List[ASTNode]):
        """Detect subprocess calls with shell=False."""
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            if func_name not in ("run", "call", "Popen", "check_output", "check_call"):
                continue
            
            # Check for shell=False in keywords
            for child in nodes:
                if child.parent_id == node.node_id and child.node_type == "keyword":
                    if child.name == "shell":
                        # Check the value - we need to look for False
                        for val_node in nodes:
                            if val_node.parent_id == child.node_id:
                                if val_node.name == "False" or (val_node.node_type == "Constant" and val_node.name == "False"):
                                    match = SanitizerMatch(
                                        sanitizer_type=SanitizerType.SHELL_FALSE,
                                        line=node.lineno,
                                        node_id=node.node_id,
                                        variable=None,
                                        description="subprocess called with shell=False",
                                        effectiveness=0.9,
                                        mitigates=["command_injection"]
                                    )
                                    self.sanitizers.append(match)
    
    def _detect_prefix_checks(self, nodes: List[ASTNode], lines: List[str]):
        """Detect path prefix/startswith checks."""
        for i, line in enumerate(lines):
            lineno = i + 1
            # Look for startswith checks that might be base directory validation
            if ".startswith(" in line and ("path" in line.lower() or "dir" in line.lower() or "base" in line.lower()):
                node_id = self._find_node_at_line(nodes, lineno)
                match = SanitizerMatch(
                    sanitizer_type=SanitizerType.PATH_PREFIX_CHECK,
                    line=lineno,
                    node_id=node_id,
                    variable=None,
                    description="Path prefix/base directory check",
                    effectiveness=0.85,
                    mitigates=["path_traversal"]
                )
                self.sanitizers.append(match)
    
    def _find_assigned_var_for_call(self, call_node: ASTNode, nodes: List[ASTNode]) -> Optional[str]:
        """Find variable that receives the call result."""
        for node in nodes:
            if node.node_type == "Assign" and node.lineno == call_node.lineno:
                targets = node.attributes.get("targets", [])
                if targets:
                    return targets[0]
        return None
    
    def _find_node_at_line(self, nodes: List[ASTNode], lineno: int) -> int:
        """Find first node at given line."""
        for node in nodes:
            if node.lineno == lineno:
                return node.node_id
        return -1
    
    def _track_sanitized_var(self, var: str, match: SanitizerMatch):
        """Track that a variable has been sanitized."""
        if var not in self.sanitized_vars:
            self.sanitized_vars[var] = []
        self.sanitized_vars[var].append(match)
    
    def is_variable_sanitized(self, var: str, vuln_type: str) -> Optional[SanitizerMatch]:
        """Check if a variable has been sanitized for a specific vulnerability type."""
        if var not in self.sanitized_vars:
            return None
        
        for match in self.sanitized_vars[var]:
            if vuln_type in match.mitigates:
                return match
        return None
    
    def get_sanitizers_at_line(self, lineno: int) -> List[SanitizerMatch]:
        """Get all sanitizers at a specific line."""
        return [s for s in self.sanitizers if s.line == lineno]
    
    def get_sanitizers_for_vuln(self, vuln_type: str) -> List[SanitizerMatch]:
        """Get all sanitizers that mitigate a specific vulnerability type."""
        return [s for s in self.sanitizers if vuln_type in s.mitigates]
