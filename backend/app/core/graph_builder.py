"""Graph Builder module for converting AST to graph representation."""
import ast
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
from app.core.ast_parser import ASTNode, ASTParser
import logging

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Representation of a graph edge."""
    source: int
    target: int
    edge_type: str  # 'ast', 'data_flow', 'control_flow', 'call'


@dataclass
class CodeGraph:
    """Graph representation of code for GNN processing."""
    nodes: List[ASTNode]
    edges: List[Edge]
    node_features: np.ndarray  # Shape: (num_nodes, feature_dim)
    edge_index: np.ndarray  # Shape: (2, num_edges)
    edge_types: List[str]
    
    # Mapping info
    node_to_line: Dict[int, int]
    line_to_nodes: Dict[int, List[int]]
    
    def to_dict(self) -> Dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_feature_dim": self.node_features.shape[1] if len(self.node_features) > 0 else 0,
            "edge_types": list(set(self.edge_types))
        }


class GraphBuilder:
    """Build graph representation from AST for GNN input."""
    
    # Node type vocabulary for encoding
    NODE_TYPE_VOCAB = {
        "Module": 0, "FunctionDef": 1, "AsyncFunctionDef": 2, "ClassDef": 3,
        "Return": 4, "Delete": 5, "Assign": 6, "AugAssign": 7, "AnnAssign": 8,
        "For": 9, "AsyncFor": 10, "While": 11, "If": 12, "With": 13,
        "AsyncWith": 14, "Raise": 15, "Try": 16, "Assert": 17, "Import": 18,
        "ImportFrom": 19, "Global": 20, "Nonlocal": 21, "Expr": 22, "Pass": 23,
        "Break": 24, "Continue": 25, "BoolOp": 26, "NamedExpr": 27, "BinOp": 28,
        "UnaryOp": 29, "Lambda": 30, "IfExp": 31, "Dict": 32, "Set": 33,
        "ListComp": 34, "SetComp": 35, "DictComp": 36, "GeneratorExp": 37,
        "Await": 38, "Yield": 39, "YieldFrom": 40, "Compare": 41, "Call": 42,
        "FormattedValue": 43, "JoinedStr": 44, "Constant": 45, "Attribute": 46,
        "Subscript": 47, "Starred": 48, "Name": 49, "List": 50, "Tuple": 51,
        "Slice": 52, "ExceptHandler": 53, "arguments": 54, "arg": 55,
        "keyword": 56, "alias": 57, "withitem": 58, "comprehension": 59
    }
    
    # Dangerous function names for enhanced features
    DANGEROUS_FUNCTIONS = {
        "eval", "exec", "compile", "open", "input",
        "os.system", "os.popen", "subprocess.call", "subprocess.run",
        "subprocess.Popen", "pickle.loads", "pickle.load",
        "yaml.load", "yaml.unsafe_load", "marshal.loads",
        "exec", "execfile", "__import__"
    }
    
    EDGE_TYPE_VOCAB = {
        "ast": 0,
        "data_flow": 1,
        "control_flow": 2,
        "call": 3,
        "return": 4
    }
    
    def __init__(self, feature_dim: int = 128):
        """Initialize graph builder with feature dimension."""
        self.feature_dim = feature_dim
        self.parser = ASTParser()
    
    def build_graph(self, source_code: str) -> CodeGraph:
        """
        Build a graph from Python source code.
        
        Args:
            source_code: Python source code string
            
        Returns:
            CodeGraph object with nodes, edges, and features
        """
        # Parse AST
        nodes, tree = self.parser.parse(source_code)
        
        if not nodes:
            return self._empty_graph()
        
        # Build edges
        edges = []
        
        # 1. AST structure edges (parent-child)
        ast_edges = self._build_ast_edges(nodes)
        edges.extend(ast_edges)
        
        # 2. Data flow edges
        data_flow_edges = self._build_data_flow_edges(nodes, tree)
        edges.extend(data_flow_edges)
        
        # 3. Control flow edges
        control_flow_edges = self._build_control_flow_edges(nodes)
        edges.extend(control_flow_edges)
        
        # 4. Function call edges
        call_edges = self._build_call_edges(nodes)
        edges.extend(call_edges)
        
        # Build node features
        node_features = self._build_node_features(nodes)
        
        # Build edge index tensor
        edge_index = np.array(
            [[e.source for e in edges], [e.target for e in edges]],
            dtype=np.int64
        ) if edges else np.zeros((2, 0), dtype=np.int64)
        
        edge_types = [e.edge_type for e in edges]
        
        # Build line mappings
        node_to_line = {n.node_id: n.lineno for n in nodes}
        line_to_nodes: Dict[int, List[int]] = {}
        for node in nodes:
            if node.lineno not in line_to_nodes:
                line_to_nodes[node.lineno] = []
            line_to_nodes[node.lineno].append(node.node_id)
        
        return CodeGraph(
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            node_to_line=node_to_line,
            line_to_nodes=line_to_nodes
        )
    
    def _empty_graph(self) -> CodeGraph:
        """Return an empty graph."""
        return CodeGraph(
            nodes=[],
            edges=[],
            node_features=np.zeros((0, self.feature_dim)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_types=[],
            node_to_line={},
            line_to_nodes={}
        )
    
    def _build_ast_edges(self, nodes: List[ASTNode]) -> List[Edge]:
        """Build AST structure edges (parent to child)."""
        edges = []
        for node in nodes:
            for child_id in node.children_ids:
                edges.append(Edge(
                    source=node.node_id,
                    target=child_id,
                    edge_type="ast"
                ))
                # Also add reverse edge for bidirectional message passing
                edges.append(Edge(
                    source=child_id,
                    target=node.node_id,
                    edge_type="ast"
                ))
        return edges
    
    def _build_data_flow_edges(self, nodes: List[ASTNode], tree: ast.AST) -> List[Edge]:
        """Build data flow edges tracking variable definitions and uses."""
        edges = []
        
        # Track variable definitions: var_name -> list of defining node IDs
        definitions: Dict[str, List[int]] = {}
        
        # First pass: find all definitions
        for node in nodes:
            if node.node_type == "Assign":
                targets = node.attributes.get("targets", [])
                for target in targets:
                    if target not in definitions:
                        definitions[target] = []
                    definitions[target].append(node.node_id)
            elif node.node_type in ("FunctionDef", "AsyncFunctionDef"):
                if node.name:
                    if node.name not in definitions:
                        definitions[node.name] = []
                    definitions[node.name].append(node.node_id)
        
        # Second pass: connect uses to definitions
        for node in nodes:
            if node.node_type == "Name" and node.name:
                # This is a variable use
                if node.name in definitions:
                    for def_id in definitions[node.name]:
                        # Only connect if definition comes before use
                        def_node = next((n for n in nodes if n.node_id == def_id), None)
                        if def_node and def_node.lineno <= node.lineno:
                            edges.append(Edge(
                                source=def_id,
                                target=node.node_id,
                                edge_type="data_flow"
                            ))
        
        return edges
    
    def _build_control_flow_edges(self, nodes: List[ASTNode]) -> List[Edge]:
        """Build control flow edges between statements."""
        edges = []
        
        # Group nodes by their parent function/module
        scope_nodes: Dict[Optional[int], List[ASTNode]] = {}
        
        for node in nodes:
            parent = node.parent_id
            if parent not in scope_nodes:
                scope_nodes[parent] = []
            scope_nodes[parent].append(node)
        
        # Within each scope, connect sequential statements
        for parent_id, scope_members in scope_nodes.items():
            # Sort by line number
            sorted_nodes = sorted(scope_members, key=lambda n: (n.lineno, n.col_offset))
            
            # Connect sequential nodes of statement types
            statement_types = {"Assign", "Expr", "Return", "If", "For", "While", 
                             "With", "Try", "Raise", "Assert"}
            
            prev_stmt = None
            for node in sorted_nodes:
                if node.node_type in statement_types:
                    if prev_stmt is not None:
                        edges.append(Edge(
                            source=prev_stmt.node_id,
                            target=node.node_id,
                            edge_type="control_flow"
                        ))
                    prev_stmt = node
        
        return edges
    
    def _build_call_edges(self, nodes: List[ASTNode]) -> List[Edge]:
        """Build edges from function definitions to their calls."""
        edges = []
        
        # Find function definitions
        func_defs: Dict[str, int] = {}
        for node in nodes:
            if node.node_type in ("FunctionDef", "AsyncFunctionDef") and node.name:
                func_defs[node.name] = node.node_id
        
        # Find calls and connect to definitions
        for node in nodes:
            if node.node_type == "Call":
                func_name = node.attributes.get("func_name")
                if func_name and func_name in func_defs:
                    edges.append(Edge(
                        source=func_defs[func_name],
                        target=node.node_id,
                        edge_type="call"
                    ))
        
        return edges
    
    def _build_node_features(self, nodes: List[ASTNode]) -> np.ndarray:
        """Build feature vectors for each node."""
        num_nodes = len(nodes)
        features = np.zeros((num_nodes, self.feature_dim), dtype=np.float32)
        
        for i, node in enumerate(nodes):
            feature_vec = []
            
            # 1. Node type encoding (one-hot, 60 dims)
            type_idx = self.NODE_TYPE_VOCAB.get(node.node_type, len(self.NODE_TYPE_VOCAB))
            type_onehot = np.zeros(60)
            if type_idx < 60:
                type_onehot[type_idx] = 1.0
            feature_vec.extend(type_onehot)
            
            # 2. Position features (4 dims)
            feature_vec.append(np.log1p(node.lineno))
            feature_vec.append(np.log1p(node.col_offset))
            feature_vec.append(len(node.children_ids) / 10.0)  # Normalized child count
            feature_vec.append(1.0 if node.parent_id is None else 0.0)  # Is root
            
            # 3. Semantic features (20 dims)
            semantic_features = self._extract_semantic_features(node)
            feature_vec.extend(semantic_features)
            
            # 4. Security-relevant features (20 dims)
            security_features = self._extract_security_features(node)
            feature_vec.extend(security_features)
            
            # 5. Name embedding (24 dims) - simple character-based
            name_embedding = self._embed_name(node.name)
            feature_vec.extend(name_embedding)
            
            # Pad or truncate to feature_dim
            feature_arr = np.array(feature_vec[:self.feature_dim], dtype=np.float32)
            if len(feature_arr) < self.feature_dim:
                feature_arr = np.pad(feature_arr, (0, self.feature_dim - len(feature_arr)))
            
            features[i] = feature_arr
        
        return features
    
    def _extract_semantic_features(self, node: ASTNode) -> List[float]:
        """Extract semantic features from node attributes."""
        features = [0.0] * 20
        
        # Is function definition
        features[0] = 1.0 if node.node_type in ("FunctionDef", "AsyncFunctionDef") else 0.0
        
        # Is class definition
        features[1] = 1.0 if node.node_type == "ClassDef" else 0.0
        
        # Is function call
        features[2] = 1.0 if node.node_type == "Call" else 0.0
        
        # Is import
        features[3] = 1.0 if node.node_type in ("Import", "ImportFrom") else 0.0
        
        # Is assignment
        features[4] = 1.0 if node.node_type in ("Assign", "AugAssign", "AnnAssign") else 0.0
        
        # Is control flow
        features[5] = 1.0 if node.node_type in ("If", "For", "While", "Try") else 0.0
        
        # Is exception handling
        features[6] = 1.0 if node.node_type in ("Try", "ExceptHandler", "Raise") else 0.0
        
        # Is return/yield
        features[7] = 1.0 if node.node_type in ("Return", "Yield", "YieldFrom") else 0.0
        
        # Has children
        features[8] = 1.0 if len(node.children_ids) > 0 else 0.0
        
        # Is string constant
        features[9] = 1.0 if node.node_type == "Constant" and isinstance(node.name, str) else 0.0
        
        # Number of arguments (for calls)
        features[10] = node.attributes.get("num_args", 0) / 10.0
        
        # Is async
        features[11] = 1.0 if node.attributes.get("is_async", False) else 0.0
        
        # Has decorators
        decorators = node.attributes.get("decorators", [])
        features[12] = len(decorators) / 5.0
        
        # Is comprehension
        features[13] = 1.0 if node.node_type in ("ListComp", "DictComp", "SetComp", "GeneratorExp") else 0.0
        
        # Is attribute access
        features[14] = 1.0 if node.node_type == "Attribute" else 0.0
        
        # Is subscript
        features[15] = 1.0 if node.node_type == "Subscript" else 0.0
        
        # Is binary operation
        features[16] = 1.0 if node.node_type == "BinOp" else 0.0
        
        # Is comparison
        features[17] = 1.0 if node.node_type == "Compare" else 0.0
        
        # Is boolean operation
        features[18] = 1.0 if node.node_type == "BoolOp" else 0.0
        
        # Is lambda
        features[19] = 1.0 if node.node_type == "Lambda" else 0.0
        
        return features
    
    def _extract_security_features(self, node: ASTNode) -> List[float]:
        """Extract security-relevant features."""
        features = [0.0] * 20
        
        # Check if this is a dangerous function call
        func_name = node.attributes.get("func_name", "")
        module = node.attributes.get("module", "")
        full_name = f"{module}.{func_name}" if module else func_name
        
        # Is dangerous function
        features[0] = 1.0 if full_name in self.DANGEROUS_FUNCTIONS or func_name in self.DANGEROUS_FUNCTIONS else 0.0
        
        # Is eval/exec
        features[1] = 1.0 if func_name in ("eval", "exec", "compile") else 0.0
        
        # Is subprocess/os call
        features[2] = 1.0 if module in ("os", "subprocess") else 0.0
        
        # Is pickle/marshal
        features[3] = 1.0 if module in ("pickle", "marshal", "cPickle") else 0.0
        
        # Is yaml
        features[4] = 1.0 if module == "yaml" else 0.0
        
        # Is open/file operation
        features[5] = 1.0 if func_name in ("open", "file", "read", "write") else 0.0
        
        # Is SQL-related
        features[6] = 1.0 if func_name in ("execute", "executemany", "executescript") else 0.0
        
        # Is request/HTTP
        features[7] = 1.0 if module in ("requests", "urllib", "urllib2", "http") else 0.0
        
        # Contains string formatting
        features[8] = 1.0 if node.node_type in ("JoinedStr", "FormattedValue") else 0.0
        
        # Is input
        features[9] = 1.0 if func_name == "input" else 0.0
        
        # Check if name looks like a secret
        if node.name:
            name_lower = node.name.lower() if isinstance(node.name, str) else ""
            features[10] = 1.0 if any(
                kw in name_lower for kw in 
                ["password", "secret", "key", "token", "api_key", "apikey", "pwd", "passwd"]
            ) else 0.0
        
        # Is constant that looks like a secret (entropy check placeholder)
        if node.node_type == "Constant" and isinstance(node.name, str):
            features[11] = 1.0 if len(node.name) > 10 and any(c.isdigit() for c in node.name) else 0.0
        
        # Is path operation
        features[12] = 1.0 if module in ("os.path", "pathlib") or func_name in ("join", "abspath") else 0.0
        
        # Is socket operation
        features[13] = 1.0 if module == "socket" else 0.0
        
        # Is shell operation
        features[14] = 1.0 if func_name in ("system", "popen", "spawn") else 0.0
        
        # Is import of dangerous module
        if node.node_type == "ImportFrom":
            dangerous_modules = {"os", "subprocess", "pickle", "marshal", "yaml", "eval"}
            imported_module = node.attributes.get("module", "")
            features[15] = 1.0 if imported_module in dangerous_modules else 0.0
        
        # Is string concatenation with variable
        features[16] = 1.0 if node.node_type == "BinOp" and node.attributes.get("op") == "Add" else 0.0
        
        # Is format string
        features[17] = 1.0 if func_name == "format" else 0.0
        
        # Is environment variable access
        features[18] = 1.0 if func_name in ("getenv", "environ") else 0.0
        
        # Is database connection
        features[19] = 1.0 if func_name in ("connect", "cursor") else 0.0
        
        return features
    
    def _embed_name(self, name: Optional[str]) -> List[float]:
        """Simple character-based name embedding."""
        embedding = [0.0] * 24
        
        if not name or not isinstance(name, str):
            return embedding
        
        # Character distribution features
        name_lower = name.lower()
        
        # Length feature
        embedding[0] = min(len(name) / 50.0, 1.0)
        
        # Contains underscore
        embedding[1] = 1.0 if '_' in name else 0.0
        
        # Is all caps
        embedding[2] = 1.0 if name.isupper() and len(name) > 1 else 0.0
        
        # Starts with underscore
        embedding[3] = 1.0 if name.startswith('_') else 0.0
        
        # Contains digit
        embedding[4] = 1.0 if any(c.isdigit() for c in name) else 0.0
        
        # First few chars hash (simple distribution)
        for i, char in enumerate(name_lower[:6]):
            if i + 5 < 24:
                embedding[i + 5] = (ord(char) - ord('a')) / 26.0 if char.isalpha() else 0.5
        
        # Word pattern features
        words = name.split('_')
        embedding[11] = min(len(words) / 5.0, 1.0)
        
        # Common prefixes
        prefixes = ["get", "set", "is", "has", "do", "make", "create", "delete", "update"]
        for i, prefix in enumerate(prefixes[:6]):
            if i + 12 < 24:
                embedding[i + 12] = 1.0 if name_lower.startswith(prefix) else 0.0
        
        # Common security-related keywords
        security_keywords = ["auth", "crypt", "hash", "sign", "verify", "validate"]
        for i, kw in enumerate(security_keywords):
            if i + 18 < 24:
                embedding[i + 18] = 1.0 if kw in name_lower else 0.0
        
        return embedding
