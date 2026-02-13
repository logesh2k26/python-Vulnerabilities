"""AST Parser module for Python source code analysis."""
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """Representation of an AST node with metadata."""
    node_id: int
    node_type: str
    name: Optional[str]
    lineno: int
    col_offset: int
    end_lineno: Optional[int]
    end_col_offset: Optional[int]
    parent_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "end_lineno": self.end_lineno,
            "end_col_offset": self.end_col_offset,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "attributes": self.attributes
        }


class ASTParser:
    """Parse Python source code into structured AST representation."""
    
    # Node types we care about for vulnerability detection
    IMPORTANT_NODE_TYPES = {
        # Function-related
        "FunctionDef", "AsyncFunctionDef", "Lambda",
        # Calls
        "Call", "Attribute",
        # Names and literals
        "Name", "Constant", "Str", "Bytes",
        # Assignments
        "Assign", "AnnAssign", "AugAssign",
        # Control flow
        "If", "For", "While", "Try", "With",
        # Imports
        "Import", "ImportFrom",
        # Expressions
        "BinOp", "Compare", "BoolOp", "UnaryOp",
        # Subscript for dict/list access
        "Subscript", "Index", "Slice",
        # F-strings
        "JoinedStr", "FormattedValue",
        # Class
        "ClassDef",
        # Return/Yield
        "Return", "Yield", "YieldFrom",
        # Exception handling
        "Raise", "ExceptHandler",
        # Comprehensions
        "ListComp", "DictComp", "SetComp", "GeneratorExp"
    }
    
    def __init__(self):
        self.nodes: List[ASTNode] = []
        self.node_counter = 0
        self.source_lines: List[str] = []
        
    def parse(self, source_code: str) -> Tuple[List[ASTNode], ast.AST]:
        """
        Parse Python source code and return structured AST nodes.
        
        Args:
            source_code: Python source code string
            
        Returns:
            Tuple of (list of ASTNode, raw AST tree)
        """
        self.nodes = []
        self.node_counter = 0
        self.source_lines = source_code.split('\n')
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise ValueError(f"Invalid Python syntax: {e}")
        
        # Walk the tree and extract nodes
        self._walk_tree(tree, parent_id=None)
        
        return self.nodes, tree
    
    def _walk_tree(self, node: ast.AST, parent_id: Optional[int]) -> Optional[int]:
        """Recursively walk AST and build node list."""
        node_type = type(node).__name__
        
        # Get node position info
        lineno = getattr(node, 'lineno', 0)
        col_offset = getattr(node, 'col_offset', 0)
        end_lineno = getattr(node, 'end_lineno', lineno)
        end_col_offset = getattr(node, 'end_col_offset', col_offset)
        
        # Extract name if available
        name = self._extract_name(node)
        
        # Extract additional attributes
        attributes = self._extract_attributes(node)
        
        # Create node
        current_id = self.node_counter
        self.node_counter += 1
        
        ast_node = ASTNode(
            node_id=current_id,
            node_type=node_type,
            name=name,
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent_id=parent_id,
            children_ids=[],
            attributes=attributes
        )
        
        self.nodes.append(ast_node)
        
        # Process children
        for child in ast.iter_child_nodes(node):
            child_id = self._walk_tree(child, current_id)
            if child_id is not None:
                ast_node.children_ids.append(child_id)
        
        return current_id
    
    def _extract_name(self, node: ast.AST) -> Optional[str]:
        """Extract the name/identifier from a node if available."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            return node.name
        elif isinstance(node, ast.ClassDef):
            return node.name
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.alias):
            return node.name
        elif isinstance(node, ast.arg):
            return node.arg
        elif isinstance(node, ast.Constant):
            # Return string representation of constant
            if isinstance(node.value, str):
                return node.value[:50] if len(str(node.value)) > 50 else node.value
            return str(node.value)[:20]
        elif isinstance(node, ast.keyword):
            return node.arg
        return None
    
    def _extract_attributes(self, node: ast.AST) -> Dict[str, Any]:
        """Extract relevant attributes from a node."""
        attrs = {}
        
        if isinstance(node, ast.Call):
            # Extract function being called
            if isinstance(node.func, ast.Name):
                attrs["func_name"] = node.func.id
            elif isinstance(node.func, ast.Attribute):
                attrs["func_name"] = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    attrs["module"] = node.func.value.id
            attrs["num_args"] = len(node.args)
            attrs["num_kwargs"] = len(node.keywords)
            
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs["num_args"] = len(node.args.args)
            attrs["is_async"] = isinstance(node, ast.AsyncFunctionDef)
            attrs["decorators"] = [
                self._get_decorator_name(d) for d in node.decorator_list
            ]
            
        elif isinstance(node, ast.Import):
            attrs["modules"] = [alias.name for alias in node.names]
            
        elif isinstance(node, ast.ImportFrom):
            attrs["module"] = node.module
            attrs["names"] = [alias.name for alias in node.names]
            
        elif isinstance(node, ast.Assign):
            targets = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
            attrs["targets"] = targets
            
        elif isinstance(node, ast.Compare):
            attrs["ops"] = [type(op).__name__ for op in node.ops]
            
        elif isinstance(node, ast.BinOp):
            attrs["op"] = type(node.op).__name__
            
        return attrs
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return "unknown"
    
    def get_source_segment(self, lineno: int, end_lineno: int) -> str:
        """Get source code segment for given line range."""
        if lineno <= 0 or lineno > len(self.source_lines):
            return ""
        start_idx = lineno - 1
        end_idx = min(end_lineno, len(self.source_lines))
        return '\n'.join(self.source_lines[start_idx:end_idx])
    
    def get_call_chain(self, node: ast.AST) -> List[str]:
        """Extract the full call chain from a Call node."""
        chain = []
        current = node.func if isinstance(node, ast.Call) else node
        
        while current:
            if isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            elif isinstance(current, ast.Name):
                chain.append(current.id)
                break
            elif isinstance(current, ast.Call):
                current = current.func
            else:
                break
        
        return list(reversed(chain))
    
    def find_nodes_by_type(self, node_type: str) -> List[ASTNode]:
        """Find all nodes of a specific type."""
        return [n for n in self.nodes if n.node_type == node_type]
    
    def find_nodes_by_name(self, name: str) -> List[ASTNode]:
        """Find all nodes with a specific name."""
        return [n for n in self.nodes if n.name == name]
    
    def get_node_by_id(self, node_id: int) -> Optional[ASTNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
