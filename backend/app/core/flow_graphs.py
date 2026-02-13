"""Control Flow and Data Flow Graph extraction."""
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from app.core.ast_parser import ASTNode
import logging

logger = logging.getLogger(__name__)


@dataclass
class CFGNode:
    """Node in the Control Flow Graph."""
    node_id: int
    node_type: str
    lineno: int
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    is_branch: bool = False
    is_loop: bool = False
    is_entry: bool = False
    is_exit: bool = False


@dataclass  
class DFGEdge:
    """Edge in the Data Flow Graph (def-use chain)."""
    def_node: int      # Node where variable is defined
    use_node: int      # Node where variable is used
    variable: str      # Variable name
    def_line: int
    use_line: int


@dataclass
class FlowGraphs:
    """Container for CFG and DFG."""
    cfg_nodes: Dict[int, CFGNode]
    cfg_edges: List[Tuple[int, int]]  # (from, to)
    dfg_edges: List[DFGEdge]
    entry_node: Optional[int]
    exit_nodes: List[int]
    
    def get_paths_between(self, start: int, end: int, max_paths: int = 10) -> List[List[int]]:
        """Find paths between two nodes in CFG."""
        paths = []
        visited = set()
        
        def dfs(current: int, path: List[int]):
            if len(paths) >= max_paths:
                return
            if current == end:
                paths.append(path.copy())
                return
            if current in visited:
                return
            
            visited.add(current)
            if current in self.cfg_nodes:
                for succ in self.cfg_nodes[current].successors:
                    dfs(succ, path + [succ])
            visited.remove(current)
        
        dfs(start, [start])
        return paths
    
    def get_reaching_definitions(self, node_id: int, variable: str) -> List[DFGEdge]:
        """Get all definitions that reach a given node for a variable."""
        return [e for e in self.dfg_edges if e.use_node == node_id and e.variable == variable]
    
    def get_uses_of_definition(self, def_node: int, variable: str) -> List[DFGEdge]:
        """Get all uses of a variable defined at a given node."""
        return [e for e in self.dfg_edges if e.def_node == def_node and e.variable == variable]

    def get_variable_flow(self, variable: str) -> List[DFGEdge]:
        """Get all data flow edges for a specific variable."""
        return [e for e in self.dfg_edges if e.variable == variable]


class FlowGraphExtractor:
    """Extract Control Flow Graph and Data Flow Graph from AST."""
    
    BRANCH_NODES = {"If", "IfExp"}
    LOOP_NODES = {"For", "While", "AsyncFor"}
    EXIT_NODES = {"Return", "Raise", "Break", "Continue"}
    
    def __init__(self):
        self.cfg_nodes: Dict[int, CFGNode] = {}
        self.cfg_edges: List[Tuple[int, int]] = []
        self.dfg_edges: List[DFGEdge] = []
        self.definitions: Dict[str, List[Tuple[int, int]]] = {}  # var -> [(node_id, line)]
    
    def extract(self, nodes: List[ASTNode]) -> FlowGraphs:
        """
        Extract CFG and DFG from AST nodes.
        
        Args:
            nodes: List of ASTNode from parser
            
        Returns:
            FlowGraphs containing CFG and DFG
        """
        self.cfg_nodes = {}
        self.cfg_edges = []
        self.dfg_edges = []
        self.definitions = {}
        
        if not nodes:
            return FlowGraphs({}, [], [], None, [])
        
        # Build CFG
        self._build_cfg(nodes)
        
        # Build DFG
        self._build_dfg(nodes)
        
        # Find entry and exit nodes
        entry = self._find_entry_node(nodes)
        exits = self._find_exit_nodes()
        
        return FlowGraphs(
            cfg_nodes=self.cfg_nodes,
            cfg_edges=self.cfg_edges,
            dfg_edges=self.dfg_edges,
            entry_node=entry,
            exit_nodes=exits
        )
    
    def _build_cfg(self, nodes: List[ASTNode]):
        """Build Control Flow Graph."""
        # Create CFG nodes for statement-level AST nodes
        statement_types = {
            "FunctionDef", "AsyncFunctionDef", "ClassDef",
            "Assign", "AugAssign", "AnnAssign",
            "For", "AsyncFor", "While", "If",
            "With", "AsyncWith", "Try",
            "Return", "Yield", "YieldFrom",
            "Raise", "Assert", "Expr",
            "Pass", "Break", "Continue"
        }
        
        # First pass: create CFG nodes
        for node in nodes:
            if node.node_type in statement_types:
                is_branch = node.node_type in self.BRANCH_NODES
                is_loop = node.node_type in self.LOOP_NODES
                
                self.cfg_nodes[node.node_id] = CFGNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    lineno=node.lineno,
                    is_branch=is_branch,
                    is_loop=is_loop
                )
        
        # Second pass: connect nodes based on control flow
        sorted_nodes = sorted(
            [n for n in nodes if n.node_id in self.cfg_nodes],
            key=lambda n: (n.lineno, n.col_offset)
        )
        
        for i, node in enumerate(sorted_nodes[:-1]):
            next_node = sorted_nodes[i + 1]
            cfg_node = self.cfg_nodes[node.node_id]
            
            # Skip exit nodes
            if node.node_type in self.EXIT_NODES:
                continue
            
            # Handle branches
            if node.node_type == "If":
                # Connect to next node (can be then/else or fall-through)
                cfg_node.successors.append(next_node.node_id)
                self.cfg_edges.append((node.node_id, next_node.node_id))
            elif node.node_type in self.LOOP_NODES:
                # Loop connects to body and to node after loop
                cfg_node.successors.append(next_node.node_id)
                self.cfg_edges.append((node.node_id, next_node.node_id))
                # Also back-edge at end of loop (simplified)
            else:
                # Sequential connection
                cfg_node.successors.append(next_node.node_id)
                self.cfg_nodes[next_node.node_id].predecessors.append(node.node_id)
                self.cfg_edges.append((node.node_id, next_node.node_id))
    
    def _build_dfg(self, nodes: List[ASTNode]):
        """Build Data Flow Graph with def-use chains."""
        # First pass: find all definitions
        for node in nodes:
            if node.node_type == "Assign":
                targets = node.attributes.get("targets", [])
                for target in targets:
                    if target not in self.definitions:
                        self.definitions[target] = []
                    self.definitions[target].append((node.node_id, node.lineno))
            
            elif node.node_type in ("FunctionDef", "AsyncFunctionDef"):
                if node.name:
                    if node.name not in self.definitions:
                        self.definitions[node.name] = []
                    self.definitions[node.name].append((node.node_id, node.lineno))
            
            elif node.node_type == "For":
                # Loop variable is defined
                for child in nodes:
                    if child.parent_id == node.node_id and child.node_type == "Name":
                        var = child.name
                        if var:
                            if var not in self.definitions:
                                self.definitions[var] = []
                            self.definitions[var].append((node.node_id, node.lineno))
                        break
        
        # Second pass: find all uses and create def-use edges
        for node in nodes:
            if node.node_type == "Name" and node.name:
                var = node.name
                if var in self.definitions:
                    # Find the most recent definition before this use
                    for def_id, def_line in reversed(self.definitions[var]):
                        if def_line <= node.lineno:
                            self.dfg_edges.append(DFGEdge(
                                def_node=def_id,
                                use_node=node.node_id,
                                variable=var,
                                def_line=def_line,
                                use_line=node.lineno
                            ))
                            break
    
    def _find_entry_node(self, nodes: List[ASTNode]) -> Optional[int]:
        """Find the entry node of the CFG."""
        if not self.cfg_nodes:
            return None
        
        # Find nodes with no predecessors
        for node_id, cfg_node in self.cfg_nodes.items():
            if not cfg_node.predecessors:
                cfg_node.is_entry = True
                return node_id
        
        # Fallback: first node
        return min(self.cfg_nodes.keys())
    
    def _find_exit_nodes(self) -> List[int]:
        """Find exit nodes of the CFG."""
        exits = []
        for node_id, cfg_node in self.cfg_nodes.items():
            if not cfg_node.successors or cfg_node.node_type in self.EXIT_NODES:
                cfg_node.is_exit = True
                exits.append(node_id)
        return exits
    
    def get_control_flow_between(self, start_line: int, end_line: int) -> List[int]:
        """Get nodes in the control flow path between two lines."""
        nodes_in_range = []
        for node_id, cfg_node in self.cfg_nodes.items():
            if start_line <= cfg_node.lineno <= end_line:
                nodes_in_range.append(node_id)
        return sorted(nodes_in_range, key=lambda n: self.cfg_nodes[n].lineno)
