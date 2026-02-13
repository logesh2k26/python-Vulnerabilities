"""Taint Analysis Engine for tracking untrusted data flow."""
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from app.core.ast_parser import ASTNode, ASTParser
from app.core.flow_graphs import FlowGraphExtractor
import logging

logger = logging.getLogger(__name__)


class TaintSource(Enum):
    """Types of taint sources (untrusted input origins)."""
    USER_INPUT = "user_input"      # input(), raw_input()
    REQUEST_DATA = "request_data"  # request.args, request.form, request.data
    COMMAND_ARGS = "command_args"  # sys.argv, argparse
    FILE_INPUT = "file_input"      # file.read(), open().read()
    ENV_VARS = "env_vars"          # os.environ, os.getenv
    DATABASE = "database"          # cursor.fetchone(), query results
    NETWORK = "network"            # socket.recv(), requests.get()


class TaintSink(Enum):
    """Types of taint sinks (dangerous operations)."""
    CODE_EXEC = "code_exec"        # eval, exec, compile
    COMMAND_EXEC = "command_exec"  # os.system, subprocess
    FILE_ACCESS = "file_access"    # open, pathlib operations
    SQL_QUERY = "sql_query"        # cursor.execute
    DESERIALIZATION = "deser"      # pickle.loads, yaml.load
    NETWORK_REQUEST = "network"    # requests.get with user URL


@dataclass
class TaintedValue:
    """Represents a tainted value with its origin and propagation history."""
    variable_name: str
    source_type: TaintSource
    source_line: int
    source_node_id: int
    propagation_path: List[Tuple[int, str]] = field(default_factory=list)
    is_sanitized: bool = False
    sanitizer_line: Optional[int] = None
    sanitizer_type: Optional[str] = None
    
    def add_propagation(self, node_id: int, description: str):
        self.propagation_path.append((node_id, description))
    
    def mark_sanitized(self, line: int, sanitizer_type: str):
        self.is_sanitized = True
        self.sanitizer_line = line
        self.sanitizer_type = sanitizer_type


@dataclass 
class TaintPath:
    """Complete taint path from source to sink."""
    source: TaintedValue
    sink_type: TaintSink
    sink_line: int
    sink_node_id: int
    sink_function: str
    path_nodes: List[int]
    is_mitigated: bool
    mitigation_details: Optional[str] = None
    severity: str = "critical"  # critical, high, medium, low, info
    
    def to_dict(self) -> Dict:
        return {
            "source_variable": self.source.variable_name,
            "source_type": self.source.source_type.value,
            "source_line": self.source.source_line,
            "sink_type": self.sink_type.value,
            "sink_line": self.sink_line,
            "sink_function": self.sink_function,
            "propagation_path": [
                {"node_id": n, "description": d}
                for n, d in self.source.propagation_path
            ],
            "is_sanitized": self.source.is_sanitized,
            "sanitizer_line": self.source.sanitizer_line,
            "sanitizer_type": self.source.sanitizer_type,
            "is_mitigated": self.is_mitigated,
            "mitigation_details": self.mitigation_details,
            "severity": self.severity
        }


class TaintAnalyzer:
    """Performs taint analysis to track data flow from sources to sinks using DFG."""
    
    # Source definitions: function/attribute -> TaintSource type
    TAINT_SOURCES = {
        # User input
        "input": TaintSource.USER_INPUT,
        "raw_input": TaintSource.USER_INPUT,
        # Request data (Flask, Django, FastAPI)
        "request.args": TaintSource.REQUEST_DATA,
        "request.form": TaintSource.REQUEST_DATA,
        "request.data": TaintSource.REQUEST_DATA,
        "request.json": TaintSource.REQUEST_DATA,
        "request.get_json": TaintSource.REQUEST_DATA,
        "request.cookies": TaintSource.REQUEST_DATA,
        "request.headers": TaintSource.REQUEST_DATA,
        "request.query_params": TaintSource.REQUEST_DATA,
        # Command line arguments
        "sys.argv": TaintSource.COMMAND_ARGS,
        "argparse.parse_args": TaintSource.COMMAND_ARGS,
        # File input
        "read": TaintSource.FILE_INPUT,
        "readline": TaintSource.FILE_INPUT,
        "readlines": TaintSource.FILE_INPUT,
        # Environment
        "os.environ": TaintSource.ENV_VARS,
        "os.getenv": TaintSource.ENV_VARS,
    }
    
    # Sink definitions: function/attribute -> TaintSink type
    TAINT_SINKS = {
        # Code execution
        "eval": TaintSink.CODE_EXEC,
        "exec": TaintSink.CODE_EXEC,
        "compile": TaintSink.CODE_EXEC,
        # Command execution
        "os.system": TaintSink.COMMAND_EXEC,
        "os.popen": TaintSink.COMMAND_EXEC,
        "subprocess.call": TaintSink.COMMAND_EXEC,
        "subprocess.run": TaintSink.COMMAND_EXEC,
        "subprocess.Popen": TaintSink.COMMAND_EXEC,
        # SQL
        "execute": TaintSink.SQL_QUERY,
        "executemany": TaintSink.SQL_QUERY,
        "executescript": TaintSink.SQL_QUERY,
        # Deserialization
        "pickle.loads": TaintSink.DESERIALIZATION,
        "pickle.load": TaintSink.DESERIALIZATION,
        "yaml.load": TaintSink.DESERIALIZATION,
        "unsafe_load": TaintSink.DESERIALIZATION,
    }
    
    def __init__(self):
        self.tainted_vars: Dict[str, TaintedValue] = {}
        self.taint_paths: List[TaintPath] = []
        self.nodes: List[ASTNode] = []
        self.source_lines: List[str] = []
        self.flow_extractor = FlowGraphExtractor()
        self.flow_graphs = None
    
    def analyze(self, nodes: List[ASTNode], source_code: str) -> List[TaintPath]:
        """
        Perform taint analysis on AST nodes using Data Flow Graph.
        """
        self.nodes = nodes
        self.source_lines = source_code.split('\n')
        self.tainted_vars = {}
        self.taint_paths = []
        
        # Build CFG and DFG
        self.flow_graphs = self.flow_extractor.extract(nodes)
        
        # Build node lookup
        node_map = {n.node_id: n for n in nodes}
        
        # Phase 1: Identify Initial Taint Sources
        worklist = []  # List of TaintedValue
        
        for node in nodes:
            source_type = self._is_taint_source(node)
            if source_type:
                # Find the variable being assigned
                var_name = self._find_assigned_variable(node, nodes)
                if var_name:
                    taint = TaintedValue(
                        variable_name=var_name,
                        source_type=source_type,
                        source_line=node.lineno,
                        source_node_id=node.node_id
                    )
                    taint.add_propagation(node.node_id, f"Source: {source_type.value}")
                    self.tainted_vars[var_name] = taint
                    worklist.append(taint)
                    
        # Phase 2: Propagate Taint via DFG
        # We use a worklist algorithm to propagate taint through the DFG
        processed_defs = set()
        
        while worklist:
            current_taint = worklist.pop(0)
            
            # Find all uses of this tainted variable
            # We need to find the definition node for this variable at this line
            # Optimization: look at all def-use edges where variable matches
            
            # Get all uses of this variable from the DFG
            uses = self.flow_graphs.get_variable_flow(current_taint.variable_name)
            
            for edge in uses:
                # Only follow edges where definition is at or after the source line (simple flow)
                # In a real DFG, edges already encode valid flow
                
                # Check if this use enters a Sink
                use_node = node_map.get(edge.use_node)
                if not use_node:
                    continue
                
                # Check for Sink
                sink_type = self._check_node_is_sink_arg(use_node, nodes)
                if sink_type:
                    # Found a path to sink!
                    path = self._build_taint_path(current_taint, use_node, sink_type)
                    self.taint_paths.append(path)
                
                # Check for propagation (Assignment to new variable)
                # If the use node is part of an assignment RHS, tainted flows to LHS
                lhs_var = self._find_assigned_variable_from_rhs(use_node, nodes)
                if lhs_var and lhs_var != current_taint.variable_name:
                    # Propagate
                    new_taint = TaintedValue(
                        variable_name=lhs_var,
                        source_type=current_taint.source_type,
                        source_line=current_taint.source_line,
                        source_node_id=current_taint.source_node_id,
                        propagation_path=current_taint.propagation_path.copy()
                    )
                    new_taint.add_propagation(edge.use_node, f"Flow to {lhs_var} at line {edge.use_line}")
                    
                    # Avoid infinite loops
                    state_key = (lhs_var, edge.use_line)
                    if state_key not in processed_defs:
                        processed_defs.add(state_key)
                        worklist.append(new_taint)
                        self.tainted_vars[lhs_var] = new_taint

        return self.taint_paths

    def _is_taint_source(self, node: ASTNode) -> Optional[TaintSource]:
        """Check if a node is a taint source."""
        if node.node_type != "Call":
            return None
        
        func_name = node.attributes.get("func_name", "")
        module = node.attributes.get("module", "")
        # Full name check
        full_name = f"{module}.{func_name}" if module else func_name
        
        # Check standard sources
        if full_name in self.TAINT_SOURCES: return self.TAINT_SOURCES[full_name]
        if func_name in self.TAINT_SOURCES: return self.TAINT_SOURCES[func_name]
        
        # Special case: request.args.get() 
        # The parser might represent this as Call -> Attribute "get" -> Attribute "args" -> Name "request"
        # Since we have flattened attributes in some parser versions, let's check name
        if "request" in full_name and "get" in func_name:
            return TaintSource.REQUEST_DATA
            
        return None

    def _find_assigned_variable(self, source_node: ASTNode, nodes: List[ASTNode]) -> Optional[str]:
        """Find the variable assigned to the result of a node."""
        # Find the parent Assign node
        parent = next((n for n in nodes if n.node_id == source_node.parent_id), None)
        if parent and parent.node_type == "Assign":
            targets = parent.attributes.get("targets", [])
            if targets: return targets[0]
        return None

    def _find_assigned_variable_from_rhs(self, rhs_node: ASTNode, nodes: List[ASTNode]) -> Optional[str]:
        """If rhs_node is part of an expression assigned to a variable, return that variable."""
        # Walk up parents until we hit an Assign or stop
        curr = rhs_node
        while curr:
            if curr.node_type == "Assign":
                targets = curr.attributes.get("targets", [])
                if targets: return targets[0]
                return None
            
            # Stop at block boundaries
            if curr.node_type in ("FunctionDef", "ClassDef", "Module"):
                return None
                
            curr = next((n for n in nodes if n.node_id == curr.parent_id), None)
        return None

    def _check_node_is_sink_arg(self, node: ASTNode, nodes: List[ASTNode]) -> Optional[TaintSink]:
        """Check if a node is an argument to a sink function."""
        # Check parents to see if we are inside a Call to a Sink
        curr = node
        while curr:
            if curr.node_type == "Call":
                func_name = curr.attributes.get("func_name", "")
                module = curr.attributes.get("module", "")
                full_name = f"{module}.{func_name}" if module else func_name
                
                if full_name in self.TAINT_SINKS: return self.TAINT_SINKS[full_name]
                if func_name in self.TAINT_SINKS: return self.TAINT_SINKS[func_name]
                
                # Check for method calls checking (e.g. cursor.execute)
                if "execute" in func_name: return TaintSink.SQL_QUERY
                
                return None # Inside a call but not a sink
            
            # Stop at block boundaries
            if curr.node_type in ("FunctionDef", "ClassDef", "Module", "Assign"):
                return None
                
            curr = next((n for n in nodes if n.node_id == curr.parent_id), None)
        return None

    def _build_taint_path(self, tainted: TaintedValue, sink_node: ASTNode, sink_type: TaintSink) -> TaintPath:
        """Build path object."""
        # Find the specific line where the sink is called
        sink_call_line = sink_node.lineno
        
        # Flatten propagation path
        path_nodes = [tainted.source_node_id]
        path_nodes.extend([p[0] for p in tainted.propagation_path])
        path_nodes.append(sink_node.node_id)
        
        return TaintPath(
            source=tainted,
            sink_type=sink_type,
            sink_line=sink_call_line,
            sink_node_id=sink_node.node_id,
            sink_function=str(sink_type.value),
            path_nodes=path_nodes,
            is_mitigated=False,
            severity="critical"
        )

    def get_taint_summary(self) -> Dict:
        """Get summary of taint analysis."""
        return {
            "total_sources": len(self.tainted_vars),
            "total_paths": len(self.taint_paths),
            "paths": [p.to_dict() for p in self.taint_paths]
        }
