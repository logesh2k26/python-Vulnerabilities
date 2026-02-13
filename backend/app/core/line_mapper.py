"""Line Mapper for AST node to source line mapping."""
from typing import Dict, List, Tuple, Set
from app.core.ast_parser import ASTNode


class LineMapper:
    """Map AST nodes to source code lines for highlighting."""
    
    def __init__(self, source_code: str, nodes: List[ASTNode]):
        self.source_lines = source_code.split('\n')
        self.nodes = nodes
        self.line_to_nodes = self._build_line_mapping()
    
    def _build_line_mapping(self) -> Dict[int, List[ASTNode]]:
        mapping: Dict[int, List[ASTNode]] = {}
        for node in self.nodes:
            if node.lineno not in mapping:
                mapping[node.lineno] = []
            mapping[node.lineno].append(node)
            if node.end_lineno and node.end_lineno != node.lineno:
                for ln in range(node.lineno + 1, node.end_lineno + 1):
                    if ln not in mapping:
                        mapping[ln] = []
                    mapping[ln].append(node)
        return mapping
    
    def get_node_lines(self, node_id: int) -> List[int]:
        for node in self.nodes:
            if node.node_id == node_id:
                if node.end_lineno:
                    return list(range(node.lineno, node.end_lineno + 1))
                return [node.lineno]
        return []
    
    def get_nodes_at_line(self, line: int) -> List[ASTNode]:
        return self.line_to_nodes.get(line, [])
    
    def get_line_content(self, line: int) -> str:
        if 1 <= line <= len(self.source_lines):
            return self.source_lines[line - 1]
        return ""
    
    def get_highlighted_lines(
        self,
        node_importance: Dict[int, float],
        threshold: float = 0.3
    ) -> List[Tuple[int, float, str]]:
        line_scores: Dict[int, float] = {}
        for node_id, score in node_importance.items():
            lines = self.get_node_lines(node_id)
            for ln in lines:
                line_scores[ln] = max(line_scores.get(ln, 0), score)
        
        result = []
        for ln, score in sorted(line_scores.items()):
            if score >= threshold:
                result.append((ln, score, self.get_line_content(ln)))
        return result
    
    def get_vulnerable_spans(
        self,
        node_ids: List[int]
    ) -> List[Dict]:
        spans = []
        for node_id in node_ids:
            for node in self.nodes:
                if node.node_id == node_id:
                    spans.append({
                        "start_line": node.lineno,
                        "end_line": node.end_lineno or node.lineno,
                        "start_col": node.col_offset,
                        "end_col": node.end_col_offset or node.col_offset,
                        "node_type": node.node_type,
                        "name": node.name
                    })
        return spans
