"""Attention mapper for explainability."""
from typing import Dict, List, Tuple
import numpy as np
from app.core.ast_parser import ASTNode


class AttentionMapper:
    """Map attention weights to source code locations."""
    
    def __init__(self, nodes: List[ASTNode], attention_weights: np.ndarray):
        self.nodes = nodes
        self.attention = attention_weights
        self.node_scores = self._compute_node_scores()
    
    def _compute_node_scores(self) -> Dict[int, float]:
        scores = {}
        if len(self.attention) != len(self.nodes):
            return scores
        
        # Normalize attention
        attn_min = self.attention.min()
        attn_max = self.attention.max()
        if attn_max > attn_min:
            normalized = (self.attention - attn_min) / (attn_max - attn_min)
        else:
            normalized = np.zeros_like(self.attention)
        
        for i, node in enumerate(self.nodes):
            scores[node.node_id] = float(normalized[i])
        
        return scores
    
    def get_line_importance(self) -> Dict[int, float]:
        """Get importance score for each line."""
        line_scores: Dict[int, List[float]] = {}
        
        for node in self.nodes:
            if node.lineno not in line_scores:
                line_scores[node.lineno] = []
            
            score = self.node_scores.get(node.node_id, 0.0)
            line_scores[node.lineno].append(score)
            
            if node.end_lineno and node.end_lineno != node.lineno:
                for ln in range(node.lineno + 1, node.end_lineno + 1):
                    if ln not in line_scores:
                        line_scores[ln] = []
                    line_scores[ln].append(score * 0.5)
        
        return {ln: max(scores) for ln, scores in line_scores.items() if scores}
    
    def get_top_nodes(self, n: int = 10) -> List[Tuple[ASTNode, float]]:
        """Get the N most important nodes."""
        sorted_nodes = sorted(
            [(node, self.node_scores.get(node.node_id, 0.0)) for node in self.nodes],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_nodes[:n]
