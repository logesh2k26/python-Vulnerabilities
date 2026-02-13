"""Embedding Generator for semantic code embeddings."""
import ast
import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib
from app.core.ast_parser import ASTNode


class EmbeddingGenerator:
    """Generate semantic embeddings from AST nodes."""
    
    def __init__(self, embedding_dim: int = 128, max_paths: int = 200):
        self.embedding_dim = embedding_dim
        self.max_paths = max_paths
        self.token_vocab = self._init_vocab()
    
    def _init_vocab(self) -> Dict[str, int]:
        tokens = [
            "print", "len", "range", "str", "int", "float", "open", "input",
            "eval", "exec", "compile", "system", "popen", "subprocess",
            "pickle", "load", "loads", "yaml", "marshal", "request",
            "execute", "cursor", "connect", "password", "secret", "key", "token"
        ]
        return {t: i for i, t in enumerate(tokens)}
    
    def generate_node_embeddings(self, nodes: List[ASTNode]) -> np.ndarray:
        embeddings = np.zeros((len(nodes), self.embedding_dim), dtype=np.float32)
        for i, node in enumerate(nodes):
            embeddings[i] = self._encode_node(node)
        return embeddings
    
    def _encode_node(self, node: ASTNode) -> np.ndarray:
        emb = np.zeros(self.embedding_dim, dtype=np.float32)
        emb[:32] = self._encode_token(node.node_type)
        if node.name:
            emb[32:64] = self._encode_token(node.name)
        emb[64:96] = self._encode_attributes(node.attributes)
        emb[96:] = self._encode_position(node)
        return emb
    
    def _encode_token(self, token: Optional[str]) -> np.ndarray:
        emb = np.zeros(32, dtype=np.float32)
        if not token:
            return emb
        if token in self.token_vocab:
            emb[self.token_vocab[token] % 32] = 1.0
        else:
            h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            for i in range(4):
                emb[(h >> (i * 8)) % 32] += 0.25
        if isinstance(token, str):
            emb[0] += min(len(token) / 20.0, 1.0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    
    def _encode_attributes(self, attrs: Dict) -> np.ndarray:
        emb = np.zeros(32, dtype=np.float32)
        if "func_name" in attrs:
            emb[:16] = self._encode_token(attrs["func_name"])[:16]
        if "module" in attrs:
            emb[16:24] = self._encode_token(attrs["module"])[:8]
        emb[24] = attrs.get("num_args", 0) / 10.0
        emb[25] = attrs.get("num_kwargs", 0) / 10.0
        emb[26] = 1.0 if attrs.get("is_async") else 0.0
        return emb
    
    def _encode_position(self, node: ASTNode) -> np.ndarray:
        emb = np.zeros(32, dtype=np.float32)
        emb[0] = np.sin(node.lineno / 10.0)
        emb[1] = np.cos(node.lineno / 10.0)
        emb[2] = np.sin(node.col_offset / 10.0)
        emb[3] = np.cos(node.col_offset / 10.0)
        emb[4] = len(node.children_ids) / 10.0
        return emb
    
    def aggregate_embeddings(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        if method == "max":
            return embeddings.max(axis=0)
        return embeddings.mean(axis=0)
