"""Graph Neural Network model for vulnerability detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Wh = self.W(x)
        N = Wh.size(0)
        
        if edge_index.size(1) == 0:
            return Wh, torch.zeros(N, device=x.device)
        
        src, dst = edge_index[0], edge_index[1]
        
        # Compute attention coefficients
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]
        edge_features = torch.cat([Wh_src, Wh_dst], dim=1)
        e = self.leakyrelu(torch.matmul(edge_features, self.a).squeeze(-1))
        
        # Softmax over neighbors
        attention = torch.zeros(N, N, device=x.device)
        attention[src, dst] = e
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Node importance (sum of incoming attention)
        node_importance = attention.sum(dim=0)
        
        # Aggregate
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime), node_importance


class VulnerabilityGNN(nn.Module):
    """Graph Neural Network for vulnerability classification."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.attention_layers.append(
                nn.ModuleList([
                    GraphAttentionLayer(in_dim, hidden_dim, dropout)
                    for _ in range(num_heads)
                ])
            )
        
        # Global pooling attention
        self.pool_attn = nn.Linear(hidden_dim * num_heads, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Store attention for explainability
        self.node_attention_weights: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input projection
        h = F.relu(self.input_proj(x))
        
        # Accumulate node importance
        total_importance = torch.zeros(h.size(0), device=h.device)
        
        # Apply attention layers
        for layer_heads in self.attention_layers:
            head_outputs = []
            for head in layer_heads:
                head_out, importance = head(h, edge_index)
                head_outputs.append(head_out)
                total_importance += importance
            h = torch.cat(head_outputs, dim=-1)
        
        # Store for explainability
        self.node_attention_weights = total_importance / (self.num_layers * self.num_heads)
        
        # Global attention pooling
        attn_weights = F.softmax(self.pool_attn(h), dim=0)
        graph_emb = (attn_weights * h).sum(dim=0, keepdim=True)
        
        # Classification
        logits = self.classifier(graph_emb)
        probs = F.softmax(logits, dim=-1)
        
        if return_attention:
            return probs, self.node_attention_weights
        return probs, None
    
    def get_node_importance(self) -> Optional[torch.Tensor]:
        return self.node_attention_weights


def create_model(device: torch.device, pretrained_path: str = None) -> VulnerabilityGNN:
    """Create and optionally load pretrained model."""
    model = VulnerabilityGNN()
    
    if pretrained_path:
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass  # Use random init
    
    return model.to(device)
