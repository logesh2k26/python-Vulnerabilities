import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch_scatter import scatter_softmax, scatter_add



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
        
        # Compute attention coefficients (sparse)
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]
        edge_features = torch.cat([Wh_src, Wh_dst], dim=1)
        e = self.leakyrelu(torch.matmul(edge_features, self.a).squeeze(-1))
        
        # Sparse Softmax over neighbors (dst is the target)
        alpha = scatter_softmax(e, dst, dim=0)
        alpha = self.dropout_layer(alpha)
        
        # Aggregate: h_prime[i] = sum_{j in neighbors(i)} alpha_{ji} * Wh[j]
        # Here src are the neighbors, dst is the node being updated
        h_prime = scatter_add(Wh[src] * alpha.view(-1, 1), dst, dim=0, dim_size=N)
        
        # Node importance (sum of incoming attention weights)
        node_importance = scatter_add(alpha, dst, dim=0, dim_size=N)
        
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
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
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
        
        # Global attention pooling (per graph in batch)
        # 1. Compute attention weights per node
        node_attn_scores = self.pool_attn(h)
        # 2. Softmax within each graph in the batch
        gate = scatter_softmax(node_attn_scores, batch, dim=0)
        # 3. Weighted sum of node features per graph
        graph_emb = scatter_add(gate * h, batch, dim=0)
        
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
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass  # Use random init
    
    return model.to(device)
