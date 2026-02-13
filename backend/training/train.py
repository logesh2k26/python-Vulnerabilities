"""Training script for vulnerability detection model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.gnn_model import VulnerabilityGNN
from app.core.graph_builder import GraphBuilder
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VulnerabilityDataset:
    """Dataset for training vulnerability detection model."""
    
    def __init__(self, data_dir: Path, graph_builder: GraphBuilder):
        self.data_dir = data_dir
        self.graph_builder = graph_builder
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load samples from labeled directories."""
        label_map = {
            "safe": 0, "eval_exec": 1, "command_injection": 2,
            "unsafe_deserialization": 3, "hardcoded_secrets": 4,
            "sql_injection": 5, "path_traversal": 6
        }
        
        for label_name, label_id in label_map.items():
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                continue
            
            for py_file in label_dir.glob("*.py"):
                try:
                    source = py_file.read_text(encoding='utf-8')
                    self.samples.append({
                        "source": source,
                        "label": label_id,
                        "filename": py_file.name
                    })
                except Exception as e:
                    logger.warning(f"Error loading {py_file}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = self.graph_builder.build_graph(sample["source"])
        
        return {
            "node_features": torch.tensor(graph.node_features, dtype=torch.float32),
            "edge_index": torch.tensor(graph.edge_index, dtype=torch.long),
            "label": sample["label"]
        }


def collate_fn(batch):
    """Collate function for batching graphs."""
    # For simplicity, process one graph at a time
    return batch[0] if batch else None


def train_model(
    data_dir: str,
    output_path: str,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the vulnerability detection model."""
    logger.info(f"Training on device: {device}")
    
    graph_builder = GraphBuilder(feature_dim=settings.EMBEDDING_DIM)
    dataset = VulnerabilityDataset(Path(data_dir), graph_builder)
    
    if len(dataset) == 0:
        logger.error("No training samples found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    model = VulnerabilityGNN(
        input_dim=settings.EMBEDDING_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        num_classes=settings.NUM_CLASSES
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Accumulate predictions and labels for metrics
        all_preds = []
        all_labels = []

        for batch in dataloader:
            if batch is None:
                continue
            
            x = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            label = torch.tensor([batch["label"]], dtype=torch.long).to(device)
            
            if x.size(0) == 0:
                continue
            
            optimizer.zero_grad()
            probs, _ = model(x, edge_index)
            loss = criterion(probs, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = probs.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += 1
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        
        scheduler.step()
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} - F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
            logger.info(f"Saved best model to {output_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to training data")
    parser.add_argument("--output", default="pretrained/vulnerability_gnn.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output, args.epochs, args.lr)
