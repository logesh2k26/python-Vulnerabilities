"""Training script for vulnerability detection model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.gnn_model import VulnerabilityGNN
from app.core.graph_builder import GraphBuilder
from app.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VulnerabilityDataset:
    """Dataset for training vulnerability detection model with pre-cached graphs."""
    
    def __init__(self, data_dir: Path, graph_builder: GraphBuilder):
        self.data_dir = data_dir
        self.graph_builder = graph_builder
        self.samples = []
        self._load_and_preprocess()
    
    def _load_and_preprocess(self):
        """Load and pre-build graphs for all samples."""
        label_map = {
            "safe": 0, "eval_exec": 1, "command_injection": 2,
            "unsafe_deserialization": 3, "hardcoded_secrets": 4,
            "sql_injection": 5, "path_traversal": 6, "ssrf": 7,
            "insecure_cryptography": 8, "xxe": 9, "redos": 10, "xss": 11
        }
        
        start_time = time.time()
        logger.info("Starting dataset preprocessing (graph construction)...")
        
        raw_files = []
        for label_name, label_id in label_map.items():
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                continue
            for py_file in label_dir.glob("*.py"):
                raw_files.append((py_file, label_id))
        
        total = len(raw_files)
        for i, (py_file, label_id) in enumerate(raw_files):
            try:
                source = py_file.read_text(encoding='utf-8')
                graph = self.graph_builder.build_graph(source)
                
                if graph.node_features.shape[0] > 0:
                    from torch_geometric.data import Data
                    data = Data(
                        x=torch.tensor(graph.node_features, dtype=torch.float32),
                        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                        y=torch.tensor([label_id], dtype=torch.long)
                    )
                    self.samples.append(data)
                
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    logger.info(f"  Processed {i + 1}/{total} samples...")
                    
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"Preprocessing complete. Loaded {len(self.samples)} valid graphs in {elapsed:.2f}s")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_model(
    data_dir: str,
    output_path: str,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the vulnerability detection model."""
    logger.info(f"Training on device: {device}")
    
    graph_builder = GraphBuilder(feature_dim=settings.EMBEDDING_DIM)
    dataset = VulnerabilityDataset(Path(data_dir), graph_builder)
    
    if len(dataset) == 0:
        logger.error("No valid training samples found!")
        return
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = VulnerabilityGNN(
        input_dim=settings.EMBEDDING_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        num_classes=settings.NUM_CLASSES
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            probs, _ = model(data.x, data.edge_index, batch=data.batch)
            loss = criterion(probs, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
            pred = probs.argmax(dim=1)
            train_correct += (pred == data.y).sum().item()
            train_total += data.num_graphs
        
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                probs, _ = model(data.x, data.edge_index, batch=data.batch)
                loss = criterion(probs, data.y)
                
                val_loss += loss.item() * data.num_graphs
                pred = probs.argmax(dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.num_graphs
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(avg_val_loss)
        
        # Metrics
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        logger.info(f"Epoch {epoch+1:02d}/{epochs} | "
                   f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info(f"  --> Saved best model (Val Loss: {best_val_loss:.4f})")
    
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
