import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import sys
import argparse
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from app.models.gnn_model import VulnerabilityGNN
from app.core.graph_builder import GraphBuilder
from app.config import settings
from training.train import VulnerabilityDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, data_dir: str):
    """Evaluate a trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating on device: {device}")
    
    # Load Data
    graph_builder = GraphBuilder(feature_dim=settings.EMBEDDING_DIM)
    dataset = VulnerabilityDataset(Path(data_dir), graph_builder)
    
    if len(dataset) == 0:
        logger.error("No samples found for evaluation!")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    model = VulnerabilityGNN(
        input_dim=settings.EMBEDDING_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        num_classes=settings.NUM_CLASSES
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            x = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            label = batch["label"]
            
            if x.size(0) == 0:
                continue
            
            probs, _ = model(x, edge_index)
            pred = probs.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
            
    # Calculate Metrics
    label_map = {
        0: "safe", 1: "eval_exec", 2: "command_injection",
        3: "unsafe_deserialization", 4: "hardcoded_secrets",
        5: "sql_injection", 6: "path_traversal"
    }
    
    target_names = [label_map.get(i, str(i)) for i in sorted(list(set(all_labels) | set(all_preds)))]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--data-dir", required=True, help="Path to evaluation data")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir)
