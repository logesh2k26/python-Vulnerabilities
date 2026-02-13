# Models module
from app.models.gnn_model import VulnerabilityGNN
from app.models.inference import VulnerabilityInference

__all__ = ["VulnerabilityGNN", "VulnerabilityInference"]
