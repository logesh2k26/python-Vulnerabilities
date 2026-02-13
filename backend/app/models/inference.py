"""Inference engine for vulnerability detection."""
import torch
from typing import List, Dict, Optional
import logging

from app.config import settings, DEVICE
from app.core.ast_parser import ASTParser
from app.core.graph_builder import GraphBuilder
from app.core.line_mapper import LineMapper
from app.core.taint_analyzer import TaintAnalyzer
from app.core.sanitizer_detector import SanitizerDetector
from app.core.severity_adjuster import SeverityAdjuster, AdjustedResult, VulnStatus
from app.models.gnn_model import VulnerabilityGNN
from app.detectors import (
    EvalExecDetector, CommandInjectionDetector,
    DeserializationDetector, HardcodedSecretsDetector, LogicFlawDetector
)
from app.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class VulnerabilityInference:
    """Main inference engine combining rule-based and ML detection."""
    
    def __init__(self):
        self.parser = ASTParser()
        self.graph_builder = GraphBuilder(feature_dim=settings.EMBEDDING_DIM)
        self.taint_analyzer = TaintAnalyzer()
        self.sanitizer_detector = SanitizerDetector()
        self.severity_adjuster = SeverityAdjuster()
        
        # Initialize model
        self.model = VulnerabilityGNN(
            input_dim=settings.EMBEDDING_DIM,
            hidden_dim=settings.HIDDEN_DIM,
            num_classes=settings.NUM_CLASSES,
            num_heads=settings.NUM_ATTENTION_HEADS,
            num_layers=settings.NUM_GNN_LAYERS,
            dropout=settings.DROPOUT
        ).to(DEVICE)
        
        # Load pretrained weights if available
        self._load_pretrained()
        self.model.eval()
        
        # Initialize detectors
        self.detectors = [
            EvalExecDetector(),
            CommandInjectionDetector(),
            DeserializationDetector(),
            HardcodedSecretsDetector(),
            LogicFlawDetector()
        ]
    
    def _load_pretrained(self):
        if settings.MODEL_PATH.exists():
            try:
                state_dict = torch.load(settings.MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded pretrained model weights")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}")
    
    def analyze(self, source_code: str, filename: str = "code.py") -> Dict:
        """Analyze Python source code for vulnerabilities."""
        try:
            nodes, tree = self.parser.parse(source_code)
        except ValueError as e:
            return {"error": str(e), "filename": filename}
        
        # Build graph
        graph = self.graph_builder.build_graph(source_code)
        
        # Line mapper
        line_mapper = LineMapper(source_code, nodes)
        
        # Run advanced analysis
        taint_paths = self.taint_analyzer.analyze(nodes, source_code)
        sanitizers = self.sanitizer_detector.detect(nodes, source_code)
        
        # Run rule-based detectors
        for detector in self.detectors:
            detector.set_source(source_code)
        
        rule_detections: List[DetectionResult] = []
        for detector in self.detectors:
            rule_detections.extend(detector.detect(nodes))
        
        # Adjust results based on mitigations
        adjusted_results = self.severity_adjuster.batch_adjust(
            rule_detections, sanitizers, taint_paths, source_code
        )
        
        # Run ML model
        ml_results = self._run_model(graph)
        
        # Combine results
        combined = self._combine_results(adjusted_results, ml_results, line_mapper)
        
        # Add highlighted lines
        node_importance = self._get_node_importance(graph)
        highlights = line_mapper.get_highlighted_lines(node_importance)
        
        is_vulnerable = len(combined["vulnerabilities"]) > 0
        overall_score = max(
            (v["confidence"] for v in combined["vulnerabilities"]),
            default=0.0
        )
        
        return {
            "filename": filename,
            "is_vulnerable": is_vulnerable,
            "overall_confidence": overall_score,
            "label": "VULNERABLE" if is_vulnerable else "SAFE",
            "vulnerabilities": combined["vulnerabilities"],
            "ml_predictions": ml_results,
            "highlighted_lines": [
                {"line": ln, "score": score, "content": content}
                for ln, score, content in highlights
            ],
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(graph.edges),
                "lines_of_code": len(source_code.split('\n'))
            }
        }
    
    def _run_model(self, graph) -> Dict:
        """Run GNN model on the graph."""
        if len(graph.nodes) == 0:
            return self._empty_predictions()
        
        with torch.no_grad():
            x = torch.tensor(graph.node_features, dtype=torch.float32).to(DEVICE)
            edge_index = torch.tensor(graph.edge_index, dtype=torch.long).to(DEVICE)
            
            probs, attention = self.model(x, edge_index, return_attention=True)
            probs = probs.squeeze().cpu().numpy()
        
        return {
            vuln_type: float(probs[i])
            for i, vuln_type in enumerate(settings.VULNERABILITY_TYPES)
        }
    
    def _empty_predictions(self) -> Dict:
        return {vuln_type: 0.0 for vuln_type in settings.VULNERABILITY_TYPES}
    
    def _get_node_importance(self, graph) -> Dict[int, float]:
        attention = self.model.get_node_importance()
        if attention is None:
            return {}
        
        attention = attention.cpu().numpy()
        importance = {}
        for i, node in enumerate(graph.nodes):
            if i < len(attention):
                importance[node.node_id] = float(attention[i])
        return importance
    
    def _combine_results(
        self,
        adjusted_results: List[AdjustedResult],
        ml_results: Dict,
        line_mapper: LineMapper
    ) -> Dict:
        vulnerabilities = []
        
        for adj in adjusted_results:
            # If fully mitigated, we might skip it or include it as info
            if adj.status == VulnStatus.SAFE:
                continue
                
            ml_confidence = ml_results.get(adj.original.vulnerability_type, 0.5)
            
            # Use adjusted confidence
            combined_confidence = (adj.adjusted_confidence * 0.7) + (ml_confidence * 0.3)
            
            if combined_confidence >= settings.CONFIDENCE_THRESHOLD:
                vulnerabilities.append({
                    "type": adj.original.vulnerability_type,
                    "confidence": round(combined_confidence, 4),
                    "severity": adj.adjusted_severity,
                    "description": adj.original.description,
                    "remediation": adj.original.remediation,
                    "affected_lines": adj.original.affected_lines,
                    "code_snippet": adj.original.code_snippet,
                    "status": adj.status.value,
                    "mitigations": [m.to_dict() for m in adj.mitigations_found],
                    "taint_path": adj.taint_path.to_dict() if adj.taint_path else None,
                    "metadata": {**adj.original.metadata, "adjustment_reason": adj.adjustment_reason}
                })
        
        # Add ML-only detections (only if high confidence and not already detected/mitigated)
        for vuln_type, score in ml_results.items():
            if vuln_type == "safe":
                continue
            if score >= 0.8:  # Higher threshold for ML-only
                already_detected = any(v["type"] == vuln_type for v in vulnerabilities)
                if not already_detected:
                    vulnerabilities.append({
                        "type": vuln_type,
                        "confidence": round(score, 4),
                        "severity": "medium",
                        "description": f"ML model detected potential {vuln_type}",
                        "remediation": "Review the flagged code sections",
                        "affected_lines": [],
                        "code_snippet": "",
                        "status": "unsafe",
                        "metadata": {"source": "ml_model"}
                    })
        
        vulnerabilities.sort(key=lambda v: v["confidence"], reverse=True)
        return {"vulnerabilities": vulnerabilities}
    
    def batch_analyze(self, files: List[Dict]) -> List[Dict]:
        """Analyze multiple files."""
        results = []
        for file_info in files:
            source = file_info.get("content", "")
            filename = file_info.get("filename", "unknown.py")
            results.append(self.analyze(source, filename))
        return results
