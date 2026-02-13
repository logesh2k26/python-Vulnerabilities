# Core module for AST processing and analysis
from app.core.ast_parser import ASTParser
from app.core.graph_builder import GraphBuilder
from app.core.embeddings import EmbeddingGenerator
from app.core.line_mapper import LineMapper
from app.core.taint_analyzer import TaintAnalyzer, TaintPath, TaintSource, TaintSink
from app.core.sanitizer_detector import SanitizerDetector, SanitizerMatch, SanitizerType
from app.core.flow_graphs import FlowGraphExtractor, FlowGraphs
from app.core.severity_adjuster import SeverityAdjuster, AdjustedResult, VulnStatus

__all__ = [
    "ASTParser", "GraphBuilder", "EmbeddingGenerator", "LineMapper",
    "TaintAnalyzer", "TaintPath", "TaintSource", "TaintSink",
    "SanitizerDetector", "SanitizerMatch", "SanitizerType",
    "FlowGraphExtractor", "FlowGraphs",
    "SeverityAdjuster", "AdjustedResult", "VulnStatus"
]
