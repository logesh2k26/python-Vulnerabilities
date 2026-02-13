"""Severity Adjustment module for reducing false positives."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from app.core.taint_analyzer import TaintPath, TaintSink
from app.core.sanitizer_detector import SanitizerMatch, SanitizerType
from app.detectors.base import DetectionResult
import logging

logger = logging.getLogger(__name__)


class VulnStatus(Enum):
    """Classification of vulnerability status."""
    UNSAFE = "unsafe"           # No mitigation detected
    MITIGATED = "mitigated"     # Partial mitigation found
    SAFE = "safe"               # Properly mitigated


@dataclass
class AdjustedResult:
    """Detection result with severity adjustment."""
    original: DetectionResult
    adjusted_severity: str
    adjusted_confidence: float
    status: VulnStatus
    mitigations_found: List[SanitizerMatch]
    taint_path: Optional[TaintPath]
    adjustment_reason: str
    
    def to_dict(self) -> Dict:
        return {
            "vulnerability_type": self.original.vulnerability_type,
            "original_severity": self.original.severity,
            "adjusted_severity": self.adjusted_severity,
            "original_confidence": self.original.confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "status": self.status.value,
            "mitigations": [m.to_dict() for m in self.mitigations_found],
            "taint_path": self.taint_path.to_dict() if self.taint_path else None,
            "adjustment_reason": self.adjustment_reason,
            "affected_lines": self.original.affected_lines,
            "description": self.original.description,
            "remediation": self.original.remediation
        }


class SeverityAdjuster:
    """Adjusts vulnerability severity based on detected mitigations."""
    
    # Severity levels in order
    SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]
    
    # Vulnerability type to relevant sanitizer types mapping
    VULN_SANITIZER_MAP = {
        "path_traversal": [
            SanitizerType.PATH_NORMALIZATION,
            SanitizerType.PATH_BASENAME,
            SanitizerType.PATH_PREFIX_CHECK,
        ],
        "command_injection": [
            SanitizerType.SHELL_FALSE,
            SanitizerType.COMMAND_WHITELIST,
            SanitizerType.ESCAPE_FUNCTION,
            SanitizerType.INPUT_VALIDATION,
        ],
        "eval_exec": [
            SanitizerType.LITERAL_EVAL,
            SanitizerType.COMMAND_WHITELIST,
            SanitizerType.INPUT_VALIDATION,
            SanitizerType.REGEX_VALIDATION,
            SanitizerType.TYPE_CHECK,
        ],
        "sql_injection": [
            SanitizerType.PARAMETERIZED_QUERY,
            SanitizerType.ESCAPE_FUNCTION,
        ],
        "unsafe_deserialization": [
            SanitizerType.SAFE_YAML_LOADER,
        ],
        "ssrf": [
            SanitizerType.INPUT_VALIDATION,
            SanitizerType.COMMAND_WHITELIST,
            SanitizerType.REGEX_VALIDATION,
        ],
    }
    
    # Mitigation effectiveness thresholds
    FULL_MITIGATION_THRESHOLD = 0.85
    PARTIAL_MITIGATION_THRESHOLD = 0.5
    
    def __init__(self):
        self.adjustments_made: List[str] = []
    
    def adjust(
        self,
        detection: DetectionResult,
        sanitizers: List[SanitizerMatch],
        taint_paths: List[TaintPath],
        context_lines: Dict[int, str]
    ) -> AdjustedResult:
        """
        Adjust severity of a detection based on mitigations.
        
        Args:
            detection: Original detection result
            sanitizers: List of detected sanitizers
            taint_paths: List of taint paths
            context_lines: Line number to source code mapping
            
        Returns:
            AdjustedResult with potentially modified severity
        """
        vuln_type = detection.vulnerability_type
        affected_lines = set(detection.affected_lines)
        
        # Find relevant sanitizers
        relevant_sanitizers = self._find_relevant_sanitizers(
            vuln_type, sanitizers, affected_lines
        )
        
        # Find matching taint path
        matching_path = self._find_matching_taint_path(
            taint_paths, affected_lines
        )
        
        # Calculate mitigation score
        mitigation_score = self._calculate_mitigation_score(
            relevant_sanitizers, matching_path, vuln_type
        )
        
        # Apply contextual analysis
        context_adjustment = self._analyze_context(
            vuln_type, affected_lines, context_lines
        )
        
        total_mitigation = min(1.0, mitigation_score + context_adjustment)
        
        # Determine new status and severity
        status, new_severity, reason = self._determine_status(
            detection.severity, total_mitigation, relevant_sanitizers
        )
        
        # Adjust confidence
        new_confidence = self._adjust_confidence(
            detection.confidence, total_mitigation, status
        )
        
        return AdjustedResult(
            original=detection,
            adjusted_severity=new_severity,
            adjusted_confidence=new_confidence,
            status=status,
            mitigations_found=relevant_sanitizers,
            taint_path=matching_path,
            adjustment_reason=reason
        )
    
    def _find_relevant_sanitizers(
        self,
        vuln_type: str,
        sanitizers: List[SanitizerMatch],
        affected_lines: set
    ) -> List[SanitizerMatch]:
        """Find sanitizers relevant to this vulnerability type and location."""
        relevant_types = self.VULN_SANITIZER_MAP.get(vuln_type, [])
        
        relevant = []
        for san in sanitizers:
            # Check if sanitizer type is relevant
            if san.sanitizer_type not in relevant_types:
                continue
            
            # Check if sanitizer is near the affected lines (within 20 lines before)
            for line in affected_lines:
                if san.line <= line <= san.line + 20:
                    # Sanitizer comes before or at the vulnerable line
                    if san.line <= line:
                        relevant.append(san)
                        break
        
        return relevant
    
    def _find_matching_taint_path(
        self,
        taint_paths: List[TaintPath],
        affected_lines: set
    ) -> Optional[TaintPath]:
        """Find taint path that matches the affected lines."""
        for path in taint_paths:
            if path.sink_line in affected_lines:
                return path
        return None
    
    def _calculate_mitigation_score(
        self,
        sanitizers: List[SanitizerMatch],
        taint_path: Optional[TaintPath],
        vuln_type: str
    ) -> float:
        """Calculate overall mitigation effectiveness score."""
        if not sanitizers:
            return 0.0
        
        # Take the maximum effectiveness from relevant sanitizers
        max_effectiveness = max(s.effectiveness for s in sanitizers)
        
        # Bonus for multiple mitigations (defense in depth)
        depth_bonus = min(0.1 * (len(sanitizers) - 1), 0.2)
        
        # Check if taint path shows sanitization
        path_bonus = 0.0
        if taint_path and taint_path.is_mitigated:
            path_bonus = 0.15
        
        return min(1.0, max_effectiveness + depth_bonus + path_bonus)
    
    def _analyze_context(
        self,
        vuln_type: str,
        affected_lines: set,
        context_lines: Dict[int, str]
    ) -> float:
        """Analyze surrounding code context for additional mitigations."""
        adjustment = 0.0
        
        for line_num in affected_lines:
            # Check lines before the vulnerable line
            for offset in range(-5, 0):
                check_line = line_num + offset
                if check_line in context_lines:
                    line = context_lines[check_line].lower()
                    
                    # Path traversal context checks
                    if vuln_type == "path_traversal":
                        if "if not" in line and ("startswith" in line or "in " in line):
                            adjustment += 0.1
                        if "raise" in line and "error" in line:
                            adjustment += 0.05
                    
                    # Command injection context checks
                    if vuln_type == "command_injection":
                        if "whitelist" in line or "allowed" in line:
                            adjustment += 0.1
                        if "if " in line and " not in " in line:
                            adjustment += 0.05
                    
                    # Eval context checks
                    if vuln_type == "eval_exec":
                        if "literal_eval" in line:
                            adjustment += 0.3
                        if "isdigit" in line or "isalnum" in line:
                            adjustment += 0.1
        
        return min(0.3, adjustment)  # Cap context adjustment
    
    def _determine_status(
        self,
        original_severity: str,
        mitigation_score: float,
        sanitizers: List[SanitizerMatch]
    ) -> Tuple[VulnStatus, str, str]:
        """Determine vulnerability status and adjusted severity."""
        if mitigation_score >= self.FULL_MITIGATION_THRESHOLD:
            # Fully mitigated
            status = VulnStatus.SAFE
            new_severity = "info"
            reason = self._build_mitigation_reason(sanitizers, "fully mitigated")
            
        elif mitigation_score >= self.PARTIAL_MITIGATION_THRESHOLD:
            # Partially mitigated
            status = VulnStatus.MITIGATED
            new_severity = self._reduce_severity(original_severity, 2)
            reason = self._build_mitigation_reason(sanitizers, "partially mitigated")
            
        elif mitigation_score > 0:
            # Some mitigation but not enough
            status = VulnStatus.MITIGATED
            new_severity = self._reduce_severity(original_severity, 1)
            reason = f"Weak mitigation detected (score: {mitigation_score:.0%})"
            
        else:
            # No mitigation
            status = VulnStatus.UNSAFE
            new_severity = original_severity
            reason = "No mitigation detected"
        
        return status, new_severity, reason
    
    def _reduce_severity(self, severity: str, levels: int) -> str:
        """Reduce severity by N levels."""
        if severity not in self.SEVERITY_ORDER:
            return severity
        
        current_idx = self.SEVERITY_ORDER.index(severity)
        new_idx = max(0, current_idx - levels)
        return self.SEVERITY_ORDER[new_idx]
    
    def _build_mitigation_reason(self, sanitizers: List[SanitizerMatch], status: str) -> str:
        """Build human-readable mitigation reason."""
        if not sanitizers:
            return status
        
        sanitizer_types = set(s.sanitizer_type.value for s in sanitizers)
        types_str = ", ".join(sanitizer_types)
        return f"{status.capitalize()} by: {types_str}"
    
    def _adjust_confidence(
        self,
        original_confidence: float,
        mitigation_score: float,
        status: VulnStatus
    ) -> float:
        """Adjust confidence based on mitigation status."""
        if status == VulnStatus.SAFE:
            # Very low confidence for safe detections
            return max(0.1, original_confidence * (1 - mitigation_score))
        
        elif status == VulnStatus.MITIGATED:
            # Reduced confidence
            return original_confidence * (1 - mitigation_score * 0.5)
        
        else:
            # Unsafe - keep or slightly increase confidence
            return original_confidence
    
    def batch_adjust(
        self,
        detections: List[DetectionResult],
        sanitizers: List[SanitizerMatch],
        taint_paths: List[TaintPath],
        source_code: str
    ) -> List[AdjustedResult]:
        """Adjust all detections in batch."""
        lines = source_code.split('\n')
        context_lines = {i + 1: line for i, line in enumerate(lines)}
        
        results = []
        for detection in detections:
            adjusted = self.adjust(detection, sanitizers, taint_paths, context_lines)
            results.append(adjusted)
        
        return results
