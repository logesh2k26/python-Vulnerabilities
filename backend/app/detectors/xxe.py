"""XXE (XML External Entity) detector."""
from typing import List
from app.core.ast_parser import ASTNode
from app.detectors.base import BaseDetector, DetectionResult


class XXEDetector(BaseDetector):
    """Detects insecure XML parsing that allows External Entity expansion."""

    name = "xxe"
    description = "Detects insecure XML entity expansion (XXE)"

    XML_PARSERS = {
        "lxml.etree": ["parse", "fromstring", "XMLParser"],
        "xml.dom.minidom": ["parse", "parseString"],
        "xml.sax": ["make_parser", "parse", "parseString"],
        "xml.etree.ElementTree": ["parse", "fromstring", "XML"],
    }

    def detect(self, nodes: List[ASTNode]) -> List[DetectionResult]:
        results = []
        for node in nodes:
            if node.node_type != "Call":
                continue
            
            func_name = node.attributes.get("func_name", "")
            module = node.attributes.get("module", "")
            
            full_name = f"{module}.{func_name}" if module else func_name
            
            is_xml_call = False
            for m, funcs in self.XML_PARSERS.items():
                short_m = m.split('.')[-1]
                if (module == m or module == short_m or func_name == m) and any(f in func_name for f in funcs):
                    is_xml_call = True
                    break
            
            if is_xml_call:
                # Basic check: any XML parsing is flagged unless we find explicit safety config
                # In a more advanced version, we'd check for 'resolve_entities=False' in lxml
                results.append(DetectionResult(
                    vulnerability_type="xxe",
                    confidence=0.6,
                    affected_lines=[node.lineno],
                    affected_nodes=[node.node_id],
                    description=f"Potentially insecure XML parsing with {full_name}",
                    severity="high",
                    remediation="Disable DTDs and external entity resolution in your XML parser",
                    code_snippet=self.get_code_snippet(node.lineno, node.end_lineno or node.lineno),
                    metadata={"parser": full_name}
                ))
        return results
