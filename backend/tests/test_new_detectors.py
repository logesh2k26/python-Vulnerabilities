import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ast_parser import ASTParser
from app.detectors.cryptography import InsecureCryptographyDetector
from app.detectors.xxe import XXEDetector
from app.detectors.xss_detector import XSSDetector

class TestNewDetectors(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()
        self.crypto_detector = InsecureCryptographyDetector()
        self.xxe_detector = XXEDetector()
        self.xss_detector = XSSDetector()

    def test_insecure_cryptography_hashing(self):
        code = "import hashlib\nh = hashlib.md5(b'password')"
        nodes, _ = self.parser.parse(code)
        self.crypto_detector.set_source(code)
        results = self.crypto_detector.detect(nodes)
        self.assertTrue(any(r.vulnerability_type == "insecure_cryptography" for r in results))

    def test_insecure_cryptography_random(self):
        code = "import random\nx = random.randint(1, 10)"
        nodes, _ = self.parser.parse(code)
        self.crypto_detector.set_source(code)
        results = self.crypto_detector.detect(nodes)
        # Check metadata instead of string description
        self.assertTrue(any("randint" in str(r.metadata.get("function", "")) for r in results))

    def test_xxe_detection(self):
        code = "from lxml import etree\ntree = etree.fromstring('<xml>data</xml>')"
        nodes, _ = self.parser.parse(code)
        self.xxe_detector.set_source(code)
        results = self.xxe_detector.detect(nodes)
        self.assertTrue(any(r.vulnerability_type == "xxe" for r in results))

    def test_xss_detection(self):
        code = "from markupsafe import Markup\nhtml = Markup('<b>' + 'user' + '</b>')"
        nodes, _ = self.parser.parse(code)
        self.xss_detector.set_source(code)
        results = self.xss_detector.detect(nodes)
        self.assertTrue(any(r.vulnerability_type == "xss" for r in results))

if __name__ == "__main__":
    unittest.main()
