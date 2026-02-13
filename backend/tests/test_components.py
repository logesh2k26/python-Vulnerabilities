
import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ast_parser import ASTParser
from app.detectors.hardcoded_secrets import HardcodedSecretsDetector
from app.detectors.eval_exec import EvalExecDetector
from app.detectors.command_injection import CommandInjectionDetector
from app.core.taint_analyzer import TaintAnalyzer

class TestVulnerabilityComponents(unittest.TestCase):
    
    def setUp(self):
        self.parser = ASTParser()
        self.secret_detector = HardcodedSecretsDetector()
        self.eval_detector = EvalExecDetector()
        self.cmd_detector = CommandInjectionDetector()
        self.taint_analyzer = TaintAnalyzer()
        
        self.eval_detector.source_lines = []
        self.cmd_detector.source_lines = []

    def test_ast_parser_basic(self):
        code = "x = 1\ny = 2"
        nodes, tree = self.parser.parse(code)
        self.assertTrue(len(nodes) > 0)
        assign_nodes = [n for n in nodes if n.node_type == "Assign"]
        self.assertEqual(len(assign_nodes), 2)

    def test_hardcoded_secret_detection(self):
        vulnerable_code = "password = 'super_secret_password_123'"
        nodes, tree = self.parser.parse(vulnerable_code)
        self.secret_detector.set_source(vulnerable_code)
        results = self.secret_detector.detect(nodes)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0].vulnerability_type, "hardcoded_secrets")

    def test_eval_exec_detection(self):
        code = """
def process(user_input):
    eval(user_input)
    exec("print('hello')")
"""
        nodes, tree = self.parser.parse(code)
        self.eval_detector.set_source(code)
        results = self.eval_detector.detect(nodes)
        self.assertEqual(len(results), 2)

    def test_command_injection_detection(self):
        code = """
import os
import subprocess

def run_cmd(cmd):
    os.system(cmd)
    subprocess.call("ls", shell=True)
    subprocess.run(["ls"], shell=False)
"""
        nodes, tree = self.parser.parse(code)
        self.cmd_detector.set_source(code)
        results = self.cmd_detector.detect(nodes)
        # Expecting 3 detections (os.system, subprocess.call, subprocess.run)
        descriptions = [r.description for r in results]
        self.assertTrue(any("os.system" in d for d in descriptions))
        self.assertTrue(any("subprocess" in d for d in descriptions))

    def test_taint_analysis_flow(self):
        code = """
import os
def handle_request():
    user_data = input("enter command")
    cmd = "echo " + user_data
    os.system(cmd)
"""
        # Note: TaintAnalyzer requires nodes AND source_code
        nodes, tree = self.parser.parse(code)
        paths = self.taint_analyzer.analyze(nodes, code)
        
        # Should detect flow from input -> user_data -> cmd -> os.system
        self.assertTrue(len(paths) > 0, "No taint paths found")
        
        path = paths[0]
        self.assertEqual(path.sink_type.value, "command_exec")
        self.assertEqual(path.source.source_type.value, "user_input")
        self.assertEqual(path.source.variable_name, "cmd")

if __name__ == '__main__':
    unittest.main()
