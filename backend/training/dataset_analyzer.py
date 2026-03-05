"""Dataset Analyzer: Auto-classify raw Python files into vulnerability categories.

Usage:
    python training/dataset_analyzer.py --raw-dir data/raw --data-dir data

This script:
1. Scans all .py files in the raw directory
2. Parses each file and runs all detectors
3. Classifies based on highest-confidence detection (or 'safe' if none found)
4. Copies files into the appropriate data/<category>/ subdirectory
5. Prints a summary report
"""
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ast_parser import ASTParser
from app.detectors import (
    EvalExecDetector, CommandInjectionDetector,
    DeserializationDetector, HardcodedSecretsDetector, LogicFlawDetector
)
from app.detectors.base import DetectionResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Valid vulnerability categories (matches training label_map)
CATEGORIES = [
    "safe", "eval_exec", "command_injection",
    "unsafe_deserialization", "hardcoded_secrets",
    "sql_injection", "path_traversal"
]

# Mapping from detector vulnerability_type to category folder name
VULN_TYPE_MAP = {
    "eval_exec": "eval_exec",
    "eval_injection": "eval_exec",
    "exec_injection": "eval_exec",
    "command_injection": "command_injection",
    "unsafe_deserialization": "unsafe_deserialization",
    "hardcoded_secrets": "hardcoded_secrets",
    "hardcoded_secret": "hardcoded_secrets",
    "sql_injection": "sql_injection",
    "path_traversal": "path_traversal",
    "logic_flaw": "safe",  # Logic flaws don't have their own training category
}


def classify_file(source_code: str, parser: ASTParser, detectors: list) -> Tuple[str, float, List[DetectionResult]]:
    """Classify a Python file by running all detectors.
    
    Returns:
        (category, confidence, detections)
    """
    try:
        nodes, tree = parser.parse(source_code)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse: {e}")
    
    if not nodes:
        return "safe", 1.0, []
    
    # Run all detectors
    all_detections: List[DetectionResult] = []
    for detector in detectors:
        detector.set_source(source_code)
        try:
            results = detector.detect(nodes)
            all_detections.extend(results)
        except Exception as e:
            logger.debug(f"Detector {detector.name} failed: {e}")
    
    if not all_detections:
        return "safe", 1.0, []
    
    # Find the highest-confidence detection
    best = max(all_detections, key=lambda d: d.confidence)
    category = VULN_TYPE_MAP.get(best.vulnerability_type, "safe")
    
    return category, best.confidence, all_detections


def analyze_raw_dataset(raw_dir: Path, data_dir: Path, dry_run: bool = False) -> Dict:
    """Analyze and classify all raw Python files.
    
    Args:
        raw_dir: Path to raw (unclassified) Python files
        data_dir: Path to organized data directory with category subfolders
        dry_run: If True, only report classifications without copying files
    
    Returns:
        Summary report dictionary
    """
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        return {"error": "Raw directory not found"}
    
    # Initialize components
    parser = ASTParser()
    detectors = [
        EvalExecDetector(),
        CommandInjectionDetector(),
        DeserializationDetector(),
        HardcodedSecretsDetector(),
        LogicFlawDetector(),
    ]
    
    # Ensure category directories exist
    for category in CATEGORIES:
        (data_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Collect raw files
    raw_files = list(raw_dir.glob("*.py"))
    if not raw_files:
        logger.warning(f"No .py files found in {raw_dir}")
        return {"error": "No Python files found", "raw_dir": str(raw_dir)}
    
    logger.info(f"Found {len(raw_files)} Python files in {raw_dir}")
    
    # Classification results
    classifications = defaultdict(list)
    errors = []
    report_entries = []
    
    for py_file in sorted(raw_files):
        try:
            source = py_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError) as e:
            errors.append({"file": py_file.name, "error": f"Read error: {e}"})
            continue
        
        if not source.strip():
            errors.append({"file": py_file.name, "error": "Empty file"})
            continue
        
        try:
            category, confidence, detections = classify_file(source, parser, detectors)
        except ValueError as e:
            errors.append({"file": py_file.name, "error": str(e)})
            continue
        
        classifications[category].append(py_file.name)
        
        # Build report entry
        entry = {
            "file": py_file.name,
            "category": category,
            "confidence": round(confidence, 4),
            "detections": len(detections),
        }
        if detections:
            entry["top_detection"] = detections[0].vulnerability_type
        report_entries.append(entry)
        
        # Copy file to category directory
        if not dry_run:
            dest_dir = data_dir / category
            dest_file = dest_dir / py_file.name
            
            # Handle name collisions
            if dest_file.exists():
                stem = py_file.stem
                suffix = py_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(py_file, dest_file)
            logger.info(f"  {py_file.name} -> {category}/ (confidence: {confidence:.2f})")
    
    # Build summary
    summary = {
        "total_files": len(raw_files),
        "classified": sum(len(v) for v in classifications.values()),
        "errors": len(errors),
        "categories": {cat: len(classifications.get(cat, [])) for cat in CATEGORIES},
        "error_details": errors,
        "classifications": report_entries,
    }
    
    return summary


def print_report(summary: Dict):
    """Print a human-readable classification report."""
    if "error" in summary and not summary.get("classifications"):
        print(f"\n❌ Error: {summary['error']}")
        return
    
    print("\n" + "=" * 60)
    print("  DATASET CLASSIFICATION REPORT")
    print("=" * 60)
    
    print(f"\n  Total files scanned:  {summary['total_files']}")
    print(f"  Successfully classified: {summary['classified']}")
    print(f"  Errors/skipped:       {summary['errors']}")
    
    print("\n  Category Distribution:")
    print("  " + "-" * 40)
    
    categories = summary.get("categories", {})
    total = max(summary["classified"], 1)
    
    for cat in CATEGORIES:
        count = categories.get(cat, 0)
        pct = (count / total) * 100
        bar = "#" * int(pct / 3)
        print(f"  {cat:<25} {count:>4}  ({pct:5.1f}%)  {bar}")
    
    if summary.get("error_details"):
        print(f"\n  [!] Files with errors:")
        for err in summary["error_details"][:10]:
            print(f"    - {err['file']}: {err['error']}")
        if len(summary["error_details"]) > 10:
            print(f"    ... and {len(summary['error_details']) - 10} more")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Classify raw Python files into vulnerability categories")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw Python files")
    parser.add_argument("--data-dir", default="data", help="Path to organized data directory")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't copy files")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    data_dir = Path(args.data_dir)
    
    print(f"Analyzing raw files from: {raw_dir.resolve()}")
    print(f"Output directory: {data_dir.resolve()}")
    if args.dry_run:
        print("** DRY RUN - no files will be copied **")
    
    summary = analyze_raw_dataset(raw_dir, data_dir, dry_run=args.dry_run)
    print_report(summary)
    
    if not args.dry_run and summary.get("classified", 0) > 0:
        print("\n[OK] Files have been copied to their category folders.")
        print("   Run dataset_validator.py next to check dataset quality.")
    

if __name__ == "__main__":
    main()
