"""Dataset Validator: Check dataset quality before training.

Usage:
    python training/dataset_validator.py --data-dir data

This script:
1. Scans all category directories in the data folder
2. Validates each .py file (parseable, non-empty)
3. Detects duplicate files via content hashing
4. Reports class distribution and balance
5. Returns OK / WARNINGS / ERRORS status
"""
import sys
import hashlib
import ast
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = [
    "safe", "eval_exec", "command_injection",
    "unsafe_deserialization", "hardcoded_secrets",
    "sql_injection", "path_traversal"
]


def validate_python_file(filepath: Path) -> Dict:
    """Validate a single Python file.
    
    Returns dict with keys: valid, error, size, hash
    """
    result = {"valid": True, "error": None, "size": 0, "hash": ""}
    
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError) as e:
        return {"valid": False, "error": f"Cannot read: {e}", "size": 0, "hash": ""}
    
    result["size"] = len(content)
    result["hash"] = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    if not content.strip():
        return {"valid": False, "error": "Empty file", "size": 0, "hash": result["hash"]}
    
    # Check if it's valid Python
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {e.msg} (line {e.lineno})", "size": result["size"], "hash": result["hash"]}
    
    # Check minimum complexity (at least 2 lines of non-comment code)
    code_lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
    if len(code_lines) < 2:
        result["error"] = "Warning: Very short file (< 2 code lines)"
    
    return result


def validate_dataset(data_dir: Path) -> Dict:
    """Validate the entire dataset.
    
    Returns a comprehensive validation report.
    """
    if not data_dir.exists():
        return {"status": "ERROR", "error": f"Data directory not found: {data_dir}"}
    
    report = {
        "status": "OK",
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "duplicate_files": 0,
        "categories": {},
        "issues": [],
        "duplicates": [],
    }
    
    all_hashes = {}  # hash -> list of (category, filename)
    category_counts = {}
    
    for category in CATEGORIES:
        cat_dir = data_dir / category
        if not cat_dir.exists():
            report["categories"][category] = {"count": 0, "valid": 0, "invalid": 0}
            report["issues"].append(f"Missing category directory: {category}/")
            continue
        
        py_files = list(cat_dir.glob("*.py"))
        valid_count = 0
        invalid_count = 0
        
        for py_file in py_files:
            report["total_files"] += 1
            result = validate_python_file(py_file)
            
            if result["valid"]:
                valid_count += 1
                report["valid_files"] += 1
            else:
                invalid_count += 1
                report["invalid_files"] += 1
                report["issues"].append(f"{category}/{py_file.name}: {result['error']}")
            
            # Track duplicates
            if result["hash"]:
                file_key = f"{category}/{py_file.name}"
                if result["hash"] in all_hashes:
                    existing = all_hashes[result["hash"]]
                    report["duplicates"].append({
                        "file1": existing[0],
                        "file2": file_key,
                        "hash": result["hash"]
                    })
                    report["duplicate_files"] += 1
                else:
                    all_hashes[result["hash"]] = (file_key,)
        
        category_counts[category] = len(py_files)
        report["categories"][category] = {
            "count": len(py_files),
            "valid": valid_count,
            "invalid": invalid_count,
        }
    
    # Check class balance
    counts = [c for c in category_counts.values() if c > 0]
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        if min_count > 0 and max_count / min_count > 10:
            report["issues"].append(
                f"Severe class imbalance: max={max_count}, min={min_count} (ratio {max_count/min_count:.1f}:1)"
            )
        
        empty_cats = [cat for cat, cnt in category_counts.items() if cnt == 0]
        if empty_cats:
            report["issues"].append(f"Empty categories: {', '.join(empty_cats)}")
    
    # Check minimum dataset size
    if report["total_files"] < 14:  # At least 2 per category
        report["issues"].append(
            f"Very small dataset ({report['total_files']} files). "
            f"Recommend at least 50+ samples per category for meaningful training."
        )
    
    # Determine overall status
    if report["invalid_files"] > 0 or report["duplicate_files"] > 0:
        report["status"] = "WARNINGS"
    if report["total_files"] == 0:
        report["status"] = "ERROR"
    if report["invalid_files"] > report["total_files"] * 0.3:
        report["status"] = "ERROR"
    
    return report


def print_validation_report(report: Dict):
    """Print a human-readable validation report."""
    status_icon = {"OK": "[OK]", "WARNINGS": "[WARN]", "ERROR": "[ERR]"}.get(report["status"], "[?]")
    
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n  Status: {status_icon} {report['status']}")
    print(f"\n  Total files:     {report['total_files']}")
    print(f"  Valid files:     {report['valid_files']}")
    print(f"  Invalid files:   {report['invalid_files']}")
    print(f"  Duplicate files: {report['duplicate_files']}")
    
    print("\n  Category Distribution:")
    print("  " + "-" * 45)
    
    total = max(report["total_files"], 1)
    for cat in CATEGORIES:
        info = report["categories"].get(cat, {"count": 0, "valid": 0, "invalid": 0})
        count = info["count"]
        pct = (count / total) * 100
        bar = "#" * int(pct / 2.5)
        status = ""
        if info.get("invalid", 0) > 0:
            status = f" (! {info['invalid']} invalid)"
        print(f"  {cat:<25} {count:>4}  ({pct:5.1f}%)  {bar}{status}")
    
    if report.get("duplicates"):
        print(f"\n  [DUP] Duplicate files ({len(report['duplicates'])}):")
        for dup in report["duplicates"][:5]:
            print(f"    - {dup['file1']} == {dup['file2']}")
        if len(report["duplicates"]) > 5:
            print(f"    ... and {len(report['duplicates']) - 5} more")
    
    if report.get("issues"):
        print(f"\n  [!] Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"    - {issue}")
    
    print("\n" + "=" * 60)
    
    # Training readiness
    if report["status"] == "OK":
        print("\n  [OK] Dataset is ready for training!")
        print("  Run: python training/train.py --data-dir data --epochs 100")
    elif report["status"] == "WARNINGS":
        print("\n  [WARN] Dataset has warnings but can be used for training.")
        print("  Consider fixing the issues above for better results.")
        print("  Run: python training/train.py --data-dir data --epochs 100")
    else:
        print("\n  [ERR] Dataset has critical issues. Fix them before training.")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality for training")
    parser.add_argument("--data-dir", default="data", help="Path to organized data directory")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print(f"Validating dataset at: {data_dir.resolve()}")
    
    report = validate_dataset(data_dir)
    print_validation_report(report)
    
    # Exit code for CI/CD integration
    exit_code = {"OK": 0, "WARNINGS": 0, "ERROR": 1}.get(report["status"], 1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
