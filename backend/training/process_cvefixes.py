"""Process CVEFixes.csv dataset into training data.

Usage:
    python training/process_cvefixes.py

This script:
1. Reads CVEFixes.csv from data/raw/
2. Filters Python-only samples
3. Auto-classifies vulnerable samples into specific vulnerability types
4. Saves as .py files in data/<category>/ directories
5. Prints a summary report
"""
import csv
import sys
import ast
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

csv.field_size_limit(2**30)  # Safe for Windows

from app.core.ast_parser import ASTParser
from app.detectors import (
    EvalExecDetector, CommandInjectionDetector,
    DeserializationDetector, HardcodedSecretsDetector, LogicFlawDetector
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = [
    "safe", "eval_exec", "command_injection",
    "unsafe_deserialization", "hardcoded_secrets",
    "sql_injection", "path_traversal"
]

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
    "logic_flaw": "safe",
}


def clean_nuls(f):
    """Strip NUL characters from file lines."""
    for line in f:
        yield line.replace('\x00', '')


def is_valid_python(code: str) -> bool:
    """Check if code is valid parseable Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def classify_code(source_code: str, parser: ASTParser, detectors: list) -> str:
    """Classify Python code into a vulnerability category.
    
    Returns the category name.
    """
    try:
        nodes, tree = parser.parse(source_code)
    except (ValueError, SyntaxError):
        return "safe"  # Can't parse = treat as safe
    
    if not nodes:
        return "safe"
    
    best_category = "safe"
    best_confidence = 0.0
    
    for detector in detectors:
        detector.set_source(source_code)
        try:
            results = detector.detect(nodes)
            for result in results:
                category = VULN_TYPE_MAP.get(result.vulnerability_type, None)
                if category and category != "safe" and result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_category = category
        except Exception:
            continue
    
    return best_category


def process_cvefixes(csv_path: Path, data_dir: Path, max_samples: int = 0):
    """Process CVEFixes.csv and create training dataset.
    
    Args:
        csv_path: Path to CVEFixes.csv
        data_dir: Output data directory
        max_samples: Max samples to process (0 = all)
    """
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
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
    
    # Track stats
    stats = {
        "total_rows": 0,
        "python_rows": 0,
        "valid_python": 0,
        "invalid_python": 0,
        "saved": Counter(),
        "skipped_duplicates": 0,
        "csv_safe": 0,
        "csv_vulnerable": 0,
    }
    
    seen_hashes = set()
    file_counters = Counter()
    
    logger.info(f"Processing {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(clean_nuls(f))
        
        for row in reader:
            stats["total_rows"] += 1
            
            # Filter Python only
            lang = row.get("language", "").strip().lower()
            if lang != "py":
                continue
            
            stats["python_rows"] += 1
            code = row.get("code", "").strip()
            safety = row.get("safety", "").strip().lower()
            
            if not code or len(code) < 20:  # Skip very short snippets
                continue
            
            # Check for valid Python
            if not is_valid_python(code):
                stats["invalid_python"] += 1
                continue
            
            stats["valid_python"] += 1
            
            # Deduplicate
            code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
            if code_hash in seen_hashes:
                stats["skipped_duplicates"] += 1
                continue
            seen_hashes.add(code_hash)
            
            # Determine category
            if safety == "safe":
                stats["csv_safe"] += 1
                category = "safe"
            else:
                stats["csv_vulnerable"] += 1
                # Auto-classify vulnerable code
                category = classify_code(code, parser, detectors)
                # If detector says safe but CSV says vulnerable,
                # still keep it - treat as a general vulnerability sample
                # We'll put unclassified vulnerables into the category with 
                # fewest samples for balance
            
            # Save the file
            file_counters[category] += 1
            filename = f"cve_{category}_{file_counters[category]:04d}.py"
            filepath = data_dir / category / filename
            
            # Add a comment header with metadata
            header = f'# Source: CVEFixes dataset\n# Safety: {safety}\n# Category: {category}\n\n'
            filepath.write_text(header + code, encoding='utf-8')
            
            stats["saved"][category] += 1
            
            if stats["python_rows"] % 200 == 0:
                logger.info(f"  Processed {stats['python_rows']} Python rows...")
            
            if max_samples > 0 and sum(stats["saved"].values()) >= max_samples:
                logger.info(f"  Reached max samples limit ({max_samples})")
                break
    
    # Print report
    print("\n" + "=" * 60)
    print("  CVEFixes DATASET PROCESSING REPORT")
    print("=" * 60)
    
    print(f"\n  Total CSV rows:      {stats['total_rows']}")
    print(f"  Python rows:         {stats['python_rows']}")
    print(f"  Valid Python:        {stats['valid_python']}")
    print(f"  Invalid Python:      {stats['invalid_python']}")
    print(f"  Duplicates skipped:  {stats['skipped_duplicates']}")
    print(f"\n  CSV safe samples:       {stats['csv_safe']}")
    print(f"  CSV vulnerable samples: {stats['csv_vulnerable']}")
    
    print(f"\n  Files saved per category:")
    print("  " + "-" * 40)
    total_saved = sum(stats["saved"].values())
    for cat in CATEGORIES:
        count = stats["saved"].get(cat, 0)
        pct = (count / max(total_saved, 1)) * 100
        bar = "#" * int(pct / 3)
        print(f"  {cat:<25} {count:>4}  ({pct:5.1f}%)  {bar}")
    
    print(f"\n  Total files saved:   {total_saved}")
    print("=" * 60)
    
    if total_saved > 0:
        print("\n  [OK] Dataset created successfully!")
        print("  Next steps:")
        print("    1. python training/dataset_validator.py --data-dir data")
        print("    2. python training/train.py --data-dir data --epochs 100")
    
    return stats


def main():
    parser_arg = argparse.ArgumentParser(description="Process CVEFixes.csv into training dataset")
    parser_arg.add_argument("--csv", default="data/raw/CVEFixes.csv", help="Path to CVEFixes.csv")
    parser_arg.add_argument("--data-dir", default="data", help="Output data directory")
    parser_arg.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    
    args = parser_arg.parse_args()
    
    process_cvefixes(Path(args.csv), Path(args.data_dir), args.max_samples)


if __name__ == "__main__":
    main()
