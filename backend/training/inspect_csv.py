"""Quick script to inspect CVEFixes.csv structure."""
import csv
import sys
from collections import Counter

csv.field_size_limit(2**30)

csv_path = "data/raw/CVEFixes.csv"

langs = Counter()
safety_labels = Counter()
total = 0
python_samples = []
errors = 0

def clean_nuls(f):
    """Strip NUL characters from file."""
    for line in f:
        yield line.replace('\x00', '')

with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(clean_nuls(f))
    for row in reader:
        total += 1
        try:
            lang = row.get("language", "").strip().lower()
            safe = row.get("safety", "").strip().lower()
        except Exception:
            errors += 1
            continue
        
        langs[lang] += 1
        safety_labels[safe] += 1
        
        if lang == "python" and len(python_samples) < 3:
            code = row.get("code", "")
            python_samples.append({
                "code_preview": code[:200],
                "safety": safe,
                "code_len": len(code)
            })
        
        if total % 100000 == 0:
            print(f"  Processed {total} rows...")

print(f"Total rows: {total}")
print(f"Read errors: {errors}")
print(f"\nTop languages:")
for lang, count in langs.most_common(15):
    print(f"  {lang}: {count}")

print(f"\nSafety labels: {dict(safety_labels)}")
print(f"\nPython samples: {langs.get('python', 0)}")

for i, s in enumerate(python_samples):
    print(f"\nPython sample {i+1}:")
    print(f"  Safety: {s['safety']}")
    print(f"  Code length: {s['code_len']}")
    print(f"  Preview: {s['code_preview'][:150]}...")
