# Dataset Pipeline: Walkthrough

## What Was Built

A complete pipeline to go from **raw unclassified Python files** → **organized dataset** → **validated** → **trained GNN model**.

---

## New Files

| File | Purpose |
|------|---------|
| [dataset_analyzer.py](file:///f:/python-vulnerability-detector/backend/training/dataset_analyzer.py) | Auto-classifies raw `.py` files into vulnerability categories |
| [dataset_validator.py](file:///f:/python-vulnerability-detector/backend/training/dataset_validator.py) | Validates dataset quality (balance, duplicates, parse errors) |
| [cryptography.py](file:///f:/python-vulnerability-detector/backend/app/detectors/cryptography.py) | [NEW] Detects weak hashes (MD5, SHA1) and insecure randomness |
| [xxe.py](file:///f:/python-vulnerability-detector/backend/app/detectors/xxe.py) | [NEW] Detects XML External Entity (XXE) vulnerabilities |
| [redos.py](file:///f:/python-vulnerability-detector/backend/app/detectors/redos.py) | [NEW] Detects Regex Denial of Service (ReDoS) patterns |
| [xss_detector.py](file:///f:/python-vulnerability-detector/backend/app/detectors/xss_detector.py) | [NEW] Detects Cross-Site Scripting (XSS) in web code |
| [data/raw/](file:///f:/python-vulnerability-detector/backend/data/raw/) | Drop zone for unclassified files (supports CSV/folders) |


## Modified Files

| File | Changes |
|------|---------|
| [routes.py](file:///f:/python-vulnerability-detector/backend/app/api/routes.py) | Added 4 dataset endpoints: upload, classify, status, train |
| [schemas/__init__.py](file:///f:/python-vulnerability-detector/backend/app/schemas/__init__.py) | Added `DatasetStatus`, `TrainingRequest`, `TrainingResult`, `CategoryInfo` models |

---

## How to Use

### Option 1: CLI (Recommended for large datasets)

```bash
# 1. Drop your raw .py files into:
backend/data/raw/

# 2. Auto-classify them into categories
python training/dataset_analyzer.py --raw-dir data/raw --data-dir data

# 3. Validate the dataset
python training/dataset_validator.py --data-dir data

# 4. Train the model
python training/train.py --data-dir data --output pretrained/vulnerability_gnn.pt --epochs 100
```

### Option 2: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/dataset/upload` | POST | Upload raw `.py` files (multipart) |
| `/api/v1/dataset/classify` | POST | Auto-classify uploaded raw files |
| `/api/v1/dataset/status` | GET | Get dataset stats and validation |
| `/api/v1/dataset/train` | POST | Train model (body: `{"epochs": 100, "learning_rate": 0.001}`) |

---

## Verification

- `dataset_validator.py` ran successfully on existing data, reporting 7 category directories with files correctly distributed
- `dataset_analyzer.py` imports and `classify_file()` function validated
- All modules import cleanly from the backend root
