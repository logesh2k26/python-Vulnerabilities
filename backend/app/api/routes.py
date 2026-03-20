"""API routes for vulnerability detection — hardened."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import logging
from pathlib import Path

from app.schemas import AnalysisResult, BatchAnalysisRequest, FileInput
from app.schemas import DatasetStatus, TrainingRequest, TrainingResult
from app.security.auth import verify_api_key
from app.security.rate_limiter import limiter
from app.security.validators import (
    sanitize_filename,
    validate_resolved_path,
    validate_file_size,
    validate_code_content,
    validate_file_count,
    validate_python_extension,
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

# ── Directory constants ───────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PRETRAINED_DIR = Path(__file__).parent.parent.parent / "pretrained"


# ══════════════════════════════════════════════════════════════════════════
# Analysis Endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.post("/analyze", response_model=AnalysisResult)
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def analyze_code(
    request: Request,
    file_input: FileInput,
    _key: str = Depends(verify_api_key),
):
    """Analyze a single Python file for vulnerabilities."""
    engine = request.app.state.inference_engine

    if not file_input.content.strip():
        raise HTTPException(status_code=400, detail="Empty code content")

    # Content-length validation
    validate_code_content(file_input.content)

    result = engine.analyze(file_input.content, file_input.filename)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/analyze/batch")
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def analyze_batch(
    request: Request,
    batch: BatchAnalysisRequest,
    _key: str = Depends(verify_api_key),
):
    """Analyze multiple Python files."""
    engine = request.app.state.inference_engine

    if not batch.files:
        raise HTTPException(status_code=400, detail="No files provided")

    validate_file_count(len(batch.files))

    for f in batch.files:
        validate_code_content(f.content)

    files = [{"content": f.content, "filename": f.filename} for f in batch.files]
    results = engine.batch_analyze(files)

    summary = {
        "total_files": len(results),
        "vulnerable_files": sum(1 for r in results if r.get("is_vulnerable")),
        "safe_files": sum(1 for r in results if not r.get("is_vulnerable")),
        "results": results,
    }

    return summary


@router.post("/analyze/upload")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def analyze_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    _key: str = Depends(verify_api_key),
):
    """Upload and analyze Python files."""
    engine = request.app.state.inference_engine

    validate_file_count(len(files))

    results = []
    for file in files:
        if not validate_python_extension(file.filename):
            results.append({
                "filename": file.filename,
                "error": "Only Python files are supported",
            })
            continue

        content = await file.read()
        validate_file_size(content, file.filename)

        try:
            source = content.decode("utf-8")
        except UnicodeDecodeError:
            results.append({
                "filename": file.filename,
                "error": "Could not decode file as UTF-8",
            })
            continue

        result = engine.analyze(source, file.filename)
        results.append(result)

    return {"total_files": len(results), "results": results}


@router.get("/vulnerability-types")
async def get_vulnerability_types(
    _key: str = Depends(verify_api_key),
):
    """Get list of supported vulnerability types."""
    return {
        "types": [
            {"id": "eval_exec", "name": "Eval/Exec Injection", "severity": "critical"},
            {"id": "command_injection", "name": "Command Injection", "severity": "critical"},
            {"id": "unsafe_deserialization", "name": "Unsafe Deserialization", "severity": "critical"},
            {"id": "hardcoded_secrets", "name": "Hardcoded Secrets", "severity": "high"},
            {"id": "sql_injection", "name": "SQL Injection", "severity": "critical"},
            {"id": "path_traversal", "name": "Path Traversal", "severity": "high"},
            {"id": "ssrf", "name": "SSRF (Server Side Request Forgery)", "severity": "high"},
            {"id": "insecure_cryptography", "name": "Insecure Cryptography", "severity": "medium"},
            {"id": "xxe", "name": "XXE (XML External Entity)", "severity": "high"},
            {"id": "redos", "name": "ReDoS (Regex Denial of Service)", "severity": "medium"},
            {"id": "xss", "name": "Cross-Site Scripting (XSS)", "severity": "high"},
        ]
    }


# ══════════════════════════════════════════════════════════════════════════
# Dataset Management Endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.post("/dataset/upload")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_dataset_files(
    request: Request,
    files: List[UploadFile] = File(...),
    _key: str = Depends(verify_api_key),
):
    """Upload raw Python files for dataset classification."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    validate_file_count(len(files))

    uploaded = []
    errors = []

    for file in files:
        if not validate_python_extension(file.filename):
            errors.append({"filename": file.filename, "error": "Only .py files accepted"})
            continue

        content = await file.read()
        validate_file_size(content, file.filename)

        try:
            source = content.decode("utf-8")
        except UnicodeDecodeError:
            errors.append({"filename": file.filename, "error": "Could not decode as UTF-8"})
            continue

        if not source.strip():
            errors.append({"filename": file.filename, "error": "Empty file"})
            continue

        # ── Secure filename handling ──────────────────────────────────
        safe_name = sanitize_filename(file.filename)
        dest = RAW_DIR / safe_name
        validate_resolved_path(dest, RAW_DIR)

        # Handle name collisions
        if dest.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            counter = 1
            while dest.exists():
                dest = RAW_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        dest.write_text(source, encoding="utf-8")
        uploaded.append(safe_name)

    return {
        "uploaded": len(uploaded),
        "errors": len(errors),
        "uploaded_files": uploaded,
        "error_details": errors,
        "message": f"Uploaded {len(uploaded)} files. "
                   f"Run POST /api/v1/dataset/classify to auto-classify them.",
    }


@router.post("/dataset/classify")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def classify_dataset(
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """Auto-classify raw files into vulnerability categories."""
    from training.dataset_analyzer import analyze_raw_dataset

    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.py")):
        raise HTTPException(status_code=400, detail="No raw .py files found. Upload files first.")

    summary = analyze_raw_dataset(RAW_DIR, DATA_DIR, dry_run=False)
    return summary


@router.get("/dataset/status", response_model=DatasetStatus)
async def get_dataset_status(
    _key: str = Depends(verify_api_key),
):
    """Get dataset statistics and validation status."""
    from training.dataset_validator import validate_dataset

    report = validate_dataset(DATA_DIR)
    return report


# ── Background training helper ────────────────────────────────────────────

def _run_training(data_dir: str, output_path: str, epochs: int, lr: float):
    """Synchronous training — runs inside a background thread."""
    try:
        from training.train import train_model
        train_model(data_dir=data_dir, output_path=output_path, epochs=epochs, lr=lr)
        logger.info("Background training completed successfully")
    except Exception as e:
        logger.error("Background training failed: %s", e, exc_info=True)


@router.post("/dataset/train", response_model=TrainingResult)
@limiter.limit(settings.RATE_LIMIT_TRAIN)
async def train_model_endpoint(
    request: Request,
    config: TrainingRequest,
    background_tasks: BackgroundTasks,
    _key: str = Depends(verify_api_key),
):
    """Train the GNN model on the current dataset (runs in background)."""
    from training.dataset_validator import validate_dataset

    # Validate dataset first
    report = validate_dataset(DATA_DIR)
    if report["status"] == "ERROR":
        raise HTTPException(
            status_code=400,
            detail="Dataset has critical issues. Check /dataset/status for details.",
        )

    if report["total_files"] == 0:
        raise HTTPException(status_code=400, detail="No training data found")

    output_path = str(PRETRAINED_DIR / "vulnerability_gnn.pt")
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Non-blocking: run in background ───────────────────────────────
    background_tasks.add_task(
        _run_training,
        data_dir=str(DATA_DIR),
        output_path=output_path,
        epochs=config.epochs,
        lr=config.learning_rate,
    )

    return TrainingResult(
        status="started",
        epochs_completed=0,
        model_path=output_path,
        message=f"Training started in background ({config.epochs} epochs). "
                f"Check server logs for progress.",
    )
