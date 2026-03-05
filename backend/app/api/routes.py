"""API routes for vulnerability detection."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import List
import logging

from app.schemas import AnalysisResult, BatchAnalysisRequest, FileInput

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_code(request: Request, file_input: FileInput):
    """Analyze a single Python file for vulnerabilities."""
    engine = request.app.state.inference_engine
    
    if not file_input.content.strip():
        raise HTTPException(status_code=400, detail="Empty code content")
    
    result = engine.analyze(file_input.content, file_input.filename)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/analyze/batch")
async def analyze_batch(request: Request, batch: BatchAnalysisRequest):
    """Analyze multiple Python files."""
    engine = request.app.state.inference_engine
    
    if not batch.files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    files = [{"content": f.content, "filename": f.filename} for f in batch.files]
    results = engine.batch_analyze(files)
    
    summary = {
        "total_files": len(results),
        "vulnerable_files": sum(1 for r in results if r.get("is_vulnerable")),
        "safe_files": sum(1 for r in results if not r.get("is_vulnerable")),
        "results": results
    }
    
    return summary


@router.post("/analyze/upload")
async def analyze_upload(request: Request, files: List[UploadFile] = File(...)):
    """Upload and analyze Python files."""
    engine = request.app.state.inference_engine
    
    results = []
    for file in files:
        if not file.filename.endswith('.py'):
            results.append({
                "filename": file.filename,
                "error": "Only Python files are supported"
            })
            continue
        
        content = await file.read()
        try:
            source = content.decode('utf-8')
        except UnicodeDecodeError:
            results.append({
                "filename": file.filename,
                "error": "Could not decode file as UTF-8"
            })
            continue
        
        result = engine.analyze(source, file.filename)
        results.append(result)
    
    return {
        "total_files": len(results),
        "results": results
    }


@router.get("/vulnerability-types")
async def get_vulnerability_types():
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
            {"id": "xss", "name": "Cross-Site Scripting (XSS)", "severity": "high"}
        ]
    }


# ---------------------------------------------------------------------------
# Dataset Management Endpoints
# ---------------------------------------------------------------------------

from pathlib import Path
from app.schemas import DatasetStatus, TrainingRequest, TrainingResult

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PRETRAINED_DIR = Path(__file__).parent.parent.parent / "pretrained"


@router.post("/dataset/upload")
async def upload_dataset_files(files: List[UploadFile] = File(...)):
    """Upload raw Python files for dataset classification."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    errors = []
    
    for file in files:
        if not file.filename.endswith('.py'):
            errors.append({"filename": file.filename, "error": "Only .py files accepted"})
            continue
        
        content = await file.read()
        try:
            source = content.decode('utf-8')
        except UnicodeDecodeError:
            errors.append({"filename": file.filename, "error": "Could not decode as UTF-8"})
            continue
        
        if not source.strip():
            errors.append({"filename": file.filename, "error": "Empty file"})
            continue
        
        dest = RAW_DIR / file.filename
        # Handle name collisions
        if dest.exists():
            stem = Path(file.filename).stem
            suffix = Path(file.filename).suffix
            counter = 1
            while dest.exists():
                dest = RAW_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
        
        dest.write_text(source, encoding='utf-8')
        uploaded.append(file.filename)
    
    return {
        "uploaded": len(uploaded),
        "errors": len(errors),
        "uploaded_files": uploaded,
        "error_details": errors,
        "raw_dir": str(RAW_DIR),
        "message": f"Uploaded {len(uploaded)} files to raw/. "
                   f"Run POST /api/v1/dataset/classify to auto-classify them."
    }


@router.post("/dataset/classify")
async def classify_dataset():
    """Auto-classify raw files into vulnerability categories."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from training.dataset_analyzer import analyze_raw_dataset
    
    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.py")):
        raise HTTPException(status_code=400, detail="No raw .py files found. Upload files first.")
    
    summary = analyze_raw_dataset(RAW_DIR, DATA_DIR, dry_run=False)
    return summary


@router.get("/dataset/status", response_model=DatasetStatus)
async def get_dataset_status():
    """Get dataset statistics and validation status."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from training.dataset_validator import validate_dataset
    
    report = validate_dataset(DATA_DIR)
    return report


@router.post("/dataset/train", response_model=TrainingResult)
async def train_model_endpoint(config: TrainingRequest):
    """Train the GNN model on the current dataset."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from training.dataset_validator import validate_dataset
    
    # Validate dataset first
    report = validate_dataset(DATA_DIR)
    if report["status"] == "ERROR":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has critical issues: {'; '.join(report.get('issues', []))}"
        )
    
    if report["total_files"] == 0:
        raise HTTPException(status_code=400, detail="No training data found")
    
    # Run training
    try:
        from training.train import train_model
        
        output_path = str(PRETRAINED_DIR / "vulnerability_gnn.pt")
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        
        train_model(
            data_dir=str(DATA_DIR),
            output_path=output_path,
            epochs=config.epochs,
            lr=config.learning_rate
        )
        
        return TrainingResult(
            status="success",
            epochs_completed=config.epochs,
            model_path=output_path,
            message=f"Training completed. Model saved to {output_path}"
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return TrainingResult(
            status="error",
            message=f"Training failed: {str(e)}"
        )
