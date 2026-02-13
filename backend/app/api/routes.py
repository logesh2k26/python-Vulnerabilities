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
            {"id": "ssrf", "name": "Server-Side Request Forgery", "severity": "high"}
        ]
    }
