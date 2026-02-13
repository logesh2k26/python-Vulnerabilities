import { useState, useRef } from 'react'
import CodeEditor from './CodeEditor'
import ResultsPanel from './ResultsPanel'
import FileUpload from './FileUpload'

const SAMPLE_CODE = `# 🔍 Advanced Security Analysis Demo
# This file contains both vulnerable and mitigated code patterns

import os
import subprocess
import shlex
import json
from pathlib import Path

# --- VULNERABLE PATTERNS ---

def unsafe_command(user_input):
    """Ref: Command Injection (High Severity)"""
    # DANGER: Direct shell execution
    os.system("echo " + user_input)

def unsafe_path(filename):
    """Ref: Path Traversal (High Severity)"""
    # DANGER: No validation
    with open("/var/data/" + filename) as f:
        return f.read()

# --- MITIGATED PATTERNS (Should be detected as Safe/Mitigated) ---

def safe_command(user_input):
    """Ref: Command Injection (Mitigated)"""
    # SAFE: Using shell=False and list arguments
    subprocess.run(["echo", user_input], shell=False)

def safe_path(filename):
    """Ref: Path Traversal (Mitigated)"""
    # SAFE: Path normalization and prefix check
    base_dir = Path("/var/data").resolve()
    target = (base_dir / filename).resolve()
    
    if not str(target).startswith(str(base_dir)):
        raise ValueError("Invalid path")
        
    return target.read_text()

def sanitized_command(user_input):
    """Ref: Command Injection (Sanitized)"""
    # SAFE: Input validation
    if not user_input.isalnum():
        raise ValueError("Invalid input")
    
    os.system(f"echo {user_input}")
`

function Dashboard({ analysisResult, isLoading, error, onAnalyze, onUpload }) {
    const [code, setCode] = useState(SAMPLE_CODE)
    const [showUpload, setShowUpload] = useState(false)

    const handleAnalyze = () => {
        if (code.trim()) {
            onAnalyze(code)
        }
    }

    return (
        <main className="main-grid">
            <div className="editor-panel glass-card">
                <div className="panel-header">
                    <h2 className="panel-title">
                        <span className="panel-title-icon">⌨️</span>
                        Code Editor
                    </h2>
                    <div className="btn-group">
                        <button
                            className="btn btn-secondary"
                            onClick={() => setShowUpload(!showUpload)}
                        >
                            📁 Upload
                        </button>
                        <button
                            className={`btn analyze-btn ${isLoading ? 'loading' : ''}`}
                            onClick={handleAnalyze}
                            disabled={isLoading || !code.trim()}
                        >
                            {isLoading ? (
                                <>
                                    <span className="spinner"></span>
                                    Scanning...
                                </>
                            ) : (
                                <>🛡️ Scan Code</>
                            )}
                        </button>
                    </div>
                </div>

                {showUpload && (
                    <FileUpload onUpload={(files) => {
                        onUpload(files)
                        setShowUpload(false)
                    }} />
                )}

                <div className="code-editor-container">
                    <CodeEditor
                        value={code}
                        onChange={setCode}
                        highlights={analysisResult?.highlighted_lines || []}
                    />
                </div>
            </div>

            <ResultsPanel result={analysisResult} isLoading={isLoading} />
        </main>
    )
}

export default Dashboard
