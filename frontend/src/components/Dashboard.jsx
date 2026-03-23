import { useState, useRef, useCallback } from 'react'
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
    const [targetLine, setTargetLine] = useState(null)
    const editorRef = useRef(null)
    const filename = 'code.py'

    const handleAnalyze = () => {
        if (code.trim()) {
            onAnalyze(code, filename)
        }
    }

    const handleLineClick = useCallback((lineNumber) => {
        if (editorRef.current) {
            editorRef.current.scrollToLine(lineNumber)
        }
        setTargetLine(lineNumber)
        setTimeout(() => setTargetLine(null), 100)
    }, [])

    const isVulnerable = analysisResult?.is_vulnerable
    const hasResult = !!analysisResult

    return (
        <div className="sc-container">
            {/* Breadcrumb Header */}
            <div className="sc-breadcrumb-bar">
                <div className="sc-breadcrumb-left">
                    <span className="sc-breadcrumb-path">Projects /</span>
                    <span className="sc-breadcrumb-current">{filename}</span>
                    {hasResult && (
                        <>
                            <div className="sc-breadcrumb-divider" />
                            <div className={`sc-status-badge ${isVulnerable ? 'sc-status-bad' : 'sc-status-good'}`}>
                                <span className="sc-status-dot" />
                                <span>{isVulnerable ? 'Vulnerabilities Found' : 'Code is Secure'}</span>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Two-Column Layout */}
            <div className="sc-split-layout">
                {/* Left: Code Editor */}
                <section className="sc-code-section">
                    <div className="sc-code-header">
                        <div>
                            <h1 className="sc-filename">{filename}</h1>
                            <p className="sc-filepath">src/analysis/{filename}</p>
                        </div>
                        <div className="sc-code-actions">
                            <button
                                className="sc-btn sc-btn-outline"
                                onClick={() => setShowUpload(!showUpload)}
                            >
                                <span className="sc-btn-icon">📁</span>
                                Upload
                            </button>
                            <button
                                className={`sc-btn sc-btn-primary ${isLoading ? 'sc-loading' : ''}`}
                                onClick={handleAnalyze}
                                disabled={isLoading || !code.trim()}
                            >
                                {isLoading ? (
                                    <>
                                        <span className="sc-spinner" />
                                        Scanning...
                                    </>
                                ) : (
                                    <>
                                        <span className="sc-btn-icon">🛡️</span>
                                        Scan Code
                                    </>
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

                    {/* Code Editor Canvas */}
                    <div className="sc-editor-canvas">
                        <div className="sc-editor-toolbar">
                            <div className="sc-editor-dots">
                                <div className="sc-dot sc-dot-red" />
                                <div className="sc-dot sc-dot-yellow" />
                                <div className="sc-dot sc-dot-green" />
                            </div>
                            <span className="sc-editor-lang">PYTHON 3.11 HIGHLIGHTING</span>
                        </div>
                        <div className="sc-editor-body">
                            <CodeEditor
                                ref={editorRef}
                                value={code}
                                onChange={setCode}
                                highlights={analysisResult?.highlighted_lines || []}
                                targetLine={targetLine}
                            />
                        </div>
                    </div>
                </section>

                {/* Right: Results Panel */}
                <section className="sc-results-section">
                    <ResultsPanel
                        result={analysisResult}
                        isLoading={isLoading}
                        onLineClick={handleLineClick}
                    />
                </section>
            </div>

            <style>{`
                .sc-container {
                    padding: 0;
                    min-height: 100%;
                    display: flex;
                    flex-direction: column;
                }

                .sd-main {
                    margin-left: 0;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    background: #f8fafc; /* Subtle Tonal Tint */
                }
/* Breadcrumb */
                .sc-breadcrumb-bar {
                    padding: 0 32px;
                    height: 48px;
                    display: flex;
                    align-items: center;
                    border-bottom: 1px solid rgba(168, 179, 187, 0.1);
                    background: rgba(248, 250, 252, 0.5);
                }
                .sc-breadcrumb-left {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .sc-breadcrumb-path {
                    font-size: 13px;
                    color: #566168;
                }
                .sc-breadcrumb-current {
                    font-size: 13px;
                    font-weight: 700;
                    color: #29343a;
                }
                .sc-breadcrumb-divider {
                    width: 1px;
                    height: 16px;
                    background: rgba(168, 179, 187, 0.3);
                    margin: 0 8px;
                }
                .sc-status-badge {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 14px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                }
                .sc-status-dot {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                }
                .sc-status-bad {
                    background: rgba(254, 137, 131, 0.12);
                    color: #752121;
                    border: 1px solid rgba(159, 64, 61, 0.1);
                }
                .sc-status-bad .sc-status-dot {
                    background: #9f403d;
                    box-shadow: 0 0 6px rgba(159, 64, 61, 0.4);
                }
                .sc-status-good {
                    background: rgba(209, 228, 254, 0.3);
                    color: #2f4155;
                    border: 1px solid rgba(79, 97, 118, 0.1);
                }
                .sc-status-good .sc-status-dot {
                    background: #4c56af;
                    box-shadow: 0 0 6px rgba(76, 86, 175, 0.4);
                }

                /* Split Layout */
                .sc-split-layout {
                    flex: 1;
                    display: grid;
                    grid-template-columns: 7fr 5fr;
                    padding: 32px;
                    padding-top: 24px;
                    gap: 32px;
                    min-height: 0;
                }
                @media (max-width: 1024px) {
                    .sc-split-layout {
                        grid-template-columns: 1fr;
                    }
                }

                /* Code Section */
                .sc-code-section {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    min-height: 0;
                }
                .sc-code-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                }
                .sc-filename {
                    font-family: 'Manrope', sans-serif;
                    font-size: 24px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                    letter-spacing: -0.02em;
                }
                .sc-filepath {
                    font-size: 13px;
                    color: #566168;
                    margin: 4px 0 0;
                }
                .sc-code-actions {
                    display: flex;
                    gap: 8px;
                }

                /* Buttons */
                .sc-btn {
                    padding: 10px 20px;
                    border-radius: 12px;
                    font-family: 'Inter', sans-serif;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    transition: all 0.2s;
                    border: none;
                }
                .sc-btn-icon {
                    font-size: 14px;
                }
                .sc-btn-outline {
                    background: #fff;
                    color: #29343a;
                    border: 1px solid rgba(168, 179, 187, 0.2);
                }
                .sc-btn-outline:hover {
                    background: #f0f4f8;
                    border-color: rgba(168, 179, 187, 0.35);
                }
                .sc-btn-primary {
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    color: #f9f6ff;
                    box-shadow: 0 2px 8px rgba(76, 86, 175, 0.25);
                }
                .sc-btn-primary:hover:not(:disabled) {
                    opacity: 0.9;
                    box-shadow: 0 4px 16px rgba(76, 86, 175, 0.35);
                }
                .sc-btn-primary:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .sc-spinner {
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border: 2px solid rgba(249, 246, 255, 0.3);
                    border-top-color: #f9f6ff;
                    border-radius: 50%;
                    animation: sc-spin 0.7s linear infinite;
                }
                @keyframes sc-spin { to { transform: rotate(360deg); } }

                /* Editor Canvas */
                .sc-editor-canvas {
                    flex: 1;
                    background: #fff;
                    border-radius: 16px;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    min-height: 500px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                    border: 1px solid #eef2f6;
                }
                .sc-editor-toolbar {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding: 10px 16px;
                    background: #f0f4f8;
                    border-bottom: 1px solid rgba(168, 179, 187, 0.1);
                }
                .sc-editor-dots {
                    display: flex;
                    gap: 6px;
                }
                .sc-dot {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }
                .sc-dot-red { background: rgba(159, 64, 61, 0.35); }
                .sc-dot-yellow { background: rgba(79, 97, 118, 0.35); }
                .sc-dot-green { background: rgba(76, 86, 175, 0.35); }
                .sc-editor-lang {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #717c84;
                }
                .sc-editor-body {
                    flex: 1;
                    overflow: auto;
                    background: #fbfcfd;
                }

                /* Results Section */
                .sc-results-section {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                    min-height: 0;
                    overflow-y: auto;
                    max-height: calc(100vh - 160px);
                }
                .sc-results-section::-webkit-scrollbar {
                    width: 4px;
                }
                .sc-results-section::-webkit-scrollbar-track {
                    background: transparent;
                }
                .sc-results-section::-webkit-scrollbar-thumb {
                    background: rgba(168, 179, 187, 0.3);
                    border-radius: 4px;
                }
            `}</style>
        </div>
    )
}

export default Dashboard
