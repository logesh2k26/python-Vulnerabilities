import React, { useState } from 'react';
import TraceVisualizer from './TraceVisualizer'
import ChatBot from './ChatBot';

const CWE_MAP = {
    eval_exec: { id: 'CWE-94', name: 'Code Injection', score: '8.6' },
    command_injection: { id: 'CWE-78', name: 'OS Command Injection', score: '9.8' },
    unsafe_deserialization: { id: 'CWE-502', name: 'Unsafe Deserialization', score: '8.1' },
    hardcoded_secrets: { id: 'CWE-798', name: 'Hardcoded Credentials', score: '7.5' },
    sql_injection: { id: 'CWE-89', name: 'SQL Injection', score: '9.4' },
    path_traversal: { id: 'CWE-22', name: 'Path Traversal', score: '7.8' },
    ssrf: { id: 'CWE-918', name: 'Server-Side Request Forgery', score: '8.2' }
}

const TYPE_LABELS = {
    eval_exec: 'Code Injection',
    command_injection: 'Command Injection',
    unsafe_deserialization: 'Unsafe Deserialization',
    hardcoded_secrets: 'Hardcoded Secrets',
    sql_injection: 'SQL Injection',
    path_traversal: 'Path Traversal',
    ssrf: 'SSRF Attack'
}

function getSeverityLabel(severity, confidence) {
    const score = parseFloat(CWE_MAP[severity]?.score || '0')
    if (confidence > 0.8 || score >= 9.0) return { label: 'Critical', color: '#9f403d' }
    if (confidence > 0.6 || score >= 7.0) return { label: 'High', color: '#c4501a' }
    if (confidence > 0.4 || score >= 4.0) return { label: 'Medium', color: '#b08a00' }
    return { label: 'Low', color: '#4b5d72' }
}

function ResultsPanel({ result, isLoading, onLineClick }) {
    const [chatVuln, setChatVuln] = useState(null);

    const renderContent = () => {
        if (isLoading) {
            return (
                <div className="rp-card rp-loading-card">
                    <div className="rp-loading-spinner" />
                    <p className="rp-loading-title">Analyzing with AI...</p>
                    <p className="rp-loading-sub">Parsing AST • Building graph • Running GNN</p>
                </div>
            )
        }

        if (!result) {
            return (
                <div className="rp-card rp-empty-card">
                    <div className="rp-empty-placeholder">
                        <span className="material-symbols-outlined">analytics</span>
                        <p>No analysis results yet. Scan your code to begin.</p>
                    </div>
                </div>
            )
        }

        const { is_vulnerable, overall_confidence, vulnerabilities = [], stats = {} } = result

        return (
            <>
                {vulnerabilities.length > 0 ? (
                    vulnerabilities.map((vuln, idx) => (
                        <VulnerabilityDetail
                            key={idx}
                            vulnerability={vuln}
                            index={idx}
                            onLineClick={onLineClick}
                            onExplainClick={() => setChatVuln(vuln)}
                        />
                    ))
                ) : (
                    <div className="rp-card rp-safe-card">
                        <div className="rp-safe-icon">✅</div>
                        <h3 className="rp-safe-title">Code is Secure</h3>
                        <p className="rp-safe-text">No vulnerabilities detected. Confidence: {Math.round((overall_confidence || 0) * 100)}%</p>
                    </div>
                )}

                {/* AI Engine Insight */}
                {is_vulnerable && vulnerabilities.length > 0 && (
                    <div className="rp-ai-insight">
                        <div className="rp-ai-glow" />
                        <div className="rp-ai-header">
                            <span className="rp-ai-bolt">⚡</span>
                            <span className="rp-ai-label">AI ENGINE INSIGHT</span>
                        </div>
                        <p className="rp-ai-text">
                            "Our AI model detected {vulnerabilities.length} potential {vulnerabilities.length === 1 ? 'vulnerability' : 'vulnerabilities'} in 
                            this code. {vulnerabilities.some(v => v.status === 'mitigated')
                                ? 'Some patterns appear to have mitigations in place.'
                                : 'We recommend reviewing the remediation steps for each finding.'}"
                        </p>
                        <button className="rp-ai-link" onClick={() => setChatVuln(vulnerabilities[0])}>
                            Ask AI for more details →
                        </button>
                    </div>
                )}

                {/* Stats Summary */}
                {stats.lines_of_code > 0 && (
                    <div className="rp-stats-bar">
                        <div className="rp-stat">
                            <span className="rp-stat-val">{stats.lines_of_code || 0}</span>
                            <span className="rp-stat-lbl">Lines</span>
                        </div>
                        <div className="rp-stat-divider" />
                        <div className="rp-stat">
                            <span className="rp-stat-val">{stats.total_nodes || 0}</span>
                            <span className="rp-stat-lbl">AST Nodes</span>
                        </div>
                        <div className="rp-stat-divider" />
                        <div className="rp-stat">
                            <span className="rp-stat-val">{Math.round((overall_confidence || 0) * 100)}%</span>
                            <span className="rp-stat-lbl">Confidence</span>
                        </div>
                    </div>
                )}
            </>
        )
    }

    return (
        <div className="rp-panel">
            {renderContent()}
            {chatVuln && (
                <ChatBot
                    vulnerability={chatVuln}
                    onClose={() => setChatVuln(null)}
                />
            )}

            {chatVuln && (
                <ChatBot
                    vulnerability={chatVuln}
                    onClose={() => setChatVuln(null)}
                />
            )}

            <style>{`
                .rp-panel {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                /* ── Cards ── */
                .rp-card {
                    background: #fff;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                    border: 1px solid #eef2f6;
                    padding: 24px;
                    transition: all 0.2s ease;
                }
                .rp-card:hover {
                    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.02);
                    transform: translateY(-2px);
                }

                /* ── Loading State ── */
                .rp-loading-card {
                    text-align: center;
                    padding: 60px 28px;
                }
                .rp-loading-spinner {
                    width: 36px;
                    height: 36px;
                    border: 3px solid rgba(76,86,175,0.15);
                    border-top-color: #4c56af;
                    border-radius: 50%;
                    margin: 0 auto 18px;
                    animation: rp-spin 0.8s linear infinite;
                }
                @keyframes rp-spin { to { transform: rotate(360deg); } }
                .rp-loading-title {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 700;
                    font-size: 15px;
                    color: #4c56af;
                    margin: 0;
                }
                .rp-loading-sub {
                    font-size: 12px;
                    color: #717c84;
                    margin: 8px 0 0;
                }

                /* ── Empty / Safe State ── */
                .rp-empty-card, .rp-safe-card {
                    text-align: center;
                    padding: 50px 28px;
                }
                .rp-empty-card {
                    background: #f0f7ff; /* Soft Light Blue Background */
                    border-radius: 20px;
                    padding: 80px 40px;
                    border: 2px dashed rgba(79, 70, 229, 0.2);
                }
                .rp-empty-placeholder {
                    text-align: center;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 20px;
                }
                .rp-empty-placeholder span { 
                    font-size: 84px; 
                    color: #4f46e5;
                    text-shadow: 0 4px 12px rgba(79, 70, 229, 0.1);
                    margin-bottom: 8px;
                }
                .rp-empty-placeholder p { 
                    font-size: 18px; 
                    font-weight: 800; 
                    color: #4f46e5; /* Deep Indigo Text */
                    font-family: 'Manrope', sans-serif;
                    max-width: 320px;
                    line-height: 1.4;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }

                /* ── AI Insight Card ── */
                .rp-ai-insight {
                    position: relative;
                    overflow: hidden;
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    padding: 28px;
                    border-radius: 16px;
                    color: #f9f6ff;
                }
                .rp-ai-glow {
                    position: absolute;
                    right: -32px;
                    top: -32px;
                    width: 120px;
                    height: 120px;
                    background: rgba(249,246,255,0.1);
                    border-radius: 50%;
                    filter: blur(40px);
                }
                .rp-ai-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 12px;
                    position: relative;
                }
                .rp-ai-bolt {
                    font-size: 14px;
                }
                .rp-ai-label {
                    font-size: 10px;
                    font-weight: 900;
                    text-transform: uppercase;
                    letter-spacing: 0.2em;
                }
                .rp-ai-text {
                    font-size: 14px;
                    font-weight: 500;
                    line-height: 1.7;
                    font-style: italic;
                    margin: 0 0 14px;
                    position: relative;
                    opacity: 0.92;
                }
                .rp-ai-link {
                    background: none;
                    border: none;
                    color: #f9f6ff;
                    font-size: 12px;
                    font-weight: 700;
                    text-decoration: underline;
                    text-decoration-color: rgba(249,246,255,0.3);
                    text-underline-offset: 4px;
                    cursor: pointer;
                    padding: 0;
                    position: relative;
                    transition: opacity 0.2s;
                }
                .rp-ai-link:hover { opacity: 0.8; }

                /* ── Stats Bar ── */
                .rp-stats-bar {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 24px;
                    background: #fff;
                    border-radius: 16px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(168,179,187,0.08);
                    padding: 18px 28px;
                }
                .rp-stat {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 2px;
                }
                .rp-stat-val {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    color: #29343a;
                }
                .rp-stat-lbl {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #717c84;
                }
                .rp-stat-divider {
                    width: 1px;
                    height: 32px;
                    background: rgba(168,179,187,0.2);
                }
            `}</style>
        </div>
    )
}

// ── Vulnerability Detail Card ──
function VulnerabilityDetail({ vulnerability, index, onLineClick, onExplainClick }) {
    const [expanded, setExpanded] = useState(false)
    const {
        type, confidence, severity, description, remediation,
        affected_lines, status, mitigations, taint_path, metadata
    } = vulnerability

    const cwe = CWE_MAP[type] || { id: 'CWE-???', name: TYPE_LABELS[type] || type, score: '—' }
    const sevInfo = getSeverityLabel(type, confidence)
    const scoreNum = parseFloat(cwe.score) || 0

    return (
        <>
            {/* Vulnerability Card */}
            <div className="rp-card rp-vuln-card">
                <div className="rp-vuln-top">
                    <div className="rp-vuln-meta">
                        <span className="rp-cwe-label">{cwe.id}: {cwe.name}</span>
                        <h2 className="rp-vuln-title">{TYPE_LABELS[type] || type}</h2>
                    </div>
                    <div className="rp-severity-badge" style={{ background: sevInfo.color }}>
                        <span className="rp-severity-score">{cwe.score}</span>
                        <span className="rp-severity-label">{sevInfo.label}</span>
                    </div>
                </div>
                <p className="rp-vuln-desc">{description}</p>

                {status === 'mitigated' && (
                    <div className="rp-mitigated-notice">
                        <span>🛡️</span>
                        <span>This vulnerability appears to have mitigations in place</span>
                    </div>
                )}

                {/* Pattern Comparison */}
                <div className="rp-patterns">
                    <div className="rp-pattern rp-pattern-bad" onClick={() => affected_lines?.length > 0 && onLineClick(affected_lines[0])}>
                        <span className="rp-pattern-icon">❌</span>
                        <div className="rp-pattern-content">
                            <span className="rp-pattern-label">Insecure Pattern</span>
                            <code className="rp-pattern-code">
                                {type === 'command_injection' ? 'os.system("cmd " + user_input)' :
                                 type === 'eval_exec' ? 'eval(user_input)' :
                                 type === 'sql_injection' ? 'cursor.execute("SELECT * " + query)' :
                                 type === 'path_traversal' ? 'open("/data/" + filename)' :
                                 type === 'hardcoded_secrets' ? 'password = "hardcoded_value"' :
                                 type === 'unsafe_deserialization' ? 'pickle.loads(user_data)' :
                                 'Unsafe pattern detected'}
                            </code>
                        </div>
                    </div>
                    <div className="rp-pattern rp-pattern-good">
                        <span className="rp-pattern-icon">✅</span>
                        <div className="rp-pattern-content">
                            <span className="rp-pattern-label">Recommended Pattern</span>
                            <code className="rp-pattern-code">
                                {type === 'command_injection' ? 'subprocess.run(["cmd", arg], shell=False)' :
                                 type === 'eval_exec' ? 'ast.literal_eval(user_input)' :
                                 type === 'sql_injection' ? 'cursor.execute("SELECT * WHERE id=%s", (id,))' :
                                 type === 'path_traversal' ? 'Path(base).resolve() + validation' :
                                 type === 'hardcoded_secrets' ? 'os.environ.get("PASSWORD")' :
                                 type === 'unsafe_deserialization' ? 'json.loads(user_data)' :
                                 'Use safe alternative'}
                            </code>
                        </div>
                    </div>
                </div>

                {/* Affected lines */}
                {affected_lines?.length > 0 && (
                    <div className="rp-lines">
                        {affected_lines.map(line => (
                            <button
                                key={line}
                                className="rp-line-btn"
                                onClick={(e) => { e.stopPropagation(); onLineClick?.(line) }}
                            >
                                📍 Line {line}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Remediation Steps */}
            <div className="rp-remediation">
                <h3 className="rp-remediation-title">REMEDIATION STEPS</h3>
                <ul className="rp-steps">
                    <li className="rp-step">
                        <div className="rp-step-num"><span>01</span></div>
                        <div>
                            <h4 className="rp-step-title">
                                {type === 'command_injection' ? 'Avoid Shell Execution' :
                                 type === 'eval_exec' ? 'Remove Dynamic Evaluation' :
                                 type === 'sql_injection' ? 'Use Parameterized Queries' :
                                 type === 'path_traversal' ? 'Validate File Paths' :
                                 type === 'hardcoded_secrets' ? 'Use Environment Variables' :
                                 'Fix Insecure Pattern'}
                            </h4>
                            <p className="rp-step-desc">
                                {type === 'command_injection' ? 'Whenever possible, use built-in library functions instead of calling shell commands.' :
                                 type === 'eval_exec' ? 'Avoid eval() and exec() with user-controlled input. Use safe alternatives like ast.literal_eval().' :
                                 type === 'sql_injection' ? 'Never concatenate user input into SQL queries. Use parameterized queries instead.' :
                                 type === 'path_traversal' ? 'Normalize and validate all file paths against a trusted base directory.' :
                                 type === 'hardcoded_secrets' ? 'Store secrets in environment variables or use a secret management service.' :
                                 'Replace the insecure pattern with the recommended safe alternative.'}
                            </p>
                        </div>
                    </li>
                    <li className="rp-step">
                        <div className="rp-step-num"><span>02</span></div>
                        <div>
                            <h4 className="rp-step-title">
                                {type === 'command_injection' ? 'Use Subprocess with Array' :
                                 type === 'eval_exec' ? 'Use Safe Parsers' :
                                 type === 'sql_injection' ? 'Use ORM Framework' :
                                 type === 'path_traversal' ? 'Restrict Access Scope' :
                                 type === 'hardcoded_secrets' ? 'Add to .gitignore' :
                                 'Implement Safe Alternative'}
                            </h4>
                            <p className="rp-step-desc">
                                {type === 'command_injection' ? 'Pass arguments as a list to subprocess.run with shell=False to prevent shell interpretation.' :
                                 type === 'eval_exec' ? 'Use JSON or YAML parsers for structured data. Use ast.literal_eval() for Python literals.' :
                                 type === 'sql_injection' ? 'Consider using an ORM like SQLAlchemy for safer database interactions.' :
                                 type === 'path_traversal' ? 'Use pathlib\'s resolve() and verify the path starts with the base directory.' :
                                 type === 'hardcoded_secrets' ? 'Ensure secret files and .env files are excluded from version control.' :
                                 'Follow the recommended pattern shown above.'}
                            </p>
                        </div>
                    </li>
                    <li className="rp-step">
                        <div className="rp-step-num"><span>03</span></div>
                        <div>
                            <h4 className="rp-step-title">Input Validation</h4>
                            <p className="rp-step-desc">
                                Implement a strict whitelist of allowed input values or formats before processing any user-controlled data.
                            </p>
                        </div>
                    </li>
                </ul>

                <button className="rp-autofix-btn" onClick={onExplainClick}>
                    <span>✨</span>
                    <span>Ask AI to Explain Fix</span>
                </button>
            </div>

            {/* Expandable Details */}
            {(taint_path || mitigations?.length > 0) && (
                <button className="rp-expand-btn" onClick={() => setExpanded(!expanded)}>
                    {expanded ? '▲ Hide Details' : '▼ Show Technical Details'}
                </button>
            )}

            {expanded && (
                <div className="rp-card rp-details-card">
                    {taint_path && <TraceVisualizer taintPath={taint_path} />}
                    {mitigations?.length > 0 && (
                        <div className="rp-mitigations-list">
                            <h4 className="rp-detail-subtitle">Detected Mitigations</h4>
                            {mitigations.map((m, i) => (
                                <div key={i} className="rp-mitigation-item">
                                    <span>🛡️</span>
                                    <span>{m.description} (Line {m.line})</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            <style>{`
                /* ── Vulnerability Card ── */
                .rp-vuln-card {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                    border-left: 4px solid #4c56af;
                    background: linear-gradient(to right, #ffffff, #fcfdfe);
                }
                .rp-vuln-card:hover {
                    border-left-width: 6px;
                }
                .rp-vuln-top {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: 16px;
                }
                .rp-vuln-meta {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .rp-cwe-label {
                    font-size: 10px;
                    font-weight: 900;
                    color: #9f403d;
                    text-transform: uppercase;
                    letter-spacing: 0.2em;
                }
                .rp-vuln-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                }
                .rp-severity-badge {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 8px 14px;
                    border-radius: 10px;
                    flex-shrink: 0;
                }
                .rp-severity-score {
                    font-size: 16px;
                    font-weight: 900;
                    color: #fff;
                    font-family: 'Manrope', sans-serif;
                }
                .rp-severity-label {
                    font-size: 9px;
                    font-weight: 800;
                    color: rgba(255,255,255,0.9);
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                }
                .rp-vuln-desc {
                    font-size: 13px;
                    color: #566168;
                    line-height: 1.7;
                    margin: 0;
                }

                .rp-mitigated-notice {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px 14px;
                    background: rgba(209, 228, 254, 0.25);
                    border: 1px solid rgba(79, 97, 118, 0.12);
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: 600;
                    color: #415368;
                }

                /* ── Patterns ── */
                .rp-patterns {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .rp-pattern {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 14px 16px;
                    border-radius: 12px;
                    transition: transform 0.15s ease;
                }
                .rp-pattern:hover {
                    transform: translateX(2px);
                }
                .rp-pattern-bad {
                    background: rgba(254, 137, 131, 0.08);
                    border: 1px solid rgba(159, 64, 61, 0.1);
                    cursor: pointer;
                }
                .rp-pattern-good {
                    background: rgba(209, 228, 254, 0.15);
                    border: 1px solid rgba(79, 97, 118, 0.1);
                }
                .rp-pattern-icon {
                    font-size: 18px;
                    flex-shrink: 0;
                }
                .rp-pattern-content {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    min-width: 0;
                }
                .rp-pattern-label {
                    font-size: 11px;
                    font-weight: 700;
                    color: #29343a;
                }
                .rp-pattern-code {
                    font-size: 11px;
                    font-family: 'JetBrains Mono', 'Fira Code', monospace;
                    color: #566168;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    background: none;
                    padding: 0;
                }

                /* ── Affected Lines ── */
                .rp-lines {
                    display: flex;
                    gap: 6px;
                    flex-wrap: wrap;
                }
                .rp-line-btn {
                    font-size: 11px;
                    font-weight: 600;
                    padding: 5px 12px;
                    border-radius: 8px;
                    border: 1px solid rgba(76, 86, 175, 0.12);
                    background: rgba(224, 224, 255, 0.3);
                    color: #4c56af;
                    cursor: pointer;
                    transition: all 0.15s;
                }
                .rp-line-btn:hover {
                    background: rgba(224, 224, 255, 0.5);
                    border-color: rgba(76, 86, 175, 0.25);
                }

                /* ── Remediation ── */
                .rp-remediation {
                    background: #f0f4f8;
                    border-radius: 16px;
                    padding: 28px;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .rp-remediation-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 12px;
                    font-weight: 900;
                    text-transform: uppercase;
                    letter-spacing: 0.2em;
                    color: #566168;
                    margin: 0;
                }
                .rp-steps {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .rp-step {
                    display: flex;
                    gap: 14px;
                }
                .rp-step-num {
                    width: 28px;
                    height: 28px;
                    border-radius: 50%;
                    background: rgba(76, 86, 175, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                }
                .rp-step-num span {
                    font-size: 10px;
                    font-weight: 800;
                    color: #4c56af;
                }
                .rp-step-title {
                    font-size: 14px;
                    font-weight: 700;
                    color: #29343a;
                    margin: 0 0 4px;
                }
                .rp-step-desc {
                    font-size: 12px;
                    color: #566168;
                    line-height: 1.6;
                    margin: 0;
                }

                /* ── Auto-Fix Button ── */
                .rp-autofix-btn {
                    width: 100%;
                    padding: 16px;
                    background: #0b0f11;
                    color: #f7f9fc;
                    border: none;
                    border-radius: 14px;
                    font-size: 14px;
                    font-weight: 700;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    transition: all 0.2s;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }
                .rp-autofix-btn:hover {
                    opacity: 0.88;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
                }

                /* ── Expand / Details ── */
                .rp-expand-btn {
                    background: none;
                    border: 1px solid rgba(168,179,187,0.2);
                    border-radius: 10px;
                    padding: 10px 16px;
                    font-size: 12px;
                    font-weight: 600;
                    color: #566168;
                    cursor: pointer;
                    transition: all 0.2s;
                    text-align: center;
                }
                .rp-expand-btn:hover {
                    background: #f0f4f8;
                    border-color: rgba(168,179,187,0.35);
                }
                .rp-details-card {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .rp-detail-subtitle {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                    color: #717c84;
                    margin: 0 0 8px;
                }
                .rp-mitigation-item {
                    display: flex;
                    gap: 8px;
                    font-size: 12px;
                    color: #415368;
                    padding: 4px 0;
                }
                .rp-mitigations-list {
                    margin-top: 8px;
                }
            `}</style>
        </>
    )
}

export default ResultsPanel
