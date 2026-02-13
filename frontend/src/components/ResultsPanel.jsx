import TraceVisualizer from './TraceVisualizer'

function ResultsPanel({ result, isLoading }) {
    if (isLoading) {
        return (
            <div className="results-panel">
                <div className="glass-card" style={{ padding: '60px 40px', textAlign: 'center' }}>
                    <div className="spinner" style={{ width: 36, height: 36, margin: '0 auto 20px', borderWidth: 3 }}></div>
                    <p style={{ color: 'var(--neon-blue)', fontWeight: 500 }}>Analyzing with AI...</p>
                    <p style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 8 }}>
                        Parsing AST • Building graph • Running GNN
                    </p>
                </div>
            </div>
        )
    }

    if (!result) {
        return (
            <div className="results-panel">
                <div className="glass-card empty-state">
                    <div className="empty-icon">🔐</div>
                    <h3 className="empty-title">Ready to Scan</h3>
                    <p className="empty-text">
                        Write or paste Python code, then click Scan Code to detect vulnerabilities
                    </p>
                </div>
            </div>
        )
    }

    const { is_vulnerable, overall_confidence, vulnerabilities = [], stats = {} } = result
    const riskLevel = overall_confidence > 0.7 ? 'high' : overall_confidence > 0.4 ? 'medium' : 'low'

    return (
        <div className="results-panel">
            {/* Status Badge */}
            <div className={`glass-card status-badge ${is_vulnerable ? 'vulnerable' : 'safe'}`}>
                <span className="status-icon">{is_vulnerable ? '⚠️' : '✅'}</span>
                <div className="status-text">{is_vulnerable ? 'VULNERABLE' : 'SECURE'}</div>
                <p className="status-subtext">
                    {vulnerabilities.length} {vulnerabilities.length === 1 ? 'issue' : 'issues'} detected
                </p>
            </div>

            {/* Stats */}
            <div className="stats-grid">
                <div className="glass-card stat-card">
                    <div className="stat-value blue">{stats.lines_of_code || 0}</div>
                    <div className="stat-label">Lines</div>
                </div>
                <div className="glass-card stat-card">
                    <div className="stat-value purple">{stats.total_nodes || 0}</div>
                    <div className="stat-label">AST Nodes</div>
                </div>
                <div className="glass-card stat-card">
                    <div className={`stat-value ${is_vulnerable ? 'red' : 'green'}`}>
                        {Math.round((overall_confidence || 0) * 100)}%
                    </div>
                    <div className="stat-label">Confidence</div>
                </div>
            </div>

            {/* Risk Meter */}
            <div className="glass-card risk-meter">
                <div className="risk-header">
                    <span className="risk-title">🎯 Risk Score</span>
                    <span className={`risk-value`} style={{ color: `var(--neon-${riskLevel === 'high' ? 'red' : riskLevel === 'medium' ? 'orange' : 'green'})` }}>
                        {riskLevel.toUpperCase()}
                    </span>
                </div>
                <div className="risk-bar">
                    <div
                        className={`risk-fill ${riskLevel}`}
                        style={{ width: `${(overall_confidence || 0) * 100}%` }}
                    />
                </div>
            </div>

            {/* Vulnerabilities */}
            {vulnerabilities.length > 0 && (
                <div className="glass-card vuln-section">
                    <h3 className="vuln-section-title">
                        🔴 Security Issues
                        <span className="vuln-count">{vulnerabilities.length}</span>
                    </h3>
                    <div className="vulnerabilities-list">
                        {vulnerabilities.map((vuln, idx) => (
                            <VulnerabilityCard key={idx} vulnerability={vuln} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
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

function VulnerabilityCard({ vulnerability }) {
    const {
        type, confidence, severity, description, remediation,
        affected_lines, status, mitigations, taint_path, metadata
    } = vulnerability

    return (
        <div className={`glass-card vuln-card ${severity}`}>
            <div className="vuln-header">
                <div className="vuln-type">
                    <span className={`severity-badge ${severity}`}>{severity}</span>
                    <span className="vuln-name">{TYPE_LABELS[type] || type}</span>
                    {status === 'mitigated' && (
                        <span className="severity-badge medium" style={{ background: 'rgba(48, 209, 88, 0.15)', color: 'var(--neon-green)' }}>
                            MITIGATED
                        </span>
                    )}
                </div>
                <div className="vuln-meta" style={{ textAlign: 'right' }}>
                    <div className="vuln-confidence">{Math.round(confidence * 100)}% Conf.</div>
                </div>
            </div>

            <p className="vuln-description">{description}</p>

            {metadata?.adjustment_reason && (
                <div style={{ fontSize: 12, color: 'var(--neon-green)', marginBottom: 8, fontStyle: 'italic' }}>
                    ℹ️ {metadata.adjustment_reason}
                </div>
            )}

            {affected_lines?.length > 0 && (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                    {affected_lines.map(line => (
                        <span key={line} className="vuln-line">📍 Line {line}</span>
                    ))}
                </div>
            )}

            {taint_path && (
                <TraceVisualizer taintPath={taint_path} />
            )}

            {mitigations?.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>DETECTED MITIGATIONS:</div>
                    {mitigations.map((m, i) => (
                        <div key={i} style={{ fontSize: 12, color: 'var(--neon-green)', display: 'flex', gap: 6 }}>
                            <span>🛡️</span>
                            <span>{m.description} (Line {m.line})</span>
                        </div>
                    ))}
                </div>
            )}

            {remediation && (
                <div className="vuln-fix">
                    <strong>💡 Fix:</strong> {remediation}
                </div>
            )}
        </div>
    )
}

export default ResultsPanel
