import React, { useState } from 'react';
import TraceVisualizer from './TraceVisualizer'
import ChatBot from './ChatBot';

// Circular Progress SVG Component
function CircularProgress({ value, size = 64, stroke = 5, color = 'var(--neon-cyan)' }) {
    const radius = (size - stroke) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
        <div className="circular-progress" style={{ width: size, height: size }}>
            <svg width={size} height={size}>
                <circle
                    className="progress-bg"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    strokeWidth={stroke}
                />
                <circle
                    className="progress-fill"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth={stroke}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                />
            </svg>
            <span className="progress-value" style={{ color, fontSize: size > 50 ? '18px' : '12px' }}>
                {Math.round(value)}%
            </span>
        </div>
    );
}

function ResultsPanel({ result, isLoading, onLineClick }) {
    const [chatVuln, setChatVuln] = useState(null);

    if (isLoading) {
        return (
            <div className="results-panel">
                <div className="glass-card" style={{ padding: '60px 40px', textAlign: 'center' }}>
                    <div className="spinner" style={{ width: 36, height: 36, margin: '0 auto 20px', borderWidth: 3 }}></div>
                    <p style={{ color: 'var(--neon-cyan)', fontWeight: 500, fontFamily: 'var(--font-heading)' }}>Analyzing with AI...</p>
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
    const confidencePercent = Math.round((overall_confidence || 0) * 100)

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

            {/* Stats with Circular Progress */}
            <div className="stats-grid">
                <div className="glass-card stat-card">
                    <div className="stat-value blue">{stats.lines_of_code || 0}</div>
                    <div className="stat-label">Lines</div>
                </div>
                <div className="glass-card stat-card">
                    <div className="stat-value purple">{stats.total_nodes || 0}</div>
                    <div className="stat-label">AST Nodes</div>
                </div>
                <div className="glass-card stat-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                    <CircularProgress
                        value={confidencePercent}
                        size={52}
                        stroke={4}
                        color={is_vulnerable ? 'var(--neon-red)' : 'var(--neon-green)'}
                    />
                    <div className="stat-label">Confidence</div>
                </div>
            </div>

            {/* Risk Meter */}
            <div className="glass-card risk-meter">
                <div className="risk-header">
                    <span className="risk-title">🎯 Risk Score</span>
                    <span style={{
                        color: riskLevel === 'high' ? 'var(--neon-red)' :
                               riskLevel === 'medium' ? 'var(--neon-orange)' : 'var(--neon-green)',
                        textShadow: `0 0 10px ${riskLevel === 'high' ? 'rgba(255,45,85,0.3)' :
                                     riskLevel === 'medium' ? 'rgba(255,149,0,0.3)' : 'rgba(57,255,20,0.3)'}`
                    }}>
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
                            <VulnerabilityCard
                                key={idx}
                                vulnerability={vuln}
                                onLineClick={onLineClick}
                                onExplainClick={() => setChatVuln(vuln)}
                                index={idx}
                            />
                        ))}
                    </div>
                </div>
            )}
            {/* Chatbot Overlay */}
            {chatVuln && (
                <ChatBot
                    vulnerability={chatVuln}
                    onClose={() => setChatVuln(null)}
                />
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

function VulnerabilityCard({ vulnerability, onLineClick, onExplainClick, index }) {
    const [expanded, setExpanded] = useState(false)
    const {
        type, confidence, severity, description, remediation,
        affected_lines, status, mitigations, taint_path, metadata
    } = vulnerability

    const handleCardClick = (e) => {
        if (e.target.closest('button') || e.target.closest('.clickable-line')) return;

        if (onLineClick && affected_lines?.length > 0) {
            onLineClick(affected_lines[0])
        }
    }

    return (
        <div
            className={`vuln-card ${severity}`}
            onClick={handleCardClick}
            style={{
                cursor: affected_lines?.length > 0 ? 'pointer' : 'default',
                animationDelay: `${index * 0.1}s`
            }}
            title={affected_lines?.length > 0 ? `Click to jump to line ${affected_lines[0]}` : ''}
        >
            <div className="vuln-header">
                <div className="vuln-type">
                    <span className={`severity-badge ${severity}`}>{severity}</span>
                    <span className="vuln-name">{TYPE_LABELS[type] || type}</span>
                    {status === 'mitigated' && (
                        <span className="severity-badge medium"
                            style={{ background: 'rgba(57, 255, 20, 0.1)', color: 'var(--neon-green)', borderColor: 'rgba(57, 255, 20, 0.2)' }}>
                            MITIGATED
                        </span>
                    )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div className="vuln-confidence">{Math.round(confidence * 100)}%</div>
                    <button
                        className="explain-btn"
                        onClick={(e) => { e.stopPropagation(); onExplainClick(); }}
                        style={{
                            background: 'rgba(0, 240, 255, 0.06)',
                            border: '1px solid rgba(0, 240, 255, 0.15)',
                            color: 'var(--neon-cyan)',
                            padding: '5px 12px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '0.78rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px',
                            transition: 'all 0.25s'
                        }}
                    >
                        🤖 Explain
                    </button>
                    <button
                        onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
                        style={{
                            background: 'rgba(120, 100, 255, 0.06)',
                            border: '1px solid rgba(120, 100, 255, 0.12)',
                            color: 'var(--text-muted)',
                            width: '28px',
                            height: '28px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '12px',
                            transition: 'all 0.25s',
                            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)'
                        }}
                    >
                        ▼
                    </button>
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
                        <span
                            key={line}
                            className="vuln-line clickable-line"
                            onClick={(e) => {
                                e.stopPropagation()
                                onLineClick?.(line)
                            }}
                            title={`Jump to line ${line}`}
                        >
                            📍 Line {line}
                        </span>
                    ))}
                </div>
            )}

            {/* Expandable details */}
            {expanded && (
                <div style={{
                    marginTop: 12,
                    paddingTop: 12,
                    borderTop: '1px solid rgba(120, 100, 255, 0.08)',
                    animation: 'card-float-in 0.3s ease-out'
                }}>
                    {taint_path && (
                        <TraceVisualizer taintPath={taint_path} />
                    )}

                    {mitigations?.length > 0 && (
                        <div style={{ marginTop: 10, marginBottom: 10 }}>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                                Detected Mitigations
                            </div>
                            {mitigations.map((m, i) => (
                                <div key={i} style={{
                                    fontSize: 12,
                                    color: 'var(--neon-green)',
                                    display: 'flex',
                                    gap: 6,
                                    padding: '4px 0'
                                }}>
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
            )}

            {/* Show remediation even when collapsed if no expand button content */}
            {!expanded && remediation && !taint_path && !mitigations?.length && (
                <div className="vuln-fix">
                    <strong>💡 Fix:</strong> {remediation}
                </div>
            )}
        </div>
    )
}

export default ResultsPanel
