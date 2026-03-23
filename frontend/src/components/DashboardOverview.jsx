import React, { useState, useEffect } from 'react'

function DashboardOverview({ scanHistory = [], onNavigate }) {
    const [lastAnalysis, setLastAnalysis] = useState(null)

    useEffect(() => {
        if (scanHistory.length > 0) {
            const latest = new Date(scanHistory[0].timestamp)
            setLastAnalysis(latest.toLocaleDateString('en-US', {
                day: '2-digit', month: 'short', year: 'numeric',
            }) + ', ' + latest.toLocaleTimeString('en-US', {
                hour: '2-digit', minute: '2-digit', hour12: true
            }))
        }
    }, [scanHistory])

    // Calculate stats from scan history
    const totalScans = scanHistory.length
    const criticalCount = scanHistory.filter(s => s.severity === 'critical').length
    const highCount = scanHistory.filter(s => s.severity === 'high').length
    const mediumCount = scanHistory.filter(s => s.severity === 'medium').length
    const lowCount = scanHistory.filter(s => s.is_vulnerable && s.severity === 'low').length
    const safeCount = scanHistory.filter(s => !s.is_vulnerable).length

    // Health score (higher = better)
    const healthScore = totalScans === 0 ? 82 : Math.max(0, Math.min(100,
        Math.round(100 - (criticalCount * 15 + highCount * 8 + mediumCount * 3 + lowCount * 1))
    ))

    // SVG circle math
    const radius = 80
    const circumference = 2 * Math.PI * radius
    const dashOffset = circumference - (healthScore / 100) * circumference

    // Recent 3 scans for activity log
    const recentScans = scanHistory.slice(0, 5)

    const getTimeDiff = (timestamp) => {
        const diff = Date.now() - new Date(timestamp).getTime()
        const mins = Math.floor(diff / 60000)
        if (mins < 60) return `${mins}m ago`
        const hrs = Math.floor(mins / 60)
        if (hrs < 24) return `${hrs}h ago`
        const days = Math.floor(hrs / 24)
        return `${days}d ago`
    }

    const getSeverityBadge = (scan) => {
        if (!scan.is_vulnerable) return { text: 'SECURE', cls: 'dov-badge-secure' }
        if (scan.severity === 'critical') return { text: 'CRITICAL', cls: 'dov-badge-critical' }
        if (scan.severity === 'high') return { text: 'FAILED', cls: 'dov-badge-critical' }
        if (scan.severity === 'medium') return { text: 'WARNING', cls: 'dov-badge-warning' }
        return { text: 'LOW', cls: 'dov-badge-low' }
    }

    return (
        <div className="dov-container">
            {/* Page Title */}
            <div className="dov-page-header">
                <div>
                    <span className="dov-eyebrow">OPERATIONAL HUB</span>
                    <h1 className="dov-title">Security Overview</h1>
                </div>
                <div className="dov-last-analysis">
                    <span className="dov-last-label">Last Analysis</span>
                    <span className="dov-last-value">
                        {lastAnalysis || 'No scans yet'}
                    </span>
                </div>
            </div>

            {/* Bento Grid Metrics */}
            <div className="dov-bento-grid">
                {/* Health Score */}
                <div className="dov-health-card">
                    <div className="dov-health-glow" />
                    <span className="dov-health-label">PROJECT HEALTH SCORE</span>
                    <div className="dov-health-ring">
                        <svg width="192" height="192" viewBox="0 0 192 192">
                            <circle
                                cx="96" cy="96" r={radius}
                                fill="transparent"
                                stroke="#e1e9f0"
                                strokeWidth="12"
                            />
                            <circle
                                cx="96" cy="96" r={radius}
                                fill="transparent"
                                stroke="#4c56af"
                                strokeWidth="12"
                                strokeDasharray={circumference}
                                strokeDashoffset={dashOffset}
                                strokeLinecap="round"
                                style={{
                                    transform: 'rotate(-90deg)',
                                    transformOrigin: '50% 50%',
                                    transition: 'stroke-dashoffset 1s ease'
                                }}
                            />
                        </svg>
                        <div className="dov-health-center">
                            <span className="dov-health-number">{healthScore}</span>
                            <span className="dov-health-secure">SECURE</span>
                        </div>
                    </div>
                    <p className="dov-health-desc">
                        Your repository security is <strong>{
                            healthScore >= 80 ? 'Good' : healthScore >= 60 ? 'Fair' : 'Needs Attention'
                        }</strong>{totalScans > 0
                            ? `, with ${criticalCount + highCount} issues needing attention.`
                            : '. Start scanning to see real data.'
                        }
                    </p>
                </div>

                {/* Right side metric cards */}
                <div className="dov-metrics-col">
                    {/* Critical Threats */}
                    <div className="dov-metric-card dov-metric-critical">
                        <div className="dov-metric-top">
                            <div className="dov-metric-icon dov-icon-critical">⚠️</div>
                            <span className="dov-metric-badge dov-badge-change-critical">
                                {criticalCount > 0 ? `+${criticalCount}` : '0'} total
                            </span>
                        </div>
                        <h3 className="dov-metric-number dov-num-critical">
                            {String(criticalCount).padStart(2, '0')}
                        </h3>
                        <p className="dov-metric-label">CRITICAL THREATS</p>
                    </div>

                    {/* High Risk */}
                    <div className="dov-metric-card dov-metric-high">
                        <div className="dov-metric-top">
                            <div className="dov-metric-icon dov-icon-high">⚡</div>
                            <span className="dov-metric-badge dov-badge-change-neutral">
                                {highCount} total
                            </span>
                        </div>
                        <h3 className="dov-metric-number">{String(highCount).padStart(2, '0')}</h3>
                        <p className="dov-metric-label">HIGH RISK ISSUES</p>
                    </div>
                </div>

                <div className="dov-metrics-col">
                    {/* Medium Risk */}
                    <div className="dov-metric-card">
                        <div className="dov-metric-top">
                            <div className="dov-metric-icon dov-icon-medium">ℹ️</div>
                            <span className="dov-metric-badge dov-badge-change-neutral">Stable</span>
                        </div>
                        <h3 className="dov-metric-number">{String(mediumCount).padStart(2, '0')}</h3>
                        <p className="dov-metric-label">MEDIUM RISK</p>
                    </div>

                    {/* Low Priority */}
                    <div className="dov-metric-card">
                        <div className="dov-metric-top">
                            <div className="dov-metric-icon dov-icon-low">📋</div>
                            <span className="dov-metric-badge dov-badge-change-neutral">
                                {safeCount} safe
                            </span>
                        </div>
                        <h3 className="dov-metric-number">{String(lowCount + safeCount).padStart(2, '0')}</h3>
                        <p className="dov-metric-label">LOW PRIORITY</p>
                    </div>
                </div>
            </div>

            {/* Recent Activity Log */}
            <div className="dov-activity-section">
                <div className="dov-activity-header">
                    <div className="dov-activity-title-row">
                        <span className="dov-activity-icon">🕐</span>
                        <h2 className="dov-activity-title">Recent Activity Log</h2>
                    </div>
                    <button className="dov-export-btn">Export All Logs</button>
                </div>

                <div className="dov-activity-list">
                    {recentScans.length === 0 ? (
                        <div className="dov-activity-empty">
                            <div className="dov-empty-scan-icon">🔍</div>
                            <p>No scan history yet. Click <strong>Scan Code</strong> to start your first vulnerability analysis.</p>
                            <button className="dov-start-scan-btn" onClick={() => onNavigate && onNavigate('scans')}>
                                Start First Scan
                            </button>
                        </div>
                    ) : (
                        recentScans.map((scan, i) => {
                            const badge = getSeverityBadge(scan)
                            return (
                                <div key={scan.id || i} className="dov-scan-item">
                                    <div className="dov-scan-left">
                                        <div className="dov-scan-file-icon">
                                            {scan.is_vulnerable ? '📦' : '✅'}
                                        </div>
                                        <div className="dov-scan-info">
                                            <div className="dov-scan-name-row">
                                                <h4 className="dov-scan-filename">{scan.filename}</h4>
                                                <span className={`dov-scan-badge ${badge.cls}`}>
                                                    {badge.text}
                                                </span>
                                            </div>
                                            <div className="dov-scan-meta">
                                                <span>🕐 {getTimeDiff(scan.timestamp)}</span>
                                                <span>💻 {scan.vulnerability_type || 'analysis'}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="dov-scan-right">
                                        <span className="dov-scan-issues-label">FOUND ISSUES</span>
                                        <span className={`dov-scan-issues-value ${scan.is_vulnerable ? 'dov-issues-bad' : ''}`}>
                                            {scan.is_vulnerable
                                                ? `${(scan.vulnerabilities?.length || 1)} Vulnerabilities`
                                                : 'None Found'
                                            }
                                        </span>
                                    </div>
                                </div>
                            )
                        })
                    )}
                </div>

                {recentScans.length > 0 && (
                    <div className="dov-load-more">
                        <button
                            className="dov-load-more-btn"
                            onClick={() => onNavigate && onNavigate('history')}
                        >
                            Load More Scan History
                        </button>
                    </div>
                )}
            </div>

            {/* Footer */}
            <footer className="dov-footer">
                <div className="dov-footer-left">
                    <span>System Version 4.2.0-Alpha</span>
                    <span>Database ID: #PX-9921</span>
                </div>
                <div className="dov-footer-right">
                    © 2024 PyVulnDetect Architectural Solutions
                </div>
            </footer>

            <style>{`
                .dov-container {
                    padding: 32px;
                    max-width: 1280px;
                    margin: 0 auto;
                    display: flex;
                    flex-direction: column;
                    gap: 40px;
                }

                /* Page Header */
                .dov-page-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                }
                .dov-eyebrow {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.2em;
                    color: #4c56af;
                    display: block;
                    margin-bottom: 4px;
                }
                .dov-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 36px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                    letter-spacing: -0.02em;
                }
                .dov-last-analysis {
                    text-align: right;
                }
                .dov-last-label {
                    display: block;
                    font-size: 12px;
                    font-weight: 500;
                    color: #566168;
                    margin-bottom: 4px;
                }
                .dov-last-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 700;
                    padding: 6px 14px;
                    background: #e1e9f0;
                    border-radius: 8px;
                    color: #29343a;
                }

                /* Bento Grid */
                .dov-bento-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 24px;
                }
                @media (max-width: 1024px) {
                    .dov-bento-grid {
                        grid-template-columns: 1fr;
                    }
                }

                /* Health Card */
                .dov-health-card {
                    background: #fff;
                    border-radius: 16px;
                    padding: 32px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    gap: 20px;
                    position: relative;
                    overflow: hidden;
                }
                .dov-health-glow {
                    position: absolute;
                    top: -48px;
                    right: -48px;
                    width: 128px;
                    height: 128px;
                    background: rgba(76, 86, 175, 0.06);
                    border-radius: 50%;
                    filter: blur(40px);
                    pointer-events: none;
                }
                .dov-health-label {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #566168;
                }
                .dov-health-ring {
                    position: relative;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .dov-health-center {
                    position: absolute;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .dov-health-number {
                    font-family: 'Manrope', sans-serif;
                    font-size: 56px;
                    font-weight: 900;
                    color: #4c56af;
                    line-height: 1;
                }
                .dov-health-secure {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #566168;
                    margin-top: 4px;
                }
                .dov-health-desc {
                    font-size: 14px;
                    color: #566168;
                    max-width: 220px;
                    line-height: 1.6;
                    margin: 0;
                }
                .dov-health-desc strong { color: #29343a; }

                /* Metrics Column */
                .dov-metrics-col {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }

                /* Metric Cards */
                .dov-metric-card {
                    background: #fff;
                    border-radius: 16px;
                    padding: 24px;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .dov-metric-critical {
                    background: rgba(254, 137, 131, 0.08);
                }
                .dov-metric-high {
                    background: #fff;
                    border-left: 4px solid #4c56af;
                }
                .dov-metric-top {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                }
                .dov-metric-icon {
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                }
                .dov-icon-critical { background: rgba(254, 137, 131, 0.25); }
                .dov-icon-high { background: #e0e0ff; }
                .dov-icon-medium { background: #d1e4fe; }
                .dov-icon-low { background: #e2e8f0; }

                .dov-metric-badge {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.02em;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                .dov-badge-change-critical {
                    background: rgba(254, 137, 131, 0.15);
                    color: #752121;
                }
                .dov-badge-change-neutral {
                    background: #e1e9f0;
                    color: #566168;
                }

                .dov-metric-number {
                    font-family: 'Manrope', sans-serif;
                    font-size: 32px;
                    font-weight: 900;
                    color: #29343a;
                    margin: 0;
                    line-height: 1;
                }
                .dov-num-critical { color: #752121; }

                .dov-metric-label {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: #566168;
                    margin: 0;
                }

                /* Activity Section */
                .dov-activity-section {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .dov-activity-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding-bottom: 16px;
                    border-bottom: 1px solid rgba(168, 179, 187, 0.15);
                }
                .dov-activity-title-row {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .dov-activity-icon {
                    font-size: 20px;
                    color: #4c56af;
                }
                .dov-activity-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 700;
                    color: #29343a;
                    margin: 0;
                }
                .dov-export-btn {
                    background: none;
                    border: none;
                    font-size: 14px;
                    font-weight: 700;
                    color: #4c56af;
                    cursor: pointer;
                    padding: 0;
                    transition: all 0.2s;
                }
                .dov-export-btn:hover { text-decoration: underline; }

                /* Scan Items */
                .dov-activity-list {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .dov-scan-item {
                    background: #fff;
                    border-radius: 16px;
                    padding: 20px 24px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                .dov-scan-item:hover {
                    box-shadow: 0 4px 24px rgba(148, 163, 184, 0.15);
                    transform: translateY(-1px);
                }
                .dov-scan-item:nth-child(even) {
                    background: #f0f4f8;
                }
                .dov-scan-left {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    flex: 1;
                }
                .dov-scan-file-icon {
                    width: 48px;
                    height: 48px;
                    background: #f0f4f8;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    flex-shrink: 0;
                }
                .dov-scan-info {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .dov-scan-name-row {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .dov-scan-filename {
                    font-family: 'Manrope', sans-serif;
                    font-size: 15px;
                    font-weight: 700;
                    color: #29343a;
                    margin: 0;
                }
                .dov-scan-badge {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.02em;
                    padding: 3px 8px;
                    border-radius: 4px;
                }
                .dov-badge-secure { background: #d1e4fe; color: #415368; }
                .dov-badge-critical { background: rgba(254, 137, 131, 0.2); color: #752121; }
                .dov-badge-warning { background: #e0e0ff; color: #3f48a1; }
                .dov-badge-low { background: #e1e9f0; color: #566168; }

                .dov-scan-meta {
                    display: flex;
                    gap: 16px;
                    font-size: 12px;
                    color: #566168;
                }

                .dov-scan-right {
                    text-align: right;
                    flex-shrink: 0;
                }
                .dov-scan-issues-label {
                    display: block;
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #566168;
                    margin-bottom: 4px;
                }
                .dov-scan-issues-value {
                    font-size: 14px;
                    font-weight: 900;
                    color: #29343a;
                }
                .dov-issues-bad { color: #9f403d; }

                /* Empty State */
                .dov-activity-empty {
                    background: #fff;
                    border-radius: 16px;
                    padding: 48px 32px;
                    text-align: center;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 12px;
                }
                .dov-empty-scan-icon { font-size: 40px; }
                .dov-activity-empty p {
                    font-size: 14px;
                    color: #566168;
                    max-width: 320px;
                    margin: 0;
                }
                .dov-start-scan-btn {
                    margin-top: 8px;
                    padding: 12px 28px;
                    border: none;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    color: #f9f6ff;
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .dov-start-scan-btn:hover { opacity: 0.9; }

                /* Load More */
                .dov-load-more {
                    display: flex;
                    justify-content: center;
                    padding-top: 16px;
                }
                .dov-load-more-btn {
                    padding: 14px 48px;
                    border: none;
                    border-radius: 999px;
                    background: #e1e9f0;
                    color: #566168;
                    font-size: 14px;
                    font-weight: 700;
                    letter-spacing: -0.01em;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .dov-load-more-btn:hover {
                    background: #d9e4ec;
                    transform: scale(1.02);
                }

                /* Footer */
                .dov-footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding-top: 48px;
                    padding-bottom: 16px;
                }
                .dov-footer-left, .dov-footer-right {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #566168;
                    opacity: 0.5;
                }
                .dov-footer-left {
                    display: flex;
                    gap: 24px;
                }

                @media (max-width: 768px) {
                    .dov-container { padding: 20px; gap: 28px; }
                    .dov-page-header { flex-direction: column; align-items: flex-start; gap: 16px; }
                    .dov-last-analysis { text-align: left; }
                    .dov-title { font-size: 28px; }
                    .dov-scan-item { flex-direction: column; gap: 16px; align-items: flex-start; }
                    .dov-scan-right { text-align: left; }
                    .dov-footer { flex-direction: column; gap: 8px; align-items: flex-start; }
                }
            `}</style>
        </div>
    )
}

export default DashboardOverview
