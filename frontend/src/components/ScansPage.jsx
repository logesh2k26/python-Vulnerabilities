import React, { useState, useMemo } from 'react'

function ScansPage({ scanHistory = [], onNavigate }) {
    const [currentPage, setCurrentPage] = useState(1)
    const [searchQuery, setSearchQuery] = useState('')
    const itemsPerPage = 8

    // Use scanHistory from props if available, otherwise show demo data
    const allScans = useMemo(() => {
        if (scanHistory.length > 0) return scanHistory
        return [
            { id: 1, filename: 'payment_gateway_v2.py', timestamp: '2026-03-22T14:22:00Z', is_vulnerable: false, confidence: 0.992, vulnerability_type: 'none', severity: 'low', project: 'PyCommerce', size: '2.4MB' },
            { id: 2, filename: 'user_controller.py', timestamp: '2026-03-21T09:15:00Z', is_vulnerable: true, confidence: 0.845, vulnerability_type: 'command_injection', severity: 'high', project: 'Core API', size: '1.1MB' },
            { id: 3, filename: 'database_config.yaml', timestamp: '2026-03-20T18:44:00Z', is_vulnerable: false, confidence: 1.0, vulnerability_type: 'none', severity: 'low', project: 'Infra', size: '0.2MB' },
            { id: 4, filename: 'legacy_modules.zip', timestamp: '2026-03-19T11:02:00Z', is_vulnerable: false, confidence: 0.978, vulnerability_type: 'none', severity: 'low', project: 'Archive', size: '156MB' },
            { id: 5, filename: 'auth_service.py', timestamp: '2026-03-18T16:30:00Z', is_vulnerable: true, confidence: 0.984, vulnerability_type: 'sql_injection', severity: 'high', project: 'Core API', size: '3.1MB' },
        ]
    }, [scanHistory])

    const filteredScans = useMemo(() => {
        if (!searchQuery.trim()) return allScans
        const q = searchQuery.toLowerCase()
        return allScans.filter(s => s.filename.toLowerCase().includes(q))
    }, [allScans, searchQuery])

    const totalScans = filteredScans.length
    const vulnerableCount = filteredScans.filter(s => s.is_vulnerable).length
    const totalPages = Math.max(1, Math.ceil(totalScans / itemsPerPage))
    const paged = filteredScans.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

    // Most recent critical scan for highlight card
    const criticalScan = allScans.find(s => s.is_vulnerable && s.confidence > 0.8)

    const formatDate = (ts) => {
        const d = new Date(ts)
        return {
            date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
            time: d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
        }
    }

    const timeAgo = (ts) => {
        const diff = Date.now() - new Date(ts).getTime()
        const mins = Math.floor(diff / 60000)
        if (mins < 60) return `${mins} minutes ago`
        const hrs = Math.floor(mins / 60)
        if (hrs < 24) return `${hrs} hours ago`
        const days = Math.floor(hrs / 24)
        return `${days} days ago`
    }

    const getFileIcon = (filename) => {
        if (filename.endsWith('.py')) return '📄'
        if (filename.endsWith('.yaml') || filename.endsWith('.yml')) return '⚙️'
        if (filename.endsWith('.zip') || filename.endsWith('.tar')) return '📦'
        if (filename.endsWith('.json')) return '{ }'
        return '📄'
    }

    return (
        <div className="hp-container">
            {/* Page Header */}
            <div className="hp-header">
                <div className="hp-header-left">
                    <p className="hp-label-tag">SECURITY AUDIT LEDGER</p>
                    <h2 className="hp-page-title">Scan History</h2>
                </div>
                <div className="hp-header-stats">
                    <div className="hp-stat-box">
                        <p className="hp-stat-label">Total Scans</p>
                        <p className="hp-stat-value">{totalScans.toLocaleString()}</p>
                    </div>
                    <div className="hp-stat-box">
                        <p className="hp-stat-label">Vulnerable</p>
                        <p className="hp-stat-value hp-stat-error">{vulnerableCount}</p>
                    </div>
                </div>
            </div>

            {/* Bento Grid: Highlight Cards */}
            <div className="hp-bento">
                {/* Main Highlight Card */}
                <div className="hp-highlight-main">
                    <div className="hp-highlight-glow" />
                    <div className="hp-highlight-body">
                        <div className="hp-highlight-left">
                            <span className="hp-risk-tag">
                                {criticalScan ? 'HIGH RISK ALERT' : 'RECENT SCAN'}
                            </span>
                            <h3 className="hp-highlight-filename">
                                {criticalScan?.filename || allScans[0]?.filename || 'No scans yet'}
                            </h3>
                            <p className="hp-highlight-desc">
                                {criticalScan
                                    ? `Critical vulnerability detected. ${criticalScan.vulnerability_type?.replace('_', ' ')} pattern found.`
                                    : 'Your most recent scan was clean.'}
                            </p>
                        </div>
                        <div className="hp-highlight-right">
                            <p className="hp-conf-label">CONFIDENCE SCORE</p>
                            <p className="hp-conf-value">
                                {((criticalScan?.confidence || allScans[0]?.confidence || 0) * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>
                    <div className="hp-highlight-actions">
                        <button className="hp-review-btn" onClick={() => onNavigate && onNavigate('scans')}>
                            <span>Review Details</span>
                            <span>→</span>
                        </button>
                        <span className="hp-time-ago">
                            Scanned {timeAgo(criticalScan?.timestamp || allScans[0]?.timestamp || new Date().toISOString())}
                        </span>
                    </div>
                </div>

                {/* Weekly Safety Card */}
                <div className="hp-safety-card">
                    <div className="hp-safety-gradient" />
                    <div className="hp-safety-content">
                        <span className="hp-safety-icon">🛡️</span>
                        <h3 className="hp-safety-title">Weekly Safety</h3>
                        <p className="hp-safety-desc">
                            Your repository health has improved by {Math.max(0, 100 - vulnerableCount * 3)}% since last week.
                        </p>
                        <div className="hp-progress-bar">
                            <div className="hp-progress-track">
                                <div 
                                    className="hp-progress-fill" 
                                    style={{ width: `${Math.max(0, 100 - vulnerableCount * 3)}%` }}
                                />
                            </div>
                            <span className="hp-progress-label">
                                {Math.max(0, 100 - vulnerableCount * 3)}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* History Table */}
            <div className="hp-table-wrapper">
                <div className="hp-table-inner">
                    {/* Table Header */}
                    <div className="hp-table-head">
                        <div className="hp-th hp-th-file">FILE & PROJECT NAME</div>
                        <div className="hp-th hp-th-date">SCAN DATE</div>
                        <div className="hp-th hp-th-status">STATUS</div>
                        <div className="hp-th hp-th-conf">CONFIDENCE</div>
                        <div className="hp-th hp-th-actions">ACTIONS</div>
                    </div>

                    {/* Rows */}
                    {paged.length > 0 ? paged.map((scan, idx) => {
                        const dt = formatDate(scan.timestamp)
                        return (
                            <div 
                                key={scan.id || idx} 
                                className={`hp-table-row ${idx % 2 === 1 ? 'hp-row-alt' : ''}`}
                            >
                                <div className="hp-td hp-td-file">
                                    <div className={`hp-file-icon ${scan.is_vulnerable ? 'hp-icon-vuln' : 'hp-icon-safe'}`}>
                                        {getFileIcon(scan.filename)}
                                    </div>
                                    <div>
                                        <p className="hp-file-name">{scan.filename}</p>
                                        <p className="hp-file-project">
                                            Project: {scan.project || 'Default'} {scan.size ? `• ${scan.size}` : ''}
                                        </p>
                                    </div>
                                </div>
                                <div className="hp-td hp-td-date">
                                    <p className="hp-date-main">{dt.date}</p>
                                    <p className="hp-date-time">{dt.time}</p>
                                </div>
                                <div className="hp-td hp-td-status">
                                    <span className={`hp-status-pill ${scan.is_vulnerable ? 'hp-pill-vuln' : 'hp-pill-safe'}`}>
                                        <span className="hp-pill-dot" />
                                        <span>{scan.is_vulnerable ? 'VULNERABLE' : 'SAFE'}</span>
                                    </span>
                                </div>
                                <div className="hp-td hp-td-conf">
                                    <p className={`hp-conf-num ${scan.is_vulnerable ? 'hp-conf-red' : ''}`}>
                                        {(scan.confidence * 100).toFixed(1)}%
                                    </p>
                                </div>
                                <div className="hp-td hp-td-actions">
                                    <button className="hp-action-btn" title="View details">👁️</button>
                                    <button className="hp-action-btn" title="Download report">⬇️</button>
                                </div>
                            </div>
                        )
                    }) : (
                        <div className="hp-empty-row">
                            <span className="hp-empty-icon">📋</span>
                            <p>No scan history found</p>
                        </div>
                    )}

                    {/* Pagination Footer */}
                    <div className="hp-pagination">
                        <p className="hp-page-info">
                            Showing {Math.min((currentPage - 1) * itemsPerPage + 1, totalScans)} to {Math.min(currentPage * itemsPerPage, totalScans)} of {totalScans.toLocaleString()} scans
                        </p>
                        <div className="hp-page-btns">
                            <button 
                                className="hp-page-btn hp-page-nav"
                                disabled={currentPage <= 1}
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                            >‹</button>
                            {Array.from({ length: Math.min(3, totalPages) }, (_, i) => i + 1).map(pg => (
                                <button
                                    key={pg}
                                    className={`hp-page-btn ${currentPage === pg ? 'hp-page-active' : ''}`}
                                    onClick={() => setCurrentPage(pg)}
                                >{pg}</button>
                            ))}
                            <button 
                                className="hp-page-btn hp-page-nav"
                                disabled={currentPage >= totalPages}
                                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                            >›</button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Info Footer Cards */}
            <div className="hp-footer-cards">
                <div className="hp-info-card hp-info-primary">
                    <h4 className="hp-info-title">Auto-Retention</h4>
                    <p className="hp-info-desc">Scans older than 90 days are automatically archived to the encrypted cold-storage tier.</p>
                </div>
                <div className="hp-info-card hp-info-indigo">
                    <h4 className="hp-info-title">Export Formats</h4>
                    <p className="hp-info-desc">Reports can be exported in JSON, PDF, and SARIF formats for SIEM integration.</p>
                </div>
                <div className="hp-info-card hp-info-slate">
                    <h4 className="hp-info-title">Global Coverage</h4>
                    <p className="hp-info-desc">All active regions are currently synced with the central vulnerability database v4.2.1.</p>
                </div>
            </div>

            <style>{`
                .hp-container {
                    padding: 48px;
                    max-width: 1200px;
                    margin: 0 auto;
                }

                /* ── Header ── */
                .hp-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    margin-bottom: 40px;
                }
                .hp-label-tag {
                    font-size: 10px;
                    font-weight: 800;
                    color: #4c56af;
                    letter-spacing: 0.2em;
                    text-transform: uppercase;
                    margin: 0 0 6px;
                }
                .hp-page-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 36px;
                    font-weight: 900;
                    color: #0f172a;
                    margin: 0;
                    letter-spacing: -0.02em;
                }
                .hp-header-stats {
                    display: flex;
                    gap: 16px;
                }
                .hp-stat-box {
                    background: #f0f4f8;
                    padding: 14px 24px;
                    border-radius: 14px;
                }
                .hp-stat-label {
                    font-size: 10px;
                    font-weight: 700;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin: 0 0 4px;
                }
                .hp-stat-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 24px;
                    font-weight: 900;
                    color: #0f172a;
                    margin: 0;
                }
                .hp-stat-error {
                    color: #9f403d;
                }

                /* ── Bento Grid ── */
                .hp-bento {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 20px;
                    margin-bottom: 40px;
                }
                @media (max-width: 900px) {
                    .hp-bento { grid-template-columns: 1fr; }
                }

                /* Highlight Card */
                .hp-highlight-main {
                    background: #fff;
                    border-radius: 20px;
                    padding: 32px;
                    position: relative;
                    overflow: hidden;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(168,179,187,0.08);
                }
                .hp-highlight-glow {
                    position: absolute;
                    top: -80px;
                    right: -80px;
                    width: 256px;
                    height: 256px;
                    background: rgba(76, 86, 175, 0.04);
                    border-radius: 50%;
                    pointer-events: none;
                }
                .hp-highlight-body {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    position: relative;
                    z-index: 1;
                }
                .hp-highlight-left { flex: 1; }
                .hp-risk-tag {
                    display: inline-block;
                    font-size: 10px;
                    font-weight: 800;
                    padding: 5px 12px;
                    border-radius: 999px;
                    background: #fe8983;
                    color: #752121;
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                    margin-bottom: 14px;
                }
                .hp-highlight-filename {
                    font-family: 'Manrope', sans-serif;
                    font-size: 22px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 0 0 8px;
                }
                .hp-highlight-desc {
                    font-size: 14px;
                    color: #64748b;
                    margin: 0;
                    max-width: 400px;
                    line-height: 1.6;
                }
                .hp-highlight-right { text-align: right; }
                .hp-conf-label {
                    font-size: 10px;
                    font-weight: 700;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin: 0 0 4px;
                }
                .hp-conf-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 36px;
                    font-weight: 900;
                    color: #9f403d;
                    margin: 0;
                }
                .hp-highlight-actions {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    margin-top: 24px;
                    position: relative;
                    z-index: 1;
                }
                .hp-review-btn {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px 22px;
                    background: #0f172a;
                    color: #fff;
                    border: none;
                    border-radius: 12px;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .hp-review-btn:hover {
                    background: #1e293b;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                .hp-time-ago {
                    font-size: 12px;
                    color: #94a3b8;
                    font-style: italic;
                    font-weight: 500;
                }

                /* Safety Card */
                .hp-safety-card {
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    border-radius: 20px;
                    padding: 32px;
                    position: relative;
                    overflow: hidden;
                    color: #fff;
                }
                .hp-safety-gradient {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    height: 50%;
                    background: linear-gradient(to top, rgba(0,0,0,0.2), transparent);
                    pointer-events: none;
                }
                .hp-safety-content {
                    position: relative;
                    z-index: 1;
                }
                .hp-safety-icon {
                    font-size: 32px;
                    display: block;
                    margin-bottom: 14px;
                }
                .hp-safety-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    margin: 0 0 8px;
                }
                .hp-safety-desc {
                    font-size: 13px;
                    color: rgba(255,255,255,0.7);
                    margin: 0 0 20px;
                    line-height: 1.6;
                }
                .hp-progress-bar {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .hp-progress-track {
                    flex: 1;
                    height: 4px;
                    background: rgba(255,255,255,0.2);
                    border-radius: 4px;
                    overflow: hidden;
                }
                .hp-progress-fill {
                    height: 100%;
                    background: #fff;
                    border-radius: 4px;
                    transition: width 0.6s ease;
                }
                .hp-progress-label {
                    font-size: 12px;
                    font-weight: 800;
                }

                /* ── Table ── */
                .hp-table-wrapper {
                    background: rgba(240, 244, 248, 0.5);
                    border-radius: 20px;
                    padding: 4px;
                    margin-bottom: 40px;
                }
                .hp-table-inner {
                    background: #fff;
                    border-radius: 18px;
                    overflow: hidden;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
                }

                /* Table Header */
                .hp-table-head {
                    display: grid;
                    grid-template-columns: 3fr 2fr 1.5fr 1.5fr 1fr;
                    gap: 16px;
                    padding: 18px 28px;
                    border-bottom: 1px solid rgba(168,179,187,0.12);
                    background: rgba(240,244,248,0.3);
                }
                .hp-th {
                    font-size: 10px;
                    font-weight: 800;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                }
                .hp-th-status, .hp-th-conf { text-align: center; }
                .hp-th-actions { text-align: right; }

                /* Table Rows */
                .hp-table-row {
                    display: grid;
                    grid-template-columns: 3fr 2fr 1.5fr 1.5fr 1fr;
                    gap: 16px;
                    padding: 18px 28px;
                    align-items: center;
                    transition: background 0.15s;
                    cursor: pointer;
                }
                .hp-table-row:hover {
                    background: #f8fafc;
                }
                .hp-row-alt {
                    background: rgba(240,244,248,0.2);
                }

                /* File cell */
                .hp-td-file {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }
                .hp-file-icon {
                    width: 40px;
                    height: 40px;
                    border-radius: 12px;
                    background: #f1f5f9;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    transition: all 0.2s;
                    flex-shrink: 0;
                }
                .hp-table-row:hover .hp-icon-safe {
                    background: #e0e0ff;
                }
                .hp-table-row:hover .hp-icon-vuln {
                    background: rgba(254, 137, 131, 0.2);
                }
                .hp-file-name {
                    font-size: 14px;
                    font-weight: 700;
                    color: #0f172a;
                    margin: 0;
                }
                .hp-file-project {
                    font-size: 12px;
                    color: #94a3b8;
                    margin: 2px 0 0;
                }

                /* Date cell */
                .hp-date-main {
                    font-size: 14px;
                    font-weight: 500;
                    color: #475569;
                    margin: 0;
                }
                .hp-date-time {
                    font-size: 10px;
                    color: #94a3b8;
                    margin: 2px 0 0;
                }

                /* Status cell */
                .hp-td-status {
                    display: flex;
                    justify-content: center;
                }
                .hp-status-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    padding: 5px 14px;
                    border-radius: 999px;
                    font-size: 11px;
                    font-weight: 700;
                }
                .hp-pill-dot {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                }
                .hp-pill-safe {
                    background: rgba(209, 228, 254, 0.3);
                    color: #415368;
                }
                .hp-pill-safe .hp-pill-dot {
                    background: #4c56af;
                    animation: hp-pulse 2s ease-in-out infinite;
                }
                .hp-pill-vuln {
                    background: rgba(254, 137, 131, 0.15);
                    color: #752121;
                }
                .hp-pill-vuln .hp-pill-dot {
                    background: #9f403d;
                }
                @keyframes hp-pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.4; }
                }

                /* Confidence cell */
                .hp-td-conf { text-align: center; }
                .hp-conf-num {
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 900;
                    color: #0f172a;
                    margin: 0;
                }
                .hp-conf-red { color: #9f403d; }

                /* Actions cell */
                .hp-td-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 8px;
                }
                .hp-action-btn {
                    background: none;
                    border: none;
                    font-size: 16px;
                    cursor: pointer;
                    padding: 4px;
                    opacity: 0.4;
                    transition: all 0.15s;
                }
                .hp-action-btn:hover {
                    opacity: 1;
                    transform: scale(1.15);
                }

                /* Empty row */
                .hp-empty-row {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 8px;
                    padding: 48px 28px;
                    color: #94a3b8;
                    font-size: 14px;
                }
                .hp-empty-icon {
                    font-size: 36px;
                    margin-bottom: 4px;
                }

                /* ── Pagination ── */
                .hp-pagination {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 18px 28px;
                    border-top: 1px solid rgba(168,179,187,0.08);
                }
                .hp-page-info {
                    font-size: 12px;
                    font-weight: 500;
                    color: #94a3b8;
                    margin: 0;
                }
                .hp-page-btns {
                    display: flex;
                    gap: 6px;
                }
                .hp-page-btn {
                    width: 36px;
                    height: 36px;
                    border-radius: 10px;
                    border: 1px solid rgba(168,179,187,0.25);
                    background: #fff;
                    color: #475569;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.15s;
                }
                .hp-page-btn:hover:not(:disabled) {
                    background: #f1f5f9;
                }
                .hp-page-btn:disabled {
                    opacity: 0.3;
                    cursor: not-allowed;
                }
                .hp-page-active {
                    background: #4c56af !important;
                    color: #fff !important;
                    border-color: #4c56af !important;
                }
                .hp-page-nav {
                    color: #94a3b8;
                    font-size: 18px;
                }

                /* ── Footer Info Cards ── */
                .hp-footer-cards {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-top: 48px;
                }
                @media (max-width: 768px) {
                    .hp-footer-cards { grid-template-columns: 1fr; }
                }
                .hp-info-card {
                    padding: 22px;
                    background: rgba(248, 250, 252, 0.5);
                    border-radius: 14px;
                    border-left: 4px solid transparent;
                }
                .hp-info-primary { border-left-color: #4c56af; }
                .hp-info-indigo { border-left-color: #818cf8; }
                .hp-info-slate { border-left-color: #cbd5e1; }
                .hp-info-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 0 0 8px;
                }
                .hp-info-desc {
                    font-size: 12px;
                    color: #64748b;
                    line-height: 1.7;
                    margin: 0;
                }
            `}</style>
        </div>
    )
}

export default ScansPage
