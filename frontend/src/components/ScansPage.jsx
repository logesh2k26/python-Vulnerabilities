import React, { useState } from 'react'

function ScansPage() {
    const [scans] = useState([
        { id: 1, filename: 'app.py', date: '2026-03-20 10:30', status: 'vulnerable', issues: 3, confidence: 78 },
        { id: 2, filename: 'utils.py', date: '2026-03-19 15:45', status: 'safe', issues: 0, confidence: 95 },
        { id: 3, filename: 'database.py', date: '2026-03-19 14:20', status: 'vulnerable', issues: 5, confidence: 82 },
        { id: 4, filename: 'auth.py', date: '2026-03-18 09:15', status: 'vulnerable', issues: 2, confidence: 67 },
        { id: 5, filename: 'config.py', date: '2026-03-18 08:00', status: 'safe', issues: 0, confidence: 91 },
    ])

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">🔍 Scan History</h1>
                <p className="page-subtitle">View all previous vulnerability scans</p>
            </div>

            <div className="scans-list">
                {scans.map(scan => (
                    <div key={scan.id} className="glass-card scan-item">
                        <div className="scan-info">
                            <div className="scan-filename">{scan.filename}</div>
                            <div className="scan-date">{scan.date}</div>
                        </div>
                        <div className="scan-stats">
                            <span className={`scan-status ${scan.status}`}>
                                {scan.status === 'vulnerable' ? '⚠️' : '✅'} {scan.status.toUpperCase()}
                            </span>
                            <span className="scan-issues">{scan.issues} issues</span>
                            <span className="scan-confidence">{scan.confidence}% conf.</span>
                        </div>
                    </div>
                ))}
            </div>

            {scans.length === 0 && (
                <div className="glass-card empty-state">
                    <div className="empty-icon">📋</div>
                    <h3 className="empty-title">No Scans Yet</h3>
                    <p className="empty-text">Run your first scan from the Dashboard</p>
                </div>
            )}

            <style>{`
                .scans-list {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .scan-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 18px 24px;
                    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
                    cursor: pointer;
                }
                .scan-item:hover {
                    transform: translateY(-2px) translateX(4px);
                    box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 20px rgba(0,240,255,0.06);
                    border-color: rgba(0, 240, 255, 0.15);
                }
                .scan-filename {
                    font-family: var(--font-code);
                    font-size: 15px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin-bottom: 4px;
                }
                .scan-date {
                    font-size: 12px;
                    color: var(--text-muted);
                }
                .scan-stats {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                }
                .scan-status {
                    font-size: 12px;
                    font-weight: 700;
                    padding: 4px 12px;
                    border-radius: 8px;
                    letter-spacing: 0.04em;
                }
                .scan-status.vulnerable {
                    color: var(--neon-red);
                    background: rgba(255, 45, 85, 0.1);
                    border: 1px solid rgba(255, 45, 85, 0.15);
                }
                .scan-status.safe {
                    color: var(--neon-green);
                    background: rgba(57, 255, 20, 0.08);
                    border: 1px solid rgba(57, 255, 20, 0.12);
                }
                .scan-issues {
                    font-size: 13px;
                    color: var(--text-secondary);
                }
                .scan-confidence {
                    font-size: 13px;
                    color: var(--text-muted);
                    font-family: var(--font-code);
                }
            `}</style>
        </div>
    )
}

export default ScansPage
