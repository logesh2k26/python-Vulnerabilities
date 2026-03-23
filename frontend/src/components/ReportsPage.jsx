import React, { useState, useMemo } from 'react'

function ReportsPage({ scanHistory = [] }) {
    const [timeRange, setTimeRange] = useState('14d')

    const filteredScans = useMemo(() => {
        if (timeRange === 'all') return scanHistory
        const days = timeRange === '14d' ? 14 : 30
        const cutoff = new Date(Date.now() - days * 86400000)
        return scanHistory.filter(s => new Date(s.timestamp) >= cutoff)
    }, [scanHistory, timeRange])

    const stats = useMemo(() => {
        const total = filteredScans.length
        const vulnerable = filteredScans.filter(s => s.is_vulnerable).length
        const safe = total - vulnerable

        const severity = { critical: 0, high: 0, medium: 0, low: 0 }
        filteredScans.forEach(s => {
            if (s.is_vulnerable && s.severity) {
                severity[s.severity] = (severity[s.severity] || 0) + 1
            }
        })

        const typeMap = {}
        filteredScans.forEach(s => {
            if (s.is_vulnerable && s.vulnerability_type && s.vulnerability_type !== 'none') {
                const t = s.vulnerability_type.replace(/_/g, ' ')
                typeMap[t] = (typeMap[t] || 0) + 1
            }
        })

        const dailyCounts = []
        const dailyLabels = []
        const numDays = timeRange === '30d' ? 30 : 14
        for (let i = numDays - 1; i >= 0; i--) {
            const d = new Date()
            d.setDate(d.getDate() - i)
            const key = d.toISOString().split('T')[0]
            dailyLabels.push(d.toLocaleDateString('en', { month: 'short', day: 'numeric' }))
            dailyCounts.push(filteredScans.filter(s => s.timestamp?.startsWith(key)).length)
        }

        const avgConfidence = total > 0
            ? (filteredScans.reduce((s, v) => s + (v.confidence || 0), 0) / total * 100).toFixed(1)
            : '0.0'

        const healthPercent = total > 0 ? ((safe / total) * 100).toFixed(1) : '0.0'

        return { total, vulnerable, safe, severity, dailyCounts, dailyLabels, avgConfidence, healthPercent }
    }, [filteredScans, timeRange])

    // Use demo data if no real scans
    const displayStats = stats.total > 0 ? stats : {
        total: 12842, vulnerable: 429, safe: 11201, avgConfidence: '99.4', healthPercent: '87.2',
        severity: { critical: 64, high: 193, medium: 172, low: 0 },
        dailyCounts: [40, 55, 45, 70, 60, 85, 50, 65, 40, 75, 95, 80, 55, 100],
        dailyLabels: ['Oct 01', '', '', '', '', '', 'Oct 07', '', '', '', '', '', '', 'Oct 14']
    }

    const maxBar = Math.max(...displayStats.dailyCounts, 1)
    const totalSeverity = displayStats.severity.critical + displayStats.severity.high + displayStats.severity.medium + (displayStats.severity.low || 0)
    const critHighPct = totalSeverity > 0 ? Math.round(((displayStats.severity.critical + displayStats.severity.high) / totalSeverity) * 100) : 15
    const medPct = totalSeverity > 0 ? Math.round((displayStats.severity.medium / totalSeverity) * 100) : 45
    const lowPct = 100 - critHighPct - medPct

    // SVG donut calculations
    const c = 100 // circumference approximation for stroke-dasharray
    const critHighDash = (critHighPct / 100) * c
    const medDash = (medPct / 100) * c
    const lowDash = (lowPct / 100) * c

    const recentScans = filteredScans.length > 0 ? filteredScans.slice(0, 10) : [
        { id: 1, filename: 'payroll_ledger_q4.xlsb', is_vulnerable: true, severity: 'high', vulnerability_type: 'Macro/Obfuscated Code', timestamp: new Date(Date.now() - 120000).toISOString(), confidence: 0.94 },
        { id: 2, filename: 'client_onboarding_kit.zip', is_vulnerable: true, severity: 'medium', vulnerability_type: 'Suspicious URL Redirect', timestamp: new Date(Date.now() - 840000).toISOString(), confidence: 0.72 },
        { id: 3, filename: 'logo_horizontal_final.png', is_vulnerable: true, severity: 'low', vulnerability_type: 'Metadata Inconsistency', timestamp: new Date(Date.now() - 2520000).toISOString(), confidence: 0.35 },
        { id: 4, filename: 'compliance_v2_draft.docx', is_vulnerable: false, severity: 'low', vulnerability_type: 'none', timestamp: new Date(Date.now() - 3600000).toISOString(), confidence: 0.98 },
    ]

    const timeAgo = (ts) => {
        const mins = Math.floor((Date.now() - new Date(ts).getTime()) / 60000)
        if (mins < 1) return 'Just now'
        if (mins < 60) return `${mins} mins ago`
        const hrs = Math.floor(mins / 60)
        if (hrs < 24) return `${hrs} hour${hrs > 1 ? 's' : ''} ago`
        return `${Math.floor(hrs / 24)} days ago`
    }

    const riskLabel = (sev) => {
        if (sev === 'critical' || sev === 'high') return { text: 'HIGH RISK', cls: 'rp-risk-high' }
        if (sev === 'medium') return { text: 'MEDIUM RISK', cls: 'rp-risk-medium' }
        if (sev === 'low') return { text: 'LOW RISK', cls: 'rp-risk-low' }
        return { text: 'SAFE', cls: 'rp-risk-safe' }
    }

    const getFileIcon = (fn) => {
        if (fn.match(/\.(py|js|ts|jsx)$/)) return '📄'
        if (fn.match(/\.(zip|tar|gz)$/)) return '📦'
        if (fn.match(/\.(png|jpg|svg)$/)) return '🖼️'
        if (fn.match(/\.(doc|docx|pdf)$/)) return '📝'
        if (fn.match(/\.(xls|xlsb|csv)$/)) return '📊'
        return '📄'
    }

    // PDF export (preserved from original)
    const downloadReport = () => {
        const date = new Date().toLocaleString()
        const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>PyVulnDetect Report</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:Inter,sans-serif;color:#1f2937;background:#fff;padding:40px}
.header{text-align:center;margin-bottom:36px;padding-bottom:24px;border-bottom:2px solid #4c56af}
.logo{font-size:28px;font-weight:700;color:#4c56af;margin-bottom:4px}.subtitle{font-size:13px;color:#6b7280}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}
.stat{text-align:center;padding:18px;border:1px solid #e5e7eb;border-radius:10px}
.stat .v{font-size:28px;font-weight:700}.stat .l{font-size:11px;color:#6b7280;text-transform:uppercase}
.footer{text-align:center;margin-top:40px;font-size:11px;color:#9ca3af}</style></head><body>
<div class="header"><div class="logo">🛡️ PyVulnDetect</div><div class="subtitle">System Reports — ${date}</div></div>
<div class="stats"><div class="stat"><div class="v" style="color:#4c56af">${displayStats.total.toLocaleString()}</div><div class="l">Total Scans</div></div>
<div class="stat"><div class="v" style="color:#4d626c">${displayStats.safe.toLocaleString()}</div><div class="l">Safe Files</div></div>
<div class="stat"><div class="v" style="color:#9f403d">${displayStats.vulnerable}</div><div class="l">Vulnerable</div></div>
<div class="stat"><div class="v">${displayStats.avgConfidence}%</div><div class="l">Avg Confidence</div></div></div>
<div class="footer">Generated by PyVulnDetect • ${date}</div></body></html>`
        const w = window.open('', '_blank')
        w.document.write(html)
        w.document.close()
        setTimeout(() => w.print(), 500)
    }

    return (
        <div className="rpt-container">
            {/* Page Header */}
            <div className="rpt-top-bar">
                <h2 className="rpt-page-title">System Reports</h2>
                <button className="rpt-download" onClick={downloadReport}>📥 Download Report</button>
            </div>

            {/* Summary Stat Cards */}
            <div className="rpt-stats-row">
                <div className="rpt-stat-card">
                    <p className="rpt-stat-label">TOTAL SCANS</p>
                    <h3 className="rpt-stat-value">{displayStats.total.toLocaleString()}</h3>
                    <div className="rpt-stat-trend rpt-trend-primary">
                        <span>📈</span>
                        <span>12.5% vs last month</span>
                    </div>
                </div>
                <div className="rpt-stat-card">
                    <p className="rpt-stat-label">SAFE FILES</p>
                    <h3 className="rpt-stat-value">{displayStats.safe.toLocaleString()}</h3>
                    <div className="rpt-stat-trend rpt-trend-secondary">
                        <span>🛡️</span>
                        <span>{displayStats.healthPercent}% overall health</span>
                    </div>
                </div>
                <div className="rpt-stat-card rpt-stat-error">
                    <p className="rpt-stat-label rpt-label-error">VULNERABLE</p>
                    <h3 className="rpt-stat-value rpt-value-error">{displayStats.vulnerable}</h3>
                    <div className="rpt-stat-trend rpt-trend-error">
                        <span>⚠️</span>
                        <span>Action required</span>
                    </div>
                </div>
                <div className="rpt-stat-card">
                    <p className="rpt-stat-label">AVG CONFIDENCE</p>
                    <h3 className="rpt-stat-value">{displayStats.avgConfidence}%</h3>
                    <div className="rpt-stat-trend rpt-trend-muted">
                        <span>📊</span>
                        <span>Model v2.4 Active</span>
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="rpt-charts-row">
                {/* Bar Chart: Scan Activity */}
                <div className="rpt-chart-card rpt-chart-wide">
                    <div className="rpt-chart-header">
                        <div>
                            <h4 className="rpt-chart-title">Scan Activity</h4>
                            <p className="rpt-chart-sub">Historical data for the last {timeRange === '30d' ? '30' : '14'} days</p>
                        </div>
                        <div className="rpt-time-toggle">
                            <button className={`rpt-time-btn ${timeRange === '14d' ? 'active' : ''}`} onClick={() => setTimeRange('14d')}>14D</button>
                            <button className={`rpt-time-btn ${timeRange === '30d' ? 'active' : ''}`} onClick={() => setTimeRange('30d')}>30D</button>
                        </div>
                    </div>
                    <div className="rpt-bars-area">
                        {displayStats.dailyCounts.map((val, i) => (
                            <div key={i} className="rpt-bar-col">
                                <div
                                    className="rpt-bar"
                                    style={{
                                        height: `${(val / maxBar) * 100}%`,
                                        opacity: 0.15 + (val / maxBar) * 0.85
                                    }}
                                    title={`${displayStats.dailyLabels[i] || `Day ${i + 1}`}: ${val} scans`}
                                />
                            </div>
                        ))}
                    </div>
                    <div className="rpt-bar-labels">
                        <span>{displayStats.dailyLabels[0]}</span>
                        <span>{displayStats.dailyLabels[Math.floor(displayStats.dailyLabels.length / 2)]}</span>
                        <span>{displayStats.dailyLabels[displayStats.dailyLabels.length - 1]}</span>
                    </div>
                </div>

                {/* Donut Chart: Severity Distribution */}
                <div className="rpt-chart-card rpt-chart-narrow">
                    <h4 className="rpt-chart-title">Severity Distribution</h4>
                    <div className="rpt-donut-wrap">
                        <svg className="rpt-donut-svg" viewBox="0 0 36 36">
                            {/* Background */}
                            <path
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#e1e9f0" strokeWidth="3"
                                strokeDasharray="100, 100"
                            />
                            {/* Critical/High */}
                            <path
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#9f403d" strokeWidth="3"
                                strokeDasharray={`${critHighDash}, 100`}
                                strokeDashoffset="0"
                            />
                            {/* Medium */}
                            <path
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#4c56af" strokeWidth="3"
                                strokeDasharray={`${medDash}, 100`}
                                strokeDashoffset={`${-critHighDash}`}
                            />
                            {/* Low */}
                            <path
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#4d626c" strokeWidth="3"
                                strokeDasharray={`${lowDash}, 100`}
                                strokeDashoffset={`${-(critHighDash + medDash)}`}
                            />
                        </svg>
                        <div className="rpt-donut-center">
                            <span className="rpt-donut-num">{displayStats.vulnerable}</span>
                            <span className="rpt-donut-label">ALERTS</span>
                        </div>
                    </div>
                    <div className="rpt-legend">
                        <div className="rpt-legend-row">
                            <span className="rpt-legend-dot" style={{ background: '#9f403d' }} />
                            <span className="rpt-legend-text">Critical / High</span>
                            <span className="rpt-legend-pct">{critHighPct}%</span>
                        </div>
                        <div className="rpt-legend-row">
                            <span className="rpt-legend-dot" style={{ background: '#4c56af' }} />
                            <span className="rpt-legend-text">Medium Risk</span>
                            <span className="rpt-legend-pct">{medPct}%</span>
                        </div>
                        <div className="rpt-legend-row">
                            <span className="rpt-legend-dot" style={{ background: '#4d626c' }} />
                            <span className="rpt-legend-text">Low / Minor</span>
                            <span className="rpt-legend-pct">{lowPct}%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Scan Results Table */}
            <div className="rpt-table-card">
                <div className="rpt-table-header">
                    <div>
                        <h4 className="rpt-chart-title">Recent Scan Results</h4>
                        <p className="rpt-chart-sub">Live feed of global ledger activity</p>
                    </div>
                    <button className="rpt-audit-link">
                        View Audit Log <span>›</span>
                    </button>
                </div>
                <div className="rpt-table-wrap">
                    <table className="rpt-table">
                        <thead>
                            <tr>
                                <th>RESOURCE NAME</th>
                                <th>RISK LEVEL</th>
                                <th>THREAT TYPE</th>
                                <th>DETECTION TIME</th>
                                <th className="rpt-th-right">ACTIONS</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentScans.map((scan, i) => {
                                const risk = scan.is_vulnerable ? riskLabel(scan.severity) : riskLabel(null)
                                return (
                                    <tr key={scan.id || i} className="rpt-tr">
                                        <td>
                                            <div className="rpt-resource">
                                                <div className={`rpt-resource-icon ${scan.is_vulnerable ? `rpt-icon-${scan.severity}` : 'rpt-icon-safe'}`}>
                                                    {getFileIcon(scan.filename)}
                                                </div>
                                                <div>
                                                    <p className="rpt-resource-name">{scan.filename}</p>
                                                    <p className="rpt-resource-id">ID: {(scan.id || '').toString().slice(0, 4)}-{Math.random().toString(36).slice(2, 6)}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`rpt-risk-pill ${risk.cls}`}>{risk.text}</span>
                                        </td>
                                        <td className="rpt-td-type">
                                            {scan.is_vulnerable
                                                ? (scan.vulnerability_type?.replace(/_/g, ' ') || 'Unknown')
                                                : 'None detected'}
                                        </td>
                                        <td className="rpt-td-time">{timeAgo(scan.timestamp)}</td>
                                        <td className="rpt-td-actions">
                                            <button className="rpt-more-btn">⋮</button>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            <style>{`
                .rpt-container {
                    padding: 32px 40px;
                    max-width: 1500px;
                    margin: 0 auto;
                }

                /* ── Top Bar ── */
                .rpt-top-bar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 32px;
                }
                .rpt-page-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 26px;
                    font-weight: 800;
                    color: #0f172a;
                    letter-spacing: -0.02em;
                }
                .rpt-download {
                    padding: 10px 20px;
                    border: 1px solid rgba(76,86,175,0.15);
                    border-radius: 12px;
                    background: rgba(76,86,175,0.06);
                    color: #4c56af;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .rpt-download:hover {
                    background: rgba(76,86,175,0.12);
                    border-color: rgba(76,86,175,0.25);
                }

                /* ── Stat Cards ── */
                .rpt-stats-row {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 18px;
                    margin-bottom: 24px;
                }
                @media (max-width: 900px) {
                    .rpt-stats-row { grid-template-columns: repeat(2, 1fr); }
                }
                .rpt-stat-card {
                    background: #fff;
                    padding: 24px;
                    border-radius: 16px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(168,179,187,0.08);
                    transition: transform 0.2s;
                }
                .rpt-stat-card:hover { transform: translateY(-2px); }
                .rpt-stat-error {
                    background: rgba(254, 137, 131, 0.06);
                }
                .rpt-stat-label {
                    font-size: 10px;
                    font-weight: 800;
                    color: #94a3b8;
                    letter-spacing: 0.12em;
                    margin: 0 0 12px;
                }
                .rpt-label-error { color: #9f403d; }
                .rpt-stat-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 36px;
                    font-weight: 900;
                    color: #0f172a;
                    margin: 0;
                    letter-spacing: -0.02em;
                }
                .rpt-value-error { color: #9f403d; }
                .rpt-stat-trend {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    margin-top: 14px;
                    font-size: 12px;
                    font-weight: 700;
                }
                .rpt-trend-primary { color: #4c56af; }
                .rpt-trend-secondary { color: #4d626c; }
                .rpt-trend-error { color: #9f403d; }
                .rpt-trend-muted { color: #94a3b8; }

                /* ── Charts Row ── */
                .rpt-charts-row {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 18px;
                    margin-bottom: 24px;
                }
                @media (max-width: 900px) {
                    .rpt-charts-row { grid-template-columns: 1fr; }
                }
                .rpt-chart-card {
                    background: #fff;
                    border-radius: 16px;
                    padding: 28px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(168,179,187,0.08);
                }
                .rpt-chart-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 28px;
                }
                .rpt-chart-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 17px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 0;
                }
                .rpt-chart-sub {
                    font-size: 13px;
                    color: #94a3b8;
                    margin: 4px 0 0;
                }
                .rpt-time-toggle {
                    display: flex;
                    gap: 4px;
                    background: #e1e9f0;
                    border-radius: 10px;
                    padding: 3px;
                }
                .rpt-time-btn {
                    padding: 6px 14px;
                    border: none;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: 700;
                    cursor: pointer;
                    background: transparent;
                    color: #566168;
                    transition: all 0.15s;
                }
                .rpt-time-btn.active {
                    background: #fff;
                    color: #0f172a;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                }

                /* Bar Chart */
                .rpt-bars-area {
                    display: flex;
                    align-items: flex-end;
                    gap: 6px;
                    height: 220px;
                    padding: 0 8px;
                }
                .rpt-bar-col {
                    flex: 1;
                    display: flex;
                    align-items: flex-end;
                    height: 100%;
                }
                .rpt-bar {
                    width: 100%;
                    background: #4c56af;
                    border-radius: 4px 4px 0 0;
                    min-height: 4px;
                    transition: height 0.4s ease;
                }
                .rpt-bar-labels {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 8px 0;
                    font-size: 10px;
                    font-weight: 700;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.02em;
                }

                /* Donut Chart */
                .rpt-chart-narrow {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .rpt-chart-narrow .rpt-chart-title {
                    align-self: flex-start;
                    margin-bottom: 24px;
                }
                .rpt-donut-wrap {
                    position: relative;
                    width: 180px;
                    height: 180px;
                    margin-bottom: 28px;
                }
                .rpt-donut-svg {
                    width: 100%;
                    height: 100%;
                    transform: rotate(-90deg);
                }
                .rpt-donut-center {
                    position: absolute;
                    inset: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                .rpt-donut-num {
                    font-family: 'Manrope', sans-serif;
                    font-size: 28px;
                    font-weight: 900;
                    color: #0f172a;
                }
                .rpt-donut-label {
                    font-size: 10px;
                    font-weight: 700;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                }

                /* Legend */
                .rpt-legend {
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .rpt-legend-row {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 13px;
                }
                .rpt-legend-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    flex-shrink: 0;
                }
                .rpt-legend-text {
                    flex: 1;
                    color: #475569;
                    font-weight: 500;
                }
                .rpt-legend-pct {
                    font-weight: 700;
                    color: #0f172a;
                }

                /* ── Table ── */
                .rpt-table-card {
                    background: #fff;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 0 0 1px rgba(168,179,187,0.08);
                }
                .rpt-table-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 28px;
                }
                .rpt-audit-link {
                    background: none;
                    border: none;
                    color: #4c56af;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    transition: opacity 0.15s;
                }
                .rpt-audit-link:hover { opacity: 0.7; text-decoration: underline; }

                .rpt-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .rpt-table thead tr {
                    background: rgba(240,244,248,0.5);
                }
                .rpt-table th {
                    padding: 14px 28px;
                    font-size: 10px;
                    font-weight: 800;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    text-align: left;
                    border-bottom: 1px solid rgba(168,179,187,0.1);
                }
                .rpt-th-right { text-align: right; }

                .rpt-tr {
                    transition: background 0.12s;
                }
                .rpt-tr:hover { background: rgba(240,244,248,0.3); }
                .rpt-table td {
                    padding: 18px 28px;
                    border-bottom: 1px solid rgba(168,179,187,0.06);
                    vertical-align: middle;
                }

                /* Resource cell */
                .rpt-resource {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }
                .rpt-resource-icon {
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    flex-shrink: 0;
                }
                .rpt-icon-high, .rpt-icon-critical {
                    background: rgba(254,137,131,0.1);
                }
                .rpt-icon-medium {
                    background: rgba(224,224,255,0.3);
                }
                .rpt-icon-low {
                    background: rgba(209,228,254,0.3);
                }
                .rpt-icon-safe {
                    background: rgba(207,230,242,0.3);
                }
                .rpt-resource-name {
                    font-size: 14px;
                    font-weight: 700;
                    color: #0f172a;
                    margin: 0;
                }
                .rpt-resource-id {
                    font-size: 11px;
                    color: #94a3b8;
                    margin: 2px 0 0;
                }

                /* Risk pills */
                .rpt-risk-pill {
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 999px;
                    font-size: 10px;
                    font-weight: 900;
                    text-transform: uppercase;
                    letter-spacing: 0.04em;
                }
                .rpt-risk-high {
                    background: #fe8983;
                    color: #752121;
                }
                .rpt-risk-medium {
                    background: #e0e0ff;
                    color: #3f48a1;
                }
                .rpt-risk-low {
                    background: #d1e4fe;
                    color: #415368;
                }
                .rpt-risk-safe {
                    background: #cfe6f2;
                    color: #40555f;
                }

                .rpt-td-type {
                    font-size: 14px;
                    color: #475569;
                }
                .rpt-td-time {
                    font-size: 14px;
                    color: #475569;
                }
                .rpt-td-actions {
                    text-align: right;
                }
                .rpt-more-btn {
                    background: none;
                    border: none;
                    font-size: 20px;
                    color: #94a3b8;
                    cursor: pointer;
                    padding: 4px 8px;
                    border-radius: 8px;
                    transition: background 0.15s;
                }
                .rpt-more-btn:hover {
                    background: #e1e9f0;
                    color: #475569;
                }
            `}</style>
        </div>
    )
}

export default ReportsPage
