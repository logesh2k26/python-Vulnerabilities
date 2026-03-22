import React, { useRef, useEffect, useState, useMemo } from 'react'

// ── Chart drawing helpers (pure Canvas, no library) ──

function drawDoughnutChart(canvas, data, colors, labels) {
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const size = canvas.parentElement.clientWidth
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = size + 'px'
    canvas.style.height = size + 'px'
    ctx.scale(dpr, dpr)

    const cx = size / 2, cy = size / 2
    const outerR = size / 2 - 16
    const innerR = outerR * 0.62
    const total = data.reduce((s, v) => s + v, 0)

    if (total === 0) {
        ctx.beginPath()
        ctx.arc(cx, cy, outerR, 0, Math.PI * 2)
        ctx.arc(cx, cy, innerR, 0, Math.PI * 2, true)
        ctx.fillStyle = 'rgba(120, 100, 255, 0.08)'
        ctx.fill()
        ctx.fillStyle = 'rgba(140, 140, 180, 0.5)'
        ctx.font = '14px Inter, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('No data yet', cx, cy + 5)
        return
    }

    let startAngle = -Math.PI / 2
    data.forEach((value, i) => {
        if (value === 0) return
        const sliceAngle = (value / total) * Math.PI * 2
        ctx.beginPath()
        ctx.arc(cx, cy, outerR, startAngle, startAngle + sliceAngle)
        ctx.arc(cx, cy, innerR, startAngle + sliceAngle, startAngle, true)
        ctx.closePath()
        ctx.fillStyle = colors[i]
        ctx.fill()
        // Subtle glow
        ctx.shadowColor = colors[i]
        ctx.shadowBlur = 12
        ctx.fill()
        ctx.shadowBlur = 0

        startAngle += sliceAngle
    })

    // Center text
    ctx.fillStyle = '#eef0ff'
    ctx.font = 'bold 28px Outfit, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(total.toString(), cx, cy - 6)
    ctx.fillStyle = 'rgba(140, 140, 180, 0.7)'
    ctx.font = '11px Inter, sans-serif'
    ctx.fillText('TOTAL ISSUES', cx, cy + 14)
}

function drawBarChart(canvas, data, colors, labels) {
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.parentElement.clientWidth
    const h = 220
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'
    ctx.scale(dpr, dpr)

    const padding = { top: 20, right: 20, bottom: 50, left: 40 }
    const chartW = w - padding.left - padding.right
    const chartH = h - padding.top - padding.bottom
    const maxVal = Math.max(...data, 1)

    // Grid lines
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i
        ctx.strokeStyle = 'rgba(120, 100, 255, 0.06)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(w - padding.right, y)
        ctx.stroke()

        ctx.fillStyle = 'rgba(140, 140, 180, 0.5)'
        ctx.font = '10px JetBrains Mono, monospace'
        ctx.textAlign = 'right'
        ctx.fillText(Math.round(maxVal - (maxVal / 4) * i).toString(), padding.left - 8, y + 4)
    }

    const barWidth = Math.min(36, (chartW / data.length) * 0.6)
    const gap = (chartW - barWidth * data.length) / (data.length + 1)

    data.forEach((value, i) => {
        const x = padding.left + gap * (i + 1) + barWidth * i
        const barH = (value / maxVal) * chartH
        const y = padding.top + chartH - barH

        // Bar with rounded top
        const r = Math.min(6, barWidth / 2)
        ctx.beginPath()
        ctx.moveTo(x, y + r)
        ctx.arcTo(x, y, x + barWidth, y, r)
        ctx.arcTo(x + barWidth, y, x + barWidth, y + barH, r)
        ctx.lineTo(x + barWidth, padding.top + chartH)
        ctx.lineTo(x, padding.top + chartH)
        ctx.closePath()

        const grad = ctx.createLinearGradient(x, y, x, padding.top + chartH)
        grad.addColorStop(0, colors[i % colors.length])
        grad.addColorStop(1, colors[i % colors.length] + '33')
        ctx.fillStyle = grad
        ctx.fill()

        // Glow
        ctx.shadowColor = colors[i % colors.length]
        ctx.shadowBlur = 8
        ctx.fill()
        ctx.shadowBlur = 0

        // Value on top
        if (value > 0) {
            ctx.fillStyle = '#eef0ff'
            ctx.font = 'bold 11px Outfit, sans-serif'
            ctx.textAlign = 'center'
            ctx.fillText(value.toString(), x + barWidth / 2, y - 6)
        }

        // Label below
        ctx.fillStyle = 'rgba(140, 140, 180, 0.7)'
        ctx.font = '9px Inter, sans-serif'
        ctx.textAlign = 'center'
        ctx.save()
        ctx.translate(x + barWidth / 2, padding.top + chartH + 12)
        ctx.rotate(Math.PI / 6)
        ctx.fillText(labels[i]?.substring(0, 10) || '', 0, 0)
        ctx.restore()
    })
}

function drawLineChart(canvas, data, labels) {
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.parentElement.clientWidth
    const h = 200
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'
    ctx.scale(dpr, dpr)

    const padding = { top: 20, right: 20, bottom: 36, left: 40 }
    const chartW = w - padding.left - padding.right
    const chartH = h - padding.top - padding.bottom
    const maxVal = Math.max(...data, 1)

    // Grid
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i
        ctx.strokeStyle = 'rgba(120, 100, 255, 0.06)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(w - padding.right, y)
        ctx.stroke()
    }

    if (data.length < 2) {
        ctx.fillStyle = 'rgba(140, 140, 180, 0.5)'
        ctx.font = '13px Inter, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('Need more scan data to show trends', w / 2, h / 2)
        return
    }

    const stepX = chartW / (data.length - 1)
    const points = data.map((v, i) => ({
        x: padding.left + stepX * i,
        y: padding.top + chartH - (v / maxVal) * chartH
    }))

    // Area fill
    ctx.beginPath()
    ctx.moveTo(points[0].x, padding.top + chartH)
    points.forEach(p => ctx.lineTo(p.x, p.y))
    ctx.lineTo(points[points.length - 1].x, padding.top + chartH)
    ctx.closePath()
    const areaGrad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH)
    areaGrad.addColorStop(0, 'rgba(0, 240, 255, 0.15)')
    areaGrad.addColorStop(1, 'rgba(0, 240, 255, 0.01)')
    ctx.fillStyle = areaGrad
    ctx.fill()

    // Line
    ctx.beginPath()
    points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
    ctx.strokeStyle = '#00f0ff'
    ctx.lineWidth = 2.5
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.shadowColor = '#00f0ff'
    ctx.shadowBlur = 10
    ctx.stroke()
    ctx.shadowBlur = 0

    // Dots
    points.forEach(p => {
        ctx.beginPath()
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2)
        ctx.fillStyle = '#00f0ff'
        ctx.fill()
        ctx.beginPath()
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'
        ctx.fill()
    })

    // X labels
    labels.forEach((label, i) => {
        const x = padding.left + stepX * i
        ctx.fillStyle = 'rgba(140, 140, 180, 0.6)'
        ctx.font = '9px Inter, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(label, x, h - 8)
    })
}

// ── Canvas Chart Component ──
function CanvasChart({ drawFn, deps }) {
    const canvasRef = useRef(null)

    useEffect(() => {
        if (canvasRef.current) {
            drawFn(canvasRef.current)
        }
    }, deps)

    return <canvas ref={canvasRef} style={{ display: 'block', width: '100%' }} />
}


// ── Main Reports Page ──
function ReportsPage({ scanHistory = [] }) {
    const [timeRange, setTimeRange] = useState('all') // 'all', '7d', '30d'

    // ── Download report as PDF ──
    const downloadReport = () => {
        const rangeLabel = timeRange === 'all' ? 'All Time' : timeRange === '30d' ? 'Last 30 Days' : 'Last 7 Days'
        const date = new Date().toLocaleString()

        let scanRows = ''
        filteredScans.forEach(s => {
            const status = s.is_vulnerable ? 'VULNERABLE' : 'SAFE'
            const statusColor = s.is_vulnerable ? '#ef4444' : '#22c55e'
            const sevColor = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e' }[s.severity] || '#888'
            scanRows += `<tr>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:13px">${s.filename}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:center">
                    <span style="color:${statusColor};font-weight:600;font-size:12px">${status}</span>
                </td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:center">
                    <span style="background:${sevColor}15;color:${sevColor};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">${(s.severity || '-').toUpperCase()}</span>
                </td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:center;font-size:12px">${Math.round((s.confidence || 0) * 100)}%</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280">${new Date(s.timestamp).toLocaleString()}</td>
            </tr>`
        })

        let vulnTypeRows = ''
        stats.vulnTypes.forEach(([type, count]) => {
            vulnTypeRows += `<tr><td style="padding:6px 12px;border-bottom:1px solid #f3f4f6;font-size:13px">${type}</td><td style="padding:6px 12px;border-bottom:1px solid #f3f4f6;text-align:center;font-weight:600;font-size:13px">${count}</td></tr>`
        })

        const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>PyVulnDetect Security Report</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'Inter',sans-serif; color:#1f2937; background:#fff; padding:40px; }
    .header { text-align:center; margin-bottom:36px; padding-bottom:24px; border-bottom:2px solid #3b82f6; }
    .logo { font-size:28px; font-weight:700; color:#1e40af; margin-bottom:4px; }
    .subtitle { font-size:13px; color:#6b7280; }
    .meta { display:flex; justify-content:center; gap:32px; margin-top:14px; font-size:12px; color:#6b7280; }
    .section { margin-bottom:28px; }
    .section-title { font-size:16px; font-weight:700; color:#1e3a5f; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid #e5e7eb; }
    .stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:28px; }
    .stat-card { text-align:center; padding:18px 12px; border-radius:10px; border:1px solid #e5e7eb; background:#f9fafb; }
    .stat-value { font-size:28px; font-weight:700; line-height:1; margin-bottom:4px; }
    .stat-label { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }
    .cyan { color:#0891b2; } .green { color:#16a34a; } .red { color:#dc2626; } .purple { color:#7c3aed; }
    .sev-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
    .sev-card { text-align:center; padding:14px; border-radius:8px; border:1px solid #e5e7eb; }
    .sev-card .num { font-size:22px; font-weight:700; }
    .sev-card .lbl { font-size:11px; color:#6b7280; text-transform:uppercase; margin-top:4px; }
    table { width:100%; border-collapse:collapse; }
    th { padding:10px 12px; background:#f1f5f9; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#475569; text-align:left; font-weight:600; border-bottom:2px solid #e2e8f0; }
    .footer { text-align:center; margin-top:40px; padding-top:20px; border-top:1px solid #e5e7eb; font-size:11px; color:#9ca3af; }
    @media print {
        body { padding:20px; }
        .stats-grid { break-inside:avoid; }
        table { font-size:11px; }
    }
</style>
</head><body>
    <div class="header">
        <div class="logo">🛡️ PyVulnDetect</div>
        <div class="subtitle">AI-Powered Python Security Scanner</div>
        <div class="meta">
            <span>📅 ${date}</span>
            <span>📊 ${rangeLabel}</span>
            <span>🔍 ${stats.total} Scans Analyzed</span>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card"><div class="stat-value cyan">${stats.total}</div><div class="stat-label">Total Scans</div></div>
        <div class="stat-card"><div class="stat-value green">${stats.safe}</div><div class="stat-label">Safe Files</div></div>
        <div class="stat-card"><div class="stat-value red">${stats.vulnerable}</div><div class="stat-label">Vulnerable</div></div>
        <div class="stat-card"><div class="stat-value purple">${stats.avgConfidence}%</div><div class="stat-label">Avg Confidence</div></div>
    </div>

    <div class="section">
        <div class="section-title">Severity Breakdown</div>
        <div class="sev-grid">
            <div class="sev-card"><div class="num red">${stats.severity.critical}</div><div class="lbl">Critical</div></div>
            <div class="sev-card"><div class="num" style="color:#f97316">${stats.severity.high}</div><div class="lbl">High</div></div>
            <div class="sev-card"><div class="num" style="color:#eab308">${stats.severity.medium}</div><div class="lbl">Medium</div></div>
            <div class="sev-card"><div class="num green">${stats.severity.low}</div><div class="lbl">Low</div></div>
        </div>
    </div>

    ${stats.vulnTypes.length > 0 ? `
    <div class="section">
        <div class="section-title">Vulnerability Types</div>
        <table><thead><tr><th>Type</th><th style="text-align:center">Count</th></tr></thead>
        <tbody>${vulnTypeRows}</tbody></table>
    </div>` : ''}

    <div class="section">
        <div class="section-title">Scan Details (${filteredScans.length} records)</div>
        ${filteredScans.length > 0 ? `
        <table><thead><tr>
            <th>File</th><th style="text-align:center">Status</th><th style="text-align:center">Severity</th><th style="text-align:center">Confidence</th><th>Date</th>
        </tr></thead><tbody>${scanRows}</tbody></table>
        ` : '<p style="color:#9ca3af;text-align:center;padding:24px">No scan records found.</p>'}
    </div>

    <div class="footer">
        Generated by PyVulnDetect • ${date}
    </div>
</body></html>`

        const printWindow = window.open('', '_blank')
        printWindow.document.write(html)
        printWindow.document.close()
        setTimeout(() => printWindow.print(), 500)
    }

    // Filter by time range
    const filteredScans = useMemo(() => {
        if (timeRange === 'all') return scanHistory
        const now = new Date()
        const days = timeRange === '7d' ? 7 : 30
        const cutoff = new Date(now.getTime() - days * 86400000)
        return scanHistory.filter(s => new Date(s.timestamp) >= cutoff)
    }, [scanHistory, timeRange])

    // ── Computed statistics ──
    const stats = useMemo(() => {
        const total = filteredScans.length
        const vulnerable = filteredScans.filter(s => s.is_vulnerable).length
        const safe = total - vulnerable

        // Severity counts
        const severity = { critical: 0, high: 0, medium: 0, low: 0 }
        filteredScans.forEach(s => {
            if (s.is_vulnerable && s.severity) {
                severity[s.severity] = (severity[s.severity] || 0) + 1
            }
        })

        // Vulnerability type counts
        const typeMap = {}
        filteredScans.forEach(s => {
            if (s.is_vulnerable && s.vulnerability_type && s.vulnerability_type !== 'none') {
                const t = s.vulnerability_type.replace(/_/g, ' ')
                typeMap[t] = (typeMap[t] || 0) + 1
            }
        })
        const vulnTypes = Object.entries(typeMap).sort((a, b) => b[1] - a[1]).slice(0, 8)

        // Daily scan counts (last 14 days)
        const dailyCounts = []
        const dailyLabels = []
        for (let i = 13; i >= 0; i--) {
            const d = new Date()
            d.setDate(d.getDate() - i)
            const key = d.toISOString().split('T')[0]
            dailyLabels.push(d.toLocaleDateString('en', { month: 'short', day: 'numeric' }))
            dailyCounts.push(filteredScans.filter(s => s.timestamp?.startsWith(key)).length)
        }

        const avgConfidence = total > 0
            ? Math.round(filteredScans.reduce((s, v) => s + (v.confidence || 0), 0) / total * 100)
            : 0

        return { total, vulnerable, safe, severity, vulnTypes, dailyCounts, dailyLabels, avgConfidence }
    }, [filteredScans])

    const severityColors = ['#f87171', '#fb923c', '#fbbf24', '#4ade80']
    const severityLabels = ['Critical', 'High', 'Medium', 'Low']
    const severityData = [stats.severity.critical, stats.severity.high, stats.severity.medium, stats.severity.low]

    const barColors = ['#22d3ee', '#a78bfa', '#60a5fa', '#e879f9', '#4ade80', '#fb923c', '#fbbf24', '#f87171']

    return (
        <div className="page-container">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                <div>
                    <h1 className="page-title">📊 Reports</h1>
                    <p className="page-subtitle">Real-time vulnerability analytics from your scans</p>
                </div>
                <div className="report-header-actions">
                    <button className="report-download-btn" onClick={downloadReport} title="Download Report">
                        📥 Download Report
                    </button>
                    <div className="report-time-filter">
                        {['all', '30d', '7d'].map(range => (
                            <button
                                key={range}
                                className={`report-time-btn ${timeRange === range ? 'active' : ''}`}
                                onClick={() => setTimeRange(range)}
                            >
                                {range === 'all' ? 'All Time' : range === '30d' ? '30 Days' : '7 Days'}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="report-stats-grid">
                <div className="glass-card report-stat-card">
                    <div className="report-stat-icon">🔍</div>
                    <div className="report-stat-value neon-text-cyan">{stats.total}</div>
                    <div className="report-stat-label">Total Scans</div>
                </div>
                <div className="glass-card report-stat-card">
                    <div className="report-stat-icon">🛡️</div>
                    <div className="report-stat-value neon-text-green">{stats.safe}</div>
                    <div className="report-stat-label">Safe Files</div>
                </div>
                <div className="glass-card report-stat-card">
                    <div className="report-stat-icon">⚠️</div>
                    <div className="report-stat-value neon-text-red">{stats.vulnerable}</div>
                    <div className="report-stat-label">Vulnerable</div>
                </div>
                <div className="glass-card report-stat-card">
                    <div className="report-stat-icon">🎯</div>
                    <div className="report-stat-value neon-text-violet">{stats.avgConfidence}%</div>
                    <div className="report-stat-label">Avg Confidence</div>
                </div>
            </div>

            {/* Charts Grid */}
            <div className="report-charts-grid">
                {/* Doughnut: Severity Distribution */}
                <div className="glass-card report-chart-card">
                    <h3 className="report-chart-title">Severity Distribution</h3>
                    <div className="report-doughnut-wrapper">
                        <div className="report-doughnut-canvas">
                            <CanvasChart
                                drawFn={(c) => drawDoughnutChart(c, severityData, severityColors, severityLabels)}
                                deps={[severityData.join(',')]}
                            />
                        </div>
                        <div className="report-legend">
                            {severityLabels.map((label, i) => (
                                <div key={label} className="report-legend-item">
                                    <span className="report-legend-dot" style={{ background: severityColors[i] }} />
                                    <span className="report-legend-label">{label}</span>
                                    <span className="report-legend-count">{severityData[i]}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Line: Scan Activity */}
                <div className="glass-card report-chart-card">
                    <h3 className="report-chart-title">Scan Activity (Last 14 Days)</h3>
                    <div className="report-line-canvas">
                        <CanvasChart
                            drawFn={(c) => drawLineChart(c, stats.dailyCounts, stats.dailyLabels)}
                            deps={[stats.dailyCounts.join(',')]}
                        />
                    </div>
                </div>
            </div>

            {/* Bar: Vulnerability Types */}
            {stats.vulnTypes.length > 0 && (
                <div className="glass-card report-chart-card" style={{ marginTop: 20 }}>
                    <h3 className="report-chart-title">Vulnerabilities by Type</h3>
                    <div className="report-bar-canvas">
                        <CanvasChart
                            drawFn={(c) => drawBarChart(c, stats.vulnTypes.map(v => v[1]), barColors, stats.vulnTypes.map(v => v[0]))}
                            deps={[stats.vulnTypes.map(v => v[1]).join(',')]}
                        />
                    </div>
                </div>
            )}

            {/* Recent Scan History */}
            <div className="glass-card report-history-card" style={{ marginTop: 20 }}>
                <h3 className="report-chart-title">Recent Scan Results</h3>
                {filteredScans.length === 0 ? (
                    <div className="report-empty">
                        <p>No scans yet. Go to <strong>Scan Code</strong> to analyze your first file!</p>
                    </div>
                ) : (
                    <div className="report-history-list">
                        {filteredScans.slice(0, 20).map((scan, i) => (
                            <div key={scan.id || i} className={`report-history-item ${scan.is_vulnerable ? 'vuln' : 'safe'}`}>
                                <div className="report-history-status">
                                    {scan.is_vulnerable ? '🔴' : '🟢'}
                                </div>
                                <div className="report-history-info">
                                    <span className="report-history-name">{scan.filename}</span>
                                    <span className="report-history-time">
                                        {new Date(scan.timestamp).toLocaleString()}
                                    </span>
                                </div>
                                <div className="report-history-meta">
                                    {scan.is_vulnerable && (
                                        <span className={`report-history-severity sev-${scan.severity}`}>
                                            {scan.severity?.toUpperCase()}
                                        </span>
                                    )}
                                    <span className="report-history-confidence">
                                        {Math.round((scan.confidence || 0) * 100)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <style>{`
                .report-header-actions {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                .report-download-btn {
                    padding: 10px 20px;
                    border: 1px solid rgba(34, 211, 238, 0.15);
                    border-radius: 14px;
                    background: rgba(34, 211, 238, 0.06);
                    color: var(--neon-cyan);
                    font-family: var(--font-main);
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
                    white-space: nowrap;
                }
                .report-download-btn:hover {
                    background: rgba(34, 211, 238, 0.12);
                    border-color: rgba(34, 211, 238, 0.3);
                    box-shadow: 0 0 16px rgba(34, 211, 238, 0.1);
                    transform: translateY(-1px);
                }
                .report-time-filter {
                    display: flex;
                    gap: 4px;
                    background: rgba(130, 120, 200, 0.06);
                    border-radius: 12px;
                    padding: 4px;
                }
                .report-time-btn {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 9px;
                    font-family: var(--font-main);
                    font-size: 12px;
                    font-weight: 600;
                    cursor: pointer;
                    background: transparent;
                    color: var(--text-muted);
                    transition: all 0.25s;
                }
                .report-time-btn.active {
                    background: linear-gradient(135deg, rgba(0, 240, 255, 0.12), rgba(168, 85, 247, 0.1));
                    color: var(--neon-cyan);
                    box-shadow: 0 0 12px rgba(0, 240, 255, 0.06);
                }
                .report-time-btn:hover:not(.active) { color: var(--text-secondary); }

                /* Stats */
                .report-stats-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    margin-bottom: 20px;
                }
                @media (max-width: 800px) {
                    .report-stats-grid { grid-template-columns: repeat(2, 1fr); }
                }
                .report-stat-card {
                    padding: 24px 18px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                    transition: transform 0.3s cubic-bezier(0.22, 1, 0.36, 1);
                }
                .report-stat-card:hover { transform: translateY(-4px); }
                .report-stat-card::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .report-stat-card:hover::before { opacity: 1; }
                .report-stat-icon { font-size: 24px; margin-bottom: 8px; }
                .report-stat-value {
                    font-family: var(--font-heading);
                    font-size: 32px;
                    font-weight: 700;
                    line-height: 1;
                    margin-bottom: 6px;
                }
                .report-stat-label {
                    font-size: 11px;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    font-weight: 600;
                }

                /* Charts */
                .report-charts-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                @media (max-width: 900px) {
                    .report-charts-grid { grid-template-columns: 1fr; }
                }
                .report-chart-card {
                    padding: 24px;
                }
                .report-chart-title {
                    font-family: var(--font-heading);
                    font-size: 14px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 20px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid rgba(120, 100, 255, 0.08);
                    text-transform: uppercase;
                    letter-spacing: 0.04em;
                }

                .report-doughnut-wrapper {
                    display: flex;
                    align-items: center;
                    gap: 24px;
                }
                @media (max-width: 600px) {
                    .report-doughnut-wrapper { flex-direction: column; }
                }
                .report-doughnut-canvas {
                    width: 180px;
                    height: 180px;
                    flex-shrink: 0;
                }
                .report-legend {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .report-legend-item {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 13px;
                }
                .report-legend-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    flex-shrink: 0;
                    box-shadow: 0 0 6px currentColor;
                }
                .report-legend-label {
                    color: var(--text-secondary);
                    flex: 1;
                }
                .report-legend-count {
                    font-family: var(--font-heading);
                    font-weight: 600;
                    color: var(--text-primary);
                }

                .report-line-canvas, .report-bar-canvas {
                    width: 100%;
                }

                /* History */
                .report-history-card {
                    padding: 24px;
                }
                .report-empty {
                    text-align: center;
                    padding: 40px 20px;
                    color: var(--text-secondary);
                    font-size: 14px;
                }
                .report-history-list {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                    max-height: 400px;
                    overflow-y: auto;
                }
                .report-history-list::-webkit-scrollbar { width: 4px; }
                .report-history-list::-webkit-scrollbar-track { background: transparent; }
                .report-history-list::-webkit-scrollbar-thumb {
                    background: rgba(120, 100, 255, 0.15);
                    border-radius: 2px;
                }
                .report-history-item {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    padding: 12px 16px;
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.02);
                    border: 1px solid rgba(120, 100, 255, 0.06);
                    transition: all 0.2s;
                }
                .report-history-item:hover {
                    background: rgba(255, 255, 255, 0.04);
                    border-color: rgba(120, 100, 255, 0.12);
                }
                .report-history-item.vuln { border-left: 3px solid var(--neon-red); }
                .report-history-item.safe { border-left: 3px solid var(--neon-green); }
                .report-history-status { font-size: 12px; }
                .report-history-info {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    min-width: 0;
                }
                .report-history-name {
                    font-size: 13px;
                    font-weight: 500;
                    color: var(--text-primary);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .report-history-time {
                    font-size: 11px;
                    color: var(--text-muted);
                }
                .report-history-meta {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    flex-shrink: 0;
                }
                .report-history-severity {
                    font-size: 9px;
                    font-weight: 700;
                    padding: 3px 8px;
                    border-radius: 6px;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .sev-critical {
                    background: rgba(255, 45, 85, 0.12);
                    color: var(--neon-red);
                    border: 1px solid rgba(255, 45, 85, 0.2);
                }
                .sev-high {
                    background: rgba(255, 149, 0, 0.1);
                    color: var(--neon-orange);
                    border: 1px solid rgba(255, 149, 0, 0.2);
                }
                .sev-medium {
                    background: rgba(251, 191, 36, 0.08);
                    color: var(--neon-gold);
                    border: 1px solid rgba(251, 191, 36, 0.15);
                }
                .sev-low {
                    background: rgba(57, 255, 20, 0.06);
                    color: var(--neon-green);
                    border: 1px solid rgba(57, 255, 20, 0.12);
                }
                .report-history-confidence {
                    font-family: var(--font-code);
                    font-size: 12px;
                    color: var(--neon-cyan);
                }
            `}</style>
        </div>
    )
}

export default ReportsPage
