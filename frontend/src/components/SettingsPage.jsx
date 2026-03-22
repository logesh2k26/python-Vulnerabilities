import React, { useState, useEffect, useCallback } from 'react'

const DEFAULT_SETTINGS = {
    darkMode: true,
    autoScan: false,
    notifications: true,
    apiKey: '',
    scanDepth: 'deep',
    maxFileSize: 500,
    showMinimap: true,
    fontSize: 14,
}

function loadSettings() {
    try {
        const stored = localStorage.getItem('pyvuln_settings')
        if (stored) {
            return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) }
        }
    } catch (e) {
        console.warn('Failed to load settings:', e)
    }
    return { ...DEFAULT_SETTINGS }
}

function saveSettings(settings) {
    try {
        localStorage.setItem('pyvuln_settings', JSON.stringify(settings))
        return true
    } catch (e) {
        console.error('Failed to save settings:', e)
        return false
    }
}

function SettingsPage() {
    const [settings, setSettings] = useState(loadSettings)
    const [toast, setToast] = useState(null)

    const showToast = useCallback((msg, type = 'success') => {
        setToast({ msg, type })
        setTimeout(() => setToast(null), 2500)
    }, [])

    const update = useCallback((key, value) => {
        setSettings(prev => {
            const next = { ...prev, [key]: value }
            if (saveSettings(next)) {
                showToast('Settings saved')
            } else {
                showToast('Failed to save', 'error')
            }
            return next
        })
    }, [showToast])

    const resetDefaults = useCallback(() => {
        setSettings({ ...DEFAULT_SETTINGS })
        if (saveSettings(DEFAULT_SETTINGS)) {
            showToast('Reset to defaults')
        }
    }, [showToast])

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">⚙️ Settings</h1>
                <p className="page-subtitle">Configure your scanning preferences</p>
            </div>

            <div className="st-sections">
                {/* Appearance */}
                <section className="glass-card st-section">
                    <h3 className="st-section-title">🎨 Appearance</h3>
                    <ToggleRow label="Dark Mode" desc="Use dark cosmic theme" checked={settings.darkMode} onToggle={() => update('darkMode', !settings.darkMode)} />
                    <ToggleRow label="Show Minimap" desc="Show code minimap in editor" checked={settings.showMinimap} onToggle={() => update('showMinimap', !settings.showMinimap)} />
                    <RangeRow label="Font Size" desc="Code editor font size" value={settings.fontSize} min={10} max={24} onChange={v => update('fontSize', v)} />
                </section>

                {/* Analysis */}
                <section className="glass-card st-section">
                    <h3 className="st-section-title">🔬 Analysis</h3>
                    <ToggleRow label="Auto-Scan on Paste" desc="Automatically scan when code is pasted" checked={settings.autoScan} onToggle={() => update('autoScan', !settings.autoScan)} />
                    <SelectRow
                        label="Scan Depth"
                        desc="How thorough the analysis should be"
                        value={settings.scanDepth}
                        options={[
                            { value: 'quick', label: 'Quick — AST only' },
                            { value: 'deep', label: 'Deep — AST + GNN' },
                            { value: 'full', label: 'Full — AST + GNN + Taint' },
                        ]}
                        onChange={v => update('scanDepth', v)}
                    />
                    <RangeRow label="Max File Size" desc="Maximum file size in KB" value={settings.maxFileSize} min={100} max={2000} step={100} onChange={v => update('maxFileSize', v)} />
                </section>

                {/* Notifications */}
                <section className="glass-card st-section">
                    <h3 className="st-section-title">🔔 Notifications</h3>
                    <ToggleRow label="Enable Notifications" desc="Get alerts for critical vulnerabilities" checked={settings.notifications} onToggle={() => update('notifications', !settings.notifications)} />
                </section>

                {/* API */}
                <section className="glass-card st-section">
                    <h3 className="st-section-title">🔑 API Configuration</h3>
                    <div className="st-row">
                        <div className="st-info">
                            <span className="st-label">API Key</span>
                            <span className="st-desc">Backend authentication key</span>
                        </div>
                        <input
                            type="password"
                            className="st-text-input"
                            value={settings.apiKey}
                            onChange={e => update('apiKey', e.target.value)}
                        />
                    </div>
                </section>

                {/* Actions */}
                <div className="st-actions">
                    <button className="st-reset-btn" onClick={resetDefaults}>↺ Reset to Defaults</button>
                </div>
            </div>

            {/* Toast */}
            {toast && (
                <div className={`st-toast ${toast.type}`}>
                    {toast.type === 'success' ? '✓' : '✕'} {toast.msg}
                </div>
            )}

            <style>{`
                .st-sections {
                    display: flex;
                    flex-direction: column;
                    gap: 18px;
                    max-width: 680px;
                }
                .st-section {
                    padding: 24px 28px;
                }
                .st-section-title {
                    font-family: var(--font-heading);
                    font-size: 15px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 18px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid rgba(130, 120, 200, 0.08);
                }

                .st-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 14px 0;
                    gap: 16px;
                }
                .st-row + .st-row {
                    border-top: 1px solid rgba(130, 120, 200, 0.04);
                }
                .st-info {
                    display: flex;
                    flex-direction: column;
                    gap: 3px;
                    flex: 1;
                    min-width: 0;
                }
                .st-label {
                    font-size: 14px;
                    font-weight: 500;
                    color: var(--text-primary);
                }
                .st-desc {
                    font-size: 12px;
                    color: var(--text-muted);
                }

                /* Toggle Switch */
                .st-toggle {
                    width: 48px;
                    height: 26px;
                    border-radius: 13px;
                    border: none;
                    cursor: pointer;
                    position: relative;
                    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
                    flex-shrink: 0;
                    outline: none;
                }
                .st-toggle[data-on="true"] {
                    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-violet));
                    box-shadow: 0 0 14px rgba(34, 211, 238, 0.2);
                }
                .st-toggle[data-on="false"] {
                    background: rgba(130, 120, 200, 0.15);
                }
                .st-toggle::after {
                    content: '';
                    position: absolute;
                    top: 3px;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background: white;
                    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
                    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
                }
                .st-toggle[data-on="true"]::after { left: 25px; }
                .st-toggle[data-on="false"]::after { left: 3px; }
                .st-toggle:hover {
                    filter: brightness(1.1);
                }

                /* Select */
                .st-select {
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(130, 120, 200, 0.12);
                    border-radius: 12px;
                    padding: 9px 14px;
                    color: var(--text-primary);
                    font-family: var(--font-main);
                    font-size: 13px;
                    outline: none;
                    cursor: pointer;
                    appearance: none;
                    min-width: 180px;
                    transition: all 0.25s;
                }
                .st-select:focus {
                    border-color: rgba(34, 211, 238, 0.3);
                    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.06);
                }
                .st-select option {
                    background: #12101e;
                    color: var(--text-primary);
                }

                /* Text Input */
                .st-text-input {
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(130, 120, 200, 0.12);
                    border-radius: 12px;
                    padding: 9px 14px;
                    color: var(--text-primary);
                    font-family: var(--font-code);
                    font-size: 13px;
                    outline: none;
                    width: 260px;
                    transition: all 0.25s;
                }
                .st-text-input:focus {
                    border-color: rgba(34, 211, 238, 0.3);
                    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.06);
                }

                /* Range */
                .st-range-wrap {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    flex-shrink: 0;
                }
                .st-range {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 140px;
                    height: 5px;
                    border-radius: 3px;
                    background: rgba(130, 120, 200, 0.15);
                    outline: none;
                }
                .st-range::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    width: 18px;
                    height: 18px;
                    border-radius: 50%;
                    background: var(--neon-cyan);
                    cursor: pointer;
                    box-shadow: 0 0 10px rgba(34, 211, 238, 0.25);
                    transition: transform 0.2s;
                }
                .st-range::-webkit-slider-thumb:hover {
                    transform: scale(1.2);
                }
                .st-range-val {
                    font-family: var(--font-code);
                    font-size: 13px;
                    color: var(--neon-cyan);
                    min-width: 36px;
                    text-align: right;
                    font-weight: 500;
                }

                /* Actions */
                .st-actions {
                    display: flex;
                    justify-content: flex-end;
                }
                .st-reset-btn {
                    padding: 11px 24px;
                    border: 1px solid rgba(251, 146, 60, 0.2);
                    border-radius: 14px;
                    background: rgba(251, 146, 60, 0.06);
                    color: var(--neon-orange);
                    font-family: var(--font-main);
                    font-size: 13px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.25s;
                }
                .st-reset-btn:hover {
                    background: rgba(251, 146, 60, 0.12);
                    box-shadow: 0 0 16px rgba(251, 146, 60, 0.08);
                    transform: translateY(-1px);
                }

                /* Toast */
                .st-toast {
                    position: fixed;
                    bottom: 30px;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 12px 28px;
                    border-radius: 14px;
                    font-size: 14px;
                    font-weight: 500;
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
                    z-index: 9000;
                    animation: st-toast-in 0.4s cubic-bezier(0.22, 1, 0.36, 1);
                }
                .st-toast.success {
                    background: rgba(74, 222, 128, 0.1);
                    border: 1px solid rgba(74, 222, 128, 0.2);
                    color: var(--neon-green);
                }
                .st-toast.error {
                    background: rgba(248, 113, 113, 0.1);
                    border: 1px solid rgba(248, 113, 113, 0.2);
                    color: var(--neon-red);
                }
                @keyframes st-toast-in {
                    from { transform: translateX(-50%) translateY(20px); opacity: 0; }
                    to { transform: translateX(-50%) translateY(0); opacity: 1; }
                }
            `}</style>
        </div>
    )
}

// ── Sub-components ──

function ToggleRow({ label, desc, checked, onToggle }) {
    return (
        <div className="st-row">
            <div className="st-info">
                <span className="st-label">{label}</span>
                <span className="st-desc">{desc}</span>
            </div>
            <button
                className="st-toggle"
                data-on={String(!!checked)}
                onClick={onToggle}
                role="switch"
                aria-checked={checked}
                aria-label={label}
            />
        </div>
    )
}

function SelectRow({ label, desc, value, options, onChange }) {
    return (
        <div className="st-row">
            <div className="st-info">
                <span className="st-label">{label}</span>
                <span className="st-desc">{desc}</span>
            </div>
            <select className="st-select" value={value} onChange={e => onChange(e.target.value)}>
                {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
        </div>
    )
}

function RangeRow({ label, desc, value, min, max, step = 1, onChange }) {
    return (
        <div className="st-row">
            <div className="st-info">
                <span className="st-label">{label}</span>
                <span className="st-desc">{desc}</span>
            </div>
            <div className="st-range-wrap">
                <input
                    type="range"
                    className="st-range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={e => onChange(Number(e.target.value))}
                />
                <span className="st-range-val">{value}</span>
            </div>
        </div>
    )
}

export default SettingsPage
