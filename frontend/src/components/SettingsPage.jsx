import React, { useState, useCallback, useEffect } from 'react'

const DEFAULT_SETTINGS = {
    displayName: 'Admin User',
    jobTitle: 'Security Architect',
    email: 'admin@pyvulndetect.io',
    autoScan: true,
    notifications: true,
    pushAlerts: true,
    emailDigests: true,
    scanDepth: 'standard',
    darkMode: false,
    fontSize: 14,
}

function loadSettings() {
    try {
        const stored = localStorage.getItem('pyvuln_settings')
        if (stored) return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) }
    } catch (e) { console.warn('Failed to load settings:', e) }
    return { ...DEFAULT_SETTINGS }
}

function saveSettingsToStorage(settings) {
    try {
        localStorage.setItem('pyvuln_settings', JSON.stringify(settings))
        return true
    } catch (e) {
        console.error('Failed to save settings:', e)
        return false
    }
}

function SettingsPage({ onNavigate }) {
    const [settings, setSettings] = useState(loadSettings)
    const [activeSection, setActiveSection] = useState('appearance')
    const [toast, setToast] = useState(null)

    // Apply dark mode in real-time
    useEffect(() => {
        const root = document.documentElement
        if (settings.darkMode) {
            root.classList.add('dark-mode')
        } else {
            root.classList.remove('dark-mode')
        }
    }, [settings.darkMode])

    // Apply font size in real-time (only to content area)
    useEffect(() => {
        const content = document.querySelector('.sd-content')
        if (content) {
            content.style.fontSize = settings.fontSize + 'px'
        }
        return () => {
            if (content) content.style.fontSize = ''
        }
    }, [settings.fontSize])

    const showToast = useCallback((msg, type = 'success') => {
        setToast({ msg, type })
        setTimeout(() => setToast(null), 2500)
    }, [])

    const update = useCallback((key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }))
    }, [])

    const handleSave = useCallback(() => {
        if (saveSettingsToStorage(settings)) {
            showToast('Preferences saved successfully')
        } else {
            showToast('Failed to save preferences', 'error')
        }
    }, [settings, showToast])

    const handleDiscard = useCallback(() => {
        setSettings(loadSettings())
        showToast('Changes discarded')
    }, [showToast])

    const sections = [
        { id: 'appearance', label: 'Appearance' },
        { id: 'analysis', label: 'Analysis Engine' },
        { id: 'notifications', label: 'Notifications' },
        { id: 'privacy', label: 'Privacy & Vault' },
        { id: 'advanced', label: 'Advanced Ledger' },
    ]

    return (
        <div className="st-container">
            {/* Page Header */}
            <header className="st-header">
                <h2 className="st-title">Settings</h2>
                <p className="st-subtitle">Manage your architectural security preferences and environment.</p>
            </header>

            <div className="st-layout">
                {/* Internal Navigation Column */}
                <div className="st-nav-col">
                    <nav className="st-nav">
                        {sections.map(s => (
                            <button
                                key={s.id}
                                className={`st-nav-item ${activeSection === s.id ? 'active' : ''}`}
                                onClick={() => setActiveSection(s.id)}
                            >
                                {s.label}
                            </button>
                        ))}
                        {/* App Info Link moved inside sticky nav */}
                        <button
                            className="st-appinfo-btn"
                            onClick={() => onNavigate && onNavigate('appinfo')}
                        >
                            <span className="material-symbols-outlined">info</span>
                            App Information
                        </button>
                    </nav>
                </div>

                {/* Settings Forms Column */}
                <div className="st-forms-col">
                    {/* Appearance Section */}
                    <section className="st-card">
                        <div className="st-card-header">
                            <span className="material-symbols-outlined st-card-icon">palette</span>
                            <h3 className="st-card-title">Appearance</h3>
                        </div>
                        <div className="st-card-body">
                            {/* Dark Mode Toggle */}
                            <div className="st-row">
                                <div className="st-row-info">
                                    <p className="st-row-label">Dark Mode</p>
                                    <p className="st-row-desc">Switch to a low-light interface for nightly audits.</p>
                                </div>
                                <label className="st-switch">
                                    <input
                                        type="checkbox"
                                        checked={settings.darkMode}
                                        onChange={() => update('darkMode', !settings.darkMode)}
                                    />
                                    <span className="st-switch-slider"></span>
                                </label>
                            </div>

                            {/* Font Size Slider */}
                            <div className="st-slider-block">
                                <div className="st-slider-header">
                                    <div>
                                        <p className="st-row-label">Interface Scale</p>
                                        <p className="st-row-desc">Adjust text size for optimal ledger readability.</p>
                                    </div>
                                    <span className="st-slider-badge">{settings.fontSize}PX (DEFAULT)</span>
                                </div>
                                <input
                                    type="range"
                                    min="12"
                                    max="20"
                                    value={settings.fontSize}
                                    onChange={e => update('fontSize', parseInt(e.target.value))}
                                    className="st-range"
                                />
                                <div className="st-range-labels">
                                    <span>COMPACT</span>
                                    <span>EXPANDED</span>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Analysis Engine Section */}
                    <section className="st-card">
                        <div className="st-card-header">
                            <span className="material-symbols-outlined st-card-icon">analytics</span>
                            <h3 className="st-card-title">Analysis Engine</h3>
                        </div>
                        <div className="st-card-body">
                            {/* Auto Scan Toggle */}
                            <div className="st-row">
                                <div className="st-row-info">
                                    <p className="st-row-label">Auto-Scan Incoming Triggers</p>
                                    <p className="st-row-desc">Perform background checks on every detected activity.</p>
                                </div>
                                <label className="st-switch">
                                    <input
                                        type="checkbox"
                                        checked={settings.autoScan}
                                        onChange={() => update('autoScan', !settings.autoScan)}
                                    />
                                    <span className="st-switch-slider"></span>
                                </label>
                            </div>

                            {/* Scan Depth Dropdown */}
                            <div className="st-select-block">
                                <label className="st-row-label">Scan Depth</label>
                                <div className="st-select-wrap">
                                    <select
                                        className="st-select"
                                        value={settings.scanDepth}
                                        onChange={e => update('scanDepth', e.target.value)}
                                    >
                                        <option value="surface">Surface Audit (Fast)</option>
                                        <option value="standard">Standard Architectural Review (Recommended)</option>
                                        <option value="deep">Deep Ledger Analysis (Intensive)</option>
                                    </select>
                                    <span className="material-symbols-outlined st-select-arrow">expand_more</span>
                                </div>
                                <p className="st-select-hint">Intensive scans may impact performance during high-traffic sessions.</p>
                            </div>
                        </div>
                    </section>

                    {/* Notifications Section */}
                    <section className="st-card">
                        <div className="st-card-header">
                            <span className="material-symbols-outlined st-card-icon" style={{ fontVariationSettings: "'FILL' 1" }}>notifications_active</span>
                            <h3 className="st-card-title">Notifications</h3>
                        </div>
                        <div className="st-notif-grid">
                            <div className="st-notif-card" onClick={() => update('emailDigests', !settings.emailDigests)}>
                                <div className="st-notif-top">
                                    <div className="st-notif-label">
                                        <span className="material-symbols-outlined st-notif-icon">mail</span>
                                        <span className="st-notif-name">Email Digests</span>
                                    </div>
                                    <input type="checkbox" checked={settings.emailDigests} readOnly className="st-checkbox" />
                                </div>
                                <p className="st-notif-desc">Weekly summaries of your ledger's health and detected anomalies.</p>
                            </div>
                            <div className="st-notif-card" onClick={() => update('pushAlerts', !settings.pushAlerts)}>
                                <div className="st-notif-top">
                                    <div className="st-notif-label">
                                        <span className="material-symbols-outlined st-notif-icon">smartphone</span>
                                        <span className="st-notif-name">Push Alerts</span>
                                    </div>
                                    <input type="checkbox" checked={settings.pushAlerts} readOnly className="st-checkbox" />
                                </div>
                                <p className="st-notif-desc">Immediate mobile notifications for critical security events.</p>
                            </div>
                        </div>
                    </section>

                    {/* Footer Actions */}
                    <div className="st-actions">
                        <button className="st-discard-btn" onClick={handleDiscard}>Discard Changes</button>
                        <button className="st-save-btn" onClick={handleSave}>Save Preferences</button>
                    </div>
                </div>
            </div>

            {/* Toast */}
            {toast && (
                <div className={`st-toast st-toast-${toast.type}`}>
                    {toast.type === 'success' ? '✓' : '✕'} {toast.msg}
                </div>
            )}

            <style>{`
                .st-container {
                    max-width: 1100px;
                    margin: 0 auto;
                    padding: 48px 48px 80px;
                }

                /* Header */
                .st-header { margin-bottom: 48px; }
                .st-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 40px;
                    font-weight: 900;
                    letter-spacing: -0.03em;
                    color: #29343a;
                    margin: 0 0 8px;
                }
                .st-subtitle {
                    font-size: 15px;
                    color: #566168;
                    margin: 0;
                }

                /* Layout */
                .st-layout {
                    display: grid;
                    grid-template-columns: 220px 1fr;
                    gap: 40px;
                }
                @media (max-width: 900px) {
                    .st-layout { grid-template-columns: 1fr; }
                }

                /* Nav Column */
                .st-nav-col {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }
                .st-nav {
                    position: sticky;
                    top: 100px;
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .st-nav-item {
                    padding: 10px 16px;
                    background: none;
                    border: none;
                    text-align: left;
                    font-size: 14px;
                    font-weight: 500;
                    color: #566168;
                    border-radius: 10px;
                    cursor: pointer;
                    transition: all 0.15s;
                }
                .st-nav-item:hover { background: #e8eff4; }
                .st-nav-item.active {
                    background: #d9e4ec;
                    color: #4c56af;
                    font-weight: 700;
                }

                .st-appinfo-btn {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 12px 16px;
                    background: none;
                    border: 1px solid rgba(168,179,187,0.15);
                    border-radius: 12px;
                    font-size: 13px;
                    font-weight: 600;
                    color: #566168;
                    cursor: pointer;
                    transition: all 0.2s;
                    margin-top: 24px;
                }
                .st-appinfo-btn:hover {
                    background: #f0f4f8;
                    border-color: #4c56af;
                    color: #4c56af;
                }
                .st-appinfo-btn .material-symbols-outlined {
                    font-size: 20px;
                }

                /* Forms Column */
                .st-forms-col {
                    display: flex;
                    flex-direction: column;
                    gap: 28px;
                }

                /* Card */
                .st-card {
                    background: #fff;
                    border-radius: 16px;
                    padding: 32px;
                    box-shadow: 0 4px 24px -8px rgba(0,0,0,0.04);
                }
                .st-card-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 28px;
                }
                .st-card-icon {
                    color: #4c56af;
                    font-size: 24px;
                }
                .st-card-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                }

                .st-card-body {
                    display: flex;
                    flex-direction: column;
                    gap: 28px;
                }

                /* Row (Toggle) */
                .st-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .st-row-info {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                }
                .st-row-label {
                    font-size: 15px;
                    font-weight: 600;
                    color: #29343a;
                    margin: 0;
                }
                .st-row-desc {
                    font-size: 13px;
                    color: #566168;
                    margin: 0;
                }

                /* Toggle Switch */
                .st-switch {
                    position: relative;
                    display: inline-flex;
                    align-items: center;
                    cursor: pointer;
                }
                .st-switch input { position: absolute; opacity: 0; width: 0; height: 0; }
                .st-switch-slider {
                    width: 44px;
                    height: 24px;
                    background: #d9e4ec;
                    border-radius: 12px;
                    position: relative;
                    transition: background 0.3s;
                }
                .st-switch-slider::after {
                    content: '';
                    position: absolute;
                    top: 2px;
                    left: 2px;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background: #fff;
                    border: 1px solid rgba(0,0,0,0.08);
                    transition: transform 0.3s;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                }
                .st-switch input:checked + .st-switch-slider {
                    background: #4c56af;
                }
                .st-switch input:checked + .st-switch-slider::after {
                    transform: translateX(20px);
                }

                /* Slider */
                .st-slider-block {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .st-slider-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                }
                .st-slider-badge {
                    font-size: 11px;
                    font-weight: 800;
                    color: #4c56af;
                    padding: 4px 10px;
                    background: #e0e0ff;
                    border-radius: 6px;
                    letter-spacing: 0.04em;
                }
                .st-range {
                    width: 100%;
                    height: 6px;
                    background: #e1e9f0;
                    border-radius: 3px;
                    appearance: none;
                    cursor: pointer;
                    outline: none;
                }
                .st-range::-webkit-slider-thumb {
                    appearance: none;
                    width: 18px;
                    height: 18px;
                    border-radius: 50%;
                    background: #4c56af;
                    box-shadow: 0 2px 6px rgba(76,86,175,0.3);
                    cursor: pointer;
                }
                .st-range-labels {
                    display: flex;
                    justify-content: space-between;
                    font-size: 10px;
                    font-weight: 800;
                    color: #717c84;
                    letter-spacing: 0.15em;
                    text-transform: uppercase;
                }

                /* Select */
                .st-select-block {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .st-select-wrap {
                    position: relative;
                }
                .st-select {
                    width: 100%;
                    background: #fff;
                    border: 1px solid rgba(168,179,187,0.2);
                    border-radius: 12px;
                    padding: 14px 44px 14px 16px;
                    font-size: 14px;
                    color: #29343a;
                    outline: none;
                    appearance: none;
                    cursor: pointer;
                    font-family: 'Inter', sans-serif;
                    transition: border-color 0.2s;
                }
                .st-select:focus {
                    border-color: #4c56af;
                    box-shadow: 0 0 0 3px rgba(76,86,175,0.08);
                }
                .st-select-arrow {
                    position: absolute;
                    right: 14px;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #717c84;
                    pointer-events: none;
                    font-size: 20px;
                }
                .st-select-hint {
                    font-size: 12px;
                    color: #566168;
                    font-style: italic;
                    margin: 0;
                }

                /* Notifications Grid */
                .st-notif-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                    padding: 0;
                }
                .st-notif-card {
                    padding: 20px;
                    border: 1px solid rgba(168,179,187,0.1);
                    border-radius: 14px;
                    background: #f7f9fc;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .st-notif-card:hover { background: #f0f4f8; }
                .st-notif-top {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 10px;
                }
                .st-notif-label {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .st-notif-icon {
                    color: #4c56af;
                    font-size: 22px;
                }
                .st-notif-name {
                    font-size: 14px;
                    font-weight: 700;
                    color: #29343a;
                }
                .st-notif-desc {
                    font-size: 12px;
                    color: #566168;
                    margin: 0;
                    line-height: 1.5;
                }
                .st-checkbox {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    accent-color: #4c56af;
                    cursor: pointer;
                }

                /* Actions */
                .st-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 12px;
                    padding-top: 12px;
                }
                .st-discard-btn {
                    padding: 12px 24px;
                    background: none;
                    border: none;
                    color: #566168;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    border-radius: 12px;
                    transition: background 0.15s;
                }
                .st-discard-btn:hover { background: #e8eff4; }
                .st-save-btn {
                    padding: 12px 32px;
                    background: #4c56af;
                    color: #f9f6ff;
                    border: none;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: 700;
                    cursor: pointer;
                    box-shadow: 0 4px 16px rgba(76,86,175,0.2);
                    transition: all 0.2s;
                }
                .st-save-btn:hover {
                    background: #4049a2;
                    box-shadow: 0 6px 20px rgba(76,86,175,0.3);
                }
                .st-save-btn:active { transform: scale(0.97); }

                /* Toast */
                .st-toast {
                    position: fixed;
                    bottom: 30px;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 12px 28px;
                    border-radius: 14px;
                    font-size: 14px;
                    font-weight: 600;
                    z-index: 9000;
                    animation: st-fade-in 0.35s ease;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
                }
                .st-toast-success { background: #fff; color: #16a34a; border: 1px solid rgba(22,163,74,0.2); }
                .st-toast-error { background: #fff; color: #9f403d; border: 1px solid rgba(159,64,61,0.2); }
                @keyframes st-fade-in {
                    from { transform: translateX(-50%) translateY(16px); opacity: 0; }
                    to { transform: translateX(-50%) translateY(0); opacity: 1; }
                }
            `}</style>
        </div>
    )
}

export default SettingsPage
