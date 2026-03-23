import { useState, useCallback, useEffect, useRef } from 'react'
import Dashboard from './components/Dashboard'
import DashboardOverview from './components/DashboardOverview'
import ScansPage from './components/ScansPage'
import ReportsPage from './components/ReportsPage'

import ProfilePage from './components/ProfilePage'
import AIAssistantPage from './components/AIAssistantPage'
import LoginPage from './components/LoginPage'

import FloatingChatBot from './components/FloatingChatBot'
import './index.css'

// API key for backend authentication
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const API_KEY = import.meta.env.VITE_API_KEY || ''

function App() {
    // ── Auth state (persisted in localStorage) ──
    const [isLoggedIn, setIsLoggedIn] = useState(() => {
        return localStorage.getItem('pyvuln_logged_in') === 'true'
    })
    const [user, setUser] = useState(() => {
        try {
            const stored = localStorage.getItem('pyvuln_user')
            return stored ? JSON.parse(stored) : null
        } catch { return null }
    })

    const [currentView, setCurrentView] = useState('dashboard')
    const [analysisResult, setAnalysisResult] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [sidebarOpen, setSidebarOpen] = useState(false)

    // ── Scan history (persisted in localStorage) ──
    const [scanHistory, setScanHistory] = useState(() => {
        try {
            const stored = localStorage.getItem('pyvuln_scan_history')
            return stored ? JSON.parse(stored) : []
        } catch { return [] }
    })

    const saveScanToHistory = useCallback((result) => {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            filename: result.filename || 'unknown.py',
            is_vulnerable: result.is_vulnerable || false,
            confidence: result.confidence || 0,
            vulnerability_type: result.vulnerability_type || 'none',
            severity: result.severity || 'low',
            vulnerabilities: result.vulnerabilities || [],
            risk_score: result.risk_score || 0,
        }
        setScanHistory(prev => {
            const updated = [entry, ...prev].slice(0, 200)
            localStorage.setItem('pyvuln_scan_history', JSON.stringify(updated))
            return updated
        })
    }, [])

    const handleLogin = (userData) => {
        setUser(userData)
        setIsLoggedIn(true)
    }

    const handleLogout = () => {
        setIsLoggedIn(false)
        setUser(null)
        localStorage.removeItem('pyvuln_logged_in')
        localStorage.removeItem('pyvuln_user')
        setCurrentView('dashboard')
    }

    const navigate = (view) => {
        setCurrentView(view)
        setSidebarOpen(false)
    }

    const analyzeCode = useCallback(async (code, filename = 'code.py') => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ content: code, filename })
            })

            if (!response.ok) {
                let msg = 'Analysis failed'
                try { const data = await response.json(); msg = data.detail || msg } catch {}
                throw new Error(msg)
            }

            const result = await response.json()
            setAnalysisResult(result)
            saveScanToHistory(result)
        } catch (err) {
            setError(err.message === 'Failed to fetch' ? 'Cannot connect to backend — is the server running?' : err.message)
            setAnalysisResult(null)
        } finally {
            setIsLoading(false)
        }
    }, [saveScanToHistory])

    const analyzeFiles = useCallback(async (files) => {
        setIsLoading(true)
        setError(null)

        const fileContents = await Promise.all(
            files.map(async (file) => ({
                filename: file.name,
                content: await file.text()
            }))
        )

        try {
            const response = await fetch(`${API_BASE_URL}/analyze/batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ files: fileContents })
            })

            if (!response.ok) {
                let msg = 'Batch analysis failed'
                try { const data = await response.json(); msg = data.detail || msg } catch {}
                throw new Error(msg)
            }

            const result = await response.json()
            if (result.results && result.results.length > 0) {
                setAnalysisResult(result.results[0])
                result.results.forEach(r => saveScanToHistory(r))
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setIsLoading(false)
        }
    }, [saveScanToHistory])

    // Close sidebar on escape
    useEffect(() => {
        const handleEsc = (e) => { if (e.key === 'Escape') setSidebarOpen(false) }
        window.addEventListener('keydown', handleEsc)
        return () => window.removeEventListener('keydown', handleEsc)
    }, [])

    // ── If not logged in, show login gate ──
    if (!isLoggedIn) {
        return <LoginPage onLogin={handleLogin} />
    }

    // Render current page
    const renderPage = () => {
        switch (currentView) {
            case 'dashboard':
                return (
                    <DashboardOverview
                        scanHistory={scanHistory}
                        onNavigate={navigate}
                    />
                )
            case 'scans':
                return (
                    <Dashboard
                        analysisResult={analysisResult}
                        isLoading={isLoading}
                        error={error}
                        onAnalyze={analyzeCode}
                        onUpload={analyzeFiles}
                    />
                )
            case 'history':
                return <ScansPage scanHistory={scanHistory} onNavigate={navigate} />
            case 'reports':
                return <ReportsPage scanHistory={scanHistory} />

            case 'profile':
                return <ProfilePage user={user} onLogout={handleLogout} />
            case 'ai':
                return <AIAssistantPage />

            default:
                return null
        }
    }

    // Sidebar navigation items
    const navItems = [
        { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
        { id: 'scans', label: 'Scan Code', icon: 'qr_code_scanner' },
        { id: 'ai', label: 'AI Assistant', icon: 'auto_awesome' },
        { id: 'history', label: 'History', icon: 'history' },
        { id: 'reports', label: 'Reports', icon: 'assessment' },
    ]

    const userName = user?.name || 'User'

    return (
        <>
            {/* Mobile Sidebar Overlay */}
            <div
                className={`sd-sidebar-overlay ${sidebarOpen ? 'open' : ''}`}
                onClick={() => setSidebarOpen(false)}
            />

            {/* Fixed Sidebar */}
            <aside className={`sd-sidebar ${sidebarOpen ? 'open' : ''}`}>
                <div className="sd-sidebar-brand">
                    <div className="sd-sidebar-logo" onClick={() => navigate('dashboard')}>
                        <i className="material-symbols-outlined">dataset</i>
                        <span>SkyDrop</span>
                    </div>
                    <span className="sd-sidebar-badge">Cybersecurity Ledger</span>
                </div>

                <nav className="sd-sidebar-nav">
                    {navItems.map(item => (
                        <button
                            key={item.id}
                            className={`sd-nav-item ${currentView === item.id ? 'active' : ''}`}
                            onClick={() => navigate(item.id)}
                        >
                            <span className="material-symbols-outlined sd-nav-icon">{item.icon}</span>
                            <span>{item.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="sd-sidebar-bottom">
                    <button
                        className="sd-new-scan-btn"
                        onClick={() => navigate('scans')}
                    >
                        <span>+</span>
                        <span>New Scan</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="sd-main">
                {/* Top Header Bar */}
                <header className="sd-header">
                    <div className="sd-header-left">
                        <button
                            className="sd-menu-toggle"
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            aria-label="Toggle sidebar"
                        >
                            ☰
                        </button>
                        <div className="sd-search-bar">
                            <span className="sd-search-icon">🔍</span>
                            <input
                                className="sd-search-input"
                                type="text"
                                placeholder="Search repository vulnerabilities..."
                            />
                        </div>
                    </div>

                    <div className="sd-header-right">
                        <button className="sd-header-btn" title="System Status">
                            <span className="sd-icon-sensors">◉</span>
                        </button>
                        <button className="sd-header-btn sd-notif-btn" title="Notifications">
                            🔔
                            <span className="sd-notif-dot" />
                        </button>
                        <div className="sd-header-divider" />
                        <button
                            className="sd-user-btn"
                            onClick={() => navigate('profile')}
                        >
                            <span className="sd-user-name">{userName}</span>
                            <div className="sd-user-avatar">
                                {userName.charAt(0).toUpperCase()}
                            </div>
                        </button>
                    </div>
                </header>

                {/* Page Content */}
                <div className="sd-content">
                    {renderPage()}
                </div>
            </main>

            {error && (
                <div className="sd-toast sd-toast-error">
                    <span>⚠️</span>
                    <span>{error}</span>
                </div>
            )}

            <FloatingChatBot />

            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

                .material-symbols-outlined {
                    font-family: 'Material Symbols Outlined' !important;
                    font-weight: normal;
                    font-style: normal;
                    font-size: 24px;
                    line-height: 1;
                    letter-spacing: normal;
                    text-transform: none;
                    display: inline-block;
                    white-space: nowrap;
                    word-wrap: normal;
                    direction: ltr;
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                    text-rendering: optimizeLegibility;
                    font-feature-settings: 'liga';
                    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                }

                /* ===========================
                   GLOBAL RESET/BASE
                   =========================== */
                .sd-sidebar, .sd-main, .sd-header,
                .sd-sidebar *, .sd-main *, .sd-header * {
                    font-family: 'Inter', -apple-system, sans-serif;
                    box-sizing: border-box;
                }

                /* ===========================
                   SIDEBAR
                   =========================== */
                .sd-sidebar {
                    position: fixed;
                    left: 0;
                    top: 0;
                    bottom: 0;
                    width: 256px;
                    background: #0f172a; /* Slate 950 */
                    display: flex;
                    flex-direction: column;
                    padding: 24px;
                    z-index: 40;
                    transition: transform 0.3s ease;
                    border-right: 1px solid rgba(148, 163, 184, 0.1);
                }
                @media (max-width: 1023px) {
                    .sd-sidebar {
                        transform: translateX(-100%);
                    }
                    .sd-sidebar.open {
                        transform: translateX(0);
                    }
                }

                .sd-sidebar-overlay {
                    display: none;
                    position: fixed;
                    inset: 0;
                    background: rgba(0,0,0,0.3);
                    z-index: 35;
                }
                .sd-sidebar-overlay.open {
                    display: block;
                }
                @media (min-width: 1024px) {
                    .sd-sidebar-overlay { display: none !important; }
                }

                .sd-sidebar-brand {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                    margin-bottom: 32px;
                }
                .sd-sidebar-logo {
                    font-family: 'Manrope', sans-serif;
                    font-size: 19px;
                    font-weight: 800;
                    color: #fff;
                    letter-spacing: -0.02em;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .sd-sidebar-logo i {
                    background: #4f46e5;
                    color: #fff;
                    padding: 6px;
                    border-radius: 8px;
                    font-size: 16px;
                }
                .sd-sidebar-badge {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: #94a3b8;
                    margin-top: 4px;
                    opacity: 0.8;
                }

                .sd-sidebar-nav {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }

                .sd-nav-item {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 12px 16px;
                    border: none;
                    border-radius: 12px;
                    background: transparent;
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                    letter-spacing: 0.02em;
                    color: #94a3b8; /* Steel */
                    cursor: pointer;
                    transition: all 0.2s ease;
                    text-align: left;
                    width: 100%;
                }
                .sd-nav-item:hover {
                    background: rgba(255, 255, 255, 0.05);
                    color: #f8fafc;
                    transform: translateX(4px);
                }
                .sd-nav-item.active {
                    background: #4f46e5;
                    color: #fff;
                    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
                    font-weight: 700;
                }
                .sd-nav-item.active .sd-nav-icon {
                    opacity: 1;
                }
                .sd-nav-icon {
                    font-size: 18px;
                    width: 24px;
                    text-align: center;
                    opacity: 0.8;
                }

                .sd-sidebar-bottom {
                    margin-top: auto;
                    padding-top: 16px;
                }
                .sd-new-scan-btn {
                    width: 100%;
                    padding: 14px 16px;
                    border: none;
                    border-radius: 12px;
                    background: #4f46e5;
                    color: #f9f6ff;
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 700;
                    letter-spacing: -0.01em;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    transition: all 0.2s ease;
                    box-shadow: 0 4px 12px rgba(76, 86, 175, 0.25);
                }
                .sd-new-scan-btn:hover {
                    opacity: 0.92;
                    box-shadow: 0 6px 20px rgba(76, 86, 175, 0.35);
                }
                .sd-new-scan-btn:active {
                    transform: scale(0.96);
                }

                /* ===========================
                   MAIN CONTENT AREA
                   =========================== */
                .sd-main {
                    margin-left: 0;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    background: #f7f9fc;
                }
                @media (min-width: 1024px) {
                    .sd-main {
                        margin-left: 256px;
                    }
                }

                /* ===========================
                   TOP HEADER BAR
                   =========================== */
                .sd-header {
                    position: sticky;
                    top: 0;
                    z-index: 30;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 24px 48px;
                    background: rgba(255, 255, 255, 0.8);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                }

                .sd-header-left {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                }

                .sd-menu-toggle {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 36px;
                    height: 36px;
                    border: none;
                    background: none;
                    font-size: 20px;
                    cursor: pointer;
                    border-radius: 8px;
                    color: #29343a;
                    transition: background 0.2s;
                }
                .sd-menu-toggle:hover { background: rgba(0,0,0,0.05); }
                @media (min-width: 1024px) {
                    .sd-menu-toggle { display: none; }
                }

                .sd-search-bar {
                    position: relative;
                    display: flex;
                    align-items: center;
                }
                .sd-search-icon {
                    position: absolute;
                    left: 14px;
                    font-size: 14px;
                    pointer-events: none;
                    opacity: 0.5;
                }
                .sd-search-input {
                    width: 320px;
                    padding: 12px 16px 12px 48px;
                    background: #e2e8f0;
                    border: none;
                    border-radius: 10px;
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    color: #1e293b;
                    outline: none;
                    transition: all 0.2s;
                }
                .sd-search-input:focus {
                    background: #dfe8ef;
                    box-shadow: 0 0 0 2px rgba(76, 86, 175, 0.15);
                }
                .sd-search-input::placeholder { color: #94a3b8; }
                @media (max-width: 768px) {
                    .sd-search-input { width: 180px; }
                }

                .sd-header-right {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .sd-header-btn {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 40px;
                    height: 40px;
                    border: none;
                    background: transparent;
                    border-radius: 999px;
                    cursor: pointer;
                    font-size: 18px;
                    color: #566168;
                    transition: background 0.2s;
                }
                .sd-header-btn:hover { background: rgba(148, 163, 184, 0.15); }

                .sd-icon-sensors { font-size: 20px; }

                .sd-notif-btn {
                    position: relative;
                }
                .sd-notif-dot {
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    width: 8px;
                    height: 8px;
                    background: #9f403d;
                    border-radius: 50%;
                    border: 2px solid #f8fafc;
                }

                .sd-header-divider {
                    width: 1px;
                    height: 32px;
                    background: rgba(148, 163, 184, 0.2);
                    margin: 0 8px;
                }

                .sd-user-btn {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 4px 4px 4px 16px;
                    border: 1px solid rgba(168, 179, 187, 0.15);
                    border-radius: 999px;
                    background: transparent;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .sd-user-btn:hover { background: rgba(148, 163, 184, 0.1); }

                .sd-user-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 13px;
                    font-weight: 700;
                    color: #29343a;
                }
                .sd-user-avatar {
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    background: #e0e0ff;
                    color: #4c56af;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-family: 'Manrope', sans-serif;
                    font-weight: 800;
                    font-size: 14px;
                }

                /* ===========================
                   CONTENT AREA
                   =========================== */
                .sd-content {
                    flex: 1;
                    padding: 0;
                }

                /* ===========================
                   EMPTY STATE
                   =========================== */
                .sd-page-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 80px 32px;
                    text-align: center;
                }
                .sd-empty-icon {
                    font-size: 56px;
                    margin-bottom: 16px;
                }
                .sd-empty-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 24px;
                    font-weight: 700;
                    color: #29343a;
                    margin: 0 0 8px;
                }
                .sd-empty-text {
                    font-size: 14px;
                    color: #566168;
                    max-width: 360px;
                    margin: 0;
                }

                /* ===========================
                   TOAST NOTIFICATIONS
                   =========================== */
                .sd-toast {
                    position: fixed;
                    bottom: 24px;
                    left: 50%;
                    transform: translateX(-50%);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 14px 24px;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: 500;
                    z-index: 999;
                    animation: sd-slide-up 0.3s ease;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                }
                .sd-toast-error {
                    background: #fef2f2;
                    color: #9f403d;
                    border: 1px solid rgba(159, 64, 61, 0.15);
                }
                @keyframes sd-slide-up {
                    from { transform: translateX(-50%) translateY(20px); opacity: 0; }
                    to { transform: translateX(-50%) translateY(0); opacity: 1; }
                }

                /* ===========================
                   DARK MODE
                   =========================== */
                .dark-mode body,
                .dark-mode .sd-main,
                .dark-mode .sd-content {
                    background: #0f1419 !important;
                    color: #d1d9e0 !important;
                }
                .dark-mode .sd-sidebar {
                    background: #161b22 !important;
                }
                .dark-mode .sd-sidebar-logo {
                    color: #e6edf3 !important;
                }
                .dark-mode .sd-sidebar-badge {
                    color: #8b949e !important;
                }
                .dark-mode .sd-nav-item {
                    color: #8b949e !important;
                }
                .dark-mode .sd-nav-item:hover {
                    background: rgba(255,255,255,0.06) !important;
                    color: #e6edf3 !important;
                }
                .dark-mode .sd-nav-item.active {
                    background: rgba(76,86,175,0.15) !important;
                    color: #929bfa !important;
                }
                .dark-mode .sd-header {
                    background: #0f1419 !important;
                    border-bottom-color: rgba(255,255,255,0.06) !important;
                }
                .dark-mode .sd-header input {
                    background: #161b22 !important;
                    color: #d1d9e0 !important;
                    border-color: rgba(255,255,255,0.08) !important;
                }
                .dark-mode .sd-header .material-symbols-outlined {
                    color: #8b949e !important;
                }
                .dark-mode .sd-new-scan-btn {
                    background: #4c56af !important;
                }

                /* Dark mode for settings & app info cards */
                .dark-mode .st-card,
                .dark-mode .ai-about-card {
                    background: #161b22 !important;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
                }
                .dark-mode .st-title,
                .dark-mode .st-card-title,
                .dark-mode .st-row-label,
                .dark-mode .ai-headline,
                .dark-mode .ai-about-title,
                .dark-mode .ai-cap-title,
                .dark-mode .ai-cap-name,
                .dark-mode .ai-creator-name,
                .dark-mode .ai-topbar-title {
                    color: #e6edf3 !important;
                }
                .dark-mode .st-subtitle,
                .dark-mode .st-row-desc,
                .dark-mode .st-select-hint,
                .dark-mode .ai-lead,
                .dark-mode .ai-about-text,
                .dark-mode .ai-cap-desc,
                .dark-mode .ai-creator-quote,
                .dark-mode .ai-cap-sub {
                    color: #8b949e !important;
                }
                .dark-mode .st-nav-item {
                    color: #8b949e !important;
                }
                .dark-mode .st-nav-item.active {
                    background: rgba(76,86,175,0.2) !important;
                    color: #929bfa !important;
                }
                .dark-mode .st-nav-item:hover {
                    background: rgba(255,255,255,0.05) !important;
                }
                .dark-mode .st-select {
                    background: #21262d !important;
                    color: #d1d9e0 !important;
                    border-color: rgba(255,255,255,0.1) !important;
                }
                .dark-mode .st-notif-card {
                    background: #161b22 !important;
                    border-color: rgba(255,255,255,0.06) !important;
                }
                .dark-mode .st-notif-name {
                    color: #e6edf3 !important;
                }
                .dark-mode .st-notif-desc {
                    color: #8b949e !important;
                }
                .dark-mode .st-appinfo-btn {
                    color: #8b949e !important;
                    border-color: rgba(255,255,255,0.1) !important;
                }
                .dark-mode .st-appinfo-btn:hover {
                    background: rgba(255,255,255,0.05) !important;
                    color: #929bfa !important;
                }
                .dark-mode .ai-creator {
                    background: #161b22 !important;
                }
                .dark-mode .ai-creator-link {
                    background: #21262d !important;
                    color: #d1d9e0 !important;
                }
                .dark-mode .ai-creator-link:hover {
                    color: #929bfa !important;
                }
                .dark-mode .ai-footer {
                    border-top-color: rgba(255,255,255,0.06) !important;
                }
                .dark-mode .ai-headline-outline {
                    -webkit-text-stroke-color: #e6edf3 !important;
                }
                .dark-mode .ai-badge {
                    color: #929bfa !important;
                }
                .dark-mode .ai-trusted-text,
                .dark-mode .ai-footer-name {
                    color: #d1d9e0 !important;
                }
                .dark-mode .ai-cap-icon-wrap {
                    background: #21262d !important;
                }

                /* Dark mode for all page-container cards */
                .dark-mode .glass-card,
                .dark-mode .page-container {
                    background: #161b22 !important;
                    color: #d1d9e0 !important;
                }
            `}</style>
        </>
    )
}

export default App
