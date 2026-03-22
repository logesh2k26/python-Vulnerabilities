import { useState, useCallback, useEffect, useRef } from 'react'
import Dashboard from './components/Dashboard'
import ScansPage from './components/ScansPage'
import ReportsPage from './components/ReportsPage'
import SettingsPage from './components/SettingsPage'
import ProfilePage from './components/ProfilePage'
import LoginPage from './components/LoginPage'
import AppInfoPage from './components/AppInfoPage'
import FloatingChatBot from './components/FloatingChatBot'
import './index.css'

// API key for backend authentication
const API_KEY = import.meta.env.VITE_API_KEY || ''

// Particle Background Component
function ParticleField() {
    const particles = useRef(
        Array.from({ length: 40 }, (_, i) => ({
            id: i,
            left: Math.random() * 100,
            delay: Math.random() * 15,
            duration: 12 + Math.random() * 18,
            size: 1 + Math.random() * 2,
        }))
    ).current

    return (
        <div className="particle-field" aria-hidden="true">
            {particles.map(p => (
                <div
                    key={p.id}
                    className="particle"
                    style={{
                        left: `${p.left}%`,
                        width: `${p.size}px`,
                        height: `${p.size}px`,
                        animationDelay: `${p.delay}s`,
                        animationDuration: `${p.duration}s`,
                    }}
                />
            ))}
        </div>
    )
}

// Navigation config
const NAV_ITEMS = [
    { id: 'dashboard', label: 'Dashboard', icon: '⚡' },
    { id: 'scans', label: 'Scan Code', icon: '🛡️', headerLabel: 'Scans' },
    { id: 'history', label: 'History', icon: '📋', hidden: true },
    { id: 'reports', label: 'Reports', icon: '📊' },
    { id: 'settings', label: 'Settings', icon: '⚙️', sidebarOnly: true },
    { id: 'ai', label: 'AI Assistant', icon: '🤖', sidebarOnly: true },
    { id: 'profile', label: 'Profile', icon: '👤', headerOnly: true },
]

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
            const updated = [entry, ...prev].slice(0, 200) // keep last 200 scans
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
            const response = await fetch('/api/v1/analyze', {
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
            const response = await fetch('/api/v1/analyze/batch', {
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
                // Save all batch results to history
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
                return <ScansPage />
            case 'reports':
                return <ReportsPage scanHistory={scanHistory} />
            case 'settings':
                return <SettingsPage />
            case 'profile':
                return <ProfilePage user={user} onLogout={handleLogout} />
            case 'ai':
                return (
                    <div className="page-container">
                        <div className="glass-card empty-state">
                            <div className="empty-icon">🤖</div>
                            <h3 className="empty-title">AI Assistant</h3>
                            <p className="empty-text">Use the chat bubble in the bottom-right corner to interact with the AI Security Assistant</p>
                        </div>
                    </div>
                )
            case 'appinfo':
                return <AppInfoPage />
            default:
                return null
        }
    }

    // Determine header nav items
    const headerNavItems = [
        { id: 'dashboard', label: 'Dashboard' },
    ]

    return (
        <>
            <ParticleField />

            {/* Holographic Sidebar */}
            <div className={`sidebar-overlay ${sidebarOpen ? 'open' : ''}`} onClick={() => setSidebarOpen(false)} />
            <aside className={`holo-sidebar ${sidebarOpen ? 'open' : ''}`}>
                <div className="sidebar-label">Navigation</div>
                <nav className="sidebar-nav">
                    <button className={`sidebar-item ${currentView === 'dashboard' ? 'active' : ''}`} onClick={() => navigate('dashboard')}>
                        <span>⚡</span> Dashboard
                    </button>
                    <button className={`sidebar-item ${currentView === 'scans' ? 'active' : ''}`} onClick={() => navigate('scans')}>
                        <span>🛡️</span> Scan Code
                    </button>
                    <button className={`sidebar-item ${currentView === 'history' ? 'active' : ''}`} onClick={() => navigate('history')}>
                        <span>📋</span> History
                    </button>
                    <button className={`sidebar-item ${currentView === 'reports' ? 'active' : ''}`} onClick={() => navigate('reports')}>
                        <span>📊</span> Reports
                    </button>
                </nav>
                <div className="sidebar-section-divider" />
                <div className="sidebar-label">Tools</div>
                <nav className="sidebar-nav">
                    <button className={`sidebar-item ${currentView === 'ai' ? 'active' : ''}`} onClick={() => navigate('ai')}>
                        <span>🤖</span> AI Assistant
                    </button>
                    <button className={`sidebar-item ${currentView === 'settings' ? 'active' : ''}`} onClick={() => navigate('settings')}>
                        <span>⚙️</span> Settings
                    </button>
                    <button className={`sidebar-item ${currentView === 'appinfo' ? 'active' : ''}`} onClick={() => navigate('appinfo')}>
                        <span>ℹ️</span> App Info
                    </button>
                </nav>
                <div className="sidebar-section-divider" />
                <nav className="sidebar-nav">
                    <button className={`sidebar-item ${currentView === 'profile' ? 'active' : ''}`} onClick={() => navigate('profile')}>
                        <span>👤</span> Profile
                    </button>
                </nav>
            </aside>

            <div className="app-container">
                {/* Floating Header */}
                <header className="header glass-card">
                    <div className="logo">
                        <button
                            className="sidebar-toggle"
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            aria-label="Toggle sidebar"
                        >
                            ☰
                        </button>
                        <div className="logo-icon" style={{ cursor: 'pointer' }} onClick={() => navigate('dashboard')}>🛡️</div>
                        <span className="logo-text" style={{ cursor: 'pointer' }} onClick={() => navigate('dashboard')}>PyVulnDetect</span>
                    </div>
                    <div className="header-nav">
                        {headerNavItems.map(item => (
                            <button
                                key={item.id}
                                className={`nav-link ${currentView === item.id ? 'active' : ''}`}
                                onClick={() => navigate(item.id)}
                            >
                                {item.label}
                            </button>
                        ))}
                    </div>
                    <div className="header-right">
                        <div className="header-status">
                            <span className="status-dot"></span>
                            <span>AI Engine Ready</span>
                        </div>
                        <button
                            className="header-logout-btn"
                            onClick={handleLogout}
                            title="Sign Out"
                        >
                            🚪
                        </button>
                        <button
                            className="profile-btn"
                            onClick={() => navigate('profile')}
                            aria-label="Profile"
                        >
                            {user?.name ? user.name.charAt(0).toUpperCase() : '👤'}
                        </button>
                    </div>
                </header>

                {renderPage()}

                {error && (
                    <div className="toast error">
                        <span>⚠️</span>
                        <span>{error}</span>
                    </div>
                )}

                <FloatingChatBot />
            </div>
        </>
    )
}

export default App
