import React, { useState } from 'react'

function LoginPage({ onLogin }) {
    const [mode, setMode] = useState('login')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [name, setName] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [error, setError] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)

    const handleLogin = async (e) => {
        e.preventDefault()
        setError('')

        if (!email.trim() || !password.trim()) {
            setError('Please fill in all fields')
            return
        }
        if (!email.includes('@')) {
            setError('Please enter a valid email address')
            return
        }

        setIsSubmitting(true)
        // Simulate network delay
        await new Promise(r => setTimeout(r, 800))

        const user = {
            name: email.split('@')[0],
            email,
            role: 'Security Analyst',
            scans: 0,
            joined: new Date().toISOString().split('T')[0]
        }

        localStorage.setItem('pyvuln_user', JSON.stringify(user))
        localStorage.setItem('pyvuln_logged_in', 'true')
        onLogin(user)
        setIsSubmitting(false)
    }

    const handleSignup = async (e) => {
        e.preventDefault()
        setError('')

        if (!name.trim() || !email.trim() || !password.trim() || !confirmPassword.trim()) {
            setError('Please fill in all fields')
            return
        }
        if (!email.includes('@')) {
            setError('Please enter a valid email address')
            return
        }
        if (password.length < 6) {
            setError('Password must be at least 6 characters')
            return
        }
        if (password !== confirmPassword) {
            setError('Passwords do not match')
            return
        }

        setIsSubmitting(true)
        await new Promise(r => setTimeout(r, 800))

        const user = {
            name,
            email,
            role: 'Security Analyst',
            scans: 0,
            joined: new Date().toISOString().split('T')[0]
        }

        localStorage.setItem('pyvuln_user', JSON.stringify(user))
        localStorage.setItem('pyvuln_logged_in', 'true')
        onLogin(user)
        setIsSubmitting(false)
    }

    return (
        <div className="login-gate">
            {/* Animated background orbs */}
            <div className="login-orb login-orb-1" />
            <div className="login-orb login-orb-2" />
            <div className="login-orb login-orb-3" />

            {/* Floating particles */}
            <div className="login-particles" aria-hidden="true">
                {Array.from({ length: 30 }, (_, i) => (
                    <div
                        key={i}
                        className="login-particle"
                        style={{
                            left: `${Math.random() * 100}%`,
                            animationDelay: `${Math.random() * 12}s`,
                            animationDuration: `${10 + Math.random() * 15}s`,
                            width: `${1 + Math.random() * 2}px`,
                            height: `${1 + Math.random() * 2}px`,
                        }}
                    />
                ))}
            </div>

            <div className="login-card">
                {/* Top glow bar */}
                <div className="login-card-glow" />

                {/* Logo */}
                <div className="login-logo-section">
                    <div className="login-logo-icon">🛡️</div>
                    <h1 className="login-logo-title">PyVulnDetect</h1>
                    <p className="login-logo-tagline">AI-Powered Python Security Scanner</p>
                </div>

                {/* Tabs */}
                <div className="login-tabs">
                    <button
                        className={`login-tab ${mode === 'login' ? 'active' : ''}`}
                        onClick={() => { setMode('login'); setError('') }}
                    >
                        Sign In
                    </button>
                    <button
                        className={`login-tab ${mode === 'signup' ? 'active' : ''}`}
                        onClick={() => { setMode('signup'); setError('') }}
                    >
                        Sign Up
                    </button>
                </div>

                {/* Form */}
                <form onSubmit={mode === 'login' ? handleLogin : handleSignup} className="login-form">
                    {mode === 'signup' && (
                        <div className="login-field">
                            <label className="login-label">Full Name</label>
                            <div className="login-input-wrapper">
                                <span className="login-input-icon">👤</span>
                                <input
                                    type="text"
                                    className="login-input"
                                    placeholder="John Doe"
                                    value={name}
                                    onChange={e => setName(e.target.value)}
                                    autoComplete="name"
                                />
                            </div>
                        </div>
                    )}

                    <div className="login-field">
                        <label className="login-label">Email Address</label>
                        <div className="login-input-wrapper">
                            <span className="login-input-icon">📧</span>
                            <input
                                type="email"
                                className="login-input"
                                placeholder="you@example.com"
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                                autoComplete="email"
                            />
                        </div>
                    </div>

                    <div className="login-field">
                        <label className="login-label">Password</label>
                        <div className="login-input-wrapper">
                            <span className="login-input-icon">🔒</span>
                            <input
                                type="password"
                                className="login-input"
                                placeholder="••••••••"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                            />
                        </div>
                    </div>

                    {mode === 'signup' && (
                        <div className="login-field">
                            <label className="login-label">Confirm Password</label>
                            <div className="login-input-wrapper">
                                <span className="login-input-icon">🔒</span>
                                <input
                                    type="password"
                                    className="login-input"
                                    placeholder="••••••••"
                                    value={confirmPassword}
                                    onChange={e => setConfirmPassword(e.target.value)}
                                    autoComplete="new-password"
                                />
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="login-error">
                            <span>⚠️</span> {error}
                        </div>
                    )}

                    <button type="submit" className="login-submit" disabled={isSubmitting}>
                        {isSubmitting ? (
                            <span className="login-spinner" />
                        ) : (
                            mode === 'login' ? '🔓 Sign In' : '🚀 Create Account'
                        )}
                    </button>
                </form>

                <p className="login-footer">
                    {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
                    <button
                        className="login-switch"
                        onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setError('') }}
                    >
                        {mode === 'login' ? 'Sign Up' : 'Sign In'}
                    </button>
                </p>

                {/* Security badge */}
                <div className="login-security-badge">
                    <span>🔐</span> Secured with end-to-end encryption
                </div>
            </div>

            <style>{`
                .login-gate {
                    position: fixed;
                    inset: 0;
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #06050e;
                    overflow: hidden;
                }

                /* Animated background orbs */
                .login-orb {
                    position: absolute;
                    border-radius: 50%;
                    filter: blur(80px);
                    pointer-events: none;
                }
                .login-orb-1 {
                    width: 500px;
                    height: 500px;
                    top: -10%;
                    left: -5%;
                    background: radial-gradient(circle, rgba(88, 28, 135, 0.4), transparent 70%);
                    animation: orb-float-1 18s ease-in-out infinite alternate;
                }
                .login-orb-2 {
                    width: 400px;
                    height: 400px;
                    bottom: -10%;
                    right: -5%;
                    background: radial-gradient(circle, rgba(0, 240, 255, 0.2), transparent 70%);
                    animation: orb-float-2 22s ease-in-out infinite alternate;
                }
                .login-orb-3 {
                    width: 300px;
                    height: 300px;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: radial-gradient(circle, rgba(77, 124, 255, 0.15), transparent 70%);
                    animation: orb-float-3 15s ease-in-out infinite alternate;
                }
                @keyframes orb-float-1 {
                    0% { transform: translate(0, 0) scale(1); }
                    100% { transform: translate(60px, 40px) scale(1.2); }
                }
                @keyframes orb-float-2 {
                    0% { transform: translate(0, 0) scale(1); }
                    100% { transform: translate(-50px, -30px) scale(1.15); }
                }
                @keyframes orb-float-3 {
                    0% { transform: translate(-50%, -50%) scale(1); }
                    100% { transform: translate(-40%, -60%) scale(1.3); }
                }

                /* Floating particles */
                .login-particles {
                    position: absolute;
                    inset: 0;
                    pointer-events: none;
                    overflow: hidden;
                }
                .login-particle {
                    position: absolute;
                    background: rgba(255, 255, 255, 0.5);
                    border-radius: 50%;
                    animation: login-particle-rise linear infinite;
                }
                .login-particle:nth-child(odd) { background: rgba(0, 240, 255, 0.4); }
                .login-particle:nth-child(3n) { background: rgba(168, 85, 247, 0.4); }
                @keyframes login-particle-rise {
                    0% { transform: translateY(100vh); opacity: 0; }
                    10% { opacity: 0.8; }
                    90% { opacity: 0.8; }
                    100% { transform: translateY(-10vh); opacity: 0; }
                }

                /* Card */
                .login-card {
                    position: relative;
                    width: 460px;
                    max-width: 92vw;
                    background: rgba(12, 10, 30, 0.65);
                    backdrop-filter: blur(40px);
                    -webkit-backdrop-filter: blur(40px);
                    border: 1px solid rgba(120, 100, 255, 0.15);
                    border-radius: 28px;
                    padding: 44px 40px 36px;
                    box-shadow:
                        0 24px 80px rgba(0, 0, 0, 0.6),
                        0 0 60px rgba(0, 240, 255, 0.04),
                        inset 0 1px 0 rgba(255, 255, 255, 0.04);
                    animation: login-card-appear 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
                }
                .login-card-glow {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    border-radius: 28px 28px 0 0;
                    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-magenta), var(--neon-violet));
                    background-size: 200% 100%;
                    animation: glow-sweep 4s ease-in-out infinite;
                }
                @keyframes glow-sweep {
                    0%, 100% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                }
                @keyframes login-card-appear {
                    from { transform: translateY(40px) scale(0.95); opacity: 0; }
                    to { transform: translateY(0) scale(1); opacity: 1; }
                }

                /* Logo section */
                .login-logo-section {
                    text-align: center;
                    margin-bottom: 32px;
                }
                .login-logo-icon {
                    font-size: 52px;
                    margin-bottom: 12px;
                    filter: drop-shadow(0 0 24px rgba(0, 240, 255, 0.4));
                    animation: float-icon 4s ease-in-out infinite;
                }
                .login-logo-title {
                    font-family: var(--font-heading);
                    font-size: 30px;
                    font-weight: 800;
                    margin: 0 0 6px;
                    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-violet));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.02em;
                }
                .login-logo-tagline {
                    font-size: 13px;
                    color: var(--text-secondary);
                    margin: 0;
                    letter-spacing: 0.02em;
                }

                /* Tabs */
                .login-tabs {
                    display: flex;
                    gap: 4px;
                    background: rgba(120, 100, 255, 0.06);
                    border-radius: 14px;
                    padding: 4px;
                    margin-bottom: 28px;
                }
                .login-tab {
                    flex: 1;
                    padding: 11px;
                    border: none;
                    border-radius: 11px;
                    font-family: var(--font-main);
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
                    background: transparent;
                    color: var(--text-muted);
                }
                .login-tab.active {
                    background: linear-gradient(135deg, rgba(0, 240, 255, 0.12), rgba(168, 85, 247, 0.1));
                    color: var(--neon-cyan);
                    box-shadow: 0 0 16px rgba(0, 240, 255, 0.08);
                }
                .login-tab:hover:not(.active) { color: var(--text-secondary); }

                /* Form */
                .login-form {
                    display: flex;
                    flex-direction: column;
                    gap: 18px;
                }
                .login-field {
                    display: flex;
                    flex-direction: column;
                    gap: 7px;
                }
                .login-label {
                    font-size: 11px;
                    font-weight: 600;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                }
                .login-input-wrapper {
                    position: relative;
                    display: flex;
                    align-items: center;
                }
                .login-input-icon {
                    position: absolute;
                    left: 14px;
                    font-size: 14px;
                    pointer-events: none;
                    opacity: 0.6;
                }
                .login-input {
                    width: 100%;
                    padding: 13px 16px 13px 42px;
                    border: 1px solid rgba(120, 100, 255, 0.12);
                    border-radius: 14px;
                    background: rgba(255, 255, 255, 0.03);
                    color: var(--text-primary);
                    font-family: var(--font-main);
                    font-size: 14px;
                    outline: none;
                    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
                }
                .login-input:focus {
                    border-color: rgba(0, 240, 255, 0.35);
                    box-shadow: 0 0 0 4px rgba(0, 240, 255, 0.06), 0 0 20px rgba(0, 240, 255, 0.05);
                    background: rgba(255, 255, 255, 0.05);
                }
                .login-input::placeholder { color: var(--text-muted); }

                .login-error {
                    padding: 11px 16px;
                    background: rgba(255, 45, 85, 0.08);
                    border: 1px solid rgba(255, 45, 85, 0.18);
                    border-radius: 12px;
                    color: var(--neon-red);
                    font-size: 13px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    animation: login-shake 0.4s ease;
                }
                @keyframes login-shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-6px); }
                    75% { transform: translateX(6px); }
                }

                .login-submit {
                    padding: 15px;
                    border: none;
                    border-radius: 16px;
                    background: linear-gradient(135deg, var(--neon-blue), var(--neon-violet));
                    color: white;
                    font-family: var(--font-main);
                    font-size: 15px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
                    margin-top: 4px;
                    box-shadow: 0 6px 24px rgba(77, 124, 255, 0.35);
                    min-height: 52px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .login-submit:hover:not(:disabled) {
                    transform: translateY(-3px);
                    box-shadow: 0 10px 36px rgba(77, 124, 255, 0.45), 0 0 50px rgba(168, 85, 247, 0.12);
                }
                .login-submit:active:not(:disabled) {
                    transform: translateY(0) scale(0.98);
                }
                .login-submit:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }

                .login-spinner {
                    width: 22px;
                    height: 22px;
                    border: 2.5px solid rgba(255,255,255,0.2);
                    border-top-color: white;
                    border-radius: 50%;
                    animation: spinner-rotate 0.7s linear infinite;
                }
                @keyframes spinner-rotate { to { transform: rotate(360deg); } }

                .login-footer {
                    text-align: center;
                    margin-top: 22px;
                    font-size: 13px;
                    color: var(--text-muted);
                }
                .login-switch {
                    background: none;
                    border: none;
                    color: var(--neon-cyan);
                    cursor: pointer;
                    font-family: var(--font-main);
                    font-size: 13px;
                    font-weight: 600;
                    transition: all 0.2s;
                }
                .login-switch:hover {
                    color: var(--neon-violet);
                    text-shadow: 0 0 10px rgba(168, 85, 247, 0.4);
                }

                .login-security-badge {
                    text-align: center;
                    margin-top: 20px;
                    padding-top: 16px;
                    border-top: 1px solid rgba(120, 100, 255, 0.06);
                    font-size: 11px;
                    color: var(--text-muted);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    letter-spacing: 0.03em;
                }
            `}</style>
        </div>
    )
}

export default LoginPage
