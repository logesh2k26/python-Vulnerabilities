import React, { useState } from 'react'

function LoginPage({ onLogin }) {
    const [mode, setMode] = useState('login')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [name, setName] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [error, setError] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [showPassword, setShowPassword] = useState(false)
    const [rememberMe, setRememberMe] = useState(false)

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
        <div className="skydrop-login-gate">
            <main className="skydrop-login-main">
                {/* Left Section: Brand Anchor */}
                <section className="skydrop-brand-section">
                    {/* Subtle dot pattern overlay */}
                    <div className="skydrop-brand-pattern" />
                    <div className="skydrop-brand-content">
                        <div className="skydrop-brand-header">
                            <h1 className="skydrop-brand-logo">PyVulnDetect</h1>
                            <div className="skydrop-brand-accent-bar" />
                        </div>
                        <h2 className="skydrop-brand-headline">
                            Python Vulnerability<br />Detector
                        </h2>
                        <p className="skydrop-brand-description">
                            Securing your ecosystem with enterprise-grade automated scanning and real-time threat detection.
                        </p>
                        <div className="skydrop-brand-stats">
                            <div className="skydrop-stat">
                                <div className="skydrop-stat-value">99.9%</div>
                                <div className="skydrop-stat-label">ACCURACY</div>
                            </div>
                            <div className="skydrop-stat">
                                <div className="skydrop-stat-value">Real-time</div>
                                <div className="skydrop-stat-label">MONITORING</div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Right Section: Login Form */}
                <section className="skydrop-form-section">
                    <div className="skydrop-form-container">
                        {/* Mobile Logo */}
                        <div className="skydrop-mobile-logo">
                            <h1 className="skydrop-mobile-logo-text">PyVulnDetect</h1>
                        </div>

                        <div className="skydrop-form-header">
                            <h3 className="skydrop-form-title">
                                {mode === 'login' ? 'Welcome Back' : 'Create Account'}
                            </h3>
                            <p className="skydrop-form-subtitle">
                                {mode === 'login'
                                    ? 'Sign in to your security dashboard'
                                    : 'Set up your security dashboard'}
                            </p>
                        </div>

                        {/* Tabs */}
                        <div className="skydrop-tabs">
                            <button
                                className={`skydrop-tab ${mode === 'login' ? 'active' : ''}`}
                                onClick={() => { setMode('login'); setError('') }}
                                type="button"
                            >
                                Sign In
                            </button>
                            <button
                                className={`skydrop-tab ${mode === 'signup' ? 'active' : ''}`}
                                onClick={() => { setMode('signup'); setError('') }}
                                type="button"
                            >
                                Sign Up
                            </button>
                        </div>

                        <form
                            onSubmit={mode === 'login' ? handleLogin : handleSignup}
                            className="skydrop-form"
                        >
                            {mode === 'signup' && (
                                <div className="skydrop-field">
                                    <label className="skydrop-label" htmlFor="name">Full Name</label>
                                    <input
                                        className="skydrop-input"
                                        id="name"
                                        type="text"
                                        placeholder="John Doe"
                                        value={name}
                                        onChange={e => setName(e.target.value)}
                                        autoComplete="name"
                                    />
                                </div>
                            )}

                            <div className="skydrop-field">
                                <label className="skydrop-label" htmlFor="email">
                                    {mode === 'login' ? 'Corporate Email' : 'Email Address'}
                                </label>
                                <input
                                    className="skydrop-input"
                                    id="email"
                                    type="email"
                                    placeholder="name@company.com"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    autoComplete="email"
                                />
                            </div>

                            <div className="skydrop-field">
                                <div className="skydrop-label-row">
                                    <label className="skydrop-label" htmlFor="password">Password</label>
                                    {mode === 'login' && (
                                        <button type="button" className="skydrop-forgot-link">
                                            FORGOT?
                                        </button>
                                    )}
                                </div>
                                <div className="skydrop-password-wrapper">
                                    <input
                                        className="skydrop-input"
                                        id="password"
                                        type={showPassword ? 'text' : 'password'}
                                        placeholder="••••••••"
                                        value={password}
                                        onChange={e => setPassword(e.target.value)}
                                        autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                                    />
                                    <button
                                        type="button"
                                        className="skydrop-toggle-pw"
                                        onClick={() => setShowPassword(!showPassword)}
                                        aria-label={showPassword ? 'Hide password' : 'Show password'}
                                    >
                                        {showPassword ? '🙈' : '👁️'}
                                    </button>
                                </div>
                            </div>

                            {mode === 'signup' && (
                                <div className="skydrop-field">
                                    <label className="skydrop-label" htmlFor="confirm-password">Confirm Password</label>
                                    <div className="skydrop-password-wrapper">
                                        <input
                                            className="skydrop-input"
                                            id="confirm-password"
                                            type={showPassword ? 'text' : 'password'}
                                            placeholder="••••••••"
                                            value={confirmPassword}
                                            onChange={e => setConfirmPassword(e.target.value)}
                                            autoComplete="new-password"
                                        />
                                    </div>
                                </div>
                            )}

                            {mode === 'login' && (
                                <div className="skydrop-remember">
                                    <input
                                        className="skydrop-checkbox"
                                        id="remember"
                                        type="checkbox"
                                        checked={rememberMe}
                                        onChange={e => setRememberMe(e.target.checked)}
                                    />
                                    <label className="skydrop-remember-label" htmlFor="remember">
                                        Keep me logged in for 30 days
                                    </label>
                                </div>
                            )}

                            {error && (
                                <div className="skydrop-error">
                                    <span>⚠️</span> {error}
                                </div>
                            )}

                            <button
                                type="submit"
                                className="skydrop-submit"
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? (
                                    <span className="skydrop-spinner" />
                                ) : (
                                    mode === 'login' ? 'Sign In' : 'Create Account'
                                )}
                            </button>
                        </form>

                        <div className="skydrop-form-footer">
                            <p className="skydrop-switch-text">
                                {mode === 'login'
                                    ? "Don't have an enterprise account?"
                                    : 'Already have an account?'}
                                <br />
                                <button
                                    className="skydrop-switch-btn"
                                    onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setError('') }}
                                    type="button"
                                >
                                    {mode === 'login' ? 'Request Access' : 'Sign In'}
                                </button>
                            </p>
                        </div>

                        {/* SSO/Institutional login */}
                        <div className="skydrop-sso-section">
                            <div className="skydrop-sso-divider">
                                <div className="skydrop-sso-line" />
                                <span className="skydrop-sso-label">INSTITUTIONAL LOGIN</span>
                                <div className="skydrop-sso-line" />
                            </div>
                            <button type="button" className="skydrop-sso-btn">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
                                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                                </svg>
                                <span>Continue with SSO</span>
                            </button>
                        </div>
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="skydrop-footer">
                <div className="skydrop-footer-copy">
                    <span>© 2024 PyVulnDetect Security. All rights reserved.</span>
                </div>
                <nav className="skydrop-footer-nav">
                    <a href="#">Privacy Policy</a>
                    <a href="#">Terms of Service</a>
                    <a href="#">Contact Support</a>
                </nav>
            </footer>

            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

                .skydrop-login-gate {
                    position: fixed;
                    inset: 0;
                    z-index: 9999;
                    display: flex;
                    flex-direction: column;
                    background: #f7f9fc;
                    font-family: 'Inter', sans-serif;
                    color: #29343a;
                    -webkit-font-smoothing: antialiased;
                }

                .skydrop-login-main {
                    flex: 1;
                    display: flex;
                    flex-direction: row;
                    min-height: 0;
                }

                /* === LEFT BRAND SECTION === */
                .skydrop-brand-section {
                    display: none;
                    width: 50%;
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    position: relative;
                    overflow: hidden;
                    align-items: center;
                    justify-content: center;
                    padding: 80px;
                }
                @media (min-width: 768px) {
                    .skydrop-brand-section { display: flex; }
                }
                .skydrop-brand-pattern {
                    position: absolute;
                    inset: 0;
                    opacity: 0.08;
                    pointer-events: none;
                    background-image: radial-gradient(circle at 2px 2px, white 1px, transparent 0);
                    background-size: 40px 40px;
                }
                .skydrop-brand-content {
                    position: relative;
                    z-index: 10;
                    max-width: 420px;
                    width: 100%;
                    animation: skydrop-fade-up 0.8s ease-out both;
                }
                @keyframes skydrop-fade-up {
                    from { opacity: 0; transform: translateY(24px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .skydrop-brand-header {
                    margin-bottom: 48px;
                }
                .skydrop-brand-logo {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 800;
                    font-size: 48px;
                    letter-spacing: -0.03em;
                    color: #f9f6ff;
                    margin: 0 0 16px;
                }
                .skydrop-brand-accent-bar {
                    height: 4px;
                    width: 48px;
                    background: #f9f6ff;
                    border-radius: 99px;
                }
                .skydrop-brand-headline {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 700;
                    font-size: 30px;
                    line-height: 1.25;
                    color: #f9f6ff;
                    margin: 0 0 24px;
                }
                .skydrop-brand-description {
                    color: rgba(249, 246, 255, 0.75);
                    font-size: 17px;
                    line-height: 1.6;
                    max-width: 360px;
                    margin: 0 0 48px;
                }
                .skydrop-brand-stats {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 32px;
                    padding-top: 32px;
                    border-top: 1px solid rgba(249, 246, 255, 0.1);
                }
                .skydrop-stat-value {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 800;
                    font-size: 24px;
                    letter-spacing: -0.02em;
                    color: #f9f6ff;
                }
                .skydrop-stat-label {
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 0.15em;
                    text-transform: uppercase;
                    color: rgba(249, 246, 255, 0.5);
                    margin-top: 6px;
                }

                /* === RIGHT FORM SECTION === */
                .skydrop-form-section {
                    flex: 1;
                    width: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 32px;
                    overflow-y: auto;
                    background: #f7f9fc;
                }
                @media (min-width: 768px) {
                    .skydrop-form-section {
                        width: 50%;
                        padding: 64px 96px;
                    }
                }
                .skydrop-form-container {
                    width: 100%;
                    max-width: 400px;
                    animation: skydrop-fade-up 0.8s ease-out 0.15s both;
                }

                /* Mobile Logo */
                .skydrop-mobile-logo {
                    display: block;
                    margin-bottom: 48px;
                }
                @media (min-width: 768px) {
                    .skydrop-mobile-logo { display: none; }
                }
                .skydrop-mobile-logo-text {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 800;
                    font-size: 30px;
                    letter-spacing: -0.03em;
                    color: #4c56af;
                    margin: 0;
                }

                /* Form Header */
                .skydrop-form-header {
                    margin-bottom: 32px;
                }
                .skydrop-form-title {
                    font-family: 'Manrope', sans-serif;
                    font-weight: 700;
                    font-size: 26px;
                    color: #29343a;
                    margin: 0 0 8px;
                }
                .skydrop-form-subtitle {
                    font-size: 14px;
                    color: #566168;
                    margin: 0;
                }

                /* Tabs */
                .skydrop-tabs {
                    display: flex;
                    gap: 2px;
                    background: #e8eff4;
                    border-radius: 12px;
                    padding: 3px;
                    margin-bottom: 28px;
                }
                .skydrop-tab {
                    flex: 1;
                    padding: 10px 16px;
                    border: none;
                    border-radius: 10px;
                    font-family: 'Inter', sans-serif;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.25s ease;
                    background: transparent;
                    color: #717c84;
                }
                .skydrop-tab.active {
                    background: #fff;
                    color: #4c56af;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                }
                .skydrop-tab:hover:not(.active) {
                    color: #29343a;
                }

                /* Form */
                .skydrop-form {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .skydrop-field {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .skydrop-label {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #566168;
                }
                .skydrop-label-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .skydrop-forgot-link {
                    background: none;
                    border: none;
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #4c56af;
                    cursor: pointer;
                    padding: 0;
                    transition: color 0.2s;
                }
                .skydrop-forgot-link:hover { color: #4049a2; text-decoration: underline; }

                .skydrop-input {
                    width: 100%;
                    padding: 13px 16px;
                    background: #ffffff;
                    border: 1px solid rgba(168, 179, 187, 0.2);
                    border-radius: 12px;
                    color: #29343a;
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    outline: none;
                    transition: border-color 0.25s ease, box-shadow 0.25s ease;
                    box-sizing: border-box;
                }
                .skydrop-input:focus {
                    border-color: #4c56af;
                    box-shadow: 0 0 0 3px rgba(76, 86, 175, 0.08);
                }
                .skydrop-input::placeholder {
                    color: #a8b3bb;
                }

                .skydrop-password-wrapper {
                    position: relative;
                }
                .skydrop-password-wrapper .skydrop-input {
                    padding-right: 48px;
                }
                .skydrop-toggle-pw {
                    position: absolute;
                    right: 12px;
                    top: 50%;
                    transform: translateY(-50%);
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                    padding: 4px;
                    opacity: 0.4;
                    transition: opacity 0.2s;
                }
                .skydrop-toggle-pw:hover { opacity: 0.8; }

                /* Checkbox */
                .skydrop-remember {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 4px 0;
                }
                .skydrop-checkbox {
                    width: 18px;
                    height: 18px;
                    border-radius: 5px;
                    border: 1.5px solid rgba(168, 179, 187, 0.4);
                    accent-color: #4c56af;
                    cursor: pointer;
                }
                .skydrop-remember-label {
                    font-size: 14px;
                    color: #566168;
                    cursor: pointer;
                }

                /* Error */
                .skydrop-error {
                    padding: 12px 16px;
                    background: rgba(159, 64, 61, 0.06);
                    border: 1px solid rgba(159, 64, 61, 0.15);
                    border-radius: 10px;
                    color: #9f403d;
                    font-size: 13px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    animation: skydrop-shake 0.4s ease;
                }
                @keyframes skydrop-shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-6px); }
                    75% { transform: translateX(6px); }
                }

                /* Submit Button */
                .skydrop-submit {
                    width: 100%;
                    padding: 16px;
                    border: none;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #4c56af 0%, #4049a2 100%);
                    color: #f9f6ff;
                    font-family: 'Manrope', sans-serif;
                    font-size: 15px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.25s ease;
                    min-height: 54px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-top: 4px;
                }
                .skydrop-submit:hover:not(:disabled) {
                    opacity: 0.92;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 16px rgba(76, 86, 175, 0.35);
                }
                .skydrop-submit:active:not(:disabled) {
                    transform: scale(0.98);
                }
                .skydrop-submit:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }

                /* Spinner */
                .skydrop-spinner {
                    width: 22px;
                    height: 22px;
                    border: 2.5px solid rgba(249, 246, 255, 0.3);
                    border-top-color: #f9f6ff;
                    border-radius: 50%;
                    animation: skydrop-spin 0.7s linear infinite;
                    display: inline-block;
                }
                @keyframes skydrop-spin { to { transform: rotate(360deg); } }

                /* Form Footer */
                .skydrop-form-footer {
                    margin-top: 32px;
                    padding-top: 28px;
                    border-top: 1px solid rgba(168, 179, 187, 0.15);
                    text-align: center;
                }
                .skydrop-switch-text {
                    font-size: 14px;
                    color: #566168;
                    margin: 0;
                    line-height: 1.7;
                }
                .skydrop-switch-btn {
                    background: none;
                    border: none;
                    color: #4c56af;
                    cursor: pointer;
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    font-weight: 700;
                    padding: 0;
                    transition: all 0.2s;
                }
                .skydrop-switch-btn:hover {
                    text-decoration: underline;
                    color: #4049a2;
                }

                /* SSO Section */
                .skydrop-sso-section {
                    margin-top: 32px;
                }
                .skydrop-sso-divider {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    margin-bottom: 20px;
                }
                .skydrop-sso-line {
                    flex: 1;
                    height: 1px;
                    background: rgba(168, 179, 187, 0.15);
                }
                .skydrop-sso-label {
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 0.15em;
                    text-transform: uppercase;
                    color: #566168;
                    white-space: nowrap;
                }
                .skydrop-sso-btn {
                    width: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 12px;
                    padding: 14px 16px;
                    background: #f0f4f8;
                    border: 1px solid rgba(168, 179, 187, 0.12);
                    border-radius: 12px;
                    cursor: pointer;
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                    color: #29343a;
                    transition: all 0.2s ease;
                }
                .skydrop-sso-btn:hover {
                    background: #e1e9f0;
                    border-color: rgba(168, 179, 187, 0.25);
                }

                /* Footer */
                .skydrop-footer {
                    width: 100%;
                    background: #f7f9fc;
                    border-top: 1px solid rgba(168, 179, 187, 0.15);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: space-between;
                    padding: 24px 32px;
                    gap: 16px;
                }
                @media (min-width: 768px) {
                    .skydrop-footer {
                        flex-direction: row;
                        padding: 28px 48px;
                        gap: 0;
                    }
                }
                .skydrop-footer-copy span {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #717c84;
                }
                .skydrop-footer-nav {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 28px;
                }
                .skydrop-footer-nav a {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    color: #717c84;
                    text-decoration: none;
                    transition: color 0.2s;
                }
                .skydrop-footer-nav a:hover {
                    color: #4c56af;
                }
            `}</style>
        </div>
    )
}

export default LoginPage
