import React from 'react'

function AppInfoPage() {
    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">ℹ️ App Info</h1>
                <p className="page-subtitle">About PyVulnDetect and the developer</p>
            </div>

            {/* About the App */}
            <div className="glass-card info-about-card">
                <div className="info-app-header">
                    <div className="info-app-icon">🛡️</div>
                    <div>
                        <h2 className="info-app-name">PyVulnDetect</h2>
                        <span className="info-app-version">Version 1.0.0</span>
                    </div>
                </div>
                <p className="info-app-desc">
                    PyVulnDetect is an AI-powered Python security scanner that uses Graph Neural Networks (GNN) 
                    and Abstract Syntax Tree (AST) analysis to detect vulnerabilities in Python code. 
                    It identifies critical security issues like SQL injection, command injection, 
                    cross-site scripting (XSS), path traversal, and more — helping developers write safer code.
                </p>
                <div className="info-features">
                    <div className="info-feature">
                        <span className="info-feature-icon">🔬</span>
                        <div>
                            <h4 className="info-feature-title">Deep Analysis</h4>
                            <p className="info-feature-desc">Multi-layer scanning with AST, GNN, and taint analysis</p>
                        </div>
                    </div>
                    <div className="info-feature">
                        <span className="info-feature-icon">🤖</span>
                        <div>
                            <h4 className="info-feature-title">AI-Powered</h4>
                            <p className="info-feature-desc">Uses machine learning models to detect complex vulnerability patterns</p>
                        </div>
                    </div>
                    <div className="info-feature">
                        <span className="info-feature-icon">📊</span>
                        <div>
                            <h4 className="info-feature-title">Real-time Reports</h4>
                            <p className="info-feature-desc">Visual analytics with interactive charts and downloadable reports</p>
                        </div>
                    </div>
                    <div className="info-feature">
                        <span className="info-feature-icon">💬</span>
                        <div>
                            <h4 className="info-feature-title">AI Chatbot</h4>
                            <p className="info-feature-desc">Get explanations and remediation advice for vulnerabilities</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Developer Contact */}
            <div className="glass-card info-developer-card">
                <h3 className="info-section-title">👨‍💻 Creator Info</h3>
                <div className="info-dev-profile">
                    <div className="info-dev-avatar">L</div>
                    <div>
                        <h4 className="info-dev-name">Logesh D</h4>
                        <p className="info-dev-role">Full Stack Developer & Security Researcher</p>
                    </div>
                </div>

                <div className="info-links">
                    <a href="https://github.com/logesh2k26" target="_blank" rel="noopener noreferrer" className="info-link github" title="GitHub">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                        </svg>
                    </a>
                    <a href="https://www.linkedin.com/in/logesh-d-b9ba442aa/" target="_blank" rel="noopener noreferrer" className="info-link linkedin" title="LinkedIn">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                        </svg>
                    </a>
                    <a href="mailto:logesh2k25@gmail.com" className="info-link gmail" title="Gmail">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 010 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/>
                        </svg>
                    </a>
                </div>
            </div>

            {/* Tech Stack */}
            <div className="glass-card info-tech-card">
                <h3 className="info-section-title">🛠️ Technology Stack</h3>
                <div className="info-tech-grid">
                    {[
                        { name: 'React', cat: 'Frontend' },
                        { name: 'Vite', cat: 'Build Tool' },
                        { name: 'FastAPI', cat: 'Backend' },
                        { name: 'Python', cat: 'Language' },
                        { name: 'PyTorch', cat: 'ML Framework' },
                        { name: 'GNN', cat: 'Model' },
                    ].map(t => (
                        <div key={t.name} className="info-tech-item">
                            <span className="info-tech-name">{t.name}</span>
                            <span className="info-tech-cat">{t.cat}</span>
                        </div>
                    ))}
                </div>
            </div>

            <style>{`
                .info-about-card {
                    padding: 32px;
                    margin-bottom: 20px;
                }
                .info-app-header {
                    display: flex;
                    align-items: center;
                    gap: 18px;
                    margin-bottom: 20px;
                }
                .info-app-icon {
                    font-size: 48px;
                    filter: drop-shadow(0 0 16px rgba(0, 240, 255, 0.3));
                }
                .info-app-name {
                    font-family: var(--font-heading);
                    font-size: 24px;
                    font-weight: 700;
                    color: var(--text-primary);
                    margin: 0 0 4px;
                }
                .info-app-version {
                    font-size: 12px;
                    padding: 3px 10px;
                    border-radius: 8px;
                    background: rgba(0, 240, 255, 0.08);
                    color: var(--neon-cyan);
                    border: 1px solid rgba(0, 240, 255, 0.15);
                    font-weight: 600;
                }
                .info-app-desc {
                    font-size: 14px;
                    line-height: 1.7;
                    color: var(--text-secondary);
                    margin: 0 0 24px;
                }
                .info-features {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                }
                @media (max-width: 700px) {
                    .info-features { grid-template-columns: 1fr; }
                }
                .info-feature {
                    display: flex;
                    align-items: flex-start;
                    gap: 14px;
                    padding: 16px;
                    border-radius: 14px;
                    background: rgba(255, 255, 255, 0.02);
                    border: 1px solid rgba(120, 100, 255, 0.06);
                    transition: all 0.25s;
                }
                .info-feature:hover {
                    background: rgba(255, 255, 255, 0.04);
                    border-color: rgba(120, 100, 255, 0.12);
                    transform: translateY(-2px);
                }
                .info-feature-icon {
                    font-size: 24px;
                    flex-shrink: 0;
                    margin-top: 2px;
                }
                .info-feature-title {
                    font-family: var(--font-heading);
                    font-size: 14px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 4px;
                }
                .info-feature-desc {
                    font-size: 12px;
                    color: var(--text-muted);
                    margin: 0;
                    line-height: 1.5;
                }

                /* Developer Card */
                .info-developer-card {
                    padding: 28px;
                    margin-bottom: 20px;
                }
                .info-section-title {
                    font-family: var(--font-heading);
                    font-size: 16px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 20px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid rgba(120, 100, 255, 0.08);
                }
                .info-dev-profile {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    margin-bottom: 24px;
                }
                .info-dev-avatar {
                    width: 56px;
                    height: 56px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #4d7cff, #a855f7);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-family: var(--font-heading);
                    font-size: 24px;
                    font-weight: 700;
                    color: white;
                    box-shadow: 0 0 20px rgba(77, 124, 255, 0.2);
                }
                .info-dev-name {
                    font-family: var(--font-heading);
                    font-size: 18px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 4px;
                }
                .info-dev-role {
                    font-size: 13px;
                    color: var(--text-muted);
                    margin: 0;
                }

                .info-links {
                    display: flex;
                    flex-direction: row;
                    align-items: center;
                    gap: 20px;
                }
                .info-link {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 64px;
                    height: 64px;
                    border-radius: 20px;
                    text-decoration: none;
                    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
                    border: 1px solid transparent;
                }
                .info-link svg {
                    width: 32px;
                    height: 32px;
                }
                .info-link.github {
                    background: rgba(255, 255, 255, 0.05);
                    color: #e6edf3;
                }
                .info-link.github:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: rgba(255, 255, 255, 0.15);
                    transform: translateY(-6px) scale(1.08);
                    box-shadow: 0 12px 28px rgba(255, 255, 255, 0.08);
                }
                .info-link.linkedin {
                    background: rgba(10, 102, 194, 0.08);
                    color: #0a66c2;
                }
                .info-link.linkedin:hover {
                    background: rgba(10, 102, 194, 0.16);
                    border-color: rgba(10, 102, 194, 0.25);
                    transform: translateY(-6px) scale(1.08);
                    box-shadow: 0 12px 28px rgba(10, 102, 194, 0.12);
                }
                .info-link.gmail {
                    background: rgba(234, 67, 53, 0.07);
                    color: #ea4335;
                }
                .info-link.gmail:hover {
                    background: rgba(234, 67, 53, 0.14);
                    border-color: rgba(234, 67, 53, 0.22);
                    transform: translateY(-6px) scale(1.08);
                    box-shadow: 0 12px 28px rgba(234, 67, 53, 0.1);
                }

                /* Tech Stack */
                .info-tech-card {
                    padding: 28px;
                }
                .info-tech-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 12px;
                }
                @media (max-width: 600px) {
                    .info-tech-grid { grid-template-columns: repeat(2, 1fr); }
                }
                .info-tech-item {
                    padding: 14px;
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.02);
                    border: 1px solid rgba(120, 100, 255, 0.06);
                    text-align: center;
                    transition: all 0.25s;
                }
                .info-tech-item:hover {
                    background: rgba(255, 255, 255, 0.04);
                    transform: translateY(-2px);
                }
                .info-tech-name {
                    display: block;
                    font-family: var(--font-heading);
                    font-size: 14px;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin-bottom: 4px;
                }
                .info-tech-cat {
                    font-size: 11px;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                }
            `}</style>
        </div>
    )
}

export default AppInfoPage
