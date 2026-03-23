import React from 'react'

function AppInfoPage({ onNavigate }) {
    return (
        <div className="ai-container">
            {/* Top Bar with Back Button */}
            <div className="ai-topbar">
                <button className="ai-back-btn" onClick={() => onNavigate && onNavigate('settings')}>
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>
                <h2 className="ai-topbar-title">App Information</h2>
            </div>

            {/* Editorial Header */}
            <section className="ai-hero">
                <span className="ai-badge">Institutional Grade Ledger</span>
                <h1 className="ai-headline">
                    The Future of <br />
                    <span className="ai-headline-outline">Cyber Visibility.</span>
                </h1>
                <p className="ai-lead">
                    PyVulnDetect is a high-performance architectural ledger designed for the next generation of cybersecurity teams. We bridge the gap between technical complexity and human clarity.
                </p>
            </section>

            {/* Bento Grid: About & Version */}
            <div className="ai-bento">
                <div className="ai-about-card">
                    <div className="ai-about-glow" />
                    <div className="ai-about-inner">
                        <h3 className="ai-about-title">About PyVulnDetect</h3>
                        <p className="ai-about-text">
                            Founded on the principles of transparency and speed, PyVulnDetect provides a unified interface for tracking code integrity. It functions as a decentralized source of truth for security audits, ensuring that every scan and report is logged with immutable precision.
                        </p>
                    </div>
                    <div className="ai-trusted">
                        <div className="ai-avatars">
                            <div className="ai-avatar-circle" style={{ background: '#4c56af', color: '#fff' }}>L</div>
                            <div className="ai-avatar-circle" style={{ background: '#929bfa', color: '#fff' }}>D</div>
                            <div className="ai-avatar-circle ai-avatar-count">+12</div>
                        </div>
                        <span className="ai-trusted-text">Trusted by 200+ Global Security Teams</span>
                    </div>
                </div>

                <div className="ai-version-card">
                    <span className="material-symbols-outlined ai-version-icon" style={{ fontVariationSettings: "'FILL' 1" }}>verified</span>
                    <h3 className="ai-version-title">Version 1.0.0 <br/>Stability Build</h3>
                    <p className="ai-version-date">Last updated: {new Date().toLocaleDateString('en', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                </div>
            </div>

            {/* Capabilities Section */}
            <section className="ai-capabilities">
                <div className="ai-cap-header">
                    <div>
                        <h2 className="ai-cap-title">Capabilities</h2>
                        <p className="ai-cap-sub">Advanced architectural modules for data integrity.</p>
                    </div>
                    <div className="ai-cap-divider" />
                    <span className="ai-cap-count">03 CORE MODULES</span>
                </div>

                <div className="ai-cap-grid">
                    <div className="ai-cap-item">
                        <div className="ai-cap-icon-wrap">
                            <span className="material-symbols-outlined ai-cap-icon">query_stats</span>
                        </div>
                        <h4 className="ai-cap-name">Deep Analysis</h4>
                        <p className="ai-cap-desc">
                            Our proprietary engine dissects packet headers and behavioral patterns with sub-millisecond latency, surfacing threats before they reach the core.
                        </p>
                    </div>
                    <div className="ai-cap-item">
                        <div className="ai-cap-icon-wrap">
                            <span className="material-symbols-outlined ai-cap-icon">psychology</span>
                        </div>
                        <h4 className="ai-cap-name">AI-Powered</h4>
                        <p className="ai-cap-desc">
                            Leveraging custom-trained neural networks to differentiate between legitimate high-load events and distributed denial-of-service attempts.
                        </p>
                    </div>
                    <div className="ai-cap-item">
                        <div className="ai-cap-icon-wrap">
                            <span className="material-symbols-outlined ai-cap-icon">monitoring</span>
                        </div>
                        <h4 className="ai-cap-name">Real-time Reports</h4>
                        <p className="ai-cap-desc">
                            Instantaneous ledger updates across all distributed nodes. Your dashboard reflects reality within 150ms of a network state change.
                        </p>
                    </div>
                </div>
            </section>

            {/* Creator Section */}
            <section className="ai-creator">
                <div className="ai-creator-avatar">
                    <div className="ai-creator-img-placeholder">
                        <span className="ai-creator-initial">L</span>
                    </div>
                </div>
                <div className="ai-creator-info">
                    <span className="ai-creator-label">DESIGNED & BUILT BY</span>
                    <h2 className="ai-creator-name">Logesh D</h2>
                    <p className="ai-creator-quote">
                        "We aren't just building software; we are crafting the digital infrastructure that keeps our collective data secure. Precision is not optional."
                    </p>
                    <div className="ai-creator-links">
                        <a href="https://github.com/logesh2k26" target="_blank" rel="noopener noreferrer" className="ai-creator-link">
                            <span className="material-symbols-outlined">code</span>
                            <span>View GitHub</span>
                        </a>
                        <a href="https://www.linkedin.com/in/logesh-d-b9ba442aa/" target="_blank" rel="noopener noreferrer" className="ai-creator-link">
                            <span className="material-symbols-outlined">person_search</span>
                            <span>Connect on LinkedIn</span>
                        </a>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="ai-footer">
                <div className="ai-footer-brand">
                    <div className="ai-footer-logo">
                        <span className="material-symbols-outlined" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>cloud_done</span>
                    </div>
                    <span className="ai-footer-name">PyVulnDetect</span>
                </div>
                <div className="ai-footer-links">
                    <a href="#" className="ai-footer-link">Privacy Policy</a>
                    <a href="#" className="ai-footer-link">Terms of Service</a>
                    <a href="#" className="ai-footer-link">Contact Support</a>
                </div>
                <p className="ai-footer-copy">© 2024 PyVulnDetect. All rights reserved.</p>
            </footer>

            <style>{`
                .ai-container {
                    max-width: 1100px;
                    margin: 0 auto;
                    padding: 0 48px 80px;
                }

                /* Top Bar */
                .ai-topbar {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding: 16px 0 0;
                    margin-bottom: 48px;
                }
                .ai-back-btn {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: none;
                    border: none;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: background 0.15s;
                    color: #29343a;
                }
                .ai-back-btn:hover { background: #d9e4ec; }
                .ai-topbar-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                }

                /* Hero */
                .ai-hero { margin-bottom: 60px; }
                .ai-badge {
                    display: inline-block;
                    font-size: 11px;
                    font-weight: 800;
                    letter-spacing: 0.3em;
                    text-transform: uppercase;
                    color: #4c56af;
                    margin-bottom: 20px;
                }
                .ai-headline {
                    font-family: 'Manrope', sans-serif;
                    font-size: 64px;
                    font-weight: 900;
                    letter-spacing: -0.04em;
                    line-height: 0.9;
                    color: #29343a;
                    margin: 0 0 28px;
                }
                .ai-headline-outline {
                    -webkit-text-stroke: 2px #29343a;
                    -webkit-text-fill-color: transparent;
                }
                .ai-lead {
                    font-size: 18px;
                    line-height: 1.6;
                    color: #415660;
                    max-width: 600px;
                    margin: 0;
                }

                /* Bento */
                .ai-bento {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 24px;
                    margin-bottom: 60px;
                }
                @media (max-width: 768px) {
                    .ai-bento { grid-template-columns: 1fr; }
                }

                .ai-about-card {
                    background: #fff;
                    border-radius: 24px;
                    padding: 40px;
                    position: relative;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
                }
                .ai-about-glow {
                    position: absolute;
                    top: -60px;
                    right: -60px;
                    width: 200px;
                    height: 200px;
                    background: rgba(76,86,175,0.06);
                    border-radius: 50%;
                    filter: blur(60px);
                    pointer-events: none;
                }
                .ai-about-inner { position: relative; z-index: 1; }
                .ai-about-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 22px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0 0 16px;
                }
                .ai-about-text {
                    font-size: 15px;
                    line-height: 1.7;
                    color: #415660;
                    margin: 0;
                }

                .ai-trusted {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    margin-top: 32px;
                    position: relative;
                    z-index: 1;
                }
                .ai-avatars {
                    display: flex;
                    margin-left: 0;
                }
                .ai-avatar-circle {
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    border: 3px solid #fff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    font-weight: 800;
                    margin-left: -10px;
                }
                .ai-avatar-circle:first-child { margin-left: 0; }
                .ai-avatar-count {
                    background: #4c56af;
                    color: #fff;
                    font-size: 10px;
                }
                .ai-trusted-text {
                    font-size: 13px;
                    font-weight: 600;
                    color: #566168;
                }

                .ai-version-card {
                    background: #4c56af;
                    color: #f9f6ff;
                    border-radius: 24px;
                    padding: 40px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }
                .ai-version-icon {
                    font-size: 40px;
                    margin-bottom: 24px;
                    color: #ced1ff;
                }
                .ai-version-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 24px;
                    font-weight: 800;
                    line-height: 1.2;
                    margin: 0;
                }
                .ai-version-date {
                    font-size: 13px;
                    opacity: 0.7;
                    margin: 0;
                    margin-top: 24px;
                }

                /* Capabilities */
                .ai-capabilities { margin-bottom: 60px; }
                .ai-cap-header {
                    display: flex;
                    align-items: flex-end;
                    justify-content: space-between;
                    margin-bottom: 40px;
                    gap: 24px;
                }
                .ai-cap-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 28px;
                    font-weight: 900;
                    letter-spacing: -0.02em;
                    color: #29343a;
                    margin: 0;
                }
                .ai-cap-sub {
                    font-size: 14px;
                    color: #415660;
                    margin: 4px 0 0;
                }
                .ai-cap-divider {
                    flex-grow: 1;
                    height: 1px;
                    background: #e1e9f0;
                }
                .ai-cap-count {
                    font-size: 12px;
                    font-weight: 800;
                    color: #4c56af;
                    letter-spacing: 0.15em;
                    white-space: nowrap;
                }
                .ai-cap-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 40px;
                }
                @media (max-width: 768px) {
                    .ai-cap-grid { grid-template-columns: 1fr; gap: 24px; }
                }
                .ai-cap-item {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .ai-cap-icon-wrap {
                    width: 56px;
                    height: 56px;
                    border-radius: 16px;
                    background: #e1e9f0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .ai-cap-icon {
                    font-size: 28px;
                    color: #4c56af;
                }
                .ai-cap-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 18px;
                    font-weight: 800;
                    color: #29343a;
                    margin: 0;
                }
                .ai-cap-desc {
                    font-size: 13px;
                    color: #415660;
                    line-height: 1.6;
                    margin: 0;
                }

                /* Creator */
                .ai-creator {
                    background: #f0f4f8;
                    border-radius: 28px;
                    padding: 16px 32px 16px 16px;
                    display: flex;
                    gap: 40px;
                    align-items: center;
                    margin-bottom: 80px;
                }
                @media (max-width: 768px) {
                    .ai-creator { flex-direction: column; padding: 16px; }
                }
                .ai-creator-avatar {
                    width: 280px;
                    min-height: 280px;
                    flex-shrink: 0;
                    border-radius: 20px;
                    overflow: hidden;
                }
                .ai-creator-img-placeholder {
                    width: 100%;
                    height: 100%;
                    min-height: 280px;
                    background: linear-gradient(135deg, #29343a, #4c56af);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 20px;
                }
                .ai-creator-initial {
                    font-family: 'Manrope', sans-serif;
                    font-size: 80px;
                    font-weight: 900;
                    color: rgba(255,255,255,0.3);
                }
                .ai-creator-info {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    padding: 16px 0;
                }
                .ai-creator-label {
                    font-size: 10px;
                    font-weight: 900;
                    letter-spacing: 0.4em;
                    text-transform: uppercase;
                    color: #4c56af;
                    margin-bottom: 8px;
                }
                .ai-creator-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 36px;
                    font-weight: 900;
                    letter-spacing: -0.03em;
                    color: #29343a;
                    margin: 0 0 20px;
                }
                .ai-creator-quote {
                    font-size: 16px;
                    font-style: italic;
                    line-height: 1.6;
                    color: #415660;
                    margin: 0 0 28px;
                }
                .ai-creator-links {
                    display: flex;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                .ai-creator-link {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 14px 20px;
                    background: #fff;
                    border-radius: 14px;
                    text-decoration: none;
                    color: #29343a;
                    font-size: 13px;
                    font-weight: 700;
                    letter-spacing: -0.01em;
                    transition: all 0.2s;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                }
                .ai-creator-link:hover {
                    background: #f7f9fc;
                    color: #4c56af;
                }
                .ai-creator-link .material-symbols-outlined {
                    font-size: 20px;
                    transition: color 0.2s;
                }
                .ai-creator-link:hover .material-symbols-outlined {
                    color: #4c56af;
                }

                /* Footer */
                .ai-footer {
                    padding-top: 40px;
                    border-top: 1px solid #e1e9f0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 16px;
                }
                .ai-footer-brand {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .ai-footer-logo {
                    width: 24px;
                    height: 24px;
                    background: #4c56af;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #fff;
                }
                .ai-footer-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 14px;
                    font-weight: 800;
                    color: #29343a;
                }
                .ai-footer-links {
                    display: flex;
                    gap: 28px;
                }
                .ai-footer-link {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: #a8b3bb;
                    text-decoration: none;
                    transition: color 0.15s;
                }
                .ai-footer-link:hover { color: #4c56af; }
                .ai-footer-copy {
                    font-size: 12px;
                    color: #a8b3bb;
                    margin: 0;
                }
            `}</style>
        </div>
    )
}

export default AppInfoPage
